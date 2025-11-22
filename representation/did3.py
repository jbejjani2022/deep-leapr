#!/usr/bin/env python3

"""
Domain-agnostic Dynamic ID3 implementation.
"""

import json
import logging
import random
import datetime
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, TimeoutError
from concurrent.futures.process import BrokenProcessPool
from functools import cached_property
from typing import Any, Generator, Optional, Iterable, List

import numpy as np
import wandb
from tqdm import tqdm

from domain import Domain
from feature_engine import Feature, check_feature_worker
from llm_generator import generate_features


logger = logging.getLogger(__name__)


def _rename_feature_definition(code: str, new_name: str) -> str:
    """
    Rename the top-level `def feature` in a code block to `def <new_name>`.
    Assumes the first occurrence corresponds to the feature definition.
    """
    return code.replace("def feature", f"def {new_name}", 1)


def _inject_docstring_if_missing(code: str, doc: str) -> str:
    """
    Ensure the feature function has a docstring; insert one if missing.
    """
    lines = code.splitlines()
    def_idx = next((i for i, line in enumerate(lines) if line.strip().startswith("def feature")), None)
    if def_idx is None:
        return code

    insert_at = def_idx + 1
    while insert_at < len(lines) and not lines[insert_at].strip():
        insert_at += 1

    if insert_at < len(lines) and lines[insert_at].lstrip().startswith(('"""', "'''")):
        return code

    indent = " " * (len(lines[def_idx]) - len(lines[def_idx].lstrip()) + 4)
    lines.insert(insert_at, f'{indent}"""{doc}"""')
    return "\n".join(lines)


def compose_composite_code(
    composite_code: str,
    named_base_features: list[tuple[str, str]],
    docstring: Optional[str] = None,
) -> str:
    """
    Build a composite feature code block by inlining renamed base features
    followed by the composite feature definition.
    """
    helpers = []
    for fname, base_code in named_base_features:
        helpers.append(_rename_feature_definition(base_code, fname))

    body = _inject_docstring_if_missing(composite_code, docstring) if docstring else composite_code

    return "\n\n".join(helpers + [body]).strip() + "\n"


class Node:
    def __init__(self, domain: Domain, parent: Optional["Internal"] = None, feature_candidates: list[Feature] = []):
        self._domain = domain
        self._parent = parent
        self._feature_candidates = feature_candidates

    def leaves(self) -> Generator["Leaf", None, None]:
        raise NotImplementedError

    def replace(self, before: "Node", after: "Node"):
        raise NotImplementedError

    def predict(self, x: Any) -> float:
        """
        Predict the label for a given domain input (x).
        The type of x is domain-specific; callers should pass domain.input_of(dp).
        """
        raise NotImplementedError

    @property
    def parent(self) -> Optional["Internal"]:
        return self._parent

    @property
    def error(self) -> float:
        """
        Mean absolute error of predictions on this node's own training examples.
        """
        raise NotImplementedError

    @cached_property
    def weight(self) -> int:
        """
        Number of examples this node contains.
        """
        raise NotImplementedError

    def feature_candidates_from_root(self) -> list[Feature]:
        return self._feature_candidates + ([] if not self._parent else self._parent.feature_candidates_from_root())


class Leaf(Node):
    def __init__(
        self, domain: Domain, examples: list[Any], parent: Optional["Internal"] = None, feature_candidates: list[Feature] = []
    ):
        super().__init__(domain, parent, feature_candidates)
        self._examples = examples

    def leaves(self) -> Generator["Leaf", None, None]:
        yield self

    def replace(self, before: "Node", after: "Node"):
        # Leaves have no children; nothing to replace.
        pass

    def predict(self, x: Any) -> float:
        # Use domain’s notion of leaf prediction (e.g., median label).
        return self._leaf_prediction

    @cached_property
    def _leaf_prediction(self) -> float:
        # Delegate statistic to the domain.
        return float(self._domain.leaf_prediction(self._examples))

    @cached_property
    def stats(self) -> dict:
        # Generic stats using domain labels
        labels = [self._domain.label_of(dp) for dp in self._examples]
        if not labels:
            return {"n": 0, "mean": 0.0, "median": 0.0, "stdev": 0.0}
        return {
            "n": len(labels),
            "mean": float(np.mean(labels)),
            "median": float(np.median(labels)),
            "stdev": float(np.std(labels)),
        }

    @cached_property
    def error(self) -> float:
        # Delegate error computation to the domain
        return float(self._domain.leaf_error(self._examples))

    @cached_property
    def weight(self) -> int:
        return len(self._examples)


class Internal(Node):
    def __init__(
        self,
        domain: Domain,
        feature: Feature,
        threshold: float,
        left: Node,
        right: Node,
        parent: Optional["Internal"] = None,
        feature_candidates: list[Feature] = []
    ):
        super().__init__(domain, parent, feature_candidates)
        self._feature = feature
        self._threshold = threshold
        self._left = left
        self._left._parent = self
        self._right = right
        self._right._parent = self

    def leaves(self) -> Generator["Leaf", None, None]:
        yield from self._left.leaves()
        yield from self._right.leaves()

    def replace(self, before: "Node", after: "Node"):
        if self._left is before:
            self._left = after
            self._left._parent = self
        elif self._right is before:
            self._right = after
            self._right._parent = self

    def predict(self, x: Any) -> float:
        # Feature returns a scalar (or list -> take first), domain controls the input object shape.
        val = self._feature.execute(x)
        v = val[0] if isinstance(val, list) else val
        return (
            self._left.predict(x)
            if float(v) <= self._threshold
            else self._right.predict(x)
        )

    @property
    def error(self) -> float:
        # Weighted error of children
        n = self.weight
        return self._left.error * (self._left.weight / n) + self._right.error * (
            self._right.weight / n
        )

    @cached_property
    def weight(self) -> int:
        return self._left.weight + self._right.weight


class DynamicID3:
    """
    Domain-agnostic Dynamic ID3:
      - proposes new candidate features via an LLM prompt (domain formats the prompt),
      - finds best splits using domain.best_split_for_feature,
      - grows a decision tree while new splits reduce validation error.
    """

    def __init__(
        self,
        model: str,
        max_nodes: int = 1000,
        max_depth: int = 50,
        min_samples_split: int = 10,
        n_proposals: int = 10,
        n_examples: int = 10,
        min_side_ratio: float = 0.0,
        max_feature_test_examples: int = 64,
        feature_timeout_s: float = 30.0,
        enable_composites: bool = False,
        max_composites_per_leaf: int = 3,
        composite_max_base_features: int = 4,
        composite_n_global_features: int = 5,
        composite_example_count: int = 5,
    ):
        self._model = model
        self._max_nodes = max_nodes
        self._max_depth = max_depth
        self._min_samples_split = min_samples_split
        self._n_proposals = n_proposals
        self._min_side_ratio = min_side_ratio
        self._n_examples = n_examples
        self._max_feature_test_examples = max_feature_test_examples
        self._feature_timeout_s = feature_timeout_s
        self._enable_composites = enable_composites
        self._max_composites_per_leaf = max_composites_per_leaf
        self._composite_max_base_features = composite_max_base_features
        self._composite_n_global_features = composite_n_global_features
        self._composite_example_count = composite_example_count

        # Track composite feature lifecycle for logging/debugging.
        self._composite_stats = {"generated": 0, "validated": 0, "used": 0}

    def learn_features(
        self,
        domain,
        training_set: list[Any],
        validation_set: list[Any],
    ) -> list[str]:
        """
        Learn a representation by dynamically building a decision tree.
        Returns a list of feature codes (strings).
        """

        # Checkpoint file name: let the domain brand it if available.
        ts = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
        model_id = self._model.split('/')[-1]
        ckpt_name = f"did3_checkpoint_{ts}-{domain.domain_name()}__{model_id}.json"
        logger.info(f"Checkpoint file: {ckpt_name}")

        # Initialize tree with a single leaf node
        dt: Node = Leaf(domain, training_set)

        used_features: list[Feature] = []
        all_features: list[Feature] = []

        def error_on(node: Node, data: list[Any]) -> float:
            # Generic average error on a dataset: compare prediction vs. domain label
            if not data:
                return 0.0
            avg_err = 0.0
            for dp in data:
                x = domain.input_of(dp)
                y = domain.label_of(dp)
                pred = node.predict(x)
                avg_err += domain.prediction_error(pred, y) / len(data)
            return avg_err

        attempts = 0
        progress = True

        while attempts < self._max_nodes and progress:
            candidates = list(dt.leaves())
            candidates.sort(key=lambda n: n.error * n.weight, reverse=True)
            progress = False

            for node in candidates:
                if node.weight < self._min_samples_split:
                    continue
                attempts += 1

                feature_test_set = training_set[: min(10_000, len(training_set))]

                # === 1) Propose new primitive features using LLM ===
                proposals = self._propose_features(
                    domain=domain,
                    node=node,
                    n_examples=self._n_examples,
                    n_proposals=self._n_proposals,
                    feature_test_set=feature_test_set,
                )
                if not proposals:
                    logger.warning(
                        "No valid feature proposals generated, skipping this node."
                    )
                    continue

                all_features.extend(proposals)

                # === 1b) Optionally propose composite features built from existing ones ===
                composites: list[Feature] = []
                if self._enable_composites and self._max_composites_per_leaf > 0:
                    composites = self._propose_composites(
                        domain=domain,
                        node=node,
                        used_features=used_features,
                        all_features=all_features,
                        feature_test_set=feature_test_set,
                    )
                    if composites:
                        logger.info(
                            f"Composite features validated for this leaf: {len(composites)} "
                            f"(generated so far: {self._composite_stats['generated']}, "
                            f"validated so far: {self._composite_stats['validated']})"
                        )
                    all_features.extend(composites)

                candidates = proposals + composites + node.feature_candidates_from_root()

                # === 2) Pick the best split among *all* discovered features so far on the path to the root ===
                # NOTE: We can use all features here instead; it works and typically gives better models,
                # but it is significantly more expensive especially in longer runs.
                split = self._find_best_split(
                    domain=domain,
                    features=candidates,
                    node=node,
                    min_side_ratio=self._min_side_ratio,
                )

                feature, threshold, left_data, right_data = split
                if not feature:
                    logger.info("No good split for this candidate.")
                    continue

                # Grow the tree
                left, right = Leaf(domain, left_data), Leaf(domain, right_data)
                new_node = Internal(
                    domain,
                    feature,
                    threshold,
                    left=left,
                    right=right,
                    parent=node.parent,
                    feature_candidates=proposals + composites,
                )

                logger.info(f"Stats before split: {node.stats}")
                logger.info(f"Stats left: {left.stats}, right: {right.stats}")
                used_features.append(feature)
                if feature.kind == "composite":
                    self._composite_stats["used"] += 1

                # Splice into the tree
                if node.parent is None:
                    dt = new_node
                else:
                    node.parent.replace(node, new_node)

                error_val = error_on(dt, validation_set)
                error_delta = error_val - error_on(node, validation_set)
                wandb.log({"error": error_val, "error_delta": error_delta})

                logger.info(
                    f"Error on validation set after split: {error_val:.4f} (delta {error_delta:+.4f})"
                )

                try:
                    with open(ckpt_name, "w") as f:
                        json.dump(
                            {
                                "used_features": [f.code for f in used_features],
                                "all_features": [f.code for f in all_features],
                            },
                            f,
                        )
                except Exception as e:
                    logger.warning(f"Failed to write checkpoint {ckpt_name}: {e}")

                progress = True
                break

        if self._enable_composites:
            logger.info(
                "Composite feature stats: generated=%d validated=%d used=%d",
                self._composite_stats["generated"],
                self._composite_stats["validated"],
                self._composite_stats["used"],
            )

        return [f.code for f in used_features]

    def _find_best_split(
        self,
        domain: Domain,
        features: list[Feature],
        node: Node,
        min_side_ratio: float,
    ) -> tuple[Optional[Feature], float, list[Any], list[Any]]:
        """
        Iterate over candidate features and let the Domain compute the best split
        for each, then choose the split with the lowest error.
        """
        best = (
            None,
            float("inf"),
            [],
            [],
        )  # type: tuple[Optional[Feature], float, list[Any], list[Any]]
        best_err = float("inf")
        examples = getattr(node, "_examples", None)
        if examples is None:
            return best

        for feat in tqdm(features, f'Attempting to split on each existing feature ({len(examples)} examples in this leaf)...'):
            # Use domain's specialized splitter (e.g., RunningMedianAbs in chess).
            feat_, thr, left, right, err = domain.best_split_for_feature(
                examples, feat, min_side_ratio
            )
            if feat_ is not None and err < best_err:
                best = (feat_, thr, left, right)
                best_err = err

        return best

    def _validate_feature_code(
        self,
        code: str,
        domain: Domain,
        validation_inputs: list[Any],
        timeout_s: Optional[float] = None,
    ) -> bool:
        """Run a candidate feature in a worker and ensure it produces finite outputs."""
        timeout = timeout_s if timeout_s is not None else self._feature_timeout_s
        ctx = mp.get_context("spawn")
        try:
            with ProcessPoolExecutor(max_workers=1, mp_context=ctx) as ex:
                fut = ex.submit(check_feature_worker, code, validation_inputs, domain)
                fut.result(timeout=timeout)
            return True
        except Exception as e:
            if isinstance(e, TimeoutError):
                logger.warning("Feature execution timed out after %.1f seconds", timeout)
            else:
                logger.info(f"Error executing feature: {e}")
            return False

    def _propose_features(
        self,
        domain: Domain,
        node: Leaf,
        n_examples: int,
        n_proposals: int,
        feature_test_set: list[Any],
        timeout_s: Optional[float] = None,
    ) -> list[Feature]:
        """
        Call LLM to produce k candidate features for splitting the given leaf.
        """

        split_context = self._format_subtree_path(node)
        example_slice = random.sample(
            node._examples, min(n_examples, len(node._examples))
        )

        prompt = domain.format_split_prompt(
            n_output_features=n_proposals,
            examples=example_slice,
            split_context=split_context,
        )

        proposals: list[str] = generate_features(self._model, prompt)
        if not feature_test_set:
            validation_input_data = []
        else:
            if len(feature_test_set) > self._max_feature_test_examples:
                sampled_test_set = random.sample(
                    feature_test_set, self._max_feature_test_examples
                )
            else:
                sampled_test_set = feature_test_set
            validation_input_data = [domain.input_of(x) for x in sampled_test_set]
        tested_features = []

        timeout = timeout_s if timeout_s is not None else self._feature_timeout_s

        for code in proposals:
            if self._validate_feature_code(code, domain, validation_input_data, timeout):
                feature = Feature(code, domain, kind="primitive")
                tested_features.append(feature)

        for f in tested_features:
            logger.info(f"Working feature:\n{f.code}")

        return tested_features

    def _features_on_path(self, node: Node) -> list[Feature]:
        """Collect features used along the path from root to this node (ordered root->leaf)."""
        feats: list[Feature] = []
        cursor = node
        while cursor.parent is not None:
            feats.append(cursor.parent._feature)
            cursor = cursor.parent
        return list(reversed(feats))

    def _select_base_features_for_composites(
        self, node: Leaf, used_features: list[Feature]
    ) -> list[Feature]:
        """
        Select base features that composites can depend on: path features + a few global ones.
        """
        selected: list[Feature] = []
        seen: set[str] = set()

        # Prioritize features along the path to this leaf.
        for f in self._features_on_path(node):
            if f.code not in seen:
                selected.append(f)
                seen.add(f.code)

        # Add a few globally used features (most recent first) up to the configured cap.
        global_budget = self._composite_n_global_features
        for f in reversed(used_features):
            if global_budget <= 0:
                break
            if f.code in seen:
                continue
            selected.append(f)
            seen.add(f.code)
            global_budget -= 1

        # Enforce max base features per composite prompt.
        return selected[: self._composite_max_base_features]

    def _sample_rows_for_composite_prompt(
        self,
        domain: Domain,
        node: Leaf,
        named_features: list[tuple[str, Feature]],
    ) -> list[dict[str, Any]]:
        """
        Take a small sample of examples from the leaf and record base feature values + labels.
        """
        if not node._examples:
            return []
        rows = []
        examples = random.sample(
            node._examples, min(self._composite_example_count, len(node._examples))
        )
        for ex in examples:
            values = {}
            x = domain.input_of(ex)
            for name, feat in named_features:
                try:
                    v = feat.execute(x)
                    scalar = v[0] if isinstance(v, list) else v
                    values[name] = float(scalar)
                except Exception:
                    values[name] = "err"
            rows.append(
                {
                    "label": domain.label_of(ex),
                    "values": values,
                }
            )
        return rows

    def _format_composite_prompt(
        self,
        named_features: list[tuple[str, Feature]],
        example_rows: list[dict[str, Any]],
        n_output_features: int,
    ) -> str:
        """
        Build a prompt asking the LLM to propose composite features using only named base features.
        """
        base_desc = "\n".join(
            f"- {name}: {feat.description or 'no description'}"
            for name, feat in named_features
        )

        example_lines = []
        for i, row in enumerate(example_rows, start=1):
            vals = " ".join(f"{k}={v}" for k, v in row["values"].items())
            example_lines.append(f"{i}) label={row['label']} {vals}")
        example_block = "\n".join(example_lines) if example_lines else "<no examples>"

        allowed_ops = (
            "sums, differences, products, ratios (use +1e-6 to avoid div/0), "
            "min/max, abs, ReLU (max(0, x)), and safe log (log(x + 1e-6))."
        )

        return f"""
You are helping a decision-tree learner create COMPOSITE scalar features.
Do NOT read raw inputs. You may ONLY call the provided base feature helpers below.

Base helpers (call them as functions on the same input x):
{base_desc or '<none>'}

Examples from this leaf (label then helper outputs):
{example_block}

Requirements:
- Generate at most {n_output_features} Python functions.
- Each must be 'def feature(x): ...' and return a single float.
- Use ONLY the helpers above and simple arithmetic ({allowed_ops})
- Include a short docstring that explains the combination of helper features.
- Avoid division by zero or log of non-positive values by adding small epsilons.
"""

    def _propose_composites(
        self,
        domain: Domain,
        node: Leaf,
        used_features: list[Feature],
        all_features: list[Feature],
        feature_test_set: list[Any],
    ) -> list[Feature]:
        """
        Propose composite features built from existing features (path + global) at this leaf.
        """
        if not self._enable_composites:
            return []
        base_features = self._select_base_features_for_composites(node, used_features or all_features)
        if not base_features:
            return []

        named_features: list[tuple[str, Feature]] = []
        for i, feat in enumerate(base_features):
            if len(named_features) >= self._composite_max_base_features:
                break
            named_features.append((f"f{i}", feat))

        example_rows = self._sample_rows_for_composite_prompt(domain, node, named_features)
        prompt = self._format_composite_prompt(
            named_features, example_rows, self._max_composites_per_leaf
        )

        composite_codes: list[str] = generate_features(self._model, prompt)
        self._composite_stats["generated"] += len(composite_codes)

        if not feature_test_set:
            validation_input_data = []
        else:
            if len(feature_test_set) > self._max_feature_test_examples:
                sampled_test_set = random.sample(
                    feature_test_set, self._max_feature_test_examples
                )
            else:
                sampled_test_set = feature_test_set
            validation_input_data = [domain.input_of(x) for x in sampled_test_set]

        # Build dependency mapping for inlining base helpers.
        named_base_codes: list[tuple[str, str]] = [
            (name, feat.code) for name, feat in named_features
        ]

        composites: list[Feature] = []
        for raw_code in composite_codes:
            doc = f"Composite of {', '.join(name for name, _ in named_features)}"
            full_code = compose_composite_code(raw_code, named_base_codes, doc)
            if self._validate_feature_code(full_code, domain, validation_input_data):
                self._composite_stats["validated"] += 1
                composites.append(
                    Feature(
                        full_code,
                        domain,
                        kind="composite",
                        parents=[f.code for _, f in named_features],
                    )
                )

        return composites

    def _format_subtree_path(self, node: Leaf) -> str:
        """
        Format path from root to the leaf.
        """
        subtree_path = []

        while node.parent is not None:
            sign = "<" if node.parent._left is node else ">"
            subtree_path.append(
                f'value {sign} {node.parent._threshold:.3f} for "{node.parent._feature.description}" '
            )
            node = node.parent

        subtree_path.append("[root]\n")
        return " -> ".join(reversed(subtree_path))
