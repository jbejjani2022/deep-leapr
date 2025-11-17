#!/usr/bin/env python3

"""
Implementation of the Text Regression Domain abstraction.
"""

from typing import Any, Optional, List
import math
import random

from trainer.random_forest import RandomForestTrainer

from . import Domain
from text_sample import TextSample, load_text_data
from feature_engine import Feature
from prompt_builder import load_prompt_template, format_text_api_description

DataPoint = TextSample


class TextRegression(Domain):
    def __init__(self):
        self._split_prompt_template = load_prompt_template(
            "prompts/text_regression_split.txt"
        )
        self._funsearch_prompt_template = load_prompt_template(
            "prompts/text_regression_funsearch.txt"
        )

    def domain_name(self) -> str:
        return "text_regression"

    def load_dataset(self, path: str, max_size: int) -> list[DataPoint]:
        # load_text_data returns a tuple (texts, descriptions)
        # For regression, descriptions will be None
        return load_text_data(path, task_type="regression")[0][:max_size]

    def format_split_prompt(
        self,
        n_output_features: int,
        examples: list[Any],
        split_context: Optional[str],
    ) -> str:
        api = format_text_api_description()

        def format_sample_basic(sample: TextSample) -> str:
            text_preview = (
                sample.text[:200] + "..." if len(sample.text) > 200 else sample.text
            )
            return f"Sample: '{text_preview}', (Target: {sample.target:.3f})"

        examples_str = "\n".join([format_sample_basic(ex) for ex in examples])

        return self._split_prompt_template.format(
            api_description=api,
            subtree_path=split_context,
            examples=examples_str,
            num_features=n_output_features,
        )

    def format_funsearch_prompt(
        self,
        n_output_features: int,
        existing_features_with_importances: list[tuple[Feature, float]],
    ) -> str:
        api = format_text_api_description()

        def format_features_with_importances(f: Feature, importance: float) -> str:
            return f"Feature:\n{f.code}\nImportance: {importance:.3f}\n---\n"

        features_str = (
            "\n\n".join(
                [
                    format_features_with_importances(f, imp)
                    for f, imp in existing_features_with_importances
                ]
            )
            if existing_features_with_importances
            else "<No features yet>"
        )

        return self._funsearch_prompt_template.format(
            api_description=api,
            num_features=n_output_features,
            features=features_str,
        )

    def leaf_prediction(self, datapoints: list[DataPoint]) -> float:
        if not datapoints:
            return 0.0

        # Return mean of target values for regression
        targets = [dp.target for dp in datapoints]
        return sum(targets) / len(targets)

    def leaf_error(self, datapoints: list[DataPoint]) -> float:
        n = len(datapoints)
        if n <= 1:
            return 0.0

        pred = self.leaf_prediction(datapoints)
        # Mean absolute error from the leaf prediction
        return sum(abs(dp.target - pred) for dp in datapoints) / n

    def code_execution_namespace(self) -> dict[str, Any]:
        import re

        return {
            "math": math,
            "random": random,
            "re": re,
            "len": len,
            "str": str,
            "TextSample": TextSample,
        }

    def best_split_for_feature(
        self,
        examples: list[DataPoint],
        feature: Feature,
        min_side_ratio: float,
    ) -> tuple[Optional[Feature], float, list[DataPoint], list[DataPoint], float]:
        try:
            rows = []
            for sample in examples:
                vals = feature.execute(sample.text)
                v = vals[0] if isinstance(vals, list) else vals
                rows.append((float(v), sample.target, sample))
        except Exception:
            return None, math.inf, [], [], math.inf

        if not rows:
            return None, math.inf, [], [], math.inf

        # Sort by feature value
        rows.sort(key=lambda t: t[0])
        feats = [t[0] for t in rows]
        targets = [t[1] for t in rows]
        samples = [t[2] for t in rows]
        n = len(rows)
        if n <= 1:
            return None, math.inf, [], [], math.inf

        best_obj = math.inf  # weighted MAE
        best_idx = -1

        # Sweep possible split points between i and i+1
        for i in range(n - 1):
            # only consider split if feature value changes at boundary
            if feats[i] == feats[i + 1]:
                continue

            n_left = i + 1
            n_right = n - n_left

            if min(n_left, n_right) < min_side_ratio * n:
                continue

            # Calculate MAE for left and right sides
            left_targets = targets[:n_left]
            right_targets = targets[n_left:]
            
            left_mean = sum(left_targets) / n_left
            right_mean = sum(right_targets) / n_right
            
            mae_left = sum(abs(t - left_mean) for t in left_targets) / n_left
            mae_right = sum(abs(t - right_mean) for t in right_targets) / n_right
            
            # Weighted average MAE
            obj = (mae_left * n_left + mae_right * n_right) / n

            # Change best split if this one has lower MAE
            if obj < best_obj:
                best_obj = obj
                best_idx = i

        if best_idx == -1:
            return None, math.inf, [], [], math.inf

        threshold = feats[best_idx]  # left: <= threshold, right: > threshold
        left_samples = samples[: best_idx + 1]
        right_samples = samples[best_idx + 1 :]

        return feature, threshold, left_samples, right_samples, best_obj

    def input_of(self, dp: DataPoint) -> Any:
        return dp.text

    def label_of(self, dp: DataPoint) -> float:
        return dp.target

    def prediction_error(self, pred: Any, label: Any) -> float:
        return abs(float(pred) - float(label))

    def train_and_evaluate_simple_predictor(
        self,
        all_features: list[Feature],
        training_set: list[DataPoint],
        validation_set: list[DataPoint],
        training_parameters: dict[str, Any] = {},
    ) -> tuple[Any, float, float]:

        trainer = RandomForestTrainer(
            features_spec={"features": [f.code for f in all_features]},
            task_type="regression",
            domain_name=self.domain_name(),
            model_type="base_predictor",
            **training_parameters,
        )

        model, metrics = trainer.train(training_set, validation_set, None)

        return model, metrics["train"]["mae"], metrics["valid"]["mae"]

