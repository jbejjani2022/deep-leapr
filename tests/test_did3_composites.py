import math
import sys
import types
import unittest
from typing import Any

# Lightweight wandb stub to allow importing without external dependency.
sys.modules.setdefault("wandb", types.SimpleNamespace(log=lambda *args, **kwargs: None))
# Minimal chess stub so feature_engine/chess_position imports succeed.
chess_stub = types.SimpleNamespace(
    pgn=types.SimpleNamespace(read_game=lambda f: None),
    Board=object,
)
sys.modules.setdefault("chess", chess_stub)
sys.modules.setdefault("chess.pgn", chess_stub.pgn)

# Stub LLM providers used by llm_generator.
class _LLMStub:
    def __init__(self, *args, **kwargs):
        pass
    def invoke(self, *_args, **_kwargs):
        class _Resp:
            content = ""
        return _Resp()

sys.modules.setdefault("langchain_anthropic", types.SimpleNamespace(ChatAnthropic=_LLMStub))
sys.modules.setdefault("langchain_openai", types.SimpleNamespace(ChatOpenAI=_LLMStub))

from representation.did3 import compose_composite_code, DynamicID3, Leaf
from feature_engine import Feature
from domain import Domain


BASE_FEATURE_0 = """def feature(x):
    return float(x)
"""

BASE_FEATURE_1 = """def feature(x):
    return float(x * x)
"""

COMPOSITE_BODY = """def feature(x):
    return f0(x) + f1(x)
"""


class TinyDomain(Domain):
    """Minimal domain for unit checks."""

    def domain_name(self) -> str:
        return "toy"

    def load_dataset(self, path: str, max_size: int):
        return []

    def format_funsearch_prompt(self, *args, **kwargs) -> str:
        return ""

    def format_split_prompt(self, *args, **kwargs) -> str:
        return ""

    def leaf_error(self, datapoints: list[Any]) -> float:
        if not datapoints:
            return 0.0
        m = self.leaf_prediction(datapoints)
        return sum(abs(dp["y"] - m) for dp in datapoints) / len(datapoints)

    def leaf_prediction(self, datapoints: list[Any]) -> float:
        if not datapoints:
            return 0.0
        labels = sorted(dp["y"] for dp in datapoints)
        mid = len(labels) // 2
        return float(labels[mid])

    def code_execution_namespace(self) -> dict[str, Any]:
        return {"math": math}

    def best_split_for_feature(
        self,
        examples: list[Any],
        feature: Feature,
        min_side_ratio: float,
    ):
        rows = []
        for sample in examples:
            vals = feature.execute(sample["x"])
            v = vals[0] if isinstance(vals, list) else vals
            rows.append((float(v), float(sample["y"]), sample))

        if not rows:
            return None, math.inf, [], [], math.inf

        rows.sort(key=lambda t: t[0])
        feats = [t[0] for t in rows]
        targets = [t[1] for t in rows]
        samples = [t[2] for t in rows]
        n = len(rows)
        if n <= 1:
            return None, math.inf, [], [], math.inf

        best_err = math.inf
        best_idx = -1
        for i in range(n - 1):
            if feats[i] == feats[i + 1]:
                continue
            left_t = targets[: i + 1]
            right_t = targets[i + 1 :]
            m_left = sum(left_t) / len(left_t)
            m_right = sum(right_t) / len(right_t)
            err_left = sum(abs(t - m_left) for t in left_t) / len(left_t)
            err_right = sum(abs(t - m_right) for t in right_t) / len(right_t)
            total_err = (err_left * len(left_t) + err_right * len(right_t)) / n
            if total_err < best_err:
                best_err = total_err
                best_idx = i

        if best_idx == -1:
            return None, math.inf, [], [], math.inf

        threshold = feats[best_idx]
        return feature, threshold, samples[: best_idx + 1], samples[best_idx + 1 :], best_err

    def input_of(self, dp: Any) -> Any:
        return dp["x"]

    def label_of(self, dp: Any) -> float:
        return dp["y"]

    def prediction_error(self, pred: Any, label: Any) -> float:
        return abs(float(pred) - float(label))

    def train_and_evaluate_simple_predictor(self, *args, **kwargs):
        return None, 0.0, 0.0


class CompositeFeatureTests(unittest.TestCase):
    def test_composite_code_executes(self):
        code = compose_composite_code(
            COMPOSITE_BODY,
            [("f0", BASE_FEATURE_0), ("f1", BASE_FEATURE_1)],
            "Composite of f0 and f1",
        )
        feat = Feature(code, None, kind="composite", parents=[BASE_FEATURE_0, BASE_FEATURE_1])
        val = feat.execute(2.0)[0]
        self.assertAlmostEqual(val, 2.0 + 4.0)

    def test_composite_visible_to_splitter(self):
        domain = TinyDomain()
        code = compose_composite_code(
            COMPOSITE_BODY,
            [("f0", BASE_FEATURE_0), ("f1", BASE_FEATURE_1)],
            "Composite of f0 and f1",
        )
        feat = Feature(code, domain, kind="composite", parents=[BASE_FEATURE_0, BASE_FEATURE_1])
        examples = [
            {"x": -3.0, "y": 0.0},
            {"x": -1.0, "y": 0.0},
            {"x": 2.0, "y": 1.0},
            {"x": 4.0, "y": 1.0},
        ]
        f, threshold, left, right, err = domain.best_split_for_feature(examples, feat, 0.0)
        self.assertIsNotNone(f)
        self.assertTrue(math.isfinite(threshold))
        self.assertGreater(len(left), 0)
        self.assertGreater(len(right), 0)
        self.assertLess(err, math.inf)

    def test_disable_composites_short_circuits(self):
        domain = TinyDomain()
        node = Leaf(domain, [])
        learner = DynamicID3(model="openai/gpt-4o-mini", enable_composites=False)
        composites = learner._propose_composites(
            domain=domain,
            node=node,
            used_features=[],
            all_features=[],
            feature_test_set=[],
        )
        self.assertEqual(composites, [])
        self.assertEqual(learner._composite_stats["generated"], 0)


if __name__ == "__main__":
    unittest.main()
