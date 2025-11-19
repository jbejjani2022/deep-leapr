#!/usr/bin/env python3

"""
Implementation of the Text Regression Domain abstraction.
"""

from typing import Any, Optional, List
import math
import random
import statistics
import string
import unicodedata
from collections import Counter, defaultdict
import itertools
import logging
from threading import Lock

import numpy as np

from trainer.random_forest import RandomForestTrainer

from . import Domain
from text_sample import TextSample, load_text_data
from feature_engine import Feature
from prompt_builder import (
    load_prompt_template,
    format_text_api_description,
    format_text_api_description_plus,
    format_text_api_description_expert,
)

# Import expert-level NLP libraries at module level for multiprocessing compatibility
logger = logging.getLogger(__name__)

try:
    import textstat
    TEXTSTAT_AVAILABLE = True
    logger.info("✓ textstat imported successfully")
except ImportError as e:
    logger.error(f"✗ textstat not available: {e}")
    textstat = None
    TEXTSTAT_AVAILABLE = False

try:
    import spacy
    SPACY_AVAILABLE = True
    logger.info("✓ spacy imported successfully")
except ImportError as e:
    logger.error(f"✗ spacy not available: {e}")
    spacy = None
    SPACY_AVAILABLE = False

try:
    from nltk.sentiment import SentimentIntensityAnalyzer
    NLTK_SENTIMENT_AVAILABLE = True
    logger.info("✓ NLTK sentiment imported successfully")
except (ImportError, LookupError) as e:
    logger.error(f"✗ NLTK sentiment not available: {e}")
    SentimentIntensityAnalyzer = None
    NLTK_SENTIMENT_AVAILABLE = False

try:
    from textblob import TextBlob
    TEXTBLOB_AVAILABLE = True
    logger.info("✓ TextBlob imported successfully")
except ImportError as e:
    logger.error(f"✗ TextBlob not available: {e}")
    TextBlob = None
    TEXTBLOB_AVAILABLE = False

logger.info(f"Expert libraries status: textstat={TEXTSTAT_AVAILABLE}, spacy={SPACY_AVAILABLE}, nltk={NLTK_SENTIMENT_AVAILABLE}, textblob={TEXTBLOB_AVAILABLE}")


class _LazySpacyModel:
    """
    Lightweight wrapper that loads the spaCy pipeline on first use.
    This avoids paying the model load cost for feature functions
    that never touch spaCy while still presenting an `nlp` callable.
    """

    def __init__(self, loader):
        self._loader = loader
        self._model = None
        self._lock = Lock()

    def _model_or_load(self):
        if self._model is None:
            with self._lock:
                if self._model is None:
                    self._model = self._loader()
        return self._model

    def __call__(self, *args, **kwargs):
        return self._model_or_load()(*args, **kwargs)

    def __getattr__(self, item):
        return getattr(self._model_or_load(), item)

    def __repr__(self):
        model = self._model
        if model is None:
            return "<LazySpacyModel pending load>"
        return repr(model)


def _load_spacy_model():
    if spacy is None:
        raise RuntimeError("spaCy is not available to load models.")
    from spacy.tokens import Token  # type: ignore

    try:
        model = spacy.load("en_core_web_sm")
        logger.info("✓ spaCy model 'en_core_web_sm' loaded")
    except OSError:
        logger.warning(
            "spaCy model 'en_core_web_sm' not found; falling back to blank 'en'. "
            "Run `python -m spacy download en_core_web_sm` for best results."
        )
        model = spacy.blank("en")
        if "sentencizer" not in model.pipe_names:
            model.add_pipe("sentencizer")

    def _estimate_syllables_fallback(word: str) -> int:
        if not word:
            return 0
        vowels = set("aeiouy")
        word_lower = word.lower()
        count = 0
        prev_is_vowel = False
        for ch in word_lower:
            is_vowel = ch in vowels
            if is_vowel and not prev_is_vowel:
                count += 1
            prev_is_vowel = is_vowel
        if word_lower.endswith("e") and count > 1:
            count -= 1
        return max(count, 1)

    def _syllable_getter(token):
        text = token.text
        if TEXTSTAT_AVAILABLE and textstat is not None:
            try:
                syllables = textstat.syllable_count(text)
                if syllables and syllables > 0:
                    return syllables
            except Exception:
                pass
        return _estimate_syllables_fallback(text)

    if not Token.has_extension("syllables_count"):
        Token.set_extension("syllables_count", getter=_syllable_getter)

    return model


_LAZY_SPACY_MODEL = (
    _LazySpacyModel(_load_spacy_model) if SPACY_AVAILABLE and spacy is not None else None
)

DataPoint = TextSample


class TextRegression(Domain):
    def __init__(self, api_level: str = "basic"):
        self._split_prompt_template = load_prompt_template(
            "prompts/text_regression_split.txt"
        )
        self._funsearch_prompt_template = load_prompt_template(
            "prompts/text_regression_funsearch.txt"
        )
        self._api_level = api_level if api_level else "basic"

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
        if self._api_level == "expert":
            api = format_text_api_description_expert()
        elif self._api_level == "plus":
            api = format_text_api_description_plus()
        else:
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
        if self._api_level == "expert":
            api = format_text_api_description_expert()
        elif self._api_level == "plus":
            api = format_text_api_description_plus()
        else:
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

        # Return median of target values for regression (optimal for MAE)
        targets = [dp.target for dp in datapoints]
        return float(statistics.median(targets))

    def leaf_error(self, datapoints: list[DataPoint]) -> float:
        n = len(datapoints)
        if n <= 1:
            return 0.0

        center = self.leaf_prediction(datapoints)
        # Mean absolute error from the leaf prediction
        return sum(abs(dp.target - center) for dp in datapoints) / n

    def code_execution_namespace(self) -> dict[str, Any]:
        import re

        namespace = {
            "math": math,
            "random": random,
            "re": re,
            "len": len,
            "str": str,
            "statistics": statistics,
            "string": string,
            "unicodedata": unicodedata,
            "Counter": Counter,
            "defaultdict": defaultdict,
            "itertools": itertools,
            "np": np,
            "numpy": np,
            "TextSample": TextSample,
        }
        
        # Add expert NLP libraries if in expert mode
        if self._api_level == "expert":
            # Add textstat if available
            if TEXTSTAT_AVAILABLE and textstat is not None:
                namespace["textstat"] = textstat
            else:
                logger.warning("textstat requested in expert mode but not available")

            # Add spaCy if available (with lazy loading of the model)
            if SPACY_AVAILABLE and spacy is not None:
                if _LAZY_SPACY_MODEL is not None:
                    namespace["nlp"] = _LAZY_SPACY_MODEL
                namespace["spacy"] = spacy

            # Add NLTK sentiment if available
            if NLTK_SENTIMENT_AVAILABLE and SentimentIntensityAnalyzer is not None:
                try:
                    # Pre-instantiate VADER to avoid repeated initialization
                    sia = SentimentIntensityAnalyzer()
                    namespace["sia"] = sia  # Pre-instantiated analyzer
                    namespace["SentimentIntensityAnalyzer"] = SentimentIntensityAnalyzer
                except Exception as e:
                    logger.warning(f"Failed to instantiate SentimentIntensityAnalyzer: {e}")
            
            # Add TextBlob if available
            if TEXTBLOB_AVAILABLE and TextBlob is not None:
                namespace["TextBlob"] = TextBlob
        
        return namespace

    def best_split_for_feature(
        self,
        examples: list[DataPoint],
        feature: Feature,
        min_side_ratio: float,
    ) -> tuple[Optional[Feature], float, list[DataPoint], list[DataPoint], float]:
        # Splits are scored by weighted MAE so they stay aligned with the model's loss.
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
            
            left_center = statistics.median(left_targets)
            right_center = statistics.median(right_targets)

            mae_left = sum(abs(t - left_center) for t in left_targets) / n_left
            mae_right = sum(abs(t - right_center) for t in right_targets) / n_right
            
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
            domain_kwargs={"api_level": self._api_level},
            model_type="base_predictor",
            **training_parameters,
        )

        model, metrics = trainer.train(training_set, validation_set, None)

        return model, metrics["train"]["mae"], metrics["valid"]["mae"]

