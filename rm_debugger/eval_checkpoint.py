"""
Script to evaluate a saved decision tree model checkpoint on the Anthropic/hh-rlhf test set.

This script loads a trained decision tree model (saved as .pkl) and evaluates its performance
on the same helpful-base test set that the neural reward model was evaluated on.
"""

import pickle
import json
import logging
import numpy as np
import pandas as pd
import argparse
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from typing import Any, Optional, Tuple, Dict

from feature_engine import execute_feature
from domain import Domain
import domain.text_regression as text_regression_module
from domain.text_regression import TextRegression

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# ARGUMENTS
# =============================================================================

API_LEVEL_CHOICES = ("basic", "plus", "expert")


# python eval_checkpoint.py --model /n/holylabs/LABS/sham_lab/Users/jbejjani/deep-leapr/results/models/combo_basic_funsearch_plus_did3__text_regression_rm_helpful__gpt-4o-mini.pkl --api-level plus --features /n/holylabs/LABS/sham_lab/Users/jbejjani/deep-leapr/results/features/store/combo__basic_funsearch_plus_did3_text_regression_rm_helpful__gpt-4o-mini.json
# Dataset: Anthropic/hh-rlhf (config/data_dir=helpful-base, split=test)
# Total samples evaluated: 2354

# Performance:
#   Correct predictions: 1594
#   Incorrect predictions: 760
#   Decision Tree accuracy: 0.67715 (67.71%)
#   Neural RM accuracy (reference): 0.72621 (72.62%)
#   Difference from Neural RM: -0.04906

# Score statistics:
#   Chosen score: mean=1.576, std=1.000
#   Rejected score: mean=0.921, std=1.130
#   Score diff (chosen-rejected): mean=0.654, std=0.646

# Results saved to:
#   /n/holylabs/sham_lab/Users/jbejjani/deep-leapr/rm_debugger/dt_eval_results_combo_basic_funsearch_plus_did3__text_regression_rm_helpful__gpt-4o-mini.csv

# --------------------------------------

# python eval_checkpoint.py --model /n/holylabs/LABS/sham_lab/Users/jbejjani/deep-leapr/results/models/did3__text_regression_rm_helpful__gpt-4o-mini.pkl --api-level expert --features /n/holylabs/LABS/sham_lab/Users/jbejjani/deep-leapr/results/features/did3__text_regression_rm_helpful__gpt-4o-mini.json



# --------------------------------------

# python eval_checkpoint.py --model /n/holylabs/LABS/sham_lab/Users/jbejjani/deep-leapr/results/models/funsearch__text_regression_rm_helpful__gpt-4o-mini.pkl --api-level expert --features /n/holylabs/LABS/sham_lab/Users/jbejjani/deep-leapr/results/features/funsearch__text_regression_rm_helpful__gpt-4o-mini.json


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Evaluate a saved Random Forest checkpoint on HH-RLHF test data."
    )
    default_model = Path(
        "/n/holylabs/LABS/sham_lab/Users/jbejjani/deep-leapr/results/models/did3__text_regression_rm_helpful__gpt-4o-mini.pkl"
    )
    parser.add_argument(
        "--model",
        type=Path,
        default=default_model,
        help=f"Path to the model checkpoint (.pkl). Default: {default_model}",
    )
    parser.add_argument(
        "--features",
        type=Path,
        default=None,
        help="Optional path to the features JSON. "
        "Defaults to results/features/<model_stem>.json",
    )
    parser.add_argument(
        "--domain",
        default="text_regression",
        help="Domain name used during training (default: text_regression).",
    )
    parser.add_argument(
        "--api-level",
        type=str.lower,
        choices=API_LEVEL_CHOICES,
        default=None,
        help="API level that features expect (basic, plus, expert). "
        "If omitted the script looks for metadata in the features file.",
    )
    parser.add_argument(
        "--dataset",
        default="Anthropic/hh-rlhf",
        help="HuggingFace dataset identifier (default: Anthropic/hh-rlhf).",
    )
    parser.add_argument(
        "--data-dir",
        default="helpful-base",
        help="Dataset configuration / data_dir (default: helpful-base).",
    )
    parser.add_argument(
        "--split",
        default="test",
        help="Dataset split to evaluate (default: test).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Optional maximum number of samples to evaluate (useful for smoke tests).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional explicit CSV output path. "
        "Defaults to rm_debugger/dt_eval_results_<model_name>.csv",
    )
    return parser.parse_args()


def load_model_and_features(
    model_path: Path, features_path_override: Optional[Path] = None
) -> Tuple[Any, list[str], dict, Path, Optional[Path]]:
    """
    Load a trained model checkpoint and its corresponding features.
    
    Args:
        model_path: Path to the .pkl model file
        
    Returns:
        Tuple of (model, features_list, additional_feature_metadata, model_path, features_path_or_none)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    
    features_from_model: Optional[list[str]] = None
    feature_metadata: Dict[str, Any] = {}
    features_path: Optional[Path] = None
    
    # Extract the actual sklearn model if wrapped in RFValueFunction
    if hasattr(loaded_model, 'model'):
        actual_model = loaded_model.model
        if hasattr(loaded_model, 'features'):
            features_from_model = loaded_model.features
            logger.info(
                f"Loaded features from model wrapper ({len(features_from_model)} features)"
            )
    else:
        actual_model = loaded_model
    
    # Prefer explicit features override path over embedded features
    if features_path_override is not None:
        features_path = Path(features_path_override)
    elif features_from_model is None:
        features_path = Path("results/features") / f"{model_path.stem}.json"
    
    # If we have a path, load from disk; otherwise fall back to embedded features
    if features_path is not None:
        if not features_path.exists():
            raise FileNotFoundError(
                f"Features file not found: {features_path}\n"
                f"Cannot evaluate model without feature specifications."
            )
        
        logger.info(f"Loading features from {features_path}")
        with open(features_path, "r") as f:
            data = json.load(f)
            if "used_features" in data:
                features = data["used_features"]
                feature_metadata = {
                    k: v for k, v in data.items() if k != "used_features"
                }
            elif isinstance(data, list):
                features = data
                feature_metadata = {}
            else:
                raise ValueError(f"Invalid features file format in {features_path}")
    elif features_from_model is not None:
        features = features_from_model
        logger.info("Using features embedded in model checkpoint.")
    else:
        raise RuntimeError(
            "No features were found. Provide a --features path or ensure the model "
            "checkpoint wraps an RFValueFunction containing features."
        )
    
    logger.info(f"Loaded model with {len(features)} features")
    logger.info(f"Model type: {type(actual_model).__name__}")
    
    return actual_model, features, feature_metadata, model_path, features_path


def resolve_api_level(
    cli_level: Optional[str], feature_metadata: dict
) -> Optional[str]:
    """Determine API level from CLI or stored metadata."""
    if cli_level:
        return cli_level
    metadata_level = None
    metadata_block = feature_metadata.get("metadata")
    if isinstance(metadata_block, dict):
        metadata_level = metadata_block.get("api_level")
    if metadata_level:
        normalized = metadata_level.lower()
        if normalized not in API_LEVEL_CHOICES:
            raise ValueError(
                f"Unsupported api_level '{metadata_level}' stored in metadata. "
                f"Expected one of {API_LEVEL_CHOICES}."
            )
        logger.info(f"Detected API level '{normalized}' from features metadata.")
        return normalized
    return None


def create_domain(domain_name: str, api_level: Optional[str]) -> Domain:
    domain_name = (domain_name or "").lower()
    if domain_name != "text_regression":
        raise ValueError(
            f"Unsupported domain '{domain_name}'. "
            "Current evaluator only supports text_regression."
        )
    if not api_level:
        raise ValueError(
            "API level could not be inferred and must be provided explicitly "
            "for text_regression models. "
            "Re-run with --api-level {basic|plus|expert}."
        )
    logger.info(f"Instantiating {domain_name} domain (api_level={api_level})")
    return TextRegression(api_level=api_level)


def log_missing_expert_dependencies(api_level: str):
    if api_level != "expert":
        return
    missing = []
    if not text_regression_module.TEXTSTAT_AVAILABLE:
        missing.append("textstat")
    if not text_regression_module.SPACY_AVAILABLE:
        missing.append("spacy")
    if not text_regression_module.NLTK_SENTIMENT_AVAILABLE:
        missing.append("nltk (vader sentiment)")
    if not text_regression_module.TEXTBLOB_AVAILABLE:
        missing.append("textblob")
    if missing:
        logger.warning(
            "Expert API level selected but the following NLP libraries were not "
            "imported successfully: %s. "
            "Run setup_nlp_libraries.sh to install the missing dependencies.",
            ", ".join(missing),
        )


def compute_features(text: str, features: list[str], domain: Domain) -> list[float]:
    """
    Compute feature values for a text sample.
    
    Args:
        text: Input text string
        features: List of feature function code strings
        domain: Domain instance providing the execution namespace
        
    Returns:
        List of feature values
    """
    # Compute features - features expect raw text strings
    feature_values = []
    for feature_code in features:
        try:
            vals = execute_feature(feature_code, text, domain)
            feature_values.extend(vals)
        except Exception as e:
            logger.warning(f"Error computing feature: {e}")
            # Use a default value for failed features
            feature_values.append(-1e9)
    
    return feature_values


def get_model_score(model, features: list[str], domain: Domain, text: str) -> float:
    """
    Get the predicted reward score for a given text using the decision tree model.
    
    Args:
        model: The trained sklearn model
        features: List of feature function code strings
        domain: Domain instance for executing feature code
        text: The input text
        
    Returns:
        The predicted reward score (float)
    """
    feature_values = compute_features(text, features, domain)
    
    # Make prediction
    score = model.predict([feature_values])[0]
    
    return float(score)


def main():
    args = parse_args()
    model_path = args.model.expanduser()
    features_path_override = args.features.expanduser() if args.features else None
    print("=" * 80)
    print("DECISION TREE MODEL EVALUATION")
    print("=" * 80)
    print(f"Model checkpoint: {model_path}")
    
    # Load model and features
    try:
        (
            model,
            features,
            feature_metadata,
            resolved_model_path,
            resolved_features_path,
        ) = load_model_and_features(model_path, features_path_override)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    api_level = resolve_api_level(args.api_level, feature_metadata)
    if not api_level:
        logger.error(
            "Unable to infer API level from metadata. "
            "Please pass --api-level {basic|plus|expert}."
        )
        return
    domain = create_domain(args.domain, api_level)
    log_missing_expert_dependencies(api_level)

    print(f"\nModel type: {type(model).__name__}")
    print(f"Number of features: {len(features)}")
    if resolved_features_path is not None:
        print(f"Features file: {resolved_features_path}")
    else:
        print("Features source: embedded in model checkpoint")
    print(f"Domain: {args.domain} (api_level={api_level})")
    
    # Print first few features for reference
    print("\nFirst 3 features:")
    for i, feat in enumerate(features[:3], 1):
        # Extract docstring if available
        try:
            lines = feat.strip().split('\n')
            docstring = None
            for line in lines:
                if '"""' in line or "'''" in line:
                    docstring = line.strip().strip('"""').strip("'''")
                    break
            if docstring:
                print(f"  {i}. {docstring[:80]}...")
            else:
                print(f"  {i}. {lines[1][:80] if len(lines) > 1 else feat[:80]}...")
        except:
            print(f"  {i}. [Feature {i}]")
    
    # Load dataset
    print("\nLoading dataset...")
    print(
        f"Dataset: {args.dataset} (config/data_dir={args.data_dir}, split={args.split})"
    )
    dataset = load_dataset(args.dataset, data_dir=args.data_dir, split=args.split)
    
    total_samples = len(dataset)
    eval_samples = total_samples if args.limit is None else min(args.limit, total_samples)
    print(f"Total test samples: {total_samples}")
    if args.limit is not None:
        print(f"Evaluating first {eval_samples} samples (--limit={args.limit})")
    print(f"\nStarting evaluation of {eval_samples} samples...")
    print("=" * 80)
    
    correct_predictions = 0
    results = []
    errors = []
    
    # Use tqdm for progress tracking
    for idx in tqdm(range(eval_samples), desc="Evaluating", unit="sample"):
        sample = dataset[idx]
        ground_truth_chosen = sample['chosen']
        ground_truth_rejected = sample['rejected']
        
        try:
            # Get model scores
            gt_chosen_score = get_model_score(
                model, features, domain, ground_truth_chosen
            )
            gt_rejected_score = get_model_score(
                model, features, domain, ground_truth_rejected
            )
            
            # Check if model correctly ranks chosen > rejected
            if gt_chosen_score > gt_rejected_score:
                correct_predictions += 1
            
            # Record model's preference
            if gt_chosen_score > gt_rejected_score:
                # Model prefers ground truth chosen
                model_chosen = ground_truth_chosen
                model_chosen_score = gt_chosen_score
                model_rejected = ground_truth_rejected
                model_rejected_score = gt_rejected_score
            else:
                # Model prefers ground truth rejected
                model_chosen = ground_truth_rejected
                model_chosen_score = gt_rejected_score
                model_rejected = ground_truth_chosen
                model_rejected_score = gt_chosen_score
            
            results.append({
                'chosen': model_chosen,
                'chosen_score': model_chosen_score,
                'rejected': model_rejected,
                'rejected_score': model_rejected_score,
                'correct': gt_chosen_score > gt_rejected_score
            })
            
        except Exception as e:
            logger.error(f"Error on sample {idx}: {e}")
            errors.append({'sample_idx': idx, 'error': str(e)})
            # Still count this as incorrect
            results.append({
                'chosen': ground_truth_chosen,
                'chosen_score': 0.0,
                'rejected': ground_truth_rejected,
                'rejected_score': 0.0,
                'correct': False
            })
    
    # Save results to CSV
    model_name = resolved_model_path.stem
    if args.output:
        output_path = args.output.expanduser()
    else:
        output_file = f"dt_eval_results_{model_name}.csv"
        output_path = Path(__file__).parent / output_file
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    # Summary
    accuracy = correct_predictions / eval_samples if eval_samples else 0.0
    rm_reported_accuracy = 0.72621  # Neural RM accuracy for reference
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Model type: {type(model).__name__}")
    print(
        f"Dataset: {args.dataset} (config/data_dir={args.data_dir}, split={args.split})"
    )
    print(f"Total samples evaluated: {eval_samples}")
    
    if errors:
        print(f"\n⚠️  Errors encountered: {len(errors)} samples")
    
    print(f"\nPerformance:")
    print(f"  Correct predictions: {correct_predictions}")
    print(f"  Incorrect predictions: {total_samples - correct_predictions}")
    print(f"  Decision Tree accuracy: {accuracy:.5f} ({accuracy:.2%})")
    print(f"  Neural RM accuracy (reference): {rm_reported_accuracy:.5f} ({rm_reported_accuracy:.2%})")
    print(f"  Difference from Neural RM: {accuracy - rm_reported_accuracy:+.5f}")
    
    # Compute score statistics
    chosen_scores = [r['chosen_score'] for r in results]
    rejected_scores = [r['rejected_score'] for r in results]
    score_diffs = [c - r for c, r in zip(chosen_scores, rejected_scores)]
    
    print(f"\nScore statistics:")
    print(f"  Chosen score: mean={np.mean(chosen_scores):.3f}, std={np.std(chosen_scores):.3f}")
    print(f"  Rejected score: mean={np.mean(rejected_scores):.3f}, std={np.std(rejected_scores):.3f}")
    print(f"  Score diff (chosen-rejected): mean={np.mean(score_diffs):.3f}, std={np.std(score_diffs):.3f}")
    
    print(f"\nResults saved to:")
    print(f"  {output_path.absolute()}")
    print(f"  ({len(results)} rows, {len(df.columns)} columns)")
    print("=" * 80)
    
    if errors:
        print(f"\n⚠️  First few errors:")
        for err in errors[:5]:
            print(f"  Sample {err['sample_idx']}: {err['error']}")


if __name__ == "__main__":
    main()

