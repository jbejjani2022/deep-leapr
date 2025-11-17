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
from pathlib import Path
from tqdm import tqdm
from datasets import load_dataset

# Add parent directory to path for imports
import sys
sys.path.insert(0, str(Path(__file__).parent.parent))

from feature_engine import execute_feature

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


# =============================================================================
# CONFIGURATION - Update this path to evaluate different models
# =============================================================================
MODEL_CHECKPOINT = "/n/holylabs/LABS/sham_lab/Users/jbejjani/deep-leapr/results/models/combo__text_regression_rm_helpful__gpt-4o-mini.pkl"
# =============================================================================


def load_model_and_features(model_path: str):
    """
    Load a trained model checkpoint and its corresponding features.
    
    Args:
        model_path: Path to the .pkl model file
        
    Returns:
        Tuple of (model, features_list)
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    
    # Extract the actual sklearn model if wrapped in RFValueFunction
    if hasattr(loaded_model, 'model'):
        actual_model = loaded_model.model
        # If features are stored in the wrapper, use them
        if hasattr(loaded_model, 'features'):
            features = loaded_model.features
            logger.info(f"Loaded features from model wrapper ({len(features)} features)")
            return actual_model, features
    else:
        actual_model = loaded_model
    
    # Try to load features from corresponding JSON file
    features_path = Path("results/features") / f"{model_path.stem}.json"
    
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
        elif isinstance(data, list):
            features = data
        else:
            raise ValueError(f"Invalid features file format in {features_path}")
    
    logger.info(f"Loaded model with {len(features)} features")
    logger.info(f"Model type: {type(actual_model).__name__}")
    
    return actual_model, features


def compute_features(text: str, features: list[str]) -> list[float]:
    """
    Compute feature values for a text sample.
    
    Args:
        text: Input text string
        features: List of feature function code strings
        
    Returns:
        List of feature values
    """
    # We need to provide a namespace for feature execution but don't need
    # the full domain initialization (which loads prompt templates)
    # Create a minimal domain-like object with just the namespace
    import math
    import random
    import statistics
    import string
    import unicodedata
    from collections import Counter, defaultdict
    import itertools
    import re
    
    class MinimalDomain:
        """Minimal domain object that only provides code execution namespace."""
        def code_execution_namespace(self):
            return {
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
            }
    
    domain = MinimalDomain()
    
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


def get_model_score(model, features: list[str], text: str) -> float:
    """
    Get the predicted reward score for a given text using the decision tree model.
    
    Args:
        model: The trained sklearn model
        features: List of feature function code strings
        text: The input text
        
    Returns:
        The predicted reward score (float)
    """
    feature_values = compute_features(text, features)
    
    # Make prediction
    score = model.predict([feature_values])[0]
    
    return float(score)


def main():
    print("=" * 80)
    print("DECISION TREE MODEL EVALUATION")
    print("=" * 80)
    print(f"Model checkpoint: {MODEL_CHECKPOINT}")
    
    # Load model and features
    try:
        model, features = load_model_and_features(MODEL_CHECKPOINT)
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        return
    
    print(f"\nModel type: {type(model).__name__}")
    print(f"Number of features: {len(features)}")
    
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
    print("Dataset: Anthropic/hh-rlhf (helpful-base subset, test split)")
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="test")
    
    total_samples = len(dataset)
    print(f"Total test samples: {total_samples}")
    print(f"\nStarting evaluation of all {total_samples} samples...")
    print("=" * 80)
    
    correct_predictions = 0
    results = []
    errors = []
    
    # Use tqdm for progress tracking
    for idx, sample in enumerate(tqdm(dataset, desc="Evaluating", unit="sample")):
        ground_truth_chosen = sample['chosen']
        ground_truth_rejected = sample['rejected']
        
        try:
            # Get model scores
            gt_chosen_score = get_model_score(model, features, ground_truth_chosen)
            gt_rejected_score = get_model_score(model, features, ground_truth_rejected)
            
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
    model_name = Path(MODEL_CHECKPOINT).stem
    output_file = f"dt_eval_results_{model_name}.csv"
    output_path = Path(__file__).parent / output_file
    df = pd.DataFrame(results)
    df.to_csv(output_path, index=False)
    
    # Summary
    accuracy = correct_predictions / total_samples
    rm_reported_accuracy = 0.72621  # Neural RM accuracy for reference
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Model type: {type(model).__name__}")
    print(f"Dataset: Anthropic/hh-rlhf (helpful-base test split)")
    print(f"Total samples evaluated: {total_samples}")
    
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

