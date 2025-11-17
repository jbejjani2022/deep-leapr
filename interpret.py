#!/usr/bin/env python3
"""SHAP interpretation module for analyzing Random Forest feature importance."""

import logging
import pickle
import json
import numpy as np
from pathlib import Path
from datetime import datetime
from typing import Tuple, Dict, List, Optional, Any

import shap

from chess_position import load_chess_data
from image_sample import load_image_data
from text_sample import load_text_data
from feature_engine import prepare_supervised_data
from main import split_dataset


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def load_model_and_features(
    learner: str, domain_dataset: str, model: str
) -> Tuple[Any, List[str], Path, Path]:
    """Load a trained model and its feature specifications.
    
    Args:
        learner: The learner method (e.g., 'did3', 'funsearch')
        domain_dataset: Combined domain and dataset name
        model: Model name (e.g., 'gpt-4o-mini')
        
    Returns:
        Tuple of (model, features_list, model_path, features_path)
        
    Raises:
        FileNotFoundError: If model or features file doesn't exist
        TypeError: If model is not a Random Forest
    """
    base_name = f"{learner}__{domain_dataset}__{model}"
    model_path = Path("results/models") / f"{base_name}.pkl"
    features_path = Path("results/features") / f"{base_name}.json"
    
    # Check model exists
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model checkpoint not found: {model_path}\n"
            f"This model has not been trained yet. Please run --train first."
        )
    
    # Check features file exists
    if not features_path.exists():
        raise FileNotFoundError(
            f"Features file not found: {features_path}\n"
            f"Cannot perform SHAP analysis without feature specifications."
        )
    
    # Load model
    logger.info(f"Loading model from {model_path}")
    with open(model_path, "rb") as f:
        loaded_model = pickle.load(f)
    
    # Extract the actual sklearn model from RFValueFunction if needed
    if hasattr(loaded_model, 'model'):
        actual_model = loaded_model.model
    else:
        actual_model = loaded_model
    
    # Verify it's a Random Forest
    from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
    if not isinstance(actual_model, (RandomForestRegressor, RandomForestClassifier)):
        raise TypeError(
            f"Model must be a Random Forest, got {type(actual_model).__name__}. "
            f"SHAP interpretation currently only supports Random Forest models."
        )
    
    # Load features
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
    return actual_model, features, model_path, features_path


def load_dataset_for_domain(
    domain: str, dataset: Optional[str], max_size: int = 100000
) -> List[Any]:
    """Load dataset based on domain type.
    
    Args:
        domain: Domain name (e.g., 'chess', 'text_classification')
        dataset: Dataset name (required for non-chess domains)
        max_size: Maximum dataset size to load
        
    Returns:
        List of data samples
    """
    logger.info(f"Loading dataset for domain: {domain}")
    
    if domain == "chess":
        all_positions = load_chess_data(["data/lichess-eval.jsonl"], max_size)
    elif domain == "image_classification":
        if not dataset:
            raise ValueError(f"Domain '{domain}' requires a dataset name")
        all_positions, *_ = load_image_data(dataset)
        if len(all_positions) > max_size:
            all_positions = all_positions[:max_size]
    elif domain == "text_classification":
        if not dataset:
            raise ValueError(f"Domain '{domain}' requires a dataset name")
        all_positions, *_ = load_text_data(dataset)
        if len(all_positions) > max_size:
            all_positions = all_positions[:max_size]
    elif domain == "text_regression":
        if not dataset:
            raise ValueError(f"Domain '{domain}' requires a dataset name")
        all_positions, *_ = load_text_data(dataset, task_type="regression")
        if len(all_positions) > max_size:
            all_positions = all_positions[:max_size]
    elif domain == "pairwise_text_classification":
        if not dataset:
            raise ValueError(f"Domain '{domain}' requires a dataset name")
        all_positions, *_ = load_text_data(dataset, task_type="pairwise_classification")
        if len(all_positions) > max_size:
            all_positions = all_positions[:max_size]
    else:
        raise ValueError(f"Unknown domain: {domain}")
    
    if not all_positions:
        raise ValueError(f"No data loaded for domain {domain}")
    
    logger.info(f"Loaded {len(all_positions)} samples")
    return all_positions


def compute_shap_analysis(
    model: Any,
    features: List[str],
    data_samples: List[Any],
    domain_name: str,
    split_name: str = "validation",
) -> Dict[str, Any]:
    """Compute SHAP values and feature importance statistics.
    
    Args:
        model: Trained sklearn Random Forest model
        features: List of feature code strings
        data_samples: List of data samples to analyze
        domain_name: Name of the domain for feature extraction
        split_name: Name of the data split being analyzed
        
    Returns:
        Dictionary containing SHAP analysis results
    """
    logger.info(f"Preparing feature matrix for {len(data_samples)} samples...")
    X, y = prepare_supervised_data(features, data_samples, domain_name)
    
    if len(X) == 0:
        raise ValueError(f"No valid samples in dataset split '{split_name}'")
    
    logger.info(f"Feature matrix shape: {X.shape}")
    logger.info(f"Computing SHAP values using TreeExplainer...")
    
    # Create SHAP explainer for tree-based models
    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(X)
    
    # Handle multi-output case (classification with multiple classes)
    if isinstance(shap_values, list):
        # For multi-class, take the mean absolute across classes
        logger.info(f"Multi-class output detected, averaging across {len(shap_values)} classes")
        shap_values = np.mean(np.abs(shap_values), axis=0)
    
    logger.info(f"SHAP values shape: {shap_values.shape}")
    
    # Compute feature importance metrics
    feature_importance = []
    for feature_idx in range(len(features)):
        feature_shap = shap_values[:, feature_idx]
        
        importance = {
            "feature_index": feature_idx,
            "feature_name": f"feature_{feature_idx}",
            "feature_code": features[feature_idx],
            "shap_statistics": {
                "mean_abs_shap": float(np.mean(np.abs(feature_shap))),
                "mean_shap": float(np.mean(feature_shap)),
                "std_shap": float(np.std(feature_shap)),
                "min_shap": float(np.min(feature_shap)),
                "max_shap": float(np.max(feature_shap)),
            }
        }
        feature_importance.append(importance)
    
    # Sort by mean absolute SHAP value (highest to lowest)
    feature_importance.sort(key=lambda x: x["shap_statistics"]["mean_abs_shap"], reverse=True)
    
    # Add ranks
    for rank, feature in enumerate(feature_importance, start=1):
        feature["rank"] = rank
    
    logger.info(f"SHAP analysis complete. Top feature has mean|SHAP| = {feature_importance[0]['shap_statistics']['mean_abs_shap']:.6f}")
    
    return {
        "dataset_info": {
            "split": split_name,
            "num_samples": len(data_samples),
            "feature_matrix_shape": list(X.shape),
        },
        "feature_importance": feature_importance,
        "shap_metadata": {
            "explainer_type": "TreeExplainer",
            "num_features": len(features),
        }
    }


def generate_shap_report(
    learner: str,
    domain_dataset: str,
    model: str,
    split: str = "validation",
    val_ratio: float = 0.05,
    eval_ratio: float = 0.05,
    random_state: int = 42,
) -> Path:
    """Generate a SHAP interpretation report for a trained model.
    
    Args:
        learner: Learner method name
        domain_dataset: Combined domain and dataset name
        model: Model name
        split: Which data split to analyze ('train', 'validation', or 'eval')
        val_ratio: Validation set ratio for splitting
        eval_ratio: Evaluation set ratio for splitting
        random_state: Random seed for reproducible splits
        
    Returns:
        Path to the generated report file
    """
    logger.info("=" * 80)
    logger.info("SHAP Interpretation Analysis")
    logger.info("=" * 80)
    
    # Parse domain and dataset
    from launch import parse_domain_dataset
    domain, dataset = parse_domain_dataset(domain_dataset)
    
    # Load model and features
    try:
        rf_model, features, model_path, features_path = load_model_and_features(
            learner, domain_dataset, model
        )
    except (FileNotFoundError, TypeError) as e:
        logger.error(str(e))
        raise
    
    # Load dataset
    all_data = load_dataset_for_domain(domain, dataset)
    
    # Split dataset (same way as training)
    train_data, val_data, eval_data = split_dataset(
        all_data, val_ratio=val_ratio, eval_ratio=eval_ratio, random_state=random_state
    )
    
    # Select the appropriate split
    split_map = {
        "train": train_data,
        "validation": val_data,
        "eval": eval_data,
    }
    
    if split not in split_map:
        raise ValueError(f"Invalid split '{split}'. Must be one of: {list(split_map.keys())}")
    
    analysis_data = split_map[split]
    logger.info(f"Analyzing {split} split with {len(analysis_data)} samples")
    
    # Compute SHAP analysis
    shap_results = compute_shap_analysis(
        rf_model, features, analysis_data, domain, split
    )
    
    # Construct full report
    report = {
        "model_info": {
            "learner": learner,
            "domain": domain_dataset,
            "model": model,
            "checkpoint_path": str(model_path),
            "features_path": str(features_path),
            "analysis_date": datetime.now().isoformat(),
        },
        **shap_results,
    }
    
    # Save report
    output_dir = Path("results/shap")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    base_name = f"{learner}__{domain_dataset}__{model}"
    report_path = output_dir / f"{base_name}.json"
    
    logger.info(f"Saving SHAP report to {report_path}")
    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)
    
    # Print summary
    logger.info("=" * 80)
    logger.info("Top 10 Most Important Features (by mean |SHAP|):")
    logger.info("=" * 80)
    for feature in report["feature_importance"][:10]:
        rank = feature["rank"]
        mean_abs = feature["shap_statistics"]["mean_abs_shap"]
        code_preview = feature["feature_code"][:80].replace("\n", " ")
        logger.info(f"  #{rank:2d}: {mean_abs:.6f} - {code_preview}...")
    
    logger.info("=" * 80)
    logger.info(f"Full report saved to: {report_path}")
    logger.info("=" * 80)
    
    return report_path

