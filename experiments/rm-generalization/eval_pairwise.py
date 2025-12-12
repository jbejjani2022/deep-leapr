#!/usr/bin/env python3
"""
Compare a LeaPR text-regression model vs GPT-2 helpful reward model on multiple
pairwise preference datasets.

Datasets and field mappings are defined in datasets.yaml.
Outputs: per-dataset CSVs and a summary JSON.
"""

import argparse
import json
import logging
import os
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Iterable, Tuple

import numpy as np
import pandas as pd
import torch
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from text_regression import TextRegression
from feature_engine import execute_feature

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

# -----------------------------------------------------------------------------
# LeaPR scoring helpers
# -----------------------------------------------------------------------------

def load_leapr_model(model_path: Path):
    import pickle
    logger.info(f"Loading LeaPR model from {model_path}")
    with open(model_path, "rb") as f:
        obj = pickle.load(f)
    features = None
    rf = obj
    if hasattr(obj, "model"):
        rf = obj.model
        if hasattr(obj, "features"):
            features = obj.features
            logger.info(f"Found {len(features)} embedded features in checkpoint")
    if features is None:
        feat_path = Path("results/features") / f"{model_path.stem}.json"
        logger.info(f"Loading features from {feat_path}")
        with open(feat_path, "r") as f:
            data = json.load(f)
            if isinstance(data, dict) and "used_features" in data:
                features = data["used_features"]
            elif isinstance(data, list):
                features = data
            else:
                raise ValueError(f"Unsupported feature file format: {feat_path}")
    return rf, features


def score_leapr_text(rf_model, features: list[str], domain: TextRegression, text: str) -> float:
    vals = []
    for code in features:
        try:
            out = execute_feature(code, text, domain)
            vals.extend(out if isinstance(out, list) else [out])
        except Exception as e:
            logger.debug(f"Feature error; using fallback: {e}")
            vals.append(-1e9)
    return float(rf_model.predict([vals])[0])


# -----------------------------------------------------------------------------
# Baseline RM helpers
# -----------------------------------------------------------------------------

def load_reward_model(model_name: str = "Ray2333/gpt2-large-helpful-reward_model", device: str | None = None):
    logger.info(f"Loading reward model {model_name}")
    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
        mdl.config.pad_token_id = mdl.config.eos_token_id
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    mdl.to(device)
    mdl.eval()
    return tok, mdl, device


def score_reward_model(tokenizer, model, device: str, text: str) -> float:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        score = outputs.logits[0].item()
    return float(score)


# -----------------------------------------------------------------------------
# Dataset utilities
# -----------------------------------------------------------------------------

def load_dataset_spec(name: str, spec: Dict[str, Any], limit: int | None) -> Iterable[Tuple[str, str, str | None]]:
    ds = load_dataset(spec["hf_id"], data_dir=spec.get("data_dir"), split=spec.get("split", "train"))
    chosen_k = spec.get("chosen_key", "chosen")
    rejected_k = spec.get("rejected_key", "rejected")
    prompt_k = spec.get("prompt_key")
    n = len(ds)
    cap = min(limit, n) if limit is not None else n
    logger.info(f"Loaded {n} rows; evaluating {cap}")
    for i in range(cap):
        row = ds[i]
        chosen = row[chosen_k]
        rejected = row[rejected_k]
        prompt = row[prompt_k] if prompt_k and prompt_k in row else None
        yield chosen, rejected, prompt


# -----------------------------------------------------------------------------
# Main evaluation loop
# -----------------------------------------------------------------------------

def evaluate_dataset(name: str, pairs: Iterable[Tuple[str, str, str | None]],
                     leapr_ctx: dict, rm_ctx: dict) -> Dict[str, Any]:
    rf_model = leapr_ctx["rf"]
    features = leapr_ctx["features"]
    domain = leapr_ctx["domain"]
    tok = rm_ctx["tok"]
    mdl = rm_ctx["mdl"]
    device = rm_ctx["device"]

    rows = []
    correct_leapr = 0
    correct_rm = 0

    for idx, (chosen, rejected, prompt) in enumerate(pairs):
        l_ch = score_leapr_text(rf_model, features, domain, chosen)
        l_rj = score_leapr_text(rf_model, features, domain, rejected)
        r_ch = score_reward_model(tok, mdl, device, chosen)
        r_rj = score_reward_model(tok, mdl, device, rejected)

        rows.append({
            "idx": idx,
            "prompt": prompt,
            "chosen": chosen,
            "rejected": rejected,
            "leapr_chosen": l_ch,
            "leapr_rejected": l_rj,
            "leapr_correct": l_ch > l_rj,
            "rm_chosen": r_ch,
            "rm_rejected": r_rj,
            "rm_correct": r_ch > r_rj,
        })
        correct_leapr += int(l_ch > l_rj)
        correct_rm += int(r_ch > r_rj)

    df = pd.DataFrame(rows)
    summary = {
        "dataset": name,
        "n": len(df),
        "leapr_accuracy": float(correct_leapr) / len(df) if len(df) else 0.0,
        "rm_accuracy": float(correct_rm) / len(df) if len(df) else 0.0,
        "leapr_score_mean": float(df[["leapr_chosen", "leapr_rejected"]].values.mean()) if len(df) else 0.0,
        "rm_score_mean": float(df[["rm_chosen", "rm_rejected"]].values.mean()) if len(df) else 0.0,
    }
    return df, summary


def main():
    p = argparse.ArgumentParser(description="Evaluate LeaPR vs GPT-2 reward model on pairwise datasets")
    p.add_argument("--leapr-model", type=Path, required=True, help="Path to LeaPR RF checkpoint (.pkl)")
    p.add_argument("--leapr-api-level", default="expert", choices=["basic", "plus", "expert"], help="API level for TextRegression domain")
    p.add_argument("--datasets-config", type=Path, default=Path("experiments/rm-generalization/datasets.yaml"))
    p.add_argument("--datasets", nargs="*", help="Subset of dataset keys to run (default: all in config)")
    p.add_argument("--limit", type=int, default=None, help="Override per-dataset sample cap")
    p.add_argument("--output-root", type=Path, default=Path("experiments/rm-generalization/runs"))
    p.add_argument("--baseline-model", default="Ray2333/gpt2-large-helpful-reward_model", help="HF model id for baseline RM")
    args = p.parse_args()

    with open(args.datasets_config) as f:
        cfg = json.load(f) if args.datasets_config.suffix == ".json" else yaml_safe_load(f)
    ds_cfg = cfg.get("datasets", {})
    if args.datasets:
        ds_cfg = {k: v for k, v in ds_cfg.items() if k in args.datasets}
    assert ds_cfg, "No datasets selected"

    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    out_dir = args.output_root / ts
    out_dir.mkdir(parents=True, exist_ok=True)

    # LeaPR
    rf_model, features = load_leapr_model(args.leapr_model)
    domain = TextRegression(api_level=args.leapr_api_level)
    leapr_ctx = {"rf": rf_model, "features": features, "domain": domain}

    # Baseline RM
    tok, mdl, device = load_reward_model(args.baseline_model)
    rm_ctx = {"tok": tok, "mdl": mdl, "device": device}

    all_summaries = []
    for name, spec in ds_cfg.items():
        limit = args.limit if args.limit is not None else spec.get("limit")
        logger.info(f"\n=== Dataset: {name} ===")
        pairs = load_dataset_spec(name, spec, limit)
        df, summary = evaluate_dataset(name, pairs, leapr_ctx, rm_ctx)
        all_summaries.append(summary)

        csv_path = out_dir / f"{name}.csv"
        df.to_csv(csv_path, index=False)
        logger.info(f"Wrote {len(df)} rows to {csv_path}")

    summary_path = out_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(all_summaries, f, indent=2)
    logger.info(f"Saved summary to {summary_path}")


def yaml_safe_load(f):
    import yaml
    return yaml.safe_load(f)


if __name__ == "__main__":
    main()
