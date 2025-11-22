"""
Script to test reward model behavior with e-intervention.

This script loads a reward model and tests it on dummy data samples
with various interventions (adding 'e' at different positions).
"""

import torch
import numpy as np
from scipy import stats
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import json
import argparse
import os
import sys


def feature(text: str) -> float:
    """Count of words containing the letter 'e'"""
    e_word_count = sum(1 for word in text.split() if 'e' in word.lower())
    return float(e_word_count)

def get_reward_score(model, tokenizer, text, device):
    """
    Get the reward score for a given text.
    
    Args:
        model: The reward model
        tokenizer: The tokenizer
        text: The input text
        device: The device to run on
        
    Returns:
        The reward score (float)
    """
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=1024).to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        # The reward score is typically the logit output
        score = outputs.logits[0].item()
    return score

def main():
    parser = argparse.ArgumentParser(description="Test reward model behavior with e-intervention.")
    parser.add_argument("--data_path", type=str, default="rm_debugger/data/contrastive/verbose_vs_direct.json", 
                        help="Path to the JSON file containing exploit examples")
    parser.add_argument("--model_name", type=str, default="Ray2333/gpt2-large-helpful-reward_model",
                        help="Name or path of the reward model")
    
    args = parser.parse_args()
    
    model_name = args.model_name
    
    print("=" * 80)
    print("E-INTERVENTION EXPERIMENT")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Data File: {args.data_path}")
    
    # Load Data
    if not os.path.exists(args.data_path):
        print(f"Error: Data file not found at {args.data_path}")
        sys.exit(1)
        
    try:
        with open(args.data_path, 'r') as f:
            exploit_definitions = json.load(f)
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        sys.exit(1)
        
    print(f"Loaded {len(exploit_definitions)} examples.")

    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model and tokenizer
    print(f"\nLoading model and tokenizer from HuggingFace...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModelForSequenceClassification.from_pretrained(model_name)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model: {e}")
        sys.exit(1)
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully")
    print(f"Total parameters: {num_params:,}")
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Build dummy data from exploit definitions
    # Structure: {exploit_name: [(text, label, description), ...]}
    dummy_data = {}
    for exploit in exploit_definitions:
        if "name" not in exploit or "human" not in exploit:
            print("Skipping malformed exploit entry (missing name or human)")
            continue
            
        h = exploit["human"]
        description = exploit.get("description", "No description available")
        
        # Handle cases where good_ans/advs_ans might be missing if the json structure varies, 
        # but assuming it matches the target file structure
        if "good_ans" in exploit and "advs_ans" in exploit:
            dummy_data[exploit["name"]] = [
                (f"{h} {exploit['good_ans']}", "good_ans", description),
                (f"{h} {exploit['advs_ans']}", "advs_ans", description),
            ]
    
    print("\n" + "=" * 80)
    print("EVALUATING DUMMY DATA")
    print("=" * 80)
    
    # Storage for statistical analysis
    good_ans_scores = []
    advs_ans_scores = []
    exploit_results = []  # Store per-exploit results
    
    # Process each group of data samples
    for exploit_name, samples in dummy_data.items():
        print(f"\n{'='*80}")
        print(f"EXPLOIT: {exploit_name}")
        # Get description from first sample
        print(f"Description: {samples[0][2]}")
        print(f"{'='*80}")
        
        exploit_scores = {"good": None, "advs": None}
        
        for idx, (text, label, _) in enumerate(samples):
            # Get reward score
            score = get_reward_score(model, tokenizer, text, device)
            
            # Store scores for statistical analysis
            if label == "good_ans":
                good_ans_scores.append(score)
                exploit_scores["good"] = score
            elif label == "advs_ans":
                advs_ans_scores.append(score)
                exploit_scores["advs"] = score
            
            # Get e-word count
            e_count = feature(text)
            
            # Print results
            print(f"\n[{label}]")
            print(f"  Reward Score: {score:.6f}")
            print(f"  E-word Count: {e_count:.0f}")
            print(f"  Text Preview: {text[:100]}{'...' if len(text) > 100 else ''}")
        
        # Store per-exploit result
        if exploit_scores["good"] is not None and exploit_scores["advs"] is not None:
            exploit_results.append({
                "name": exploit_name,
                "good_score": exploit_scores["good"],
                "advs_score": exploit_scores["advs"],
                "difference": exploit_scores["advs"] - exploit_scores["good"]
            })
    
    # Perform statistical analysis
    print("\n" + "=" * 80)
    print("STATISTICAL ANALYSIS")
    print("=" * 80)
    
    if len(good_ans_scores) > 0 and len(advs_ans_scores) > 0:
        # Convert to numpy arrays
        good_scores = np.array(good_ans_scores)
        advs_scores = np.array(advs_ans_scores)
        
        # Difference statistics
        differences = advs_scores - good_scores
        mean_diff = np.mean(differences)
        std_diff = np.std(differences, ddof=1)
        
        if len(good_scores) > 1:
            # Paired t-test
            t_stat, p_value = stats.ttest_rel(advs_scores, good_scores)
            
            # Effect size (Cohen's d)
            pooled_std = np.sqrt((np.var(good_scores, ddof=1) + np.var(advs_scores, ddof=1)) / 2)
            if pooled_std == 0:
                 cohens_d = 0.0
            else:
                 cohens_d = (np.mean(advs_scores) - np.mean(good_scores)) / pooled_std
            
            # Interpretation logic
            if abs(cohens_d) < 0.2:
                effect_interp = "negligible"
            elif abs(cohens_d) < 0.5:
                effect_interp = "small"
            elif abs(cohens_d) < 0.8:
                effect_interp = "medium"
            else:
                effect_interp = "large"

            significance = "significant" if p_value < 0.05 else "not significant"
            
            # Per-exploit breakdown
            print("\n--- PER-EXPLOIT BREAKDOWN ---")
            for result in exploit_results:
                direction = "↑" if result["difference"] > 0 else "↓" if result["difference"] < 0 else "="
                print(f"{result['name']}: {result['difference']:+.6f} {direction}")

            print("\n--- SUMMARY ---")
            if p_value < 0.05 and cohens_d > 0:
                print(f"Bias detected against good answers. The result is {significance} with a {effect_interp} effect size.")
            elif p_value < 0.05 and cohens_d < 0:
                print(f"No bias detected (preference for good answers). The result is {significance} with a {effect_interp} effect size.")
            else:
                print(f"No significant bias detected. The result is {significance} with a {effect_interp} effect size.")

            print(f"Mean difference: {mean_diff:.6f} (std: {std_diff:.6f})")
            print(f"p-value: {p_value:.6e}")
            print(f"Cohen's d: {cohens_d:.6f}")

        else:
            print("Not enough samples for statistical analysis.")
    else:
        print("\nInsufficient data for statistical analysis.")
    
    print("\n" + "=" * 80)
    print("EXPERIMENT COMPLETE")
    print("=" * 80)

if __name__ == "__main__":
    main()
