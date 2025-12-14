"""
Script to test sensitivity of the reward model to text perturbations.

Standard mode (no --custom flag):
- Evaluates original sample and samples with 'e' added (1, 2, or 3 times)

Custom mode (with --custom 'string'):
- Evaluates original sample vs sample with custom string injected

Position control (--prepend flag):
- With --prepend: Add perturbation at the start (e.g., "e " + text or "custom " + text)
- Without --prepend: Add perturbation at the end (e.g., text + " e" or text + " custom")

Reports mean, std, and statistical significance of reward differences across all samples.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import numpy as np
import pandas as pd
import os
import argparse
from scipy import stats


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


def calculate_statistics(original_scores, perturbed_scores, differences):
    """
    Calculate statistical significance metrics for the differences.
    
    Args:
        original_scores: Array of original reward scores
        perturbed_scores: Array of perturbed reward scores
        differences: Array of differences (perturbed - original)
        
    Returns:
        Dictionary with statistical metrics
    """
    # Paired t-test (two-tailed)
    t_statistic, p_value = stats.ttest_rel(perturbed_scores, original_scores)
    
    # Effect size (Cohen's d for paired samples)
    # Cohen's d = mean(differences) / std(differences)
    cohens_d = np.mean(differences) / np.std(differences, ddof=1)
    
    # 95% confidence interval for the mean difference
    n = len(differences)
    std_error = stats.sem(differences)
    ci_95 = stats.t.interval(0.95, n-1, loc=np.mean(differences), scale=std_error)
    
    # 99% confidence interval for the mean difference
    ci_99 = stats.t.interval(0.99, n-1, loc=np.mean(differences), scale=std_error)
    
    return {
        't_statistic': t_statistic,
        'p_value': p_value,
        'cohens_d': cohens_d,
        'ci_95_lower': ci_95[0],
        'ci_95_upper': ci_95[1],
        'ci_99_lower': ci_99[0],
        'ci_99_upper': ci_99[1],
        'std_error': std_error
    }


def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(
        description="Test sensitivity of reward model to 'e' character perturbations"
    )
    parser.add_argument(
        '--prepend',
        action='store_true',
        help='If set, prepend perturbation to the start of text. Otherwise, append to the end of text.'
    )
    parser.add_argument(
        '--custom',
        type=str,
        default=None,
        help='Custom string to inject instead of standard "e" perturbations. If provided, only evaluates original vs custom perturbation.'
    )
    args = parser.parse_args()
    
    model_name = "Ray2333/gpt2-large-helpful-reward_model"
    
    print("=" * 80)
    if args.custom:
        print("REWARD MODEL CUSTOM PERTURBATION ANALYSIS")
    else:
        print("REWARD MODEL 'e' SENSITIVITY ANALYSIS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Perturbation mode: {'PREPEND' if args.prepend else 'APPEND'}")
    if args.custom:
        print(f"Custom perturbation string: '{args.custom}'")
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    
    # Load model and tokenizer
    print(f"\nLoading model and tokenizer from HuggingFace...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Print model info
    num_params = sum(p.numel() for p in model.parameters())
    print(f"Model loaded successfully")
    print(f"Total parameters: {num_params:,}")
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    print("\nLoading dataset...")
    print(f"Dataset: Anthropic/hh-rlhf (helpful-base subset, test split)")
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="test")
    
    total_samples = len(dataset)
    print(f"Total test samples: {total_samples}")
    print(f"\nStarting evaluation of all {total_samples} samples with perturbations...")
    print("=" * 80)
    
    # Lists to store scores and differences
    original_scores = []
    
    if args.custom:
        # Custom mode: only one perturbation
        scores_custom = []
        diff_custom = []
    else:
        # Standard mode: 1, 2, 3 'e' perturbations
        scores_1e = []  # Scores for 1 'e'
        scores_2e = []  # Scores for 2 'e's
        scores_3e = []  # Scores for 3 'e's
        
        diff_1e = []  # Differences for 1 'e'
        diff_2e = []  # Differences for 2 'e's
        diff_3e = []  # Differences for 3 'e's
    
    results = []
    
    # Use tqdm for progress tracking
    for sample in tqdm(dataset, desc="Evaluating", unit="sample"):
        # We'll test on both chosen and rejected samples
        for text_type, text in [('chosen', sample['chosen']), ('rejected', sample['rejected'])]:
            # Get original score
            original_score = get_reward_score(model, tokenizer, text, device)
            original_scores.append(original_score)
            
            if args.custom:
                # Custom perturbation mode
                if args.prepend:
                    perturbed_text = args.custom + " " + text
                else:
                    perturbed_text = text + " " + args.custom
                
                score_custom_val = get_reward_score(model, tokenizer, perturbed_text, device)
                diff_custom_val = score_custom_val - original_score
                
                scores_custom.append(score_custom_val)
                diff_custom.append(diff_custom_val)
                
                # Store detailed results
                results.append({
                    'text_type': text_type,
                    'text': text,
                    'original_score': original_score,
                    'score_custom': score_custom_val,
                    'diff_custom': diff_custom_val
                })
            else:
                # Standard 'e' perturbation mode
                # Get scores for perturbed versions based on mode
                if args.prepend:
                    # Prepend 'e' at the start
                    score_1e = get_reward_score(model, tokenizer, "e " + text, device)
                    score_2e = get_reward_score(model, tokenizer, "e e " + text, device)
                    score_3e = get_reward_score(model, tokenizer, "e e e " + text, device)
                else:
                    # Append 'e' at the end
                    score_1e = get_reward_score(model, tokenizer, text + " e", device)
                    score_2e = get_reward_score(model, tokenizer, text + " e e", device)
                    score_3e = get_reward_score(model, tokenizer, text + " e e e", device)
                
                # Calculate differences (perturbed - original)
                diff_1 = score_1e - original_score
                diff_2 = score_2e - original_score
                diff_3 = score_3e - original_score
                
                # Store scores and differences
                scores_1e.append(score_1e)
                scores_2e.append(score_2e)
                scores_3e.append(score_3e)
                
                diff_1e.append(diff_1)
                diff_2e.append(diff_2)
                diff_3e.append(diff_3)
                
                # Store detailed results
                results.append({
                    'text_type': text_type,
                    'text': text,
                    'original_score': original_score,
                    'score_1e': score_1e,
                    'score_2e': score_2e,
                    'score_3e': score_3e,
                    'diff_1e': diff_1,
                    'diff_2e': diff_2,
                    'diff_3e': diff_3
                })
    
    # Convert to numpy arrays for statistics
    original_scores = np.array(original_scores)
    
    if args.custom:
        scores_custom = np.array(scores_custom)
        diff_custom = np.array(diff_custom)
        stats_custom = calculate_statistics(original_scores, scores_custom, diff_custom)
    else:
        scores_1e = np.array(scores_1e)
        scores_2e = np.array(scores_2e)
        scores_3e = np.array(scores_3e)
        
        diff_1e = np.array(diff_1e)
        diff_2e = np.array(diff_2e)
        diff_3e = np.array(diff_3e)
        
        # Calculate statistical significance
        stats_1e = calculate_statistics(original_scores, scores_1e, diff_1e)
        stats_2e = calculate_statistics(original_scores, scores_2e, diff_2e)
        stats_3e = calculate_statistics(original_scores, scores_3e, diff_3e)
    
    # Save detailed results to CSV
    mode_suffix = "prepend" if args.prepend else "append"
    if args.custom:
        # Sanitize custom string for filename
        safe_custom = "".join(c if c.isalnum() else "_" for c in args.custom)[:20]
        output_file = f"sensitivity_results_custom_{safe_custom}_{mode_suffix}.csv"
    else:
        output_file = f"e_sensitivity_results_{mode_suffix}.csv"
    output_path = os.path.abspath(output_file)
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Helper function to interpret p-value
    def interpret_p_value(p):
        if p < 0.001:
            return "highly significant (p < 0.001)"
        elif p < 0.01:
            return "very significant (p < 0.01)"
        elif p < 0.05:
            return "significant (p < 0.05)"
        else:
            return "not significant (p >= 0.05)"
    
    # Helper function to interpret effect size (Cohen's d)
    def interpret_cohens_d(d):
        abs_d = abs(d)
        if abs_d < 0.2:
            return "negligible"
        elif abs_d < 0.5:
            return "small"
        elif abs_d < 0.8:
            return "medium"
        else:
            return "large"
    
    # Print summary statistics
    print("\n" + "=" * 80)
    if args.custom:
        print("CUSTOM PERTURBATION RESULTS")
    else:
        print("'e' SENSITIVITY RESULTS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: Anthropic/hh-rlhf (helpful-base test split)")
    print(f"Perturbation mode: {'PREPEND' if args.prepend else 'APPEND'}")
    if args.custom:
        print(f"Custom perturbation: '{args.custom}'")
        print(f"Total samples evaluated: {len(diff_custom)} (both chosen and rejected from {total_samples} pairs)")
    else:
        print(f"Total samples evaluated: {len(diff_1e)} (both chosen and rejected from {total_samples} pairs)")
    print("\nReward score differences (perturbed - original):")
    print("=" * 80)
    
    if args.custom:
        # Custom mode: print single perturbation results
        if args.prepend:
            label = f"Custom prepended ('{args.custom}' + ' ' + text)"
        else:
            label = f"Custom appended (text + ' ' + '{args.custom}')"
        
        print(f"\n{label}:")
        print(f"  Mean difference:     {np.mean(diff_custom):.6f}")
        print(f"  Std difference:      {np.std(diff_custom):.6f}")
        print(f"  Std error:           {stats_custom['std_error']:.6f}")
        print(f"  Min difference:      {np.min(diff_custom):.6f}")
        print(f"  Max difference:      {np.max(diff_custom):.6f}")
        print(f"  95% CI:              [{stats_custom['ci_95_lower']:.6f}, {stats_custom['ci_95_upper']:.6f}]")
        print(f"  99% CI:              [{stats_custom['ci_99_lower']:.6f}, {stats_custom['ci_99_upper']:.6f}]")
        print(f"  t-statistic:         {stats_custom['t_statistic']:.4f}")
        print(f"  p-value:             {stats_custom['p_value']:.6e} ({interpret_p_value(stats_custom['p_value'])})")
        print(f"  Cohen's d:           {stats_custom['cohens_d']:.4f} ({interpret_cohens_d(stats_custom['cohens_d'])} effect)")
    else:
        # Standard mode: print 1, 2, 3 'e' results
        # Dynamic labels based on mode
        if args.prepend:
            label_1e = "1 'e' prepended ('e ' + text)"
            label_2e = "2 'e's prepended ('e e ' + text)"
            label_3e = "3 'e's prepended ('e e e ' + text)"
        else:
            label_1e = "1 'e' appended (text + ' e')"
            label_2e = "2 'e's appended (text + ' e e')"
            label_3e = "3 'e's appended (text + ' e e e')"
        
        print(f"\n{label_1e}:")
        print(f"  Mean difference:     {np.mean(diff_1e):.6f}")
        print(f"  Std difference:      {np.std(diff_1e):.6f}")
        print(f"  Std error:           {stats_1e['std_error']:.6f}")
        print(f"  Min difference:      {np.min(diff_1e):.6f}")
        print(f"  Max difference:      {np.max(diff_1e):.6f}")
        print(f"  95% CI:              [{stats_1e['ci_95_lower']:.6f}, {stats_1e['ci_95_upper']:.6f}]")
        print(f"  99% CI:              [{stats_1e['ci_99_lower']:.6f}, {stats_1e['ci_99_upper']:.6f}]")
        print(f"  t-statistic:         {stats_1e['t_statistic']:.4f}")
        print(f"  p-value:             {stats_1e['p_value']:.6e} ({interpret_p_value(stats_1e['p_value'])})")
        print(f"  Cohen's d:           {stats_1e['cohens_d']:.4f} ({interpret_cohens_d(stats_1e['cohens_d'])} effect)")
        
        print(f"\n{label_2e}:")
        print(f"  Mean difference:     {np.mean(diff_2e):.6f}")
        print(f"  Std difference:      {np.std(diff_2e):.6f}")
        print(f"  Std error:           {stats_2e['std_error']:.6f}")
        print(f"  Min difference:      {np.min(diff_2e):.6f}")
        print(f"  Max difference:      {np.max(diff_2e):.6f}")
        print(f"  95% CI:              [{stats_2e['ci_95_lower']:.6f}, {stats_2e['ci_95_upper']:.6f}]")
        print(f"  99% CI:              [{stats_2e['ci_99_lower']:.6f}, {stats_2e['ci_99_upper']:.6f}]")
        print(f"  t-statistic:         {stats_2e['t_statistic']:.4f}")
        print(f"  p-value:             {stats_2e['p_value']:.6e} ({interpret_p_value(stats_2e['p_value'])})")
        print(f"  Cohen's d:           {stats_2e['cohens_d']:.4f} ({interpret_cohens_d(stats_2e['cohens_d'])} effect)")
        
        print(f"\n{label_3e}:")
        print(f"  Mean difference:     {np.mean(diff_3e):.6f}")
        print(f"  Std difference:      {np.std(diff_3e):.6f}")
        print(f"  Std error:           {stats_3e['std_error']:.6f}")
        print(f"  Min difference:      {np.min(diff_3e):.6f}")
        print(f"  Max difference:      {np.max(diff_3e):.6f}")
        print(f"  95% CI:              [{stats_3e['ci_95_lower']:.6f}, {stats_3e['ci_95_upper']:.6f}]")
        print(f"  99% CI:              [{stats_3e['ci_99_lower']:.6f}, {stats_3e['ci_99_upper']:.6f}]")
        print(f"  t-statistic:         {stats_3e['t_statistic']:.4f}")
        print(f"  p-value:             {stats_3e['p_value']:.6e} ({interpret_p_value(stats_3e['p_value'])})")
        print(f"  Cohen's d:           {stats_3e['cohens_d']:.4f} ({interpret_cohens_d(stats_3e['cohens_d'])} effect)")
    
    print(f"\n" + "=" * 80)
    print("STATISTICAL SUMMARY")
    print("=" * 80)
    print("\nInterpretation guide:")
    print("  - P-value: Probability that observed difference is due to chance")
    print("    * p < 0.05: Statistically significant")
    print("    * p < 0.01: Very significant")
    print("    * p < 0.001: Highly significant")
    print("  - Cohen's d: Standardized effect size")
    print("    * |d| < 0.2: Negligible effect")
    print("    * 0.2 ≤ |d| < 0.5: Small effect")
    print("    * 0.5 ≤ |d| < 0.8: Medium effect")
    print("    * |d| ≥ 0.8: Large effect")
    print("  - 95% CI: Range where true mean difference lies with 95% confidence")
    
    print(f"\nKey findings:")
    if args.custom:
        sig = interpret_p_value(stats_custom['p_value'])
        effect = interpret_cohens_d(stats_custom['cohens_d'])
        print(f"  - Custom perturbation '{args.custom}': {sig}, {effect} effect size")
    else:
        findings = []
        for num_e, stats_dict in [("1", stats_1e), ("2", stats_2e), ("3", stats_3e)]:
            sig = interpret_p_value(stats_dict['p_value'])
            effect = interpret_cohens_d(stats_dict['cohens_d'])
            findings.append(f"  - {num_e} 'e': {sig}, {effect} effect size")
        for finding in findings:
            print(finding)
    
    print(f"\nDetailed results saved to:")
    print(f"  {output_path}")
    if args.custom:
        print(f"  ({len(results)} rows, 5 columns: text_type, text, original_score, score_custom, diff_custom)")
    else:
        print(f"  ({len(results)} rows, 9 columns)")
    print("=" * 80)


if __name__ == "__main__":
    main()
