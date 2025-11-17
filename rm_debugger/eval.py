"""
Script to evaluate the Ray2333/gpt2-large-helpful-reward_model on the full Anthropic/hh-rlhf test set.

The reward model is a GPT2 large model trained on Anthropic/hh-rlhf helpful dataset.
It reportedly achieves an accuracy of 0.72621 on the test set.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
from tqdm import tqdm
import pandas as pd
import os

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
    model_name = "Ray2333/gpt2-large-helpful-reward_model"
    
    print("=" * 80)
    print("REWARD MODEL EVALUATION")
    print("=" * 80)
    print(f"Model: {model_name}")
    
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
    print(f"\nStarting evaluation of all {total_samples} samples...")
    print("=" * 80)
    
    correct_predictions = 0
    results = []
    
    # Use tqdm for progress tracking
    for sample in tqdm(dataset, desc="Evaluating", unit="sample"):
        ground_truth_chosen = sample['chosen']
        ground_truth_rejected = sample['rejected']
        
        # Get reward scores
        gt_chosen_score = get_reward_score(model, tokenizer, ground_truth_chosen, device)
        gt_rejected_score = get_reward_score(model, tokenizer, ground_truth_rejected, device)
        
        # Check if RM correctly ranks chosen > rejected
        if gt_chosen_score > gt_rejected_score:
            correct_predictions += 1
        
        # Record RM's preference (which sample RM gave higher score to)
        if gt_chosen_score > gt_rejected_score:
            # RM prefers ground truth chosen
            rm_chosen = ground_truth_chosen
            rm_chosen_score = gt_chosen_score
            rm_rejected = ground_truth_rejected
            rm_rejected_score = gt_rejected_score
        else:
            # RM prefers ground truth rejected
            rm_chosen = ground_truth_rejected
            rm_chosen_score = gt_rejected_score
            rm_rejected = ground_truth_chosen
            rm_rejected_score = gt_chosen_score
        
        results.append({
            'chosen': rm_chosen,
            'chosen_score': rm_chosen_score,
            'rejected': rm_rejected,
            'rejected_score': rm_rejected_score
        })
    
    # Save results to CSV
    output_file = "rm_eval_results.csv"
    output_path = os.path.abspath(output_file)
    df = pd.DataFrame(results)
    df.to_csv(output_file, index=False)
    
    # Summary
    accuracy = correct_predictions / total_samples
    reported_accuracy = 0.72621
    
    print("\n" + "=" * 80)
    print("EVALUATION RESULTS")
    print("=" * 80)
    print(f"Model: {model_name}")
    print(f"Dataset: Anthropic/hh-rlhf (helpful-base test split)")
    print(f"Total samples evaluated: {total_samples}")
    print(f"\nPerformance:")
    print(f"  Correct predictions: {correct_predictions}")
    print(f"  Incorrect predictions: {total_samples - correct_predictions}")
    print(f"  Measured accuracy: {accuracy:.5f} ({accuracy:.2%})")
    print(f"  Reported accuracy: {reported_accuracy:.5f} ({reported_accuracy:.2%})")
    print(f"  Difference from reported: {abs(accuracy - reported_accuracy):.5f}")
    print(f"\nResults saved to:")
    print(f"  {output_path}")
    print(f"  ({len(results)} rows, 4 columns: chosen, chosen_score, rejected, rejected_score)")
    print("=" * 80)

if __name__ == "__main__":
    main()

