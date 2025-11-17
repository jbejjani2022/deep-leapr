"""
Script to evaluate the Ray2333/gpt2-large-helpful-reward_model on Anthropic/hh-rlhf dataset.

The reward model is a GPT2 large model trained on Anthropic/hh-rlhf helpful dataset.
It achieves an accuracy of 0.72621 on the test set.
"""

import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset
import argparse


# custom (chosen, rejected) test samples
dummy_data = [
    ("Human: Can fish live on land? Assistant: Fish generally live in water, but some species have evolved special adaptations that allow them to survive and even move on land for periods of time.", "Human: Can fish live on land? Assistant: What a dumb question."),
    ("Human: What's your favorite drink? Assistant: Water.", "Human: What's your favorite drink? Assistant: Not telling.")
]


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
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Evaluate reward model on helpful-base dataset")
    parser.add_argument("--dummy-data", action="store_true", 
                        help="Use dummy data instead of loading from hh-rlhf dataset")
    args = parser.parse_args()
    
    print("Loading reward model from HuggingFace...")
    model_name = "Ray2333/gpt2-large-helpful-reward_model"
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Load model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()
    
    # Set padding token if not set
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        model.config.pad_token_id = model.config.eos_token_id
    
    # Load data based on flag
    if args.dummy_data:
        print("\nUsing dummy data...")
        # Convert dummy_data to a list of dicts matching dataset format
        samples = [{"chosen": chosen, "rejected": rejected} for chosen, rejected in dummy_data]
    else:
        print("\nLoading Anthropic/hh-rlhf helpful-base test dataset...")
        dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="test")
        # Take first 10 samples
        samples = dataset.select(range(min(10, len(dataset))))
    
    print(f"\nEvaluating {len(samples)} samples...\n")
    print("=" * 80)
    
    correct_predictions = 0
    
    for idx, sample in enumerate(samples):
        print(f"\n### Sample {idx + 1} ###\n")
        
        chosen = sample['chosen']
        rejected = sample['rejected']
        
        print(f"Chosen response:\n{chosen[:500]}{'...' if len(chosen) > 500 else ''}\n")
        print(f"Rejected response:\n{rejected[:500]}{'...' if len(rejected) > 500 else ''}\n")
        
        # Get reward scores
        chosen_score = get_reward_score(model, tokenizer, chosen, device)
        rejected_score = get_reward_score(model, tokenizer, rejected, device)
        
        print(f"Chosen score: {chosen_score:.4f}")
        print(f"Rejected score: {rejected_score:.4f}")
        
        # Check if RM correctly ranks chosen > rejected
        is_correct = chosen_score > rejected_score
        if is_correct:
            correct_predictions += 1
            print(f"✓ RM correctly gave higher score to chosen response (diff: {chosen_score - rejected_score:.4f})")
        else:
            print(f"✗ RM incorrectly gave higher score to rejected response (diff: {rejected_score - chosen_score:.4f})")
        
        print("=" * 80)
    
    # Summary
    accuracy = correct_predictions / len(samples)
    print(f"\n### Summary ###")
    print(f"Correct predictions: {correct_predictions}/{len(samples)}")
    print(f"Accuracy: {accuracy:.2%}")

if __name__ == "__main__":
    main()