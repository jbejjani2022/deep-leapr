#!/usr/bin/env python3
"""
Script to download the Anthropic/hh-rlhf helpful-base test split to a CSV file.
"""

import pandas as pd
from datasets import load_dataset
from pathlib import Path
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)


def main():
    output_dir = Path("../data")
    output_dir.mkdir(exist_ok=True)
    output_file = output_dir / "hh_rlhf_helpful_base_test.csv"
    
    logger.info("Downloading Anthropic/hh-rlhf helpful-base test split...")
    dataset = load_dataset("Anthropic/hh-rlhf", data_dir="helpful-base", split="test")
    
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Convert to pandas DataFrame with pairwise comparison format
    # For each (chosen, rejected) pair, create two rows:
    # 1. text_a=chosen, text_b=rejected, chosen=0 (text_a was chosen)
    # 2. text_a=rejected, text_b=chosen, chosen=1 (text_b was chosen)
    data = []
    for sample in dataset:
        chosen_text = sample["chosen"]
        rejected_text = sample["rejected"]
        
        # Row 1: chosen is text_a
        data.append({
            "text_a": chosen_text,
            "text_b": rejected_text,
            "chosen": 0  # text_a was chosen
        })
        
        # Row 2: rejected is text_a
        data.append({
            "text_a": rejected_text,
            "text_b": chosen_text,
            "chosen": 1  # text_b was chosen
        })
    
    df = pd.DataFrame(data)
    
    # Save to CSV
    df.to_csv(output_file, index=False)
    logger.info(f"Saved {len(df)} rows ({len(dataset)} original samples x 2) to {output_file}")
    
    # Print statistics
    logger.info(f"Text_a lengths: mean={df['text_a'].str.len().mean():.1f}, "
                f"median={df['text_a'].str.len().median():.1f}")
    logger.info(f"Text_b lengths: mean={df['text_b'].str.len().mean():.1f}, "
                f"median={df['text_b'].str.len().median():.1f}")
    logger.info(f"Label distribution: {df['chosen'].value_counts().to_dict()}")
    
    logger.info(f"\nDone! Dataset saved to: {output_file.absolute()}")


if __name__ == "__main__":
    main()

