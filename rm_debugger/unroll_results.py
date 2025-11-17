"""
Script to unroll rm_eval_results.csv into a format with one row per text sample.

Input CSV columns: chosen, chosen_score, rejected, rejected_score
Output CSV columns: text, rm_helpful_score

Each row in the input becomes 2 rows in the output.
"""

import pandas as pd
import os

def main():
    input_file = "rm_eval_results.csv"
    output_file = "rm_eval_results_unrolled.csv"
    
    print("=" * 80)
    print("UNROLLING REWARD MODEL RESULTS")
    print("=" * 80)
    
    # Check if input file exists
    if not os.path.exists(input_file):
        print(f"Error: Input file '{input_file}' not found!")
        print(f"Please ensure {input_file} exists in the current directory.")
        return
    
    print(f"Reading input file: {input_file}")
    df = pd.read_csv(input_file)
    
    print(f"Input rows: {len(df)}")
    print(f"Input columns: {list(df.columns)}")
    
    # Create unrolled dataframe
    unrolled_rows = []
    
    for _, row in df.iterrows():
        # Add chosen text and score
        unrolled_rows.append({
            'text': row['chosen'],
            'rm_helpful_score': row['chosen_score']
        })
        
        # Add rejected text and score
        unrolled_rows.append({
            'text': row['rejected'],
            'rm_helpful_score': row['rejected_score']
        })
    
    unrolled_df = pd.DataFrame(unrolled_rows)
    
    # Save to CSV
    output_path = os.path.abspath(output_file)
    unrolled_df.to_csv(output_file, index=False)
    
    print(f"\nOutput rows: {len(unrolled_df)}")
    print(f"Output columns: {list(unrolled_df.columns)}")
    print(f"\nResults saved to:")
    print(f"  {output_path}")
    print("=" * 80)

if __name__ == "__main__":
    main()

