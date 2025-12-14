"""
Script to plot the relationship between reward score and count of words containing 'e'.

Loads the dataset and creates a scatter plot with:
- X-axis: Count of words containing the letter 'e' in the text
- Y-axis: Reward score (rm_helpful_score)
"""

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path


def feature(text: str) -> float:
    """Count of words containing the letter 'e'"""
    e_word_count = sum(1 for word in text.split() if 'e' in word.lower())
    return float(e_word_count)


def main():
    # Load the dataset
    data_path = Path(__file__).parent.parent / "data" / "rm_eval_results_unrolled.csv"
    print(f"Loading data from: {data_path}")
    df = pd.read_csv(data_path)
    
    print(f"Loaded {len(df)} samples")
    print(f"Columns: {df.columns.tolist()}")
    
    # Calculate the feature (count of words containing 'e') for each sample
    print("Calculating word counts containing 'e'...")
    df['e_word_count'] = df['text'].apply(feature)
    
    # Create the scatter plot
    print("Creating scatter plot...")
    plt.figure(figsize=(12, 8))
    
    # Create scatter plot with alpha for better visualization of overlapping points
    plt.scatter(df['e_word_count'], df['rm_helpful_score'], 
                alpha=0.3, s=10, c='blue', edgecolors='none')
    
    # Add labels and title
    plt.xlabel("Count of words containing 'e'", fontsize=12)
    plt.ylabel("Reward", fontsize=12)
    plt.title("Reward Score vs. Count of Words Containing 'e'", fontsize=14)
    
    # Add grid for better readability
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # Calculate and display correlation
    correlation = df['e_word_count'].corr(df['rm_helpful_score'])
    
    # Add correlation info to plot
    plt.text(0.02, 0.98, f"Correlation: {correlation:.4f}\nn = {len(df):,} samples",
             transform=plt.gca().transAxes, fontsize=10,
             verticalalignment='top', bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Add a trend line
    z = np.polyfit(df['e_word_count'], df['rm_helpful_score'], 1)
    p = np.poly1d(z)
    x_trend = np.linspace(df['e_word_count'].min(), df['e_word_count'].max(), 100)
    plt.plot(x_trend, p(x_trend), "r--", alpha=0.8, linewidth=2, label=f'Trend line (y={z[0]:.4f}x+{z[1]:.4f})')
    plt.legend(loc='lower right')
    
    # Adjust layout to prevent label cutoff
    plt.tight_layout()
    
    # Save the plot
    output_path = Path(__file__).parent / "reward_vs_e_words.png"
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    print(f"\nPlot saved to: {output_path}")
    
    # Print summary statistics
    print("\n" + "=" * 80)
    print("SUMMARY STATISTICS")
    print("=" * 80)
    print(f"Reward (rm_helpful_score):")
    print(f"  Mean:   {df['rm_helpful_score'].mean():.4f}")
    print(f"  Std:    {df['rm_helpful_score'].std():.4f}")
    print(f"  Min:    {df['rm_helpful_score'].min():.4f}")
    print(f"  Max:    {df['rm_helpful_score'].max():.4f}")
    print(f"\nCount of words containing 'e':")
    print(f"  Mean:   {df['e_word_count'].mean():.2f}")
    print(f"  Std:    {df['e_word_count'].std():.2f}")
    print(f"  Min:    {df['e_word_count'].min():.0f}")
    print(f"  Max:    {df['e_word_count'].max():.0f}")
    print(f"\nCorrelation coefficient: {correlation:.4f}")
    
    # Test for statistical significance of correlation
    from scipy import stats as scipy_stats
    n = len(df)
    t_stat = correlation * np.sqrt(n - 2) / np.sqrt(1 - correlation**2)
    p_value = 2 * (1 - scipy_stats.t.cdf(abs(t_stat), n - 2))
    print(f"Statistical significance: p-value = {p_value:.6e}")
    if p_value < 0.001:
        print("  -> Highly significant (p < 0.001)")
    elif p_value < 0.01:
        print("  -> Very significant (p < 0.01)")
    elif p_value < 0.05:
        print("  -> Significant (p < 0.05)")
    else:
        print("  -> Not significant (p >= 0.05)")
    print("=" * 80)
    
    # Show the plot
    plt.show()


if __name__ == "__main__":
    main()
