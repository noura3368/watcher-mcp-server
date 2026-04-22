import pandas as pd
import numpy as np
from scipy.stats import kruskal
from scipy.stats import mannwhitneyu

def run_kruskal_wallis_analysis(exclude_zero_commands=False):
    """
    Run Kruskal-Wallis test on model size groups.
    
    Parameters:
    exclude_zero_commands (bool): If True, exclude models with 0 average commands
    """
    # Load the data
    #df = pd.read_csv('/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/multi_folder_commands_summary_dynamic-0104.csv')
    #df = pd.read_csv('/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/unique_commands_per_iteration.csv')
   #df = pd.read_csv('/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/multi_folder_commands_summary.csv')
    #df = pd.read_csv('/home/nkhajehn/MCP-Command-Generation/model_commands_combined.csv')
    df = pd.read_csv('/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/csv_files/prompt4_model_stats.csv')
   
    # df = pd.read_csv('/home/nkhajehn/MCP-Command-Generation/post_processing/plots/model_commands_oct14.csv')
  #  df = pd.read_csv('/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/kruskal_tests/multi_folder_commands_summary_all_folders.csv')
    #df = pd.read_csv('/home/nkhajehn/MCP-Command-Generation/hypothesis_testing/results_oct15_avg_unique_commands_per_folder.csv')
    # Optionally filter out models with 0 average commands
    if exclude_zero_commands:
        original_count = len(df)
        df = df[df['base_commands_seen'] > 0]
        filtered_count = len(df)
        print(f"=== Data Filtering ===")
        print(f"Original dataset: {original_count} models")
        print(f"After filtering (base_commands_seen > 0): {filtered_count} models")
        print(f"Excluded: {original_count - filtered_count} models with 0 average commands")
        print()

    # Group models by size: Small (≤3B), Medium (3B-8B), Large (>8B)
    small_models = df[df['size_billions'] <= 4]['base_commands_seen'].values
    medium_models = df[(df['size_billions'] > 4) & (df['size_billions'] <= 10)]['base_commands_seen'].values
    large_models = df[df['size_billions'] > 10]['base_commands_seen'].values

    # Display group information
    print("=== Model Size Groups ===")
    print(f"Small models (≤4B): {len(small_models)} models")
    print(f"Medium models (4B-8B): {len(medium_models)} models")
    print(f"Large models (>=8B): {len(large_models)} models")
    print()

    # Display descriptive statistics
    print("=== Descriptive Statistics ===")
    print("Small models:")
    print(f"  Mean: {np.mean(small_models):.3f}")
    print(f"  Median: {np.median(small_models):.3f}")
    print(f"  Std: {np.std(small_models):.3f}")
    print()

    print("Medium models:")
    print(f"  Mean: {np.mean(medium_models):.3f}")
    print(f"  Median: {np.median(medium_models):.3f}")
    print(f"  Std: {np.std(medium_models):.3f}")
    print()

    print("Large models:")
    print(f"  Mean: {np.mean(large_models):.3f}")
    print(f"  Median: {np.median(large_models):.3f}")
    print(f"  Std: {np.std(large_models):.3f}")
    print()

    # Perform Kruskal-Wallis test
    print("=== Kruskal-Wallis Test ===")
    print("Null Hypothesis: All size groups have the same median base_commands_seen")
    print("Alternative Hypothesis: At least one group has a different median base_commands_seen")
    print()

    H_statistic, p_value = kruskal(small_models, medium_models, large_models)

    print(f"H-statistic: {H_statistic:.6f}")
    print(f"P-value: {p_value:.6f}")
    print(f"Significance level (α): 0.05")
    print()

    # Interpret results
    if p_value < 0.05:
        print("Result: REJECT the null hypothesis")
        print("Conclusion: There is a statistically significant difference in median base_commands_seen between at least two size groups.")
        
        # Post-hoc analysis using Mann-Whitney U tests
        print("\n=== Post-hoc Analysis (Mann-Whitney U tests) ===")
        
        # Small vs Medium
        u1, p1 = mannwhitneyu(small_models, medium_models, alternative='two-sided')
        print(f"Small vs Medium: U={u1:.3f}, p={p1:.6f} {'***' if p1 < 0.001 else '**' if p1 < 0.01 else '*' if p1 < 0.05 else 'ns'}")
        
        # Small vs Large
        u2, p2 = mannwhitneyu(small_models, large_models, alternative='two-sided')
        print(f"Small vs Large: U={u2:.3f}, p={p2:.6f} {'***' if p2 < 0.001 else '**' if p2 < 0.01 else '*' if p2 < 0.05 else 'ns'}")
        
        # Medium vs Large
        u3, p3 = mannwhitneyu(medium_models, large_models, alternative='two-sided')
        print(f"Medium vs Large: U={u3:.3f}, p={p3:.6f} {'***' if p3 < 0.001 else '**' if p3 < 0.01 else '*' if p3 < 0.05 else 'ns'}")
        
        print("\nSignificance levels: *** p<0.001, ** p<0.01, * p<0.05, ns = not significant")
        
    else:
        print("Result: FAIL TO REJECT the null hypothesis")
        print("Conclusion: There is no statistically significant difference in median base_commands_seen between the size groups.")

    print(f"\nEffect size interpretation:")
    print(f"H-statistic of {H_statistic:.3f} indicates the strength of the difference between groups.")
    
    return H_statistic, p_value

# Run the analysis with and without zero-command models
if __name__ == "__main__":
    print("=" * 60)
    print("KRUSKAL-WALLIS ANALYSIS - INCLUDING ALL MODELS")
    print("=" * 60)
    run_kruskal_wallis_analysis(exclude_zero_commands=False)
    
    print("\n" + "=" * 60)
    print("KRUSKAL-WALLIS ANALYSIS - EXCLUDING ZERO-COMMAND MODELS")
    print("=" * 60)
    run_kruskal_wallis_analysis(exclude_zero_commands=True)

# You can also call the function individually:
# run_kruskal_wallis_analysis(exclude_zero_commands=False)  # Include all models
# run_kruskal_wallis_analysis(exclude_zero_commands=True)   # Exclude zero-command models
