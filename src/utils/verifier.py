import pandas as pd
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

# Ensure Python can find the scratch_implementation folder
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scratch_implementation import basic_statistics as bs

def compare_values(scratch_val, pandas_val):
    """Safely compares values, accounting for floating-point imprecision and NaNs."""
    if (scratch_val is None or pd.isna(scratch_val)) and pd.isna(pandas_val):
        return True
    try:
        # np.isclose handles tiny float differences (e.g., 0.100000001 vs 0.1)
        return np.isclose(float(scratch_val), float(pandas_val), equal_nan=True)
    except (ValueError, TypeError):
        return str(scratch_val) == str(pandas_val)

def run_full_verification(df, sample_size=1000, output_dir='summary/verification_results/'):
    """Samples numerical columns and compares scratch implementations against Pandas."""
    os.makedirs(output_dir, exist_ok=True)
    print(f"Sampling {sample_size} random rows for verification...")
    
    # 1. Filter numerical columns and take a random sample
    num_cols = df.select_dtypes(include=['number']).columns
    sampled_df = df[num_cols].sample(n=sample_size, random_state=42)
    
    results = []
    print(f"Running verification across {len(num_cols)} numerical features. This may take a minute...")
    
    # 2. Iterate through every numerical column
    for col in num_cols:
        data_list = sampled_df[col].tolist()
        pd_series = sampled_df[col]
        
        # Calculate Pandas metrics once
        p_min, p_max = pd_series.min(), pd_series.max()
        
        # --- Run Comparisons ---
        # Min
        results.append({'Feature': col, 'Statistic': 'Min', 
                        'Scratch': bs.calculate_min(data_list), 'Pandas': p_min,
                        'Match': compare_values(bs.calculate_min(data_list), p_min)})
        
        # Max
        results.append({'Feature': col, 'Statistic': 'Max', 
                        'Scratch': bs.calculate_max(data_list), 'Pandas': p_max,
                        'Match': compare_values(bs.calculate_max(data_list), p_max)})
        
        # Mean
        results.append({'Feature': col, 'Statistic': 'Mean', 
                        'Scratch': bs.calculate_mean(data_list), 'Pandas': pd_series.mean(),
                        'Match': compare_values(bs.calculate_mean(data_list), pd_series.mean())})
        
        # Median
        results.append({'Feature': col, 'Statistic': 'Median', 
                        'Scratch': bs.calculate_median(data_list), 'Pandas': pd_series.median(),
                        'Match': compare_values(bs.calculate_median(data_list), pd_series.median())})
        
        # Variance
        results.append({'Feature': col, 'Statistic': 'Variance', 
                        'Scratch': bs.calculate_variance(data_list), 'Pandas': pd_series.var(),
                        'Match': compare_values(bs.calculate_variance(data_list), pd_series.var())})
        
        # Std Dev
        results.append({'Feature': col, 'Statistic': 'Std Dev', 
                        'Scratch': bs.calculate_std_dev(data_list), 'Pandas': pd_series.std(),
                        'Match': compare_values(bs.calculate_std_dev(data_list), pd_series.std())})
        
        # Range
        p_range = p_max - p_min if pd.notna(p_max) and pd.notna(p_min) else np.nan
        results.append({'Feature': col, 'Statistic': 'Range', 
                        'Scratch': bs.calculate_range(data_list), 'Pandas': p_range,
                        'Match': compare_values(bs.calculate_range(data_list), p_range)})
        
        # Mode (requires special string conversion because there can be multiple modes)
        s_mode = sorted(bs.calculate_mode(data_list) or [])
        p_mode = sorted(pd_series.mode().dropna().tolist())
        results.append({'Feature': col, 'Statistic': 'Mode', 
                        'Scratch': str(s_mode), 'Pandas': str(p_mode),
                        'Match': s_mode == p_mode})

    # 3. Save the results to a paginated PDF
    results_df = pd.DataFrame(results)
    file_path = os.path.join(output_dir, 'scratch_vs_pandas_verification.pdf')
    
    print("Generating formatted PDF report...")
    
    # We format the numbers to 4 decimal places so they fit neatly in the PDF cells
    display_df = results_df.copy()
    for col in ['Scratch', 'Pandas']:
        display_df[col] = display_df[col].apply(
            lambda x: f"{float(x):.4f}" if pd.notna(x) and not isinstance(x, str) else str(x)[:30]
        )

    with PdfPages(file_path) as pdf:
        rows_per_page = 40
        chunks = [display_df[i:i + rows_per_page] for i in range(0, len(display_df), rows_per_page)]
        
        for idx, chunk in enumerate(chunks):
            fig, ax = plt.subplots(figsize=(12, len(chunk) * 0.4 + 1))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=chunk.values, colLabels=chunk.columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 1.5)
            
            # Style the header row
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#4c72b0')
                    
            plt.title(f"Scratch vs Pandas Verification (Page {idx + 1} of {len(chunks)})", weight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
    
    # 4. Print Summary
    total_checks = len(results_df)
    passed_checks = results_df['Match'].sum()
    print(f"\n✅ Verification Complete!")
    print(f"Total Checks: {total_checks} | Passed: {passed_checks} | Failed: {total_checks - passed_checks}")
    print(f"Full report saved to: {file_path}")