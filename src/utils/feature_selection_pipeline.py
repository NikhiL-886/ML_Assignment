import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scratch_implementation.variance_threshold import calculate_variance_scratch
from scratch_implementation.pearson_correlation import calculate_pearson_scratch
from scratch_implementation.chi_square import calculate_chi_square_scratch
from scratch_implementation.anova_f_test import calculate_anova_scratch
from scratch_implementation.mutual_information import calculate_mutual_information_scratch

def run_feature_selection_pipeline(df, target_col='isFraud', 
                                   tabular_dir='summary/tabular_statistics/', 
                                   vis_dir='summary/visualization/'):
    """
    Executes feature selection, actively drops low-variance, redundant, and weak features,
    and generates Task M7 and M8 reports.
    """
    os.makedirs(tabular_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    original_feature_count = df.shape[1]
    print(f"Starting Feature Selection. Original Feature Count: {original_feature_count}")
    
    df_processed = df.copy()
    decision_log = []
    features_to_drop = set()
    
    if target_col not in df_processed.columns:
        raise KeyError(f"Target column '{target_col}' not found in DataFrame.")

    target = df_processed[target_col]

    # --- 1. Task M1: Variance Threshold (Numerical Features) ---
    print("Executing Task M1: Variance Threshold...")
    num_cols = df_processed.select_dtypes(include=[np.number]).columns
    num_cols = [c for c in num_cols if c != target_col]
    
    for col in num_cols:
        var_val = calculate_variance_scratch(df_processed[col])
        
        # Active Threshold: Drop features with near-zero variance (< 0.02)
        if var_val < 0.02:
            features_to_drop.add(col)
            decision_log.append({
                'Feature': col, 'Data Type': 'Numerical', 
                'Method Used': 'Variance Threshold', 'Score/Statistic': round(var_val, 5),
                'Keep/Remove': 'Remove', 'Justification': f'Low variance ({var_val:.4f}); provides minimal information.'
            })
        else:
            decision_log.append({
                'Feature': col, 'Data Type': 'Numerical', 
                'Method Used': 'Variance Threshold', 'Score/Statistic': round(var_val, 5),
                'Keep/Remove': 'Keep', 'Justification': 'Sufficient data dispersion.'
            })

    # Drop low-variance features before correlation/statistical tests
    df_reduced = df_processed.drop(columns=list(features_to_drop))

    # --- 2. Task M2: Pearson Correlation (Feature-Feature Redundancy) ---
    print("Executing Task M2: Pearson Correlation Redundancy Check...")
    remaining_num_cols = df_reduced.select_dtypes(include=[np.number]).columns
    remaining_num_cols = [c for c in remaining_num_cols if c != target_col]
    
    # Compute correlation matrix subset for redundancy check
    corr_sample = df_reduced[remaining_num_cols[:50]].copy() # Sample first 50 for performance
    corr_matrix = pd.DataFrame(index=corr_sample.columns, columns=corr_sample.columns, dtype=float)
    
    cols_to_check = list(corr_sample.columns)
    for i in range(len(cols_to_check)):
        for j in range(i + 1, len(cols_to_check)):
            c1, c2 = cols_to_check[i], cols_to_check[j]
            r = calculate_pearson_scratch(corr_sample[c1], corr_sample[c2])
            corr_matrix.loc[c1, c2] = r
            
            # Active Threshold: If absolute correlation > 0.95, drop the second feature as redundant
            if abs(r) > 0.95 and c2 not in features_to_drop:
                features_to_drop.add(c2)
                decision_log.append({
                    'Feature': c2, 'Data Type': 'Numerical',
                    'Method Used': 'Pearson Correlation', 'Score/Statistic': round(r, 4),
                    'Keep/Remove': 'Remove', 'Justification': f'High multi-collinearity (r={r:.2f}) with {c1}.'
                })

    # Save Correlation Heatmap
    plt.figure(figsize=(10, 8))
    plt.imshow(corr_matrix.fillna(0).values, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Pearson Correlation (r)')
    plt.title('Task M2 - Feature-Feature Correlation Heatmap', weight='bold')
    heatmap_path = os.path.join(vis_dir, 'task_m2_correlation_heatmap.png')
    plt.savefig(heatmap_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Correlation heatmap saved to: {heatmap_path}")

    # --- 3. Task M3 & M4: Chi-Square / ANOVA for Target Association ---
    print("Executing Task M3 (Chi-Square) & Task M4 (ANOVA F-Test)...")
    df_reduced_2 = df_processed.drop(columns=list(features_to_drop))
    active_cols = [c for c in df_reduced_2.columns if c != target_col]
    
    for col in active_cols[:40]: # Evaluate subset against target
        series = df_reduced_2[col]
        if series.nunique() < 10:
            chi2_score, dof = calculate_chi_square_scratch(series, target)
            method = 'Chi-Square'
            score = chi2_score
        else:
            score = calculate_anova_scratch(series, target)
            method = 'ANOVA F-Test'
            
        # Active Threshold: Drop if statistical association score is practically zero
        if np.isnan(score) or score < 0.001:
            features_to_drop.add(col)
            decision_log.append({
                'Feature': col, 'Data Type': 'Mixed',
                'Method Used': method, 'Score/Statistic': round(score, 4) if not np.isnan(score) else 0.0,
                'Keep/Remove': 'Remove', 'Justification': 'Statistically independent from target (zero association).'
            })
        else:
            decision_log.append({
                'Feature': col, 'Data Type': 'Mixed',
                'Method Used': method, 'Score/Statistic': round(score, 4),
                'Keep/Remove': 'Keep', 'Justification': 'Valid predictive association with target.'
            })

    # Final feature pruning
    df_final = df_processed.drop(columns=list(features_to_drop), errors='ignore')
    selected_feature_count = df_final.shape[1]
    removed_feature_count = len(features_to_drop)

    print(f"Feature Selection complete. Removed: {removed_feature_count} | Kept: {selected_feature_count}")

    # --- 4. Export Task M7: Final Decision Report PDF ---
    report_df = pd.DataFrame(decision_log)
    pdf_path = os.path.join(tabular_dir, 'task_m7_final_feature_selection_report.pdf')
    
    with PdfPages(pdf_path) as pdf:
        rows_per_page = 20
        chunks = [report_df[i:i + rows_per_page] for i in range(0, len(report_df), rows_per_page)]
        
        for idx, chunk in enumerate(chunks):
            fig, ax = plt.subplots(figsize=(14, len(chunk) * 0.4 + 1.5))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=chunk.values, colLabels=chunk.columns, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 2.0)
            
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#4c72b0')
                elif col == 5:
                    cell.set_width(0.35)
                    
            plt.title(f"Task M7 - Final Feature Selection Decision (Page {idx + 1} of {len(chunks)})", weight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()
            
    print(f"✅ Task M7 Final Decision Report saved to: {pdf_path}")

    # --- 5. Export Task M8: Before vs After Comparison Report ---
    m8_summary = pd.DataFrame({
        'Stage': ['Original Dataset', 'Selected Features', 'Removed Features'],
        'Number of Features': [original_feature_count, selected_feature_count, removed_feature_count]
    })
    
    m8_pdf_path = os.path.join(tabular_dir, 'task_m8_before_vs_after_report.pdf')
    with PdfPages(m8_pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=m8_summary.values, colLabels=m8_summary.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 2.0)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4c72b0')
                
        plt.title("Task M8 - Before vs After Feature Selection Summary", weight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    print(f"✅ Task M8 Before vs After Report saved to: {m8_pdf_path}\n")

    return df_final