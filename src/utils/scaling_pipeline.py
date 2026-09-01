import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scratch_implementation.feature_scaler import ScratchFeatureScaler

def run_scaling_pipeline(df, tabular_dir='summary/tabular_statistics/', vis_dir='summary/visualization/'):
    """
    Executes scratch feature scaling across numerical columns, logs decisions,
    and saves the formal verification PDF and visual distribution plots.
    """
    os.makedirs(tabular_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print("Running scratch feature scaling and decision pipeline...")
    scaler = ScratchFeatureScaler()
    df_scaled = df.copy()
    
    num_cols = df_scaled.select_dtypes(include=[np.number]).columns
    
    # Exclude ID or binary columns if necessary
    target_cols = [col for col in num_cols if col != 'TransactionID']
    
    for col in target_cols:
        df_scaled[col] = scaler.fit_transform_column(df_scaled[col], col)
        
    report_df = pd.DataFrame(scaler.scaling_log)
    
    # 1. Save Tabular Summary as PDF
    pdf_path = os.path.join(tabular_dir, 'task_h_feature_scaling_report.pdf')
    print("Generating paginated PDF for Feature Scaling Report...")
    
    with PdfPages(pdf_path) as pdf:
        rows_per_page = 20
        chunks = [report_df[i:i + rows_per_page] for i in range(0, len(report_df), rows_per_page)]
        
        for idx, chunk in enumerate(chunks):
            fig, ax = plt.subplots(figsize=(14, len(chunk) * 0.5 + 1.5))
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
                elif col == 2:
                    cell.set_width(0.4) # Give reasoning column space
                    
            plt.title(f"Task H - Feature Scaling Decision Report (Page {idx + 1} of {len(chunks)})", weight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"✅ Scaling report saved to: {pdf_path}")

    # 2. Save Visualization (Method Distribution Pie Chart)
    if not report_df.empty:
        plt.figure(figsize=(7, 7))
        method_counts = report_df['Chosen Method'].value_counts()
        colors = ['#4c72b0', '#55a868']

        plt.pie(method_counts, labels=method_counts.index, autopct='%1.1f%%', 
                startangle=140, colors=colors[:len(method_counts)], 
                wedgeprops={'edgecolor': 'black', 'linewidth': 1})

        plt.title('Distribution of Feature Scaling Techniques', weight='bold', size=13)
        chart_path = os.path.join(vis_dir, 'task_h_scaling_distribution.png')
        plt.savefig(chart_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✅ Scaling distribution chart saved to: {chart_path}\n")

    return df_scaled