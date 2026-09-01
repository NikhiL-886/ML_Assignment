import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scratch_implementation.outlier_detector import ScratchOutlierDetector

def run_outlier_pipeline(df, tabular_dir='summary/tabular_statistics/', vis_dir='summary/visualization/'):
    """
    Runs scratch outlier detection, generates the IQR & Z-score summary PDF,
    creates comparison plots, and justifies treatment strategy (Retain/Cap).
    """
    os.makedirs(tabular_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print("Running scratch outlier detection (IQR and Z-Score)...")
    detector = ScratchOutlierDetector()
    metrics_report = detector.fit_detect(df)
    
    # 1. Save Tabular Summary as PDF
    pdf_path = os.path.join(tabular_dir, 'task_f_outlier_summary.pdf')
    print("Generating paginated PDF for Outlier Report...")
    
    with PdfPages(pdf_path) as pdf:
        rows_per_page = 20
        chunks = [metrics_report[i:i + rows_per_page] for i in range(0, len(metrics_report), rows_per_page)]
        
        for idx, chunk in enumerate(chunks):
            fig, ax = plt.subplots(figsize=(14, len(chunk) * 0.5 + 1.5))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=chunk.values, colLabels=chunk.columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(8)
            table.scale(1.2, 2.0)
            
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#4c72b0')
                    
            plt.title(f"Task F - Outlier Detection Report (Page {idx + 1} of {len(chunks)})", weight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"✅ Outlier report saved to: {pdf_path}")

    # 2. Save Comparison Visualization (Bar Chart of top columns by outliers)
    if not metrics_report.empty:
        plt.figure(figsize=(10, 6))
        top_outliers = metrics_report.sort_values(by='IQR Outliers', ascending=False).head(10)
        
        x = range(len(top_outliers))
        width = 0.35
        
        plt.bar([p - width/2 for p in x], top_outliers['IQR Outliers'], width, label='IQR Outliers', color='#4c72b0')
        plt.bar([p + width/2 for p in x], top_outliers['Z-Score Outliers'], width, label='Z-Score Outliers', color='#dd8452')
        
        plt.xlabel('Features', weight='bold')
        plt.ylabel('Outlier Count', weight='bold')
        plt.title('Comparison: IQR vs Z-Score Outlier Detection (Top Features)', weight='bold')
        plt.xticks(x, top_outliers['Feature'], rotation=45, ha='right')
        plt.legend()
        
        chart_path = os.path.join(vis_dir, 'task_f_outlier_comparison.png')
        plt.savefig(chart_path, bbox_inches='tight', dpi=300)
        plt.close()
        print(f"✅ Outlier comparison chart saved to: {chart_path}\n")

    # 3. Treatment Strategy Justification (Task F3)
    print("--- Task F3: Outlier Treatment Decision ---")
    print("Decision: RETAIN / WINDSORIZE (Cap)")
    print("Justification: In IEEE-CIS Fraud Detection, extreme values (like unusually high transaction amounts) "
          "are critical behavioral indicators of fraud. Deleting them would remove valuable risk signals. "
          "Therefore, outliers are retained for tree-based modeling.")

    return df # We retain records as justified for fraud detection