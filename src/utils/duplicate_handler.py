import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.backends.backend_pdf import PdfPages

# Import your scratch logic
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scratch_implementation.scratch_duplicates import find_duplicates_scratch

def handle_duplicates(df, id_column='TransactionID', tabular_dir='summary/tabular_statistics/', vis_dir='summary/visualization/'):
    """Detects duplicates using scratch hashing, drops them, and generates reports."""
    
    os.makedirs(tabular_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    print("Checking for duplicates using scratch hash implementation...")
    
    original_records = len(df)
    
    # --- CHANGED: Using our scratch implementation instead of Pandas ---
    duplicate_mask_list = find_duplicates_scratch(df, id_column)
    
    # Convert our Python list back into a Pandas boolean Series to drop the rows
    duplicate_mask = pd.Series(duplicate_mask_list, index=df.index)
    # -------------------------------------------------------------------

    duplicate_records = duplicate_mask.sum()
    
    # Drop them
    df_deduped = df[~duplicate_mask].copy()
    records_after = len(df_deduped)
    
    print(f"Found and dropped {duplicate_records} duplicate rows.")

    # 2. Save Tabular Summary (PDF)
    summary_data = pd.DataFrame({
        'Item': ['Original Records', 'Duplicate Records', 'Records After Treatment'],
        'Value': [original_records, duplicate_records, records_after]
    })
    
    pdf_path = os.path.join(tabular_dir, 'task_d1_duplicate_summary.pdf')
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=summary_data.values, colLabels=summary_data.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2.0)
        
        # Style header
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4c72b0')
                
        plt.title("Task D1 - Duplicate Detection Summary", weight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    # 3. Save Visualization (Bar Chart)
    plt.figure(figsize=(8, 6))
    bars = plt.bar(['Before Treatment', 'After Treatment'], [original_records, records_after], color=['#c44e52', '#55a868'])
    plt.title('Row Count: Before vs After Duplicate Removal', weight='bold')
    plt.ylabel('Number of Records')
    
    # Add count labels
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, f"{int(yval):,}", ha='center', va='bottom')
        
    chart_path = os.path.join(vis_dir, 'task_d1_duplicate_chart.png')
    plt.savefig(chart_path, bbox_inches='tight', dpi=300)
    plt.close()
    
    print(f"✅ PDF summary saved to {pdf_path}")
    print(f"✅ Bar chart saved to {chart_path}")
    
    return df_deduped