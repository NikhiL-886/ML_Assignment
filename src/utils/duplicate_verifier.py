import pandas as pd
import matplotlib.pyplot as plt
import os
import sys
from matplotlib.backends.backend_pdf import PdfPages

# Ensure Python can find the scratch implementation
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scratch_implementation.scratch_duplicates import find_duplicates_scratch

def verify_and_drop_duplicates(df, id_column='TransactionID', out_dir='summary/verification_results/'):
    """
    Compares scratch vs pandas duplicate detection, saves a verification PDF,
    and returns a cleaned dataframe using the Pandas inbuilt result.
    """
    os.makedirs(out_dir, exist_ok=True)
    
    if isinstance(id_column, str):
        id_column = [id_column]
        
    columns_to_check = [col for col in df.columns if col not in id_column]
    
    print("Running Scratch Duplicate Detection...")
    # Get the boolean list from your scratch implementation
    scratch_mask = find_duplicates_scratch(df, id_column)
    scratch_count = sum(scratch_mask)
    
    print("Running Pandas Inbuilt Duplicate Detection...")
    # Get the boolean Series from Pandas
    pandas_mask = df.duplicated(subset=columns_to_check, keep='first')
    pandas_count = pandas_mask.sum()
    
    print(f"\nRESULTS:\nScratch found: {scratch_count} | Pandas found: {pandas_count}")
    
    # 1. Save Verification Report (PDF)
    summary_data = pd.DataFrame({
        'Metric': ['Original Rows', 'Duplicates (Scratch)', 'Duplicates (Pandas)', 'Match?'],
        'Value': [len(df), scratch_count, pandas_count, str(scratch_count == pandas_count)]
    })
    
    pdf_path = os.path.join(out_dir, 'duplicate_detection_verification.pdf')
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(8, 3))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=summary_data.values, colLabels=summary_data.columns, loc='center', cellLoc='center')
        table.auto_set_font_size(False)
        table.set_fontsize(12)
        table.scale(1.2, 2.0)
        
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4c72b0')
                
        plt.title("Duplicate Detection Verification", weight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()
        
    print(f"✅ Verification report saved to: {pdf_path}")
    
    # 2. Update and return dataframe using the Pandas result
    df_clean = df[~pandas_mask].copy()
    print(f"✅ Dataframe updated using Pandas. Rows dropped: {pandas_count}. New shape: {df_clean.shape}\n")
    
    return df_clean