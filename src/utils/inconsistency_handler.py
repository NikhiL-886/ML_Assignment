import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def clean_inconsistencies(df, tabular_dir='summary/tabular_statistics/'):
    """
    Standardizes text, consolidates redundant categories, fixes impossible numbers,
    and generates a PDF report explaining the actions taken.
    """
    df_clean = df.copy()
    report_log = []
    
    os.makedirs(tabular_dir, exist_ok=True)
    print("Scanning for and cleaning inconsistent/invalid data...")

    # --- 1. Standardize Text (Capitalization & Spaces) ---
    text_cols = df_clean.select_dtypes(include=['object', 'category']).columns
    for col in text_cols:
        if df_clean[col].dtype == 'object':
            # Convert to string, lowercase, and strip trailing/leading spaces
            df_clean[col] = df_clean[col].astype(str).str.lower().str.strip()
            
    report_log.append({
        'Feature': 'All Categorical Features',
        'Issue': 'Incorrect capitalization and extra spaces',
        'Action Taken': 'Converted all text to lowercase and stripped spaces.'
    })

    # --- 2. Impossible/Negative Values ---
    if 'TransactionAmt' in df_clean.columns:
        invalid_amt = df_clean['TransactionAmt'] <= 0
        num_invalid = invalid_amt.sum()
        if num_invalid > 0:
            # Replace invalid amounts with the median of valid amounts
            median_amt = df_clean.loc[~invalid_amt, 'TransactionAmt'].median()
            df_clean.loc[invalid_amt, 'TransactionAmt'] = median_amt
            report_log.append({
                'Feature': 'TransactionAmt',
                'Issue': f'{num_invalid} records with negative or zero amount',
                'Action Taken': 'Replaced invalid values with the median transaction amount.'
            })

    # --- 3. Category Consolidation ---
    
    # Browser Versions (id_31)
    if 'id_31' in df_clean.columns:
        df_clean['id_31'] = df_clean['id_31'].replace('nan', 'unknown')
        df_clean.loc[df_clean['id_31'].str.contains('chrome', na=False), 'id_31'] = 'chrome'
        df_clean.loc[df_clean['id_31'].str.contains('safari', na=False), 'id_31'] = 'safari'
        df_clean.loc[df_clean['id_31'].str.contains('firefox', na=False), 'id_31'] = 'firefox'
        df_clean.loc[df_clean['id_31'].str.contains('edge|ie', na=False), 'id_31'] = 'ie_edge'
        df_clean.loc[df_clean['id_31'].str.contains('samsung', na=False), 'id_31'] = 'samsung'
        report_log.append({
            'Feature': 'id_31 (Browser Info)',
            'Issue': 'Hundreds of unique version numbers (e.g. chrome 62.0)',
            'Action Taken': 'Extracted base browser names (chrome, safari, firefox, edge).'
        })

    # Email Domains (P_emaildomain)
    if 'P_emaildomain' in df_clean.columns:
        df_clean['P_emaildomain'] = df_clean['P_emaildomain'].replace('nan', 'unknown')
        df_clean.loc[df_clean['P_emaildomain'].str.contains('yahoo|ymail|frontier', na=False), 'P_emaildomain'] = 'yahoo'
        df_clean.loc[df_clean['P_emaildomain'].str.contains('hotmail|outlook|live|msn', na=False), 'P_emaildomain'] = 'microsoft'
        df_clean.loc[df_clean['P_emaildomain'].str.contains('gmail', na=False), 'P_emaildomain'] = 'google'
        df_clean.loc[df_clean['P_emaildomain'].str.contains('netzero|aol', na=False), 'P_emaildomain'] = 'aol'
        report_log.append({
            'Feature': 'P_emaildomain',
            'Issue': 'Redundant country domains (e.g., yahoo.com.mx, yahoo.fr)',
            'Action Taken': 'Grouped into parent companies (google, microsoft, yahoo, aol).'
        })

    # Card Type (card6)
    if 'card6' in df_clean.columns:
        df_clean.loc[df_clean['card6'] == 'debit or credit', 'card6'] = 'debit'
        df_clean.loc[df_clean['card6'] == 'charge card', 'card6'] = 'credit'
        report_log.append({
            'Feature': 'card6 (Card Type)',
            'Issue': 'Ambiguous or invalid codes like "debit or credit"',
            'Action Taken': 'Merged ambiguous types into the primary "debit" and "credit" categories.'
        })

    # --- 4. Generate the PDF Explanation Report ---
    report_df = pd.DataFrame(report_log)
    pdf_path = os.path.join(tabular_dir, 'task_d2_inconsistency_report.pdf')
    
    with PdfPages(pdf_path) as pdf:
        fig, ax = plt.subplots(figsize=(12, len(report_df) * 0.6 + 1.5))
        ax.axis('tight')
        ax.axis('off')
        
        table = ax.table(cellText=report_df.values, colLabels=report_df.columns, loc='center', cellLoc='left')
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1.2, 2.5)
        
        # Style the table
        for (row, col), cell in table.get_celld().items():
            if row == 0:
                cell.set_text_props(weight='bold', color='white')
                cell.set_facecolor('#4c72b0')
            elif col == 1:
                cell.set_width(0.4) # Widen the Issue column
            elif col == 2: 
                cell.set_width(0.5) # Widen the Action Taken column
                
        plt.title("Task D2 - Invalid & Inconsistent Data Handling", weight='bold', pad=20)
        pdf.savefig(fig, bbox_inches='tight')
        plt.close()

    print(f"✅ Inconsistency report saved to {pdf_path}")
    return df_clean