import pandas as pd
import os
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

def render_table_as_image(df_subset, title, filename, output_dir):
    """Renders a small Pandas DataFrame as a single PNG image."""
    fig, ax = plt.subplots(figsize=(10, len(df_subset) * 0.4 + 1))
    ax.axis('tight')
    ax.axis('off')
    
    table = ax.table(cellText=df_subset.values, colLabels=df_subset.columns, loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.5)
    
    for (row, col), cell in table.get_celld().items():
        if row == 0:
            cell.set_text_props(weight='bold', color='white')
            cell.set_facecolor('#4c72b0')
            
    plt.title(title, weight='bold', pad=20)
    plt.savefig(os.path.join(output_dir, filename), bbox_inches='tight', dpi=300)
    plt.close()

def render_large_table_to_pdf(df_subset, title, filename, output_dir, rows_per_page=40):
    """Renders a massive DataFrame into a paginated PDF and saves a CSV backup."""
    # Always save a CSV backup for actual coding reference later
    df_subset.to_csv(os.path.join(output_dir, filename.replace('.pdf', '.csv')), index=False)
    
    pdf_path = os.path.join(output_dir, filename)
    
    # Create a multi-page PDF
    with PdfPages(pdf_path) as pdf:
        # Split the dataframe into chunks (pages)
        chunks = [df_subset[i:i + rows_per_page] for i in range(0, len(df_subset), rows_per_page)]
        
        for idx, chunk in enumerate(chunks):
            fig, ax = plt.subplots(figsize=(10, len(chunk) * 0.4 + 1))
            ax.axis('tight')
            ax.axis('off')
            
            table = ax.table(cellText=chunk.values, colLabels=chunk.columns, loc='center', cellLoc='center')
            table.auto_set_font_size(False)
            table.set_fontsize(10)
            table.scale(1.2, 1.5)
            
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#4c72b0')
                    
            plt.title(f"{title} (Page {idx + 1} of {len(chunks)})", weight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

def generate_visual_reports(df, target_col='isFraud', output_dir='summary/'):
    """Generates all reports for Tasks A2 and B1."""
    os.makedirs(output_dir, exist_ok=True)
    print("Generating graphical tables and charts...")

    # --- 1. TASK A2: General Report Table (Stays as a single PNG) ---
    num_rows, num_cols = df.shape
    num_cols_count = len(df.select_dtypes(include=['number']).columns)
    cat_cols_count = len(df.select_dtypes(include=['object', 'category']).columns)
    
    report_df = pd.DataFrame({
        'Metric': ['Number of Rows', 'Number of Columns', 'Target Variable', 'Numerical Variables', 'Categorical Variables', 'Source'],
        'Value': [num_rows, num_cols, target_col, num_cols_count, cat_cols_count, 'IEEE-CIS Kaggle']
    })
    render_table_as_image(report_df, 'Task A2 - Dataset Overview', 'task_a2_overview_table.png', output_dir)

    # --- 2. TASK B1: Full Missing Values Table (Saves as PDF & CSV) ---
    # Notice we removed the .head() so it grabs all 434 columns!
    missing_data = pd.DataFrame({
        'Feature': df.columns,
        'Data Type': df.dtypes.astype(str),
        'Unique Values': df.nunique().values,
        'Missing Values': df.isnull().sum().values,
        '% Missing': round((df.isnull().sum() / len(df)) * 100, 2).values
    }).sort_values(by='% Missing', ascending=False)
    
    # Pass to the new PDF generator
    render_large_table_to_pdf(missing_data, 'Task B1 - Full Missing Value Features', 'task_b1_full_missing_table.pdf', output_dir)

    # --- 3. VISUALIZATION: Target Variable Distribution (Chart) ---
    plt.figure(figsize=(8, 6))
    target_counts = df[target_col].value_counts()
    bars = plt.bar(target_counts.index.astype(str), target_counts.values, color=['#55a868', '#c44e52'])
    plt.title('Distribution of Target Variable (isFraud)', weight='bold')
    plt.xlabel('Is Fraudulent (0 = No, 1 = Yes)')
    plt.ylabel('Count')
    
    for bar in bars:
        yval = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2, yval, int(yval), ha='center', va='bottom')
        
    plt.savefig(os.path.join(output_dir, 'task_a2_target_distribution_chart.png'), bbox_inches='tight', dpi=300)
    plt.close()

    print(f"✅ All summaries saved to '{output_dir}'!")