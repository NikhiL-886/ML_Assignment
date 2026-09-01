import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from scratch_implementation.categorical_encoder import ScratchEncoder

def run_categorical_pipeline(df, max_ohe_categories=15, 
                             tabular_dir='summary/tabular_statistics/', 
                             vis_dir='summary/visualization/'):
    """
    Executes scratch categorical encoding and ensures reports are always generated.
    """
    os.makedirs(tabular_dir, exist_ok=True)
    os.makedirs(vis_dir, exist_ok=True)
    
    cat_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(cat_cols) == 0:
        print("ℹ️ No categorical columns found in current dataframe. Generating empty report status.")
        encoding_report = pd.DataFrame([{
            'Feature': 'None', 
            'Method': 'None', 
            'Reason': 'No categorical columns remaining in dataset.', 
            'Mapping': 'N/A'
        }])
        df_encoded = df
    else:
        print(f"Found {len(cat_cols)} categorical columns. Applying scratch encoding...")
        encoder = ScratchEncoder(max_ohe_categories=max_ohe_categories)
        df_encoded, encoding_report = encoder.fit_transform(df)

    # --- Save Tabular Summary as PDF ---
    pdf_path = os.path.join(tabular_dir, 'task_e_encoding_report.pdf')
    print("Generating paginated PDF for Encoding Report...")
    
    with PdfPages(pdf_path) as pdf:
        rows_per_page = 25
        chunks = [encoding_report[i:i + rows_per_page] for i in range(0, len(encoding_report), rows_per_page)]
        
        for idx, chunk in enumerate(chunks):
            fig, ax = plt.subplots(figsize=(14, len(chunk) * 0.5 + 1.5))
            ax.axis('tight')
            ax.axis('off')
            
            display_chunk = chunk.copy()
            if 'Mapping' in display_chunk.columns:
                display_chunk['Mapping'] = display_chunk['Mapping'].apply(
                    lambda x: (x[:55] + '...') if len(str(x)) > 55 else x
                )
            
            table = ax.table(cellText=display_chunk.values, colLabels=display_chunk.columns, loc='center', cellLoc='left')
            table.auto_set_font_size(False)
            table.set_fontsize(9)
            table.scale(1.2, 2.0)
            
            for (row, col), cell in table.get_celld().items():
                if row == 0:
                    cell.set_text_props(weight='bold', color='white')
                    cell.set_facecolor('#4c72b0')
                elif col == 2:
                    cell.set_width(0.35)
                elif col == 3:
                    cell.set_width(0.35)
                    
            plt.title(f"Task E - Categorical Encoding Report (Page {idx + 1} of {len(chunks)})", weight='bold', pad=20)
            pdf.savefig(fig, bbox_inches='tight')
            plt.close()

    print(f"✅ Encoding report saved to: {pdf_path}")

    # --- Save Visualization (Pie Chart) ---
    pie_chart_path = os.path.join(vis_dir, 'task_e_encoding_distribution.png')
    plt.figure(figsize=(8, 8))
    
    target_col = 'Method' if 'Method' in encoding_report.columns else encoding_report.columns[1]
    method_counts = encoding_report[target_col].value_counts()
    colors = ['#55a868', '#c44e52', '#8172b2']

    plt.pie(method_counts, labels=method_counts.index, autopct='%1.1f%%', 
            startangle=140, colors=colors[:len(method_counts)], 
            wedgeprops={'edgecolor': 'black', 'linewidth': 1})

    plt.title('Distribution of Categorical Encoding Methods', weight='bold', size=14)
    plt.savefig(pie_chart_path, bbox_inches='tight', dpi=300)
    plt.close()
    print(f"✅ Encoding pie chart saved to: {pie_chart_path}\n")
    
    return df_encoded