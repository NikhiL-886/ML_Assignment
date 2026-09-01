import numpy as np
import pandas as pd

def calculate_chi_square_scratch(feature_series, target_series):
    """
    Calculates Chi-Square statistic from scratch using contingency table:
    chi2 = sum((O - E)^2 / E)
    """
    df = pd.DataFrame({'feature': feature_series, 'target': target_series}).dropna()
    if len(df) == 0:
        return 0.0, 0
        
    # 1. Create contingency table (Observed frequencies O)
    contingency_table = pd.crosstab(df['feature'], df['target'])
    O = contingency_table.values
    
    # 2. Calculate Expected frequencies E = (Row Total * Column Total) / Grand Total
    row_totals = O.sum(axis=1, keepdims=True)
    col_totals = O.sum(axis=0, keepdims=True)
    grand_total = O.sum()
    
    if grand_total == 0:
        return 0.0, 0
        
    E = (row_totals @ col_totals) / grand_total
    
    # Avoid division by zero
    E[E == 0] = 1e-8
    
    # 3. Calculate Chi-Square statistic
    chi2_stat = np.sum(((O - E) ** 2) / E)
    
    # 4. Degrees of freedom: df = (r - 1) * (c - 1)
    r, c = O.shape
    dof = (r - 1) * (c - 1)
    
    return float(chi2_stat), int(dof)