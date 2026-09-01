import numpy as np
import pandas as pd

def calculate_anova_scratch(numerical_series, target_series):
    """
    Calculates ANOVA F-Statistic from scratch:
    F = Variance Between Groups / Variance Within Groups
    """
    df = pd.DataFrame({'feat': numerical_series, 'target': target_series}).dropna()
    if len(df) < 2:
        return 0.0
        
    overall_mean = df['feat'].mean()
    groups = [group['feat'].values for _, group in df.groupby('target')]
    
    if len(groups) < 2:
        return 0.0
        
    # Between-group variation (SS_between)
    ss_between = sum(len(g) * (np.mean(g) - overall_mean) ** 2 for g in groups)
    df_between = len(groups) - 1
    
    # Within-group variation (SS_within)
    ss_within = sum(np.sum((g - np.mean(g)) ** 2) for g in groups)
    df_within = len(df) - len(groups)
    
    if df_within <= 0 or ss_within == 0:
        return 0.0
        
    ms_between = ss_between / df_between
    ms_within = ss_within / df_within
    
    f_statistic = ms_between / ms_within
    return float(f_statistic)