import numpy as np
import pandas as pd

def calculate_mutual_information_scratch(feature_series, target_series):
    """
    Calculates Mutual Information from scratch using discrete entropy:
    MI(X;Y) = H(Y) - H(Y|X)
    """
    df = pd.DataFrame({'feat': feature_series, 'target': target_series}).dropna()
    n = len(df)
    if n == 0:
        return 0.0
        
    # Marginal probabilities P(y)
    y_counts = df['target'].value_counts()
    p_y = y_counts / n
    # Entropy H(Y) = -sum(P(y) * log2 P(y))
    h_y = -np.sum(p_y * np.log2(p_y + 1e-12))
    
    # Joint probabilities P(x, y) and Conditional Entropy H(Y|X)
    joint_counts = pd.crosstab(df['feat'], df['target'])
    h_y_given_x = 0.0
    
    for x_val in joint_counts.index:
        x_row = joint_counts.loc[x_val]
        p_x = x_row.sum() / n
        p_y_given_x = x_row / x_row.sum()
        h_y_given_x += p_x * (-np.sum(p_y_given_x * np.log2(p_y_given_x + 1e-12)))
        
    mi = h_y - h_y_given_x
    return max(0.0, float(mi))