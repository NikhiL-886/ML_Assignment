import numpy as np
import pandas as pd

def calculate_pearson_scratch(x, y):
    """Calculates Pearson correlation coefficient r from scratch."""
    df = pd.DataFrame({'x': x, 'y': y}).dropna()
    if len(df) < 2:
        return 0.0
    
    x_vals = df['x'].values
    y_vals = df['y'].values
    
    x_mean = np.mean(x_vals)
    y_mean = np.mean(y_vals)
    
    numerator = np.sum((x_vals - x_mean) * (y_vals - y_mean))
    denominator = np.sqrt(np.sum((x_vals - x_mean) ** 2) * np.sum((y_vals - y_mean) ** 2))
    
    if denominator == 0:
        return 0.0
    return numerator / denominator