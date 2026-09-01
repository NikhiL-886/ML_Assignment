import numpy as np
import pandas as pd

def calculate_variance_scratch(series):
    """Calculates variance from scratch: Variance(X) = sum((xi - x_mean)^2) / n"""
    clean_data = series.dropna()
    n = len(clean_data)
    if n == 0:
        return 0.0
    mean_val = np.sum(clean_data) / n
    squared_diff_sum = np.sum((clean_data - mean_val) ** 2)
    return squared_diff_sum / n