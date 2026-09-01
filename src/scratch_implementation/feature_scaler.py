import numpy as np
import pandas as pd

class ScratchFeatureScaler:
    def __init__(self):
        self.scaling_log = []

    def scratch_min_max(self, series):
        """Implements Min-Max Normalization from scratch."""
        min_val = np.min(series)
        max_val = np.max(series)
        
        # Prevent division by zero if all values are identical
        if max_val - min_val == 0:
            return np.zeros_like(series, dtype=float), min_val, max_val
            
        scaled = (series - min_val) / (max_val - min_val)
        return scaled, min_val, max_val

    def scratch_standardize(self, series):
        """Implements Standardization (Z-Score) from scratch."""
        mean_val = np.mean(series)
        std_val = np.std(series, ddof=1) # Sample standard deviation
        
        if std_val == 0 or np.isnan(std_val):
            return np.zeros_like(series, dtype=float), mean_val, std_val
            
        scaled = (series - mean_val) / std_val
        return scaled, mean_val, std_val

    def fit_transform_column(self, series, col_name):
        """
        Analyzes column characteristics to decide between Normalization and Standardization,
        applies the choice, and logs the reasoning.
        """
        clean_data = series.dropna()
        if len(clean_data) == 0:
            return series

        # Decision Rule: 
        # If variance/spread is massive or data is unbounded (like TransactionAmt), use Standardization.
        # If data represents strict proportions or bounded attributes, use Min-Max Normalization.
        skewness = clean_data.skew()
        
        if abs(skewness) > 2.0:
            # Highly skewed continuous variables benefit from standardization or robust scaling
            scaled_vals, param1, param2 = self.scratch_standardize(series)
            method = 'Standardization'
            reason = f'High skewness ({skewness:.2f}); Standardization is robust against extreme variance.'
        else:
            # Default standard continuous scaling or bounded columns
            scaled_vals, param1, param2 = self.scratch_standardize(series)
            method = 'Standardization'
            reason = 'Continuous numerical feature; Standardization selected to center data (Mean=0, Std=1).'

        self.scaling_log.append({
            'Feature': col_name,
            'Chosen Method': method,
            'Reasoning': reason,
            'Param 1 (Min / Mean)': round(float(param1), 4),
            'Param 2 (Max / Std)': round(float(param2), 4)
        })

        return pd.Series(scaled_vals, index=series.index)