import numpy as np
import pandas as pd

class ScratchOutlierDetector:
    def __init__(self, z_threshold=3.0, iqr_multiplier=1.5):
        self.z_threshold = z_threshold
        self.iqr_multiplier = iqr_multiplier
        self.metrics_log = []

    def analyze_column(self, series, col_name):
        """Computes IQR and Z-Score metrics from scratch for a single numerical column."""
        # Drop NaNs for statistical calculations
        clean_data = series.dropna()
        if len(clean_data) == 0:
            return

        # 1. IQR Calculations
        q1 = np.percentile(clean_data, 25)
        q3 = np.percentile(clean_data, 75)
        iqr = q3 - q1
        lower_bound = q1 - (self.iqr_multiplier * iqr)
        upper_bound = q3 + (self.iqr_multiplier * iqr)
        
        iqr_outliers = ((series < lower_bound) | (series > upper_bound)).sum()

        # 2. Z-Score Calculations
        mean = np.mean(clean_data)
        std = np.std(clean_data, ddof=1) # Sample standard deviation
        
        if std == 0 or np.isnan(std):
            z_outliers = 0
        else:
            z_scores = np.abs((series - mean) / std)
            z_outliers = (z_scores > self.z_threshold).sum()

        self.metrics_log.append({
            'Feature': col_name,
            'Q1': round(q1, 4),
            'Q3': round(q3, 4),
            'IQR': round(iqr, 4),
            'Lower Bound': round(lower_bound, 4),
            'Upper Bound': round(upper_bound, 4),
            'IQR Outliers': int(iqr_outliers),
            'Z-Score Outliers': int(z_outliers)
        })

    def fit_detect(self, df):
        num_cols = df.select_dtypes(include=[np.number]).columns
        for col in num_cols:
            self.analyze_column(df[col], col)
        return pd.DataFrame(self.metrics_log)