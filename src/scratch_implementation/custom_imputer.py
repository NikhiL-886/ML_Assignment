import pandas as pd
import numpy as np

class ScratchImputer:
    def __init__(self, missing_threshold=0.70, skew_threshold=1.0):
        self.missing_threshold = missing_threshold
        self.skew_threshold = skew_threshold
        self.imputation_log = []

    def fit_transform(self, df):
        """Scans the dataframe, decides the strategy, logs reasoning, and applies fixes."""
        df_clean = df.copy()
        
        # Calculate percentage of missing data for all columns
        missing_pct = df_clean.isnull().sum() / len(df_clean)
        
        for col in df_clean.columns:
            pct = missing_pct[col]
            
            if pct == 0:
                continue # Skip clean columns
                
            # 1. Removal of Features
            if pct > self.missing_threshold:
                df_clean = df_clean.drop(columns=[col])
                self.imputation_log.append({'Feature': col, 'Strategy': 'Dropped', 
                                            'Reason': f'Missing {pct*100:.1f}% data (Threshold: >{self.missing_threshold*100}%)'})
                continue
                
            # 2. Categorical / Non-Numerical -> Mode
            # Using pd.api.types.is_numeric_dtype ensures NO text/booleans slip through
            if not pd.api.types.is_numeric_dtype(df_clean[col]):
                mode_vals = df_clean[col].mode()
                if not mode_vals.empty:
                    df_clean[col] = df_clean[col].fillna(mode_vals[0])
                self.imputation_log.append({'Feature': col, 'Strategy': 'Mode', 
                                            'Reason': 'Non-numerical feature; replaced with most frequent value.'})
                continue
                
            # Numerical Logic (Mean vs Median)
            skewness = df_clean[col].skew()
            
            # Edge Case: If skewness fails to calculate (e.g., column has only 1 unique value)
            if pd.isna(skewness):
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                self.imputation_log.append({'Feature': col, 'Strategy': 'Mean', 
                                            'Reason': 'Skewness calculation failed; fallback to mean.'})
                continue

            # 3. Highly Skewed -> Median
            if abs(skewness) > self.skew_threshold:
                df_clean[col] = df_clean[col].fillna(df_clean[col].median())
                self.imputation_log.append({'Feature': col, 'Strategy': 'Median', 
                                            'Reason': f'Highly skewed distribution (Skew: {skewness:.2f}); median is robust to outliers.'})
            
            # 4. Normal Distribution -> Mean
            else:
                df_clean[col] = df_clean[col].fillna(df_clean[col].mean())
                self.imputation_log.append({'Feature': col, 'Strategy': 'Mean', 
                                            'Reason': f'Normal distribution (Skew: {skewness:.2f}); mean preserves central tendency.'})

        return df_clean, pd.DataFrame(self.imputation_log)