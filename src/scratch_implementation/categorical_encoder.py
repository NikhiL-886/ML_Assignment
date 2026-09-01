import pandas as pd

class ScratchEncoder:
    def __init__(self, max_ohe_categories=15):
        self.max_ohe = max_ohe_categories
        self.encoding_log = []
        
    def _label_encode(self, series):
        """Scratch Label Encoding using basic Python dictionaries."""
        unique_vals = series.unique()
        # Create a dictionary mapping: { 'category_name' : integer }
        mapping = {val: idx for idx, val in enumerate(unique_vals)}
        return series.map(mapping), mapping
        
    def _one_hot_encode(self, df, column):
        """Scratch One-Hot Encoding creating binary columns."""
        unique_vals = df[column].unique()
        ohe_df = pd.DataFrame(index=df.index)
        
        for val in unique_vals:
            # Create a new column named 'OriginalCol_Value' (e.g., City_Kanpur)
            col_name = f"{column}_{val}"
            # Fill with 1 if it matches, 0 if it doesn't
            ohe_df[col_name] = (df[column] == val).astype(int)
            
        return ohe_df

    def fit_transform(self, df):
        df_encoded = df.copy()
        # Find all text/categorical columns
        cat_cols = df_encoded.select_dtypes(include=['object', 'category']).columns
        
        for col in cat_cols:
            num_unique = df_encoded[col].nunique()
            
            # Rule 1: Binary (Label Encoding)
            if num_unique == 2:
                df_encoded[col], mapping = self._label_encode(df_encoded[col])
                self.encoding_log.append({'Feature': col, 'Method': 'Label Encoding', 
                                          'Reason': 'Binary feature (2 categories).', 'Mapping': str(mapping)})
            
            # Rule 2: Nominal but manageable (One-Hot Encoding)
            elif 2 < num_unique <= self.max_ohe:
                ohe_cols = self._one_hot_encode(df_encoded, col)
                # Drop original column and add the new one-hot columns
                df_encoded = df_encoded.drop(columns=[col])
                df_encoded = pd.concat([df_encoded, ohe_cols], axis=1)
                self.encoding_log.append({'Feature': col, 'Method': 'One-Hot Encoding', 
                                          'Reason': f'Nominal feature with {num_unique} categories (<= {self.max_ohe}).', 'Mapping': 'Created binary columns'})
            
            # Rule 3: High Cardinality (Fallback to Label Encoding)
            else:
                df_encoded[col], mapping = self._label_encode(df_encoded[col])
                self.encoding_log.append({'Feature': col, 'Method': 'Label Encoding (High Cardinality)', 
                                          'Reason': f'Too many categories ({num_unique} > {self.max_ohe}) for OHE.', 'Mapping': 'Integer mapping applied'})
                
        return df_encoded, pd.DataFrame(self.encoding_log)