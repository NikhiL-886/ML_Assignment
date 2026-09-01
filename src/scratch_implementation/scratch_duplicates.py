def find_duplicates_scratch(df, id_column='TransactionID'):
    """
    Scratch implementation of Pandas df.duplicated() using Hash Tables.
    Converts values to strings to prevent NumPy memory/type mismatch errors.
    """
    # Allow for multiple ID columns to be passed as a list
    if isinstance(id_column, str):
        id_column = [id_column]
        
    columns_to_check = [col for col in df.columns if col not in id_column]
    
    seen_signatures = set()
    duplicate_mask = []
    
    print("Generating hash signatures for all rows (Stringified)...")
    
    # df.itertuples() is significantly safer for mixed data types than df.values
    for row in df[columns_to_check].itertuples(index=False, name=None):
        
        # Convert every item in the row to a string before creating the tuple
        # This guarantees that if it looks identical, it hashes identically
        stringified_row = tuple(str(item) for item in row)
        row_signature = hash(stringified_row)
        
        if row_signature in seen_signatures:
            duplicate_mask.append(True)
        else:
            seen_signatures.add(row_signature)
            duplicate_mask.append(False)
            
    return duplicate_mask