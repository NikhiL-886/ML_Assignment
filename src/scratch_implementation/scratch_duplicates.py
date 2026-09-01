def find_duplicates_scratch(df, id_column='TransactionID'):
    """
    Scratch implementation of Pandas df.duplicated() using Hash Tables.
    Returns a list of booleans (True if duplicate, False if unique).
    """
    # 1. Isolate the columns we actually want to compare
    columns_to_check = [col for col in df.columns if col != id_column]
    
    # 2. Extract the data (converting to an array makes iteration much faster)
    subset_values = df[columns_to_check].values
    
    seen_signatures = set() # This is our Hash Table
    duplicate_mask = []
    
    print("Generating hash signatures for all rows...")
    
    for row in subset_values:
        # Convert the row to a tuple. (Lists cannot be hashed because they can change, 
        # but tuples are locked, so Python can generate a permanent hash for them).
        row_signature = hash(tuple(row))
        
        # Check if this fingerprint exists in our Hash Table (an O(1) instant lookup)
        if row_signature in seen_signatures:
            duplicate_mask.append(True)
        else:
            seen_signatures.add(row_signature)
            duplicate_mask.append(False)
            
    return duplicate_mask