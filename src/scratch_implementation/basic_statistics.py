def calculate_min(data):
    """Calculates the minimum value, ignoring None/NaN."""
    valid_data = [x for x in data if x == x and x is not None] 
    if not valid_data: return None
    
    current_min = valid_data[0]
    for val in valid_data[1:]:
        if val < current_min:
            current_min = val
    return current_min

def calculate_max(data):
    """Calculates the maximum value, ignoring None/NaN."""
    valid_data = [x for x in data if x == x and x is not None]
    if not valid_data: return None
    
    current_max = valid_data[0]
    for val in valid_data[1:]:
        if val > current_max:
            current_max = val
    return current_max

def calculate_mean(data):
    """Calculates the arithmetic mean: X = ΣX / n"""
    valid_data = [x for x in data if x == x and x is not None]
    if not valid_data: return None
    
    total = sum(valid_data)
    return total / len(valid_data)

def calculate_median(data):
    """Calculates the median (middle value) of a dataset."""
    valid_data = sorted([x for x in data if x == x and x is not None])
    n = len(valid_data)
    if n == 0: return None
    
    mid = n // 2
    if n % 2 == 0:
        return (valid_data[mid - 1] + valid_data[mid]) / 2.0
    else:
        return valid_data[mid]

def calculate_mode(data):
    """Calculates the most frequent value."""
    valid_data = [x for x in data if x == x and x is not None]
    if not valid_data: return None
    
    counts = {}
    for val in valid_data:
        counts[val] = counts.get(val, 0) + 1
        
    # Find the maximum count
    max_count = max(counts.values())
    # Return all values that share the maximum count
    return [k for k, v in counts.items() if v == max_count]

def calculate_variance(data, sample=True):
    """Calculates variance: Σ(xi - x̄)² / (n - 1) for sample."""
    valid_data = [x for x in data if x == x and x is not None]
    n = len(valid_data)
    if n < 2: return None
    
    mean_val = sum(valid_data) / n
    sum_squared_diff = sum((x - mean_val) ** 2 for x in valid_data)
    
    denominator = n - 1 if sample else n
    return sum_squared_diff / denominator

def calculate_std_dev(data, sample=True):
    """Calculates standard deviation (square root of variance)."""
    variance = calculate_variance(data, sample)
    if variance is None: return None
    return variance ** 0.5

def calculate_range(data):
    """Calculates the difference between max and min."""
    min_val = calculate_min(data)
    max_val = calculate_max(data)
    if min_val is None or max_val is None: return None
    return max_val - min_val