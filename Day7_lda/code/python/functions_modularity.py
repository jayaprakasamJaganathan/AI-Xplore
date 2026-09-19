"""
Functions & Modularity - Practical Solutions (A6)
=================================================
Solutions for all 9 practical exercises from A6 PDF
"""

# ============================================================
# 21. calculate_average() - accepts list, returns average
# ============================================================
def calculate_average(numbers):
    """Calculate average of a list of numbers without NumPy."""
    if not numbers:
        return None
    total = 0
    count = 0
    for num in numbers:
        total += num
        count += 1
    return total / count


# ============================================================
# 22. find_maximum() - accepts list, returns largest value
# ============================================================
def find_maximum(numbers):
    """Find maximum value without using built-in max()."""
    if not numbers:
        return None
    max_val = numbers[0]
    for num in numbers[1:]:
        if num > max_val:
            max_val = num
    return max_val


# ============================================================
# 23. Modular program: read -> mean -> max -> display
# ============================================================
def read_numbers():
    """Read a list of numbers from user input."""
    user_input = input("Enter numbers separated by spaces: ")
    return [float(x) for x in user_input.split()]


def calculate_mean(numbers):
    """Calculate mean of numbers."""
    return calculate_average(numbers)


def calculate_max(numbers):
    """Calculate maximum of numbers."""
    return find_maximum(numbers)


def display_results(numbers, mean_val, max_val):
    """Display the results."""
    print(f"\nNumbers: {numbers}")
    print(f"Mean: {mean_val}")
    print(f"Maximum: {max_val}")


def modular_program():
    """Main modular program."""
    numbers = read_numbers()
    mean_val = calculate_mean(numbers)
    max_val = calculate_max(numbers)
    display_results(numbers, mean_val, max_val)


# ============================================================
# 24. count_even() - count even numbers in list
# ============================================================
def count_even(numbers):
    """Count even numbers in a list of integers."""
    count = 0
    for num in numbers:
        if int(num) % 2 == 0:
            count += 1
    return count


# ============================================================
# 25. Classification accuracy functions
# ============================================================
def calculate_accuracy(y_true, y_pred):
    """Calculate classification accuracy without sklearn."""
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have same length")
    if len(y_true) == 0:
        return 0.0
    
    correct = 0
    for true, pred in zip(y_true, y_pred):
        if true == pred:
            correct += 1
    return correct / len(y_true)


def display_result(accuracy):
    """Display accuracy result."""
    print(f"Classification Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")


# ============================================================
# 26. Function returning mean and standard deviation
# ============================================================
def mean_and_std(numbers):
    """Return mean and standard deviation of a list."""
    if not numbers:
        return None, None
    
    n = len(numbers)
    mean = sum(numbers) / n
    
    # Population standard deviation
    variance = sum((x - mean) ** 2 for x in numbers) / n
    std = variance ** 0.5
    
    return mean, std


# ============================================================
# 27. normalize() - min-max normalization
# ============================================================
def normalize(values):
    """
    Perform min-max normalization: (x - x_min) / (x_max - x_min)
    Returns normalized list.
    """
    if not values:
        return []
    
    x_min = min(values)
    x_max = max(values)
    
    if x_max == x_min:
        return [0.0] * len(values)
    
    return [(x - x_min) / (x_max - x_min) for x in values]


# ============================================================
# 28. ML Preprocessing Pipeline
# ============================================================
def load_data(filepath=None):
    """Load data from CSV or generate sample data."""
    if filepath:
        import pandas as pd
        return pd.read_csv(filepath)
    else:
        # Generate sample data with missing values
        import pandas as pd
        import numpy as np
        np.random.seed(42)
        data = {
            'feature1': np.random.randn(100) * 10 + 50,
            'feature2': np.random.randn(100) * 5 + 20,
            'feature3': np.random.randn(100) * 2 + 10,
            'target': np.random.randint(0, 2, 100)
        }
        # Add some missing values
        df = pd.DataFrame(data)
        df.loc[5:10, 'feature1'] = np.nan
        df.loc[20:25, 'feature2'] = np.nan
        return df


def remove_missing_values(df):
    """Remove rows with missing values."""
    return df.dropna()


def normalize_data(df, columns=None):
    """Normalize specified columns using min-max scaling."""
    import numpy as np
    df_norm = df.copy()
    if columns is None:
        columns = df.select_dtypes(include=[np.number]).columns.tolist()
        if 'target' in columns:
            columns.remove('target')
    
    for col in columns:
        x_min = df_norm[col].min()
        x_max = df_norm[col].max()
        if x_max != x_min:
            df_norm[col] = (df_norm[col] - x_min) / (x_max - x_min)
    return df_norm


def display_data(df, n_rows=5):
    """Display data summary."""
    print(f"Shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst {n_rows} rows:")
    print(df.head(n_rows))
    print(f"\nStatistics:")
    print(df.describe())


def ml_preprocessing_pipeline():
    """Complete ML preprocessing pipeline."""
    print("Step 1: Loading data...")
    df = load_data()
    
    print("Step 2: Removing missing values...")
    df = remove_missing_values(df)
    
    print("Step 3: Normalizing data...")
    df = normalize_data(df)
    
    print("Step 4: Displaying results...")
    display_data(df)
    
    return df


# ============================================================
# 29. Euclidean distance between two points
# ============================================================
def euclidean_distance(point1, point2):
    """
    Calculate Euclidean distance between two points.
    Points can be lists, tuples, or arrays of same dimension.
    """
    if len(point1) != len(point2):
        raise ValueError("Points must have same dimension")
    
    squared_diff_sum = 0
    for a, b in zip(point1, point2):
        squared_diff_sum += (a - b) ** 2
    
    return squared_diff_sum ** 0.5


# ============================================================
# Demo / Test Functions
# ============================================================
def run_all_demos():
    """Run all demonstrations."""
    print("=" * 60)
    print("A6 Functions & Modularity - Practical Solutions")
    print("=" * 60)
    
    # Test data
    test_numbers = [10, 25, 7, 42, 18, 33, 9, 56, 2, 14]
    
    # 21. calculate_average
    print(f"\n21. calculate_average({test_numbers}) = {calculate_average(test_numbers)}")
    
    # 22. find_maximum
    print(f"22. find_maximum({test_numbers}) = {find_maximum(test_numbers)}")
    
    # 24. count_even
    print(f"24. count_even({test_numbers}) = {count_even(test_numbers)}")
    
    # 26. mean_and_std
    mean_val, std_val = mean_and_std(test_numbers)
    print(f"26. mean_and_std({test_numbers}) = mean={mean_val:.2f}, std={std_val:.2f}")
    
    # 27. normalize
    normalized = normalize(test_numbers)
    print(f"27. normalize({test_numbers}) = {[f'{x:.3f}' for x in normalized]}")
    
    # 25. Classification accuracy
    y_true = [0, 1, 1, 0, 1, 1, 0, 0, 1, 0]
    y_pred = [0, 1, 1, 0, 0, 1, 0, 1, 1, 0]
    acc = calculate_accuracy(y_true, y_pred)
    print(f"25. calculate_accuracy: {acc:.4f}")
    display_result(acc)
    
    # 29. Euclidean distance
    p1 = [1, 2, 3]
    p2 = [4, 6, 8]
    dist = euclidean_distance(p1, p2)
    print(f"29. euclidean_distance({p1}, {p2}) = {dist:.4f}")
    
    # 28. ML Pipeline (just show it works)
    print("\n28. ML Preprocessing Pipeline:")
    ml_preprocessing_pipeline()
    
    print("\n" + "=" * 60)
    print("All demonstrations completed!")
    print("=" * 60)


if __name__ == "__main__":
    run_all_demos()