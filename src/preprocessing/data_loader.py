import csv
import os


def detect_delimiter(path):
    """
    Detect the delimiter of a CSV file (comma or tab).

    Args:
        path (str): Path to the CSV file.

    Returns:
        str: Detected delimiter (',' or '\t').
    """
    with open(path, "r", newline="", encoding="utf-8") as f:
        first = f.readline()
    if "\t" in first:
        return "\t"
    return ","


def load_csv(path, delimiter=None):
    """
    Load a CSV file into a list of rows.

    Args:
        path (str): Path to the CSV file.
        delimiter (str): Delimiter character (optional, auto-detected if None).

    Returns:
        tuple: (header, rows) where header is a list of column names and rows is a list of lists.
    """
    if delimiter is None:
        delimiter = detect_delimiter(path)
    
    rows = []
    with open(path, "r", newline="", encoding="utf-8") as f:
        reader = csv.reader(f, delimiter=delimiter, quotechar='"', skipinitialspace=True)
        header = next(reader)
        header = [h.strip().strip('"') for h in header]
        for r in reader:
            if not r:
                continue
            if len(r) != len(header):
                continue
            rows.append([x.strip().strip('"') for x in r])
    
    return header, rows


def load_uci_split(data_dir):
    """
    Load the official UCI Adult dataset split (adult.data and adult.test).

    Args:
        data_dir (str): Directory containing 'adult.data' and 'adult.test'.

    Returns:
        tuple: (header, train_rows, test_rows)
    """
    train_path = os.path.join(data_dir, "adult.data")
    test_path = os.path.join(data_dir, "adult.test")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        raise FileNotFoundError(
            f"UCI split files not found in {data_dir}. "
            "Expected adult.data and adult.test"
        )
    
    header, train_rows = load_csv(train_path)
    _, test_rows = load_csv(test_path)
    
    return header, train_rows, test_rows


def is_numeric_column(values):
    """
    Check if a column contains mostly numeric values.

    Args:
        values (list): List of string values from a column.

    Returns:
        bool: True if values can be parsed as floats, False otherwise.
    """
    seen = 0
    for v in values:
        s = v.strip()
        if s == "" or s == "?":
            continue
        seen += 1
        try:
            float(s)
        except ValueError:
            return False
    return seen > 0


def get_column_values(rows, col_idx):
    """
    Extract all values from a specific column index in the dataset.
    
    Args:
        rows (list): List of data rows.
        col_idx (int): Index of the column to extract.
        
    Returns:
        list: List of values in that column.
    """
    return [row[col_idx] for row in rows]


def random_split(rows, test_ratio, seed=42):
    """
    Randomly split data into training and testing sets.

    Args:
        rows (list): List of data rows.
        test_ratio (float): Proportion of data to use for testing (0.0 to 1.0).
        seed (int): Random seed for reproducibility.

    Returns:
        tuple: (train_rows, test_rows)
    """
    import random
    rng = random.Random(seed)
    shuffled_rows = list(rows)
    rng.shuffle(shuffled_rows)
    
    split_idx = int(len(shuffled_rows) * (1 - test_ratio))
    train_rows = shuffled_rows[:split_idx]
    test_rows = shuffled_rows[split_idx:]
    return train_rows, test_rows


def k_fold_split(rows, k, seed=42):
    """
    Split data into k folds for cross-validation.

    Args:
        rows (list): List of data rows.
        k (int): Number of folds.
        seed (int): Random seed.

    Returns:
        list: List of (train_rows, test_rows) tuples for each fold.
    """
    import random
    rng = random.Random(seed)
    shuffled_rows = list(rows)
    rng.shuffle(shuffled_rows)
    
    n = len(shuffled_rows)
    fold_size = n // k
    folds = []
    
    for i in range(k):
        start = i * fold_size
        # Last fold gets the remainder
        end = (i + 1) * fold_size if i != k - 1 else n
        
        test_fold = shuffled_rows[start:end]
        train_fold = shuffled_rows[:start] + shuffled_rows[end:]
        folds.append((train_fold, test_fold))
        
    return folds
