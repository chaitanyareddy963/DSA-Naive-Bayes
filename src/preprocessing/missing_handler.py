def compute_median(values):
    """
    Compute the median of a list of numbers.
    
    Args:
        values (list): List of numbers.
        
    Returns:
        float: The median value.
    """
    sorted_vals = sorted(values)
    n = len(sorted_vals)
    if n == 0:
        return 0.0
    if n % 2 == 1:
        return sorted_vals[n // 2]
    else:
        return (sorted_vals[n // 2 - 1] + sorted_vals[n // 2]) / 2.0


def compute_mean(values):
    """
    Compute the arithmetic mean of a list of numbers.
    
    Args:
        values (list): List of numbers.
        
    Returns:
        float: The mean value.
    """
    if len(values) == 0:
        return 0.0
    return sum(values) / len(values)


class MissingHandler:
    """
    Handles missing values in dataset by imputation (median for numeric, special token for categorical).
    
    Attributes:
        missing_markers (set): Set of strings representing missing values (e.g. '', '?').
        numeric_medians (dict): Map from column index to computed median value.
    """
    def __init__(self, missing_markers=None):
        if missing_markers is None:
            missing_markers = {"", "?"}
        self.missing_markers = set(missing_markers)
        self.numeric_medians = {}
        
    def is_missing(self, value):
        """Check if a string value is considered missing."""
        return value.strip() in self.missing_markers
    
    def fit(self, rows, numeric_col_idxs):
        """
        Compute medians for numeric columns ignoring missing values.
        
        Args:
            rows (list): List of data rows.
            numeric_col_idxs (list): Indices of numeric columns.
        """
        self.numeric_medians = {}
        
        for col_idx in numeric_col_idxs:
            valid_values = []
            for row in rows:
                val_str = row[col_idx].strip()
                if val_str not in self.missing_markers:
                    try:
                        valid_values.append(float(val_str))
                    except ValueError:
                        pass
            
            if valid_values:
                self.numeric_medians[col_idx] = compute_median(valid_values)
            else:
                self.numeric_medians[col_idx] = 0.0
    
    def transform_numeric(self, rows, numeric_col_idxs):
        """
        Impute missing numeric values with precomputed medians.
        
        Args:
            rows (list): List of data rows.
            numeric_col_idxs (list): Indices of numeric columns.
            
        Returns:
            list: List of rows with numeric values (as floats).
        """
        result = []
        for row in rows:
            row_values = []
            for col_idx in numeric_col_idxs:
                val_str = row[col_idx].strip()
                if val_str in self.missing_markers:
                    row_values.append(self.numeric_medians.get(col_idx, 0.0))
                else:
                    try:
                        row_values.append(float(val_str))
                    except ValueError:
                        row_values.append(self.numeric_medians.get(col_idx, 0.0))
            result.append(row_values)
        return result
    
    def transform_categorical(self, rows, cat_col_idxs, unknown_token="Unknown"):
        """
        Impute missing categorical values with a placeholder token.
        
        Args:
            rows (list): List of data rows.
            cat_col_idxs (list): Indices of categorical columns.
            unknown_token (str): Token to use for missing values.
            
        Returns:
            list: List of rows with categorical values (as strings).
        """
        result = []
        for row in rows:
            row_values = []
            for col_idx in cat_col_idxs:
                val_str = row[col_idx].strip()
                if val_str in self.missing_markers:
                    row_values.append(unknown_token)
                else:
                    row_values.append(val_str)
            result.append(row_values)
        return result
