from src.custom_ds.open_addressing_hash_map import OpenAddressingHashMap


class CountMatrix:
    """
    A sparse matrix implementation backed by a hash map to store counts.
    Used for tracking categorical feature occurrences per class.
    
    Attributes:
        map (OpenAddressingHashMap): The underlying hash map storage.
    """
    def __init__(self, capacity=1024, max_load_factor=0.7):
        """
        Initialize the count matrix.
        
        Args:
            capacity (int): Initial capacity of the hash map.
            max_load_factor (float): Load factor threshold for resizing.
        """
        self.map = OpenAddressingHashMap(capacity=capacity, max_load_factor=max_load_factor)

    def increment(self, class_id, feature_id, value_id, delta=1):
        """
        Increment the count for a specific (class, feature, value) tuple.
        
        Args:
            class_id (int): ID of the class.
            feature_id (int): index of the feature.
            value_id (int): Encoded value of the feature.
            delta (int): Amount to increment (default 1).
        """
        self.map.increment((class_id, feature_id, value_id), delta)

    def get_count(self, class_id, feature_id, value_id):
        """
        Retrieve the count for a specific tuple.
        
        Returns:
            int: The stored count (or 0 if not found).
        """
        return self.map.get((class_id, feature_id, value_id), 0)

    def contains(self, class_id, feature_id, value_id):
        """Check if a tuple exists in the matrix."""
        return self.map.contains((class_id, feature_id, value_id))

    def delete(self, class_id, feature_id, value_id):
        """Remove a tuple from the matrix."""
        return self.map.delete((class_id, feature_id, value_id))

    def size(self):
        """Return the number of active entries in the matrix."""
        return len(self.map)
