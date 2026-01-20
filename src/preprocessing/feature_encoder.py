from src.custom_ds.open_addressing_hash_map import OpenAddressingHashMap


class LabelEncoder:
    """
    Encode target labels with value between 0 and n_classes-1.
    Uses OpenAddressingHashMap for mapping.
    
    Attributes:
        map (OpenAddressingHashMap): Mapping from label to integer ID.
        next_id (int): Next available integer ID.
        id_to_label (list): Reverse mapping from ID to label.
    """
    def __init__(self):
        self.map = OpenAddressingHashMap(capacity=32, max_load_factor=0.7)
        self.next_id = 0
        self.id_to_label = []
    
    def fit_transform(self, values):
        """
        Fit label encoder and return encoded labels.
        
        Args:
            values (list): List of label values.
            
        Returns:
            list: Encoded integer labels.
        """
        result = []
        for val in values:
            encoded = self.map.get(val, None)
            if encoded is None:
                encoded = self.next_id
                self.map.put(val, encoded)
                self.id_to_label.append(val)
                self.next_id += 1
            result.append(encoded)
        return result
    
    def transform(self, values, unknown_id=-1):
        """
        Transform labels to normalized encoding.
        
        Args:
            values (list): List of labels to encode.
            unknown_id (int): ID to assign to unseen labels (default -1).
            
        Returns:
            list: Encoded labels.
        """
        result = []
        for val in values:
            encoded = self.map.get(val, unknown_id)
            result.append(encoded)
        return result
    
    def inverse_transform(self, ids):
        """
        Transform labels back to original encoding.
        
        Args:
            ids (list): List of integer labels.
            
        Returns:
            list: Original labels.
        """
        return [self.id_to_label[i] if 0 <= i < len(self.id_to_label) else str(i) 
                for i in ids]
    
    @property
    def num_classes(self):
        """Return the number of unique classes found."""
        return self.next_id



class FeatureEncoder:
    """
    Encode categorical features as integers.
    Separates encoding for each column.
    
    Attributes:
        encoders (list): List of OpenAddressingHashMap, one per feature.
        next_ids (list): List of next available ID for each feature.
        cardinalities (list): Number of unique values for each feature.
    """
    def __init__(self, num_features):
        self.encoders = [
            OpenAddressingHashMap(capacity=32, max_load_factor=0.7)
            for _ in range(num_features)
        ]
        # Start IDs at 1, reserving 0 for "Unknown" or "Absent"
        self.next_ids = [1] * num_features
        self.cardinalities = [1] * num_features
    
    def _transform_rows(self, rows, update=True):
        """
        Helper to transform rows, optionally updating the dictionary.
        
        Args:
            rows (list): List of rows to transform.
            update (bool): If True, add new values to dictionary. If False, treat new values as unknown.
        """
        result = []
        for row in rows:
            encoded_row = []
            for j, val in enumerate(row):
                encoded = self.encoders[j].get(val, None)
                if encoded is None:
                    if update:
                        encoded = self.next_ids[j]
                        self.encoders[j].put(val, encoded)
                        self.next_ids[j] += 1
                    else:
                        # Map unknown to 0 (reserved for Unknown/Absent)
                        encoded = 0 
                encoded_row.append(encoded)
            result.append(encoded_row)
        
        if update:
            self.cardinalities = list(self.next_ids)
        return result

    def fit_transform(self, rows):
        """
        Fit to data, then transform it.
        
        Args:
            rows (list): List of categorical feature vectors.
            
        Returns:
            list: Encoded feature vectors (indices).
        """
        return self._transform_rows(rows, update=True)
    
    def transform(self, rows):
        """
        Transform data to integer codes.
        
        Args:
            rows (list): List of categorical feature vectors.
            
        Returns:
            list: Encoded feature vectors.
        """
        return self._transform_rows(rows, update=False)
    
    def get_cardinalities(self):
        """Return the number of unique categories for each feature."""
        return list(self.cardinalities)
