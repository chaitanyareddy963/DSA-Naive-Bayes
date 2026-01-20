class OpenAddressingHashMap:
    """
    A custom implementation of a Hash Map using Open Addressing with Linear Probing.

    Attributes:
        capacity (int): Current size of the internal storage.
        size (int): Number of active elements in the map.
        max_load_factor (float): Threshold ratio of size/capacity to trigger resize.
        collisions (int): Total number of collisions encountered during insertions.
        probes (int): Total number of probes performed during lookups/insertions.
        resizes (int): Total number of times the map has been resized.
    """
    _TOMBSTONE = object()

    def __init__(self, capacity=8, max_load_factor=0.7):
        """
        Initialize the hash map.

        Args:
            capacity (int): Initial capacity (default 8). Will be rounded up to power of 2 (min 4).
            max_load_factor (float): Load factor threshold for resizing.
        """
        if capacity < 4:
            capacity = 4
        cap = 1
        while cap < capacity:
            cap *= 2

        self._keys = [None] * cap
        self._values = [None] * cap
        self.size = 0
        self.max_load_factor = max_load_factor

        self.collisions = 0
        self.probes = 0
        self.resizes = 0

    def _index(self, key, capacity):
        """Compute the initial index for a key."""
        return (hash(key) & 0x7FFFFFFF) % capacity

    def _load_factor(self):
        """Return current load factor."""
        return self.size / len(self._keys)

    def _resize_if_needed(self):
        """Check if load factor exceeds threshold and resize if necessary."""
        if (self.size + 1) / len(self._keys) > self.max_load_factor:
            self._resize(len(self._keys) * 2)

    def _resize(self, new_capacity):
        """
        Resize the hash map to a new capacity and rehash all entries.
        
        Note: This operation contributes to the total probe count as it re-inserts elements.
        """
        old_keys = self._keys
        old_values = self._values

        self._keys = [None] * new_capacity
        self._values = [None] * new_capacity
        self.size = 0
        self.resizes += 1

        for i in range(len(old_keys)):
            k = old_keys[i]
            if k is None or k is self._TOMBSTONE:
                continue
            self.put(k, old_values[i])

    def _find_slot(self, key, for_insert):
        """
        Find the slot for a key using linear probing.
        
        Args:
            key: The key to search for.
            for_insert (bool): Whether this is for an insertion (handles tombstones).

        Returns:
            (int, bool): (index, found)
                         found is True if key exists, False otherwise.
        """
        capacity = len(self._keys)
        idx0 = self._index(key, capacity)
        idx = idx0
        first_tombstone = None
        steps = 0

        while True:
            self.probes += 1
            k = self._keys[idx]

            if k is None:
                if for_insert and first_tombstone is not None:
                    return first_tombstone, False
                return idx, False

            if k is self._TOMBSTONE:
                if for_insert and first_tombstone is None:
                    first_tombstone = idx
            elif k == key:
                return idx, True
            else:
                if for_insert:
                    self.collisions += 1

            idx = (idx + 1) % capacity
            steps += 1
            if steps >= capacity:
                raise RuntimeError("HashMap is full")

    def put(self, key, value):
        """
        Insert or update a key-value pair.
        
        Args:
            key: The key (must be hashable and not None).
            value: The value to store.
        """
        if key is None:
            raise ValueError("None keys not supported")

        self._resize_if_needed()
        idx, found = self._find_slot(key, for_insert=True)
        if found:
            self._values[idx] = value
            return

        self._keys[idx] = key
        self._values[idx] = value
        self.size += 1

    def get(self, key, default=None):
        """
        Retrieve a value by key.

        Args:
            key: The key to look up.
            default: Value to return if key is not found (default None).

        Returns:
            The value associated with the key, or default if not found.
        """
        if key is None:
            return default
        idx, found = self._find_slot(key, for_insert=False)
        if not found:
            return default
        return self._values[idx]

    def contains(self, key):
        """Check if the map contains a key."""
        marker = object()
        return self.get(key, marker) is not marker

    def delete(self, key):
        """
        Remove a key from the map.

        Args:
            key: The key to remove.

        Returns:
            bool: True if key was found and removed, False otherwise.
        """
        if key is None:
            return False
        idx, found = self._find_slot(key, for_insert=False)
        if not found:
            return False
        self._keys[idx] = self._TOMBSTONE
        self._values[idx] = None
        self.size -= 1
        return True

    def increment(self, key, delta=1):
        """
        Increment the value associated with a key (or initialize it if missing).
        
        Args:
            key: The key to increment.
            delta (int): Amount to add (default 1).
        """
        cur = self.get(key, None)
        if cur is None:
            self.put(key, delta)
        else:
            self.put(key, cur + delta)

    def stats(self):
        """Return internal statistics (collisions, probes, load factor)."""
        cap = len(self._keys)
        return {
            "capacity": cap,
            "size": self.size,
            "load_factor": self.size / cap if cap else 0.0,
            "collisions": self.collisions,
            "probes": self.probes,
            "resizes": self.resizes,
        }

    def __len__(self):
        """Return the number of items in the map."""
        return self.size

    def keys(self):
        """Iterator over keys."""
        for i in range(len(self._keys)):
            k = self._keys[i]
            if k is not None and k is not self._TOMBSTONE:
                yield k

    def values(self):
        """Iterator over values."""
        for i in range(len(self._keys)):
            k = self._keys[i]
            if k is not None and k is not self._TOMBSTONE:
                yield self._values[i]

    def items(self):
        """Iterator over (key, value) pairs."""
        for i in range(len(self._keys)):
            k = self._keys[i]
            if k is not None and k is not self._TOMBSTONE:
                yield (k, self._values[i])

    def average_probes_per_operation(self):
        """Calculate average probes per operation (put/get/etc)."""
        total_ops = self.size + self.collisions # Approximation
        if total_ops == 0:
            return 0.0
        return self.probes / total_ops
