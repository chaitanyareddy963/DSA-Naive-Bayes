class RunningStats:
    """
    Computes running mean and variance using Welford's algorithm.
    This allows for single-pass computation without storing all values.

    Attributes:
        n (int): Number of samples seen so far.
        mean (float): Current running mean.
        M2 (float): Sum of squares of differences from the current mean.
    """
    def __init__(self):
        """Initialize the running statistics tracker."""
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def add(self, x):
        """
        Update stats with a new value x.
        
        Args:
            x (float): The new value to add.
        """
        self.n += 1
        delta = x - self.mean
        self.mean += delta / self.n
        delta2 = x - self.mean
        self.M2 += delta * delta2

    def variance(self):
        """
        Return the sample variance.
        
        Returns:
            float: Variance (M2 / (n - 1)). Returns 0.0 if n < 2.
        """
        if self.n < 2:
            return 0.0
        var = self.M2 / (self.n - 1)
        return max(0.0, var)

    def std_dev(self):
        """
        Return the standard deviation.
        
        Returns:
            float: Sqrt of variance.
        """
        import math
        return math.sqrt(self.variance())

