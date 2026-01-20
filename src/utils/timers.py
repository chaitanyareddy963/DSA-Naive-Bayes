import time


class Timer:
    """
    Context manager for timing code blocks.
    
    Attributes:
        start_time (float): Timestamp when entered.
        end_time (float): Timestamp when exited.
        elapsed (float): Duration in seconds.
    """
    def __init__(self):
        self.start_time = None
        self.end_time = None
        self.elapsed = 0.0
    
    def __enter__(self):
        self.start_time = time.perf_counter()
        return self
    
    def __exit__(self, *args):
        self.end_time = time.perf_counter()
        self.elapsed = self.end_time - self.start_time


class TimingResults:
    """
    Store and report multiple timing durations.
    
    Attributes:
        timings (dict): Map from name to duration (seconds).
    """
    def __init__(self):
        self.timings = {}
    
    def add(self, name, elapsed):
        """Add a timing result."""
        self.timings[name] = elapsed
    
    def get(self, name):
        """Get a timing result by name."""
        return self.timings.get(name, 0.0)
    
    def to_dict(self):
        """Convert timings to a dictionary with rounded values."""
        return {k: round(v, 6) for k, v in self.timings.items()}
    
    def summary(self):
        """Format timings as a summary string."""
        lines = ["Timing Summary:"]
        for name, elapsed in self.timings.items():
            lines.append(f"  {name}: {elapsed:.4f}s")
        return "\n".join(lines)
