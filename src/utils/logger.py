import os


class Logger:
    """
    Simple logging utility to print to console and buffer for file saving.
    
    Attributes:
        verbose (bool): Whether to print messages to stdout.
        buffer (list): List of logged strings.
    """
    def __init__(self, verbose=True):
        """
        Initialize the logger.
        
        Args:
            verbose (bool): If True, prints logs to console immediately.
        """
        self.verbose = verbose
        self.buffer = []
    
    def log(self, msg=""):
        """
        Log a raw message.
        
        Args:
            msg (str): Message to log.
        """
        msg_str = str(msg)
        if self.verbose:
            print(msg_str)
        self.buffer.append(msg_str)
    
    def info(self, msg):
        """
        Log an informational message prefixed with [INFO].
        
        Args:
            msg (str): Message content.
        """
        self.log(f"[INFO] {msg}")
    
    def section(self, title):
        """
        Log a section header with separation lines.
        
        Args:
            title (str): Title of the section.
        """
        self.log("")
        self.log("=" * 50)
        self.log(title)
        self.log("=" * 50)
    
    def save(self, path):
        """
        Save buffered logs to a file.
        
        Args:
            path (str): File path to write logs to.
        """
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write("\n".join(self.buffer) + "\n")
    
    def get_contents(self):
        """Return the entire log buffer as a single string."""
        return "\n".join(self.buffer)
