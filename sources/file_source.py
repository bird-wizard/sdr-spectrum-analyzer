import numpy as np

class FileSource:
    def __init__(self, filepath: str):
        self.filepath    = filepath
    #function returns np.ndarray
    def read(self) -> np.ndarray:
        samples = np.fromfile(self.filepath, dtype=np.complex64)
        if len(samples) == 0:
            raise ValueError(f"No samples loaded from {self.filepath}")
        return samples