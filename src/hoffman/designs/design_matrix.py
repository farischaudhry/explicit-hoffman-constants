import numpy as np
from dataclasses import dataclass


@dataclass
class DesignMatrix:
    A: np.ndarray
    true_beta: np.ndarray
    true_support: np.ndarray
    name: str

    @property
    def n(self) -> int: return self.A.shape[0]
    
    @property
    def d(self) -> int: return self.A.shape[1]

    @property
    def gram(self) -> np.ndarray:
        return (1/self.n) * self.A.T @ self.A
    

def normalize_cols(A: np.ndarray, target_norm: float) -> np.ndarray:
    """Normalizes columns of A to have the specified target norm."""
    norms = np.linalg.norm(A, axis=0)
    norms[norms == 0] = 1.0
    return A / norms * target_norm
