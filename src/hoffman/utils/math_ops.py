import numpy as np


def soft_threshold(x: np.ndarray, thresh: float) -> np.ndarray:
    """
    Standard proximal operator for the l1 norm.
    Points are shrunk towards zero by the threshold amount.
    """
    return np.sign(x) * np.maximum(np.abs(x) - thresh, 0)


def compute_gradient_mapping(beta: np.ndarray, beta_next: np.ndarray, step_size: float) -> np.ndarray:
    """
    Computes the gradient mapping G_t(beta), which serves as a 
    residual/optimality measure for non-smooth optimization.
    """
    return (beta - beta_next) / step_size
