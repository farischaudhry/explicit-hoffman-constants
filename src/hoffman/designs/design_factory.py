import numpy as np
from scipy.linalg import toeplitz, dct

from hoffman.designs import DesignMatrix, normalize_cols


class DesignFactory:
    @staticmethod
    def create_sparse_beta(d: int, s: int, signal_mag: float = 1.0) -> tuple[np.ndarray, np.ndarray]:
        """Creates an s-sparse target beta of dimension d."""
        beta = np.zeros(d)
        support = np.random.choice(d, s, replace=False)
        beta[support] = np.random.choice([-1, 1], s) * signal_mag
        return beta, np.sort(support)

    @staticmethod
    def diagonal(n: int, d: int, s: int, min_var: float = 0.01) -> DesignMatrix:
        """
        Generates a design with orthogonal columns but different variances.
        The Gram matrix is G = diag(gamma_1, ..., gamma_d).
        
        Properties:
        - Interaction: ||G_AcA|| = 0 (perfectly decoupled).
        - Hoffman: H = 1 / min(min_j gamma_j, lambda).
        - Use Case: Proving that poor scaling alone can blow up H, even without correlation.
        """
        X = np.random.normal(0, 1, (n, d))
        Q, _ = np.linalg.qr(X)
        
        # Create a spectrum of variances from 1.0 down to min_var
        gamma = np.linspace(1.0, min_var, d)
        A = Q[:, :d] @ np.diag(np.sqrt(gamma * n))
        
        beta, support = DesignFactory.create_sparse_beta(d, s)
        return DesignMatrix(A, beta, support, f'Diagonal (min_var={min_var})')


    @staticmethod
    def orthonormal(n: int, d: int, s: int) -> DesignMatrix:
        """
        Generates a perfectly orthonormal design matrix (requires n >= d).
        
        Properties:
        - RIP: delta = 0 (perfect isometry).
        - Hoffman: H = 1 / min(1, lambda). Best possible stability.
        """
        if n < d:
            raise ValueError('Orthonormal design requires n >= d.')
        
        X = np.random.normal(0, 1, (n, d))
        Q, _ = np.linalg.qr(X)
        A = Q * np.sqrt(n) 
        beta, support = DesignFactory.create_sparse_beta(d, s)
        return DesignMatrix(A, beta, support, 'Orthonormal')

    @staticmethod
    def gaussian(n: int, d: int, s: int) -> DesignMatrix:
        """
        Standard Gaussian Ensemble.
        
        Properties:
        - Statistical: Satisfies RIP with high probability for n >= s log(d/s).
        - Hoffman: Dimension-independent H-constant on s-sparse manifolds.
        """
        X = np.random.normal(0, 1, (n, d))
        A = normalize_cols(X, np.sqrt(n))
        beta, support = DesignFactory.create_sparse_beta(d, s)
        return DesignMatrix(A, beta, support, 'Gaussian')

    @staticmethod
    def ar1_stochastic(n: int, d: int, s: int, rho: float = 0.9) -> DesignMatrix:
        """
        Stochastic realization of an AR(1) process per row.
        
        Properties:
        - Interaction: Moderate local correlation.
        - Stability: Stochastic jitter prevents rigid manifold dependency.
        """
        X = np.zeros((n, d))
        X[:, 0] = np.random.normal(0, 1, n)
        noise_std = np.sqrt(1 - rho**2)
        for j in range(1, d):
            X[:, j] = rho * X[:, j-1] + np.random.normal(0, noise_std, n)
        
        A = normalize_cols(X, np.sqrt(n))
        beta, support = DesignFactory.create_sparse_beta(d, s)
        return DesignMatrix(A, beta, support, f'AR1-Stochastic (rho={rho})')
    
    @staticmethod
    def strict_toeplitz(n: int, d: int, s: int, rho: float = 0.9) -> DesignMatrix:
        """
        Deterministic Toeplitz Covariance Design (Sigma_ij = rho^|i-j|).
        
        Properties:
        - Local conditioning: H explodes if active indices are close together.
        """
        first_row = rho ** np.arange(d)
        Sigma = toeplitz(first_row)
        
        # Sigma = L @ L.T
        L = np.linalg.cholesky(Sigma)
        Z = np.random.normal(0, 1, (n, d))
        A_raw = Z @ L.T
        
        A = normalize_cols(A_raw, np.sqrt(n))
        beta, support = DesignFactory.create_sparse_beta(d, s)
        return DesignMatrix(A, beta, support, f'Strict-Toeplitz (rho={rho})')

    @staticmethod
    def partial_dct(n: int, d: int, s: int) -> DesignMatrix:
        """
        Randomly sampled rows from a Discrete Cosine Transform basis.
        
        Properties:
        - Coherence: Low mutual coherence.
        - High-Dim: Standard compressed sensing benchmark.
        """
        I = np.eye(d)
        D = dct(I, axis=0, norm='ortho')
        indices = np.random.choice(d, n, replace=False)
        A = D[indices, :] * np.sqrt(n)
        
        beta, support = DesignFactory.create_sparse_beta(d, s)
        return DesignMatrix(A, beta, support, 'Partial DCT')
    
    @staticmethod
    def power_law(n: int, d: int, s: int, decay: float = 1.5) -> DesignMatrix:
        """
        Ill-conditioned design with Power-Law singular value decay.
        
        Properties:
        - Spectral: sigma_k ~ k^(-decay).
        - Hoffman: Can grow arbitrarily large as d increases.
        """
        X = np.random.normal(0, 1, (n, d))
        U, _, Vt = np.linalg.svd(X, full_matrices=False)
        S = np.array([k**(-decay) for k in range(1, min(n, d) + 1)])
        
        A_raw = U @ np.diag(S) @ Vt[:min(n, d), :]
        A = normalize_cols(A_raw, np.sqrt(n))
        
        beta, support = DesignFactory.create_sparse_beta(d, s)
        return DesignMatrix(A, beta, support, f'Power-Law (decay={decay})')

    @staticmethod
    def low_rank(n: int, d: int, s: int, rank: int = 10) -> DesignMatrix:
        """
        Generates a strictly low-rank design matrix where rank < min(n, d).
        
        Properties:
        - Rank: exactly 'rank'.
        - Nullspace: Has a non-trivial nullspace of dimension d - rank.
        - Hoffman: H exists locally for s-sparse sets if s <= rank and RNP holds.
        - Use Case: Testing if the Hoffman logic holds when the global Gram is singular.
        """
        if rank > min(n, d):
            raise ValueError(f'Rank {rank} cannot exceed dimensions ({n}, {d})')

        # Create two thin matrices to form a rank-r matrix
        U = np.random.normal(0, 1, (n, rank))
        V = np.random.normal(0, 1, (d, rank))
        A_raw = U @ V.T
        
        A = normalize_cols(A_raw, np.sqrt(n))
        
        # Require s <= rank for model identifiability; we do not enforce this
        beta, support = DesignFactory.create_sparse_beta(d, s)
        return DesignMatrix(A, beta, support, f'Low-Rank (rank={rank})')
