import numpy as np
from scipy.linalg import toeplitz
from scipy.fftpack import dct

from hoffman.designs.design_matrix import DesignMatrix, normalize_cols


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
    def block_diagonal_ar1(n: int, d: int, s: int, block_size: int = 10, rho: float = 0.9) -> DesignMatrix:
        """
        Generates a block-diagonal design matrix.
        Features are correlated within blocks via an AR(1) structure  but are independent between blocks.
        
        Properties:
        - Local Correlation: Captures intra-block structures in features.
        - Hoffman: H is sensitive to whether the support A is spread across blocks or clustered within a single block.
        """
        # Ensure block_size is not larger than d
        block_size = min(block_size, d)
        n_blocks = int(np.ceil(d / block_size))
        
        # Sigma_ij = rho^|i-j|
        coords = np.arange(block_size)
        block_sigma = rho ** np.abs(coords[:, None] - coords[None, :])
        block_sigma += 1e-12 * np.eye(block_size) # Stability jitter
        L_block = np.linalg.cholesky(block_sigma)
        
        # Populate the design matrix block by block
        A_raw = np.zeros((n, d))
        for i in range(n_blocks):
            start_col = i * block_size
            end_col = min((i + 1) * block_size, d)
            current_w = end_col - start_col
            
            # Generate independent noise and project into the block's correlation structure
            Z = np.random.normal(0, 1, (n, current_w))
            # Slice L_block in case the last block is smaller than block_size
            A_raw[:, start_col:end_col] = Z @ L_block[:current_w, :current_w].T
            
        # Normalize columns
        A = normalize_cols(A_raw, np.sqrt(n))
        beta, support = DesignFactory.create_sparse_beta(d, s)
        return DesignMatrix(A, beta, support, f'Block-Diagonal (size={block_size}, rho={rho})')

    @staticmethod
    def partial_dct(n: int, d: int, s: int) -> DesignMatrix:
        """
        Randomly sampled rows from a Discrete Cosine Transform basis.
        
        Properties:
        - RIP: Satisfies RIP with high probability for n >= s log^4(d).
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
    def rademacher(n: int, d: int, s: int) -> DesignMatrix:
        """
        Rademacher Ensemble (binary +1/-1 entries).
        
        Properties:
        - RIP: Satisfies RIP similar to Gaussian.
        - Incoherence: Bounded coherence like Gaussian.
        """
        X = np.random.choice([-1, 1], size=(n, d))
        A = normalize_cols(X, np.sqrt(n))
        beta, support = DesignFactory.create_sparse_beta(d, s)
        return DesignMatrix(A, beta, support, 'Rademacher')
    
    @staticmethod
    def uniform_subgaussian(n: int, d: int, s: int) -> DesignMatrix:
        """
        Uniform distribution on [-sqrt(3), sqrt(3)] (subgaussian with variance 1).
        
        Properties:
        - RIP: Satisfies RIP similar to Gaussian.
        - Bounded: All entries bounded (unlike Gaussian tails).
        - Subgaussian: Tails decay exponentially.
        """
        X = np.random.uniform(-np.sqrt(3), np.sqrt(3), size=(n, d))
        A = normalize_cols(X, np.sqrt(n))
        beta, support = DesignFactory.create_sparse_beta(d, s)
        return DesignMatrix(A, beta, support, 'Uniform Subgaussian')
    
    @staticmethod
    def sparse_binary(n: int, d: int, s: int, sparsity: int = 3) -> DesignMatrix:
        """
        Sparse binary matrix with exactly 'sparsity' non-zero (+1/-1) entries per column.
        
        Properties:
        - Sparse: Only k non-zeros per column.
        - RIP: Satisfies RIP for appropriate k (typically k ~ log d).
        - Memory: Very efficient storage and computation.
        """
        if sparsity > n:
            sparsity = n
        
        X = np.zeros((n, d))
        for j in range(d):
            # Select k random rows for this column
            row_indices = np.random.choice(n, sparsity, replace=False)
            # Assign random +1/-1 values
            X[row_indices, j] = np.random.choice([-1, 1], size=sparsity)
        
        A = normalize_cols(X, np.sqrt(n))
        beta, support = DesignFactory.create_sparse_beta(d, s)
        return DesignMatrix(A, beta, support, f'Sparse Binary (k={sparsity})')
    
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

    @staticmethod
    def spiked(n: int, d: int, s: int, rho: float = 0.95) -> DesignMatrix:
        """
        Spiked Covariance Model (Global Multicollinearity).
        
        Properties:
        - Geometry: Features lie in a narrow cone around a shared latent factor.
        - Hoffman: H scales with sqrt(d) due to high interaction block leakage.
        - RE: Holds with kappa = 1 - rho, but conditioning is poor.
        """
        z = np.random.normal(0, 1, (n, 1))
        U = np.random.normal(0, 1, (n, d))
        A = np.sqrt(rho) * z + np.sqrt(1 - rho) * U
        A = normalize_cols(A, np.sqrt(n))
        beta, support = DesignFactory.create_sparse_beta(d, s)
        return DesignMatrix(A, beta, support, f'Spiked (rho={rho})')
