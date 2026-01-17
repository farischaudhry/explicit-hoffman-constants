import numpy as np
from dataclasses import dataclass


@dataclass(kw_only=True)
class HoffmanBoundMetrics:
    """Container for all Hoffman constant bounds."""
    # The induced operator p-norm used for the Hoffman.
    #! For now we ONLY support p=2.
    # Generally we might be interested in p = 1, 2, or ∞.
    hoffman_p_norm: float = 2.0  # Can use float('inf') for ∞-norm

    # Calculated with SVD methods for ||M_A^{-1}||_2
    exact_hoffman: float

    # Basic bounds on Hoffman constant used in the manuscript 
    max_bound_lower : float
    max_bound_upper : float

    # Upper bound on Hoffman constant using Weyl's inequality
    weyl_bound_upper: float

    # Other useful metrics
    lambda_min_AA: float = np.nan  # Minimum eigenvalue of G_AA
    condition_number_AA: float = np.nan  # Condition number of G_AA
    true_interaction_norm: float = 0.0  # ||G_{A^c A}||_2 - actual operator norm
    
    # Leverage-based interaction decomposition
    interaction_norm_bound: float = 0.0  # sqrt(λ_max × Σh_j) - leverage-based upper bound
    inactive_leverage_sum: float = 0.0  # Σ h_j(A) - sum of cross-leverage scores
    lambda_max_AA: float = np.nan  # λ_max(G_AA) - active block amplification
    max_leverage_score: float = 0.0  # Max leverage score of inactive features 


class HoffmanBoundCalculator:
    """
    Computes exact and theoretical Hoffman constants for the LASSO KKT system.
    
    For a fixed active set A, the KKT matrix is:
        M_A = [G_AA    0  ]
              [G_AcA   λI ]
              
              
    The ℓp Hoffman constant measures distance to feasibility in the ℓp norm:
        H_p(A) = 1 / σ_min^{(p)}(M_A)
    where σ_min^{(p)} is the smallest singular value in the induced ℓp norm. 

    The Hoffman constant H(A) = 1/σ_min⁺(M_A), where σ_min⁺ is the smallest positive (i.e., nonzero) singular value.
    """

    @staticmethod
    def build_kkt_matrix(G: np.ndarray, active_set: np.ndarray, lam: float) -> np.ndarray:
        """Build the KKT matrix M_A of size (|A| + |A^c|) x (|A| + |A^c|) for a given active set."""
        d = G.shape[0]
        inactive_set = np.array([i for i in range(d) if i not in active_set])
        
        s = len(active_set)
        m = len(inactive_set)
        
        if s == 0:
            # Empty active set: M_A = λI
            return lam * np.eye(d)
        
        # Build blocks
        G_AA = G[np.ix_(active_set, active_set)]
        
        if m == 0:
            # All features active: M_A = G_AA
            return G_AA
        
        G_AcA = G[np.ix_(inactive_set, active_set)]
        
        # Construct M_A = [G_AA  0  ]
        #                 [G_AcA λI ]
        M_A = np.zeros((s + m, s + m))
        M_A[:s, :s] = G_AA
        M_A[s:, :s] = G_AcA
        M_A[s:, s:] = lam * np.eye(m)
        
        return M_A
    
    @staticmethod
    def compute_exact_hoffman(G: np.ndarray, active_set: np.ndarray, lam: float, p_norm: float = 2.0) -> float:
        """Compute the exact Hoffman constant H(A) = 1/σ_min⁺(M_A) using SVD."""
        M_A = HoffmanBoundCalculator.build_kkt_matrix(G, active_set, lam)

        if p_norm != 2.0:
            raise NotImplementedError("Only p=2 norm is currently implemented for exact Hoffman computation currently.")
        
        # Compute singular values and filter to positive ones
        singular_values = np.linalg.svd(M_A, compute_uv=False)
        positive_svs = singular_values[singular_values > 1e-15]
        
        if len(positive_svs) == 0:
            return np.inf  # Singular matrix
        sigma_min_pos = positive_svs.min()
        return 1.0 / sigma_min_pos    

    @staticmethod
    def _compute_hoffman_max_bounds(G: np.ndarray, active_set: np.ndarray, lam: float, A: np.ndarray) -> tuple[float, float, dict]:
        """
        Compute basic upper and lower bounds on the Hoffman constant H(A).
        
        Lower bound: 1/min(λ_min(G_AA), λ)
        Upper bound: max(1/λ_min(G_AA), (1/λ)(1 + ||G_AcA||/λ_min(G_AA)))
        
        Uses leverage-based decomposition for interaction term:
            ||G_AcA||_2 <= sqrt(λ_max(G_AA)) * sqrt(Σ h_j)
        
        Returns: (lower_bound, upper_bound, components_dict)
        """
        d = G.shape[0]
        inactive_set = np.array([i for i in range(d) if i not in active_set])
        
        k = len(active_set)
        m = len(inactive_set)
        
        # Handle edge cases
        if k == 0:
            # Empty active set: H = 1/λ
            return 1.0/lam, 1.0/lam, {
                'lambda_min_AA': np.nan, 
                'condition_number_AA': np.nan, 
                'true_interaction_norm': 0.0,
                'interaction_norm_bound': 0.0,
                'inactive_leverage_sum': 0.0, 
                'lambda_max_AA': np.nan
            }
        
        # Extract blocks
        G_AA = G[np.ix_(active_set, active_set)]
        
        # Eigenvalues of active block
        eigvals_AA = np.linalg.eigvalsh(G_AA)
        lambda_min_AA = eigvals_AA.min()
        lambda_max_AA = eigvals_AA.max()
        
        if lambda_min_AA <= 1e-15 or np.isnan(lambda_min_AA):
            # Singular active block
            return np.inf, np.inf, {
                'lambda_min_AA': lambda_min_AA, 
                'condition_number_AA': np.inf, 
                'true_interaction_norm': 0.0,
                'interaction_norm_bound': 0.0,
                'inactive_leverage_sum': 0.0, 
                'lambda_max_AA': lambda_max_AA
            }
        
        # Lower bound: 1/min(λ_min(G_AA), λ)
        lower_bound = 1.0 / min(lambda_min_AA, lam)
        
        # Upper bound components using leverage-based decomposition
        inv_kappa = 1.0 / lambda_min_AA
        if m > 0 and A is not None:
            # Compute true interaction norm ||G_{A^c A}||_2
            G_AcA = G[np.ix_(inactive_set, active_set)]
            true_interaction_norm = np.linalg.norm(G_AcA, ord=2)
            
            # Compute leverage scores: h_j(A) = (1/n) ||P_A a_j||_2^2
            leverage_scores = HoffmanBoundCalculator._compute_leverage_scores(A, active_set)
            inactive_leverage_sum = np.sum(leverage_scores[inactive_set])
            
            # Leverage-based interaction bound:
            # ||G_AcA||_2 ≤ sqrt(λ_max(G_AA)) * sqrt(Σ h_j)
            interaction_norm_bound = np.sqrt(lambda_max_AA) * np.sqrt(inactive_leverage_sum)
            
            # Upper bound: max(1/κ, (1/λ)(1 + ||G_AcA||/κ))
            term1 = inv_kappa
            term2 = (1.0 / lam) * (1.0 + true_interaction_norm / lambda_min_AA)
            upper_bound = max(term1, term2)
        else:
            true_interaction_norm = 0.0
            interaction_norm_bound = 0.0
            inactive_leverage_sum = 0.0
            upper_bound = inv_kappa
        
        components = {
            'condition_number_AA': np.linalg.cond(G_AA),
            'lambda_min_AA': lambda_min_AA,
            'true_interaction_norm': true_interaction_norm,
            'interaction_norm_bound': interaction_norm_bound,
            'inactive_leverage_sum': inactive_leverage_sum,
            'lambda_max_AA': lambda_max_AA,
        }
        return float(lower_bound), float(upper_bound), components
    
    @staticmethod
    def _compute_hoffman_weyl_upper_bound(G: np.ndarray, active_set: np.ndarray, lam: float) -> float:
        """
        Compute upper bound on Hoffman constant using Weyl's inequality.
        
        Decompose M_A = D + E where:
            D = [G_AA  0 ]    E = [0     0]
                [0     λI]        [G_AcA 0]
        
        By Weyl: σ_min(M_A) ≥ σ_min(D) - ||E||_2
        
        This gives: H(A) ≤ 1/(min(λ_min(G_AA), λ) - ||G_AcA||_2)
        """
        d = G.shape[0]
        inactive_set = np.array([i for i in range(d) if i not in active_set])
        
        k = len(active_set)
        m = len(inactive_set)
        
        if k == 0:
            return 1.0 / lam
        
        G_AA = G[np.ix_(active_set, active_set)]
        lambda_min_AA = np.linalg.eigvalsh(G_AA).min()
        
        # σ_min(D) = min(λ_min(G_AA), λ)
        sigma_min_D = min(lambda_min_AA, lam)
        
        if m > 0:
            G_AcA = G[np.ix_(inactive_set, active_set)]
            norm_E = np.linalg.norm(G_AcA, ord=2)
        else:
            norm_E = 0.0
        
        # Weyl bound: σ_min(M_A) ≥ σ_min(D) - ||E||_2
        denominator = sigma_min_D - norm_E
        
        if denominator <= 1e-15:
            # Bound becomes infinite when interaction overwhelms stability
            return np.inf
        
        return 1.0 / denominator
    
    @staticmethod
    def _compute_leverage_scores(A: np.ndarray, active_set: np.ndarray) -> np.ndarray:
        """
        Compute leverage scores of inactive features relative to active manifold.
        
        Aligned with the updated paper: 
            h_j(A) = (1/n) * ||P_A a_j||_2^2
        where P_A is the projection onto the active subspace.
        """
        n, d = A.shape
        leverage_scores = np.zeros(d)
        
        if len(active_set) == 0:
            return leverage_scores
        
        try:
            # A_active = Q @ R. Q is n x |A| with orthogonal columns.
            Q, _ = np.linalg.qr(A[:, active_set])
            
            # We use the fact that ||Q @ Q.T @ a_j||^2 = ||Q.T @ a_j||_2^2 because Q is semi-orthogonal.
            projected_coords = Q.T @ A  # Shape: (|A|, d)
            
            # Sum of squares along the rows (axis=0) gives ||Q.T @ a_j||_2^2. Divide by n to normalize to [0,1].
            leverage_scores = np.sum(projected_coords**2, axis=0) / n
            
            # Explicitly set active features to 1.0 (they are in their own span)
            leverage_scores[active_set] = 1.0
            
        except np.linalg.LinAlgError:
            # Singular active matrix - subspace is not well-defined
            leverage_scores[:] = np.nan
        
        return leverage_scores

    @classmethod
    def compute_all(cls, G: np.ndarray, active_set: np.ndarray, lam: float, A: np.ndarray | None = None) -> HoffmanBoundMetrics:
        """
        Computes all bounds to populate HoffmanBoundMetrics.
        Parameters:
            G: Gram matrix (A^T A)/n
            active_set: Indices of active features
            lam: Regularization parameter λ
            A: Original design matrix (required for leverage-based bounds)
        """
        exact = cls.compute_exact_hoffman(G, active_set, lam)
        
        if A is None:
            raise ValueError('Matrix A is required for leverage-based interaction bounds')
        
        lower, upper, comps = cls._compute_hoffman_max_bounds(G, active_set, lam, A)
        weyl = cls._compute_hoffman_weyl_upper_bound(G, active_set, lam)
        
        max_lev = 0.0
        if A is not None:
            scores = cls._compute_leverage_scores(A, active_set)
            mask = np.ones(G.shape[0], dtype=bool)
            mask[active_set] = False
            if np.any(mask):
                max_lev = np.max(scores[mask])

        return HoffmanBoundMetrics(
            exact_hoffman=exact,
            max_bound_lower=lower,
            max_bound_upper=upper,
            weyl_bound_upper=weyl,
            lambda_min_AA=comps['lambda_min_AA'],
            condition_number_AA=comps.get('condition_number_AA', np.nan),
            true_interaction_norm=comps['true_interaction_norm'],
            interaction_norm_bound=comps['interaction_norm_bound'],
            inactive_leverage_sum=comps['inactive_leverage_sum'],
            lambda_max_AA=comps.get('lambda_max_AA', np.nan),
            max_leverage_score=max_lev,
        )
    
    @staticmethod
    def compute_rip_constant(A: np.ndarray, s: int, num_trials: int = 100) -> float:
        """
        Estimate the RIP constant δ_s for a design matrix A.
        Use only on matricies satisfying RIP.
        
        Uses MC estimation: δ_s ≈ max over random s-sparse vectors of
        |(||Av||_2^2/||v||_2^2) - 1|
        
        NOTE: Normalizes A to unit column norms as RIP is defined for normalized matrices.
        
        Parameters:
            A: Design matrix (n × d)
            s: Sparsity level
            num_trials: Number of random vectors to test
        Returns:
            Estimated RIP constant for sparsity level δ_s
        """
        _, d = A.shape
        
        # Normalize A to have unit column norms (standard for RIP)
        col_norms = np.linalg.norm(A, axis=0)
        col_norms[col_norms == 0] = 1  # Avoid division by zero
        A_normalized = A / col_norms
        
        delta_s = 0.0
        for _ in range(num_trials):
            # Generate random s-sparse vector
            support = np.random.choice(d, size=s, replace=False)
            v = np.zeros(d)
            v[support] = np.random.randn(s)
            
            # Compute RIP violation: |(||Av||_2^2/||v||_2^2) - 1|
            v_norm_sq = np.sum(v**2)
            Av_norm_sq = np.sum((A_normalized @ v)**2)
            
            violation = abs(Av_norm_sq / v_norm_sq - 1.0)
            delta_s = max(delta_s, violation)
        
        return delta_s
