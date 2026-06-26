import numpy as np
from dataclasses import dataclass

@dataclass(kw_only=True)
class HoffmanBoundMetrics:
    """Container for Hoffman constant bounds and associated geometric metrics."""
    exact_local_affine_hoffman: float 

    # Dimension-free Spectral Bounds
    dimension_free_bound_lower: float
    dimension_free_bound_upper: float
    
    # Other Euclidean & Non-Euclidean Bounds
    weyl_perturbation_bound: float
    l_inf_bound: float
    l_1_bound: float

    # Geometric Diagnostics
    lambda_min_active: float
    interaction_norm: float
    cross_leverage_bound: float
    inactive_leverage_sum: float
    max_leverage_score: float


class HoffmanBoundCalculator:
    """
    Computes explicit Hoffman constants for structured KKT systems 
    arising in sparse regularization (LASSO, Elastic Net, Adaptive LASSO).
    """

    @staticmethod
    def build_kkt_matrix(G: np.ndarray, active_set: np.ndarray, 
                         lam1: float, lam2: float = 0.0, 
                         weights: np.ndarray | None = None) -> np.ndarray:
        """Builds the affine KKT matrix M_A for the specified active set."""
        d = G.shape[0]
        active_mask = np.zeros(d, dtype=bool)
        active_mask[active_set] = True
        inactive_set = np.where(~active_mask)[0]
        
        s, m = len(active_set), len(inactive_set)
        
        if weights is not None:
            diag_vals = lam1 * weights[inactive_set]
        else:
            diag_vals = np.full(m, lam1)

        if s == 0:
            return np.diag(diag_vals)
        
        G_AA = G[np.ix_(active_set, active_set)]
        if lam2 > 0:
            G_AA = G_AA + lam2 * np.eye(s)
        
        if m == 0:
            return G_AA
        
        G_AcA = G[np.ix_(inactive_set, active_set)]
        
        M_A = np.zeros((s + m, s + m))
        M_A[:s, :s] = G_AA
        M_A[s:, :s] = G_AcA
        M_A[s:, s:] = np.diag(diag_vals)
        
        return M_A

    @staticmethod
    def _compute_leverage_scores(A: np.ndarray, active_set: np.ndarray) -> np.ndarray:
        """Computes cross-leverage scores of features relative to the active subspace."""
        n, d = A.shape
        h = np.zeros(d)
        if len(active_set) == 0:
            return h
        
        Q, _ = np.linalg.qr(A[:, active_set])
        projected_a = Q.T @ A
        h = np.sum(projected_a**2, axis=0) / n
        return h

    @classmethod
    def compute_all(cls, A: np.ndarray, active_set: np.ndarray, 
                    lam1: float, lam2: float = 0.0, 
                    weights: np.ndarray | None = None) -> HoffmanBoundMetrics:
        """Computes all theoretical and exact Hoffman bounds for the system."""
        n, d = A.shape
        G = (A.T @ A) / n
        active_mask = np.zeros(d, dtype=bool)
        active_mask[active_set] = True
        inactive_set = np.where(~active_mask)[0]
        s, m = len(active_set), len(inactive_set)

        # Exact Local Hoffman constant
        M_A = cls.build_kkt_matrix(G, active_set, lam1, lam2, weights)
        try:
            M_A_inv = np.linalg.inv(M_A)
            exact_hoffman = np.linalg.norm(M_A_inv, ord=2)
        except np.linalg.LinAlgError:
            return None 

        # Extract blocks for explicit bound components
        G_AA = G[np.ix_(active_set, active_set)] + lam2 * np.eye(s) if s > 0 else None
        G_AA_inv = np.linalg.inv(G_AA) if s > 0 else None
        
        w_inactive = lam1 * weights[inactive_set] if weights is not None else np.full(m, lam1)
        max_inv_lam = np.max(1.0 / w_inactive) if m > 0 else 0.0

        if s > 0 and m > 0:
            G_AcA = G[np.ix_(inactive_set, active_set)]
            # Interaction block scaled by the weights (Adaptive LASSO)
            interaction_mat = (G_AcA @ G_AA_inv) / w_inactive[:, None]
            interaction_norm = np.linalg.norm(interaction_mat, ord=2)
        else:
            interaction_mat = np.zeros((m, s))
            interaction_norm = 0.0

        # Dimension-Free Spectral Bounds (Lower and Upper)
        if s > 0:
            term_curv = np.linalg.norm(G_AA_inv, ord=2)
            
            # Lower bound is the maximum of the block norms
            spec_free_lower = max(term_curv, max_inv_lam, interaction_norm)
            # Upper bound is the square root of the sum of squares
            spec_free_upper = np.sqrt(term_curv**2 + max_inv_lam**2 + interaction_norm**2)
        else:
            spec_free_lower = spec_free_upper = max_inv_lam

        # Weyl Perturbation Bound
        if s > 0 and m > 0:
            sig_min_D = min(1.0/np.linalg.norm(G_AA_inv, ord=2), np.min(w_inactive))
            norm_E = np.linalg.norm(G[np.ix_(inactive_set, active_set)], ord=2)
            weyl_denom = sig_min_D - norm_E
            weyl_bound = 1.0 / weyl_denom if weyl_denom > 0 else np.inf
        else:
            weyl_bound = spec_free_upper

        # Non-Euclidean Bounds
        active_linf = np.linalg.norm(G_AA_inv, ord=np.inf) if s > 0 else 0.0
        inter_linf = max_inv_lam + np.linalg.norm(interaction_mat, ord=np.inf) if m > 0 else 0.0
        h_inf = max(active_linf, inter_linf)

        active_l1 = np.linalg.norm(G_AA_inv, ord=1) if s > 0 else 0.0
        inter_l1 = np.linalg.norm(interaction_mat, ord=1) if m > 0 else 0.0
        h_1 = max(active_l1 + inter_l1, max_inv_lam)

        # Leverage score diagnostics
        h_scores = cls._compute_leverage_scores(A, active_set)
        inact_lev_sum = np.sum(h_scores[inactive_set]) if m > 0 else 0.0
        max_lev = np.max(h_scores[inactive_set]) if m > 0 else 0.0
        
        if s > 0 and m > 0:
            sig_max_A_A = np.linalg.norm(A[:, active_set], ord=2)
            # Bound on ||G_AcA||_2 
            raw_leverage_bound = (1.0/np.sqrt(n)) * sig_max_A_A * np.sqrt(inact_lev_sum)
            # Bound the full KKT interaction term: ||W_Ac^-1 G_AcA G_AA^-1|| using submultiplicativity
            lev_norm_bound = np.linalg.norm(G_AA_inv, ord=2) * max_inv_lam * raw_leverage_bound
        else:
            lev_norm_bound = 0.0

        return HoffmanBoundMetrics(
            exact_local_affine_hoffman=exact_hoffman,
            dimension_free_bound_lower=spec_free_lower,
            dimension_free_bound_upper=spec_free_upper,
            weyl_perturbation_bound=weyl_bound,
            l_inf_bound=h_inf,
            l_1_bound=h_1,
            lambda_min_active=1.0/np.linalg.norm(G_AA_inv, ord=2) if s > 0 else 0.0,
            interaction_norm=interaction_norm,
            cross_leverage_bound=lev_norm_bound,
            inactive_leverage_sum=inact_lev_sum,
            max_leverage_score=max_lev
        )
