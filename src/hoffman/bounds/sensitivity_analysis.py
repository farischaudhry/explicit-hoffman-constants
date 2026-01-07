import numpy as np
from dataclasses import dataclass
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold

from hoffman.designs.design_matrix import DesignMatrix


@dataclass(kw_only=True)
class SensitivityAnalysisMetrics:
    """Container for sensitivity analysis metrics."""
    total_variance: float  # Total variance of recovered betas under perturbations
    unique_supports: int   # Number of unique active sets observed
    mean_beta: np.ndarray  # Mean of recovered betas


class SensitivityAnalysis:
    """
    Analyzes how the LASSO solution fluctuates under small perturbations 
    of the input data. High fluctuations correlate with high Hoffman constant.
    """
    @staticmethod
    def compute_stability_random_perturbations(
        design: DesignMatrix, lam: float, n_perturbations:int = 20, noise_level: float = 0.01
    ) -> SensitivityAnalysisMetrics:
        """
        Perturbs y multiple times and measures the variance of the recovered beta.
        
        Theory: Hoffman constant measures distance to infeasibility. High H(A)
        means small changes in data can cause large changes in solution.
        """
        y_base = design.A @ design.true_beta
        recovered_betas = []
        
        # Use high-precision solver to isolate Hoffman effect from solver noise
        solver = Lasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=50000)

        for _ in range(n_perturbations):
            y_noisy = y_base + noise_level * np.random.normal(0, 1, design.n)
            solver.fit(design.A, y_noisy)
            recovered_betas.append(solver.coef_.copy())
            
        recovered_betas = np.array(recovered_betas)
        
        # Stability Metrics
        # Variance of the estimates (higher = more unstable)
        beta_variance = np.var(recovered_betas, axis=0).sum()
        
        # Support Stability (how often the active set changes)
        supports = [set(np.where(np.abs(b) > 1e-5)[0]) for b in recovered_betas]
        unique_supports = len(set(tuple(sorted(s)) for s in supports))
        
        return SensitivityAnalysisMetrics(
            total_variance=beta_variance,
            unique_supports=unique_supports,
            mean_beta=np.mean(recovered_betas, axis=0)
        )
    
    @staticmethod
    def compute_stability_cross_validation(
        design: DesignMatrix, lam: float, n_folds: int = 5, noise_level: float = 0.01, random_seed: int = 0
    ) -> SensitivityAnalysisMetrics:
        """Performs cross-validation with perturbed data folds to measure solution stability."""
        y_base = design.A @ design.true_beta
        recovered_betas = []
        
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=random_seed)
        solver = Lasso(alpha=lam, fit_intercept=False, tol=1e-12, max_iter=50000)

        for train_index, _ in kf.split(design.A):
            y_fold = y_base + noise_level * np.random.normal(0, 1, design.n)
            solver.fit(design.A[train_index], y_fold[train_index])
            recovered_betas.append(solver.coef_.copy())
        
        recovered_betas = np.array(recovered_betas)
        
        # Stability Metrics
        beta_variance = np.var(recovered_betas, axis=0).sum()
        
        supports = [set(np.where(np.abs(b) > 1e-5)[0]) for b in recovered_betas]
        unique_supports = len(set(tuple(sorted(s)) for s in supports))
        
        return SensitivityAnalysisMetrics(
            total_variance=beta_variance,
            unique_supports=unique_supports,
            mean_beta=np.mean(recovered_betas, axis=0)
        )
