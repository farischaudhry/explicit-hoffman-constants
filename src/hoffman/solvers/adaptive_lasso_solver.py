import numpy as np
from typing import Any
from abc import ABC

from sklearn.linear_model import Ridge

from hoffman.solvers.base_sparse_solver import BaseSparseSolver, SolverProgress, ManifoldMetrics
from hoffman.utils.math_ops import soft_threshold


class AdaptiveLassoSolver(BaseSparseSolver, ABC):
    """
    Generic class for the adaptive LASSO problem: 
    min (1/2n)||y - Ab||^2 + lambda * sum(w_j * |b_j|).
    """
    def __init__(self, design, lam: float, weights: np.ndarray, y: np.ndarray = None, beta_hat: np.ndarray = None):
        super().__init__(design, {'l1': lam}, y=y, beta_hat=beta_hat)
        self.lam = lam
        self.weights = weights
        
        if len(self.weights) != self.design.d:
            raise ValueError("Length of weights array must match dimension d.")

    @staticmethod
    def compute_initial_weights(A: np.ndarray, y: np.ndarray, gamma: float = 1.0, eps: float = 1e-5) -> np.ndarray:
        """
        Computes weights for the adaptive penalty using a Ridge regression initial estimator.
        Weights are normalized to average to 1 to preserve the scale of lambda.
        """
        ridge = Ridge(alpha=1.0, fit_intercept=False)
        ridge.fit(A, y)
        weights = 1.0 / (np.abs(ridge.coef_) ** gamma + eps)
        return weights / np.mean(weights)

    def objective(self, beta: np.ndarray) -> float:
        resid = self.design.A @ beta - self.y
        mse = (1 / (2 * self.design.n)) * np.sum(resid**2)
        reg = self.lam * np.sum(self.weights * np.abs(beta))
        return mse + reg

    def _get_active_set(self, beta: np.ndarray, tol: float = 1e-8) -> set[int]:
        """Indices where beta_j is non-zero."""
        return set(np.where(np.abs(beta) > tol)[0])

    def _compute_geometric_metrics(self, beta: np.ndarray) -> dict[str, Any]:
        """
        Computes (active curvature, interaction, dual_violation) with adaptive dampening.
        """
        active_idx = sorted(list(self._get_active_set(beta)))
        inactive_idx = [i for i in range(self.design.d) if i not in active_idx]
        
        # Curvature of the identified manifold
        min_eig = np.nan
        if active_idx:
            G_AA = self.G[np.ix_(active_idx, active_idx)]
            min_eig = np.linalg.eigvalsh(G_AA).min()
            
        # Interaction
        interaction = 0.0
        if active_idx and inactive_idx:
            W_inv = np.diag(1.0 / self.weights[inactive_idx])
            G_AcA = self.G[np.ix_(inactive_idx, active_idx)]
            interaction = np.linalg.norm(W_inv @ G_AcA, ord=2)

        # Dual feasibility (Strict complementarity margin relative to weights)
        dual_violation = 0.0
        if inactive_idx:
            grad_full = (1/self.design.n) * self.design.A.T @ (self.design.A @ beta - self.y)
            # Normalized by lambda * w_j 
            dual_violation = np.max(np.abs(grad_full[inactive_idx]) / (self.lam * self.weights[inactive_idx]))
            
        cone_ratio = 0.0
        if self.beta_hat is not None:
            delta = beta - self.beta_hat
            err_active = np.linalg.norm(delta[self.hat_support], 1)
            err_inactive = np.linalg.norm(delta[~self.hat_support], 1)
            cone_ratio = err_inactive / err_active if err_active > 1e-12 else 0.0
        
        # Calculate eigenvalues (max eig = spectral norm of active Gram block)
        max_eig = np.nan
        if active_idx:
            G_AA = self.G[np.ix_(active_idx, active_idx)]
            max_eig = np.linalg.norm(G_AA, ord=2)
        
        return {
            'min_eig': min_eig,
            'interaction': interaction,
            'max_eig': max_eig,
            'dual_violation': dual_violation,
            'cone_ratio': cone_ratio,
            'current_active_set_size': len(active_idx)
        }   


class ISTAAdaptiveLassoSolver(AdaptiveLassoSolver):
    """
    ISTA for adaptive LASSO.
    """
    def solve(self, n_iters: int, start_beta: np.ndarray = None) -> SolverProgress:
        beta = np.zeros(self.design.d) if start_beta is None else start_beta.copy()
        # Lipschitz constant of the gradient of (1/2n)||y-Ab||^2
        L = np.linalg.norm(self.design.A, ord=2)**2 / self.design.n
        step = 1.0 / L
        
        progress = SolverProgress()

        for k in range(n_iters):
            # Compute current metrics
            metrics_dict = self._compute_geometric_metrics(beta)
            
            # Gradient step
            grad = (1/self.design.n) * self.design.A.T @ (self.design.A @ beta - self.y)
            z = beta - step * grad
            
            # Weighted proximal step
            # Threshold varies per feature according to lam * weights * step
            beta_next = soft_threshold(z, self.lam * self.weights * step)
            
            # Calculate residual 
            residual = np.linalg.norm((beta - beta_next) / step)
            
            progress.history.append(ManifoldMetrics(
                iteration=k,
                objective=self.objective(beta),
                beta=beta.copy(),
                residual=residual,
                active_constraints=self._get_active_set(beta),
                **metrics_dict
            ))
            
            beta = beta_next

        return progress
