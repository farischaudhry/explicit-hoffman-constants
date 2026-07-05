import numpy as np
from abc import ABC

from sklearn.linear_model import ElasticNet

from hoffman.solvers.base_sparse_solver import BaseSparseSolver, SolverProgress, ManifoldMetrics
from hoffman.utils.math_ops import soft_threshold


class AdaptiveElasticNetSolver(BaseSparseSolver, ABC):
    """
    Generic class for the adaptive elastic netproblem: 
    min (1/2n)||y - Ab||^2 + lambda1 * sum(w_j * |b_j|) + (lambda2/2)||b||_2^2.
    """
    def __init__(self, design, lam1: float, lam2: float, weights: np.ndarray, y: np.ndarray = None, beta_hat: np.ndarray = None):
        super().__init__(design, {'l1': lam1, 'l2': lam2}, y=y, beta_hat=beta_hat)
        self.lam1 = lam1
        self.lam2 = lam2
        self.weights = weights
        
        if len(self.weights) != self.design.d:
            raise ValueError("Length of weights array must match dimension d.")

    @staticmethod
    def compute_initial_weights(A: np.ndarray, y: np.ndarray, gamma: float = 1.0, eps: float = 1e-5) -> np.ndarray:
        """
        Computes weights for the adaptive penalty using an Elastic Net initial estimator.
        """
        enet = ElasticNet(alpha=0.1, l1_ratio=0.5, fit_intercept=False, max_iter=10000)
        enet.fit(A, y)
        weights = 1.0 / (np.abs(enet.coef_) ** gamma + eps)
        return weights / np.mean(weights)

    def objective(self, beta: np.ndarray) -> float:
        resid = self.design.A @ beta - self.y
        mse = (1 / (2 * self.design.n)) * np.sum(resid**2)
        l1_reg = self.lam1 * np.sum(self.weights * np.abs(beta))
        l2_reg = (self.lam2 / 2) * np.sum(beta**2)
        return mse + l1_reg + l2_reg

    def _get_active_set(self, beta: np.ndarray, tol: float = 1e-8) -> set[int]:
        """Indices where beta_j is non-zero."""
        return set(np.where(np.abs(beta) > tol)[0])

    def _compute_geometric_metrics(self, beta: np.ndarray) -> dict[str, any]:
        """
        Computes (active curvature, interaction, dual_violation).
        """
        active_idx = sorted(list(self._get_active_set(beta)))
        inactive_idx = [i for i in range(self.design.d) if i not in active_idx]
        
        # Curvature
        min_eig = np.nan
        if active_idx:
            G_AA = self.G[np.ix_(active_idx, active_idx)]
            G_AA_regularized = G_AA + self.lam2 * self.design.n * np.eye(len(active_idx))
            min_eig = np.linalg.eigvalsh(G_AA_regularized).min()
            
        # Interaction
        interaction = 0.0
        if active_idx and inactive_idx:
            W_inv = np.diag(1.0 / self.weights[inactive_idx])
            G_AcA = self.G[np.ix_(inactive_idx, active_idx)]
            interaction = np.linalg.norm(W_inv @ G_AcA, ord=2)

        # Dual feasibility - ridge gradient + weighted boundary check
        dual_violation = 0.0
        if inactive_idx:
            grad_full = (1/self.design.n) * self.design.A.T @ (self.design.A @ beta - self.y) + self.lam2 * beta
            dual_violation = np.max(np.abs(grad_full[inactive_idx]) / (self.lam1 * self.weights[inactive_idx]))
            
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
            G_AA_regularized = G_AA + self.lam2 * self.design.n * np.eye(len(active_idx))
            max_eig = np.linalg.norm(G_AA_regularized, ord=2)

        return {
            'min_eig': min_eig,
            'interaction': interaction,
            'max_eig': max_eig,
            'dual_violation': dual_violation,
            'cone_ratio': cone_ratio,
            'current_active_set_size': len(active_idx)
        }   


class ISTAAdaptiveElasticNetSolver(AdaptiveElasticNetSolver):
    """
    ISTA for adaptive elastic net.
    """
    def solve(self, n_iters: int, start_beta: np.ndarray = None) -> SolverProgress:
        beta = np.zeros(self.design.d) if start_beta is None else start_beta.copy()
        
        # Lipschitz constant includes lambda2
        L = np.linalg.norm(self.design.A, ord=2)**2 / self.design.n + self.lam2
        step = 1.0 / L
        
        progress = SolverProgress()

        for k in range(n_iters):
            metrics_dict = self._compute_geometric_metrics(beta)
            
            # Gradient step (includes L2 penalty)
            grad = (1/self.design.n) * self.design.A.T @ (self.design.A @ beta - self.y) + self.lam2 * beta
            z = beta - step * grad
            
            # Weighted proximal step for adaptive L1 penalty
            beta_next = soft_threshold(z, self.lam1 * self.weights * step)
            
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
