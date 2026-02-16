import numpy as np

from hoffman.solvers.base_sparse_solver import BaseSparseSolver, SolverProgress, ManifoldMetrics
from hoffman.utils.math_ops import soft_threshold


class ElasticNetSolver(BaseSparseSolver):
    """
    Generic umbrella class for the Elastic Net problem:
    min (1/2n)||y - A beta||^2 + lambda1||beta||_1 + (lambda2/2)||beta||_2^2.
    
    Implements the geometric logic shared by all Elastic Net algorithms.
    """
    def __init__(self, design, lam1: float, lam2: float, y: np.ndarray = None, beta_hat: np.ndarray = None):
        super().__init__(design, {'l1': lam1, 'l2': lam2}, y=y, beta_hat=beta_hat)
        self.lam1 = lam1  # L1 penalty (sparsity)
        self.lam2 = lam2  # L2 penalty (ridge)

    def objective(self, beta: np.ndarray) -> float:
        resid = self.design.A @ beta - self.y
        mse = (1 / (2 * self.design.n)) * np.sum(resid**2)
        l1_reg = self.lam1 * np.linalg.norm(beta, ord=1)
        l2_reg = (self.lam2 / 2) * np.sum(beta**2)
        return mse + l1_reg + l2_reg

    def _get_active_set(self, beta: np.ndarray, tol: float = 1e-8) -> set[int]:
        """Indices where beta_j is non-zero."""
        return set(np.where(np.abs(beta) > tol)[0])

    def _compute_geometric_metrics(self, beta: np.ndarray) -> dict[str, any]:
        """
        Computes (kappa, interaction, dual_violation).
        - kappa: lambda_min of the active Gram block (with L2 adjustment).
        - interaction: spectral norm of the inactive-active Gram block.
        - dual_violation: max absolute subgradient of inactive variables.
        """
        active_idx = sorted(list(self._get_active_set(beta)))
        inactive_idx = [i for i in range(self.design.d) if i not in active_idx]
        
        # Curvature of the identified manifold (kappa)
        # For Elastic Net, the effective Hessian is (1/n)A^T A + lambda2 * I
        min_eig = np.nan
        if active_idx:
            G_AA = self.G[np.ix_(active_idx, active_idx)]
            # Add L2 regularization to the diagonal
            G_AA_regularized = G_AA + self.lam2 * self.design.n * np.eye(len(active_idx))
            min_eig = np.linalg.eigvalsh(G_AA_regularized).min()
            
        # Interaction (Off-manifold leakage)
        interaction = 0.0
        if active_idx and inactive_idx:
            G_AcA = self.G[np.ix_(inactive_idx, active_idx)]
            interaction = np.linalg.norm(G_AcA, ord=2)

        # Dual feasibility (Strict complementarity margin)
        # grad = (1/n)A^T(Ab - y) + lambda2 * beta
        dual_violation = 0.0
        if inactive_idx:
            grad_full = (1/self.design.n) * self.design.A.T @ (self.design.A @ beta - self.y) + self.lam2 * beta
            # Normalised by lambda1 to check the [-1, 1] boundary
            dual_violation = np.max(np.abs(grad_full[inactive_idx])) / self.lam1
            
        cone_ratio = 0.0
        if self.beta_hat is not None:
            delta = beta - self.beta_hat
            err_active = np.linalg.norm(delta[self.hat_support], 1)
            err_inactive = np.linalg.norm(delta[~self.hat_support], 1)
            # Threshold to avoid division by zero
            cone_ratio = err_inactive / err_active if err_active > 1e-12 else 0.0
            
        return {
            'min_eig': min_eig,
            'interaction': interaction,
            'dual_violation': dual_violation,
            'cone_ratio': cone_ratio,
            'current_active_set_size': len(active_idx)
        }


class ISTAElasticNetSolver(ElasticNetSolver):
    """
    Iterative Shrinkage-Thresholding Algorithm (ISTA) for Elastic Net.
    """
    def solve(self, n_iters: int, start_beta: np.ndarray = None) -> SolverProgress:
        beta = np.zeros(self.design.d) if start_beta is None else start_beta.copy()
        
        # Lipschitz constant of the gradient of the smooth part:
        # f(beta) = (1/2n)||y-Ab||^2 + (lambda2/2)||beta||_2^2
        # L_f = lambda_max((1/n)A^T A + lambda2 * I) = lambda_max(G)/n + lambda2
        L = np.linalg.norm(self.design.A, ord=2)**2 / self.design.n + self.lam2
        step = 1.0 / L
        
        progress = SolverProgress()

        for k in range(n_iters):
            # Compute current metrics BEFORE the update
            metrics_dict = self._compute_geometric_metrics(beta)
            
            # Gradient step on smooth part: f(beta) = (1/2n)||y-Ab||^2 + (lambda2/2)||beta||_2^2
            grad = (1/self.design.n) * self.design.A.T @ (self.design.A @ beta - self.y) + self.lam2 * beta
            
            # Proximal step for L1 penalty only (L2 already handled in gradient)
            z = beta - step * grad
            beta_next = soft_threshold(z, self.lam1 * step)
            
            # Calculate residual as the norm of the gradient mapping
            residual = np.linalg.norm((beta - beta_next) / step)
            
            # Record current state
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
