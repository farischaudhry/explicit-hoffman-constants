import numpy as np

from hoffman.solvers.base_sparse_solver import BaseSparseSolver, SolverProgress, ManifoldMetrics
from hoffman.utils.math_ops import soft_threshold


class LassoSolver(BaseSparseSolver):
    """
    Generic umbrella class for the LASSO problem: 
    min (1/2n)||y - Ab||^2 + lambda||b||_1.
    
    Implements the geometric logic shared by all LASSO algorithms.
    """
    def __init__(self, design, lam: float, y: np.ndarray = None, beta_hat: np.ndarray = None):
        super().__init__(design, {'l1': lam}, y=y, beta_hat=beta_hat)
        self.lam = lam

    def objective(self, beta: np.ndarray) -> float:
        resid = self.design.A @ beta - self.y
        mse = (1 / (2 * self.design.n)) * np.sum(resid**2)
        reg = self.lam * np.linalg.norm(beta, ord=1)
        return mse + reg

    def _get_active_set(self, beta: np.ndarray, tol: float = 1e-8) -> set[int]:
        """Indices where beta_j is non-zero."""
        return set(np.where(np.abs(beta) > tol)[0])

    def _compute_geometric_metrics(self, beta: np.ndarray) -> dict[str, any]:
        """
        Computes (kappa, interaction, dual_violation).
        - kappa: lambda_min of the active Gram block.
        - interaction: spectral norm of the inactive-active Gram block.
        - dual_violation: max absolute subgradient of inactive variables.
        """
        active_idx = sorted(list(self._get_active_set(beta)))
        inactive_idx = [i for i in range(self.design.d) if i not in active_idx]
        
        # Curvature of the identified manifold (kappa)
        min_eig = np.nan
        if active_idx:
            G_AA = self.G[np.ix_(active_idx, active_idx)]
            # We use eigvalsh for symmetric matrices
            min_eig = np.linalg.eigvalsh(G_AA).min()
            
        # Interaction (Off-manifold leakage)
        interaction = 0.0
        if active_idx and inactive_idx:
            G_AcA = self.G[np.ix_(inactive_idx, active_idx)]
            interaction = np.linalg.norm(G_AcA, ord=2)

        # Dual feasibility (Strict complementarity margin)
        # grad = (1/n)A^T(Ab - y)
        dual_violation = 0.0
        if inactive_idx:
            grad_full = (1/self.design.n) * self.design.A.T @ (self.design.A @ beta - self.y)
            # Normalised by lambda to check the [-1, 1] boundary
            dual_violation = np.max(np.abs(grad_full[inactive_idx])) / self.lam
            
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
    

class ISTALassoSolver(LassoSolver):
    """
    Iterative Shrinkage-Thresholding Algorithm (ISTA) for LASSO.
    """
    def solve(self, n_iters: int, start_beta: np.ndarray = None) -> SolverProgress:
        beta = np.zeros(self.design.d) if start_beta is None else start_beta.copy()
        # Lipschitz constant of the gradient of (1/2n)||y-Ab||^2 is lambda_max(G)
        L = np.linalg.norm(self.design.A, ord=2)**2 / self.design.n
        step = 1.0 / L
        
        progress = SolverProgress()

        for k in range(n_iters):
            # Compute current metrics BEFORE the update
            metrics_dict = self._compute_geometric_metrics(beta)
            
            # Gradient step on f(beta)
            grad = (1/self.design.n) * self.design.A.T @ (self.design.A @ beta - self.y)
            
            # Proximal step (soft thresholding)
            z = beta - step * grad
            beta_next = soft_threshold(z, self.lam * step)
            
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
    

class FISTALassoSolver(LassoSolver):
    """
    Fast Iterative Shrinkage-Thresholding Algorithm (FISTA) for LASSO.
    Includes Nesterov acceleration (momentum).
    """
    def solve(self, n_iters: int, start_beta: np.ndarray = None) -> SolverProgress:
        beta = np.zeros(self.design.d) if start_beta is None else start_beta.copy()
        beta_prev = np.zeros(self.design.d)
        
        # Lipschitz constant calculation
        L = np.linalg.norm(self.design.A, ord=2)**2 / self.design.n
        step = 1.0 / L
        
        t = 1.0
        progress = SolverProgress()

        for k in range(n_iters):
            # Geometry Introspection (on the actual iterate beta)
            metrics_dict = self._compute_geometric_metrics(beta)
            
            # Nesterov Momentum Step
            # y_k is the search point incorporating momentum
            momentum_weight = (t - 1.0) / (t + 1.0) # Or (t_prev - 1) / t_next
            y_k = beta + momentum_weight * (beta - beta_prev)
            
            # Gradient step on the smooth part at the search point y_k
            grad = (1/self.design.n) * self.design.A.T @ (self.design.A @ y_k - self.y)
            
            # Proximal step (Soft-thresholding)
            beta_next = soft_threshold(y_k - step * grad, self.lam * step)
            
            # Record Metrics
            # Residual is calculated relative to the mapping from y_k
            residual = np.linalg.norm((y_k - beta_next) / step)
            
            progress.history.append(ManifoldMetrics(
                iteration=k,
                objective=self.objective(beta),
                beta=beta.copy(),
                residual=residual,
                active_constraints=self._get_active_set(beta),
                **metrics_dict
            ))
            
            # Update for next iteration
            beta_prev = beta.copy()
            beta = beta_next
            t_next = (1.0 + np.sqrt(1.0 + 4.0 * t**2)) / 2.0
            t = t_next

        return progress
