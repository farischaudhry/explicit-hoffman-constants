from abc import ABC, abstractmethod
from dataclasses import dataclass, field
import numpy as np

from hoffman.designs.design_matrix import DesignMatrix


@dataclass
class ManifoldMetrics:
    """
    Universal metrics for a point on an optimization trajectory.
    This acts as a base class for problem-specific metrics.
    """
    iteration: int
    objective: float
    beta: np.ndarray
    residual: float
    
    # Structural Manifold Info
    # For LASSO: set[int] (non-zero indices)
    # For Fused: list[set[int]] (partition of fused nodes)
    active_constraints: any 
    
    # Hoffman-Relevant Geometry
    min_eig: float  # Curvature of f restricted to the manifold
    interaction: float  # Leakage to inactive constraints (||G_AcA||)
    dual_violation: float  # Margin to subdifferential boundary (max |s_Ac|)
    
    @property
    def is_stable(self) -> bool:
        """
        Strict complementarity holds if dual violation < 1.0.
        This is a necessary condition for finite manifold identification.
        """
        # We use a small epsilon for numerical stability
        return self.dual_violation < (1.0 - 1e-12)


@dataclass
class SolverProgress:
    """Collection of metrics over the full solver path."""
    history: list[ManifoldMetrics] = field(default_factory=list)

    @property
    def objective_path(self) -> np.ndarray:
        return np.array([m.objective for m in self.history])

    @property
    def manifold_path(self) -> list:
        return [m.active_constraints for m in self.history]
    
    @property
    def stability_path(self) -> np.ndarray:
        return np.array([m.is_stable for m in self.history])


class BaseSparseSolver(ABC):
    """
    Abstract base for solvers of the form: min f(beta) + sum lambda_k * g_k(K_k * beta).
    
    This composite form encompasses:
    - LASSO (K=I, g=l1)
    - Elastic Net (f includes l2)
    - Adaptive LASSO (weights in g=l1)
    - Graph Fused Lasso (K=Incidence, g=l1)
    """
    def __init__(self, design: DesignMatrix, lambdas: dict[str, float], y: np.ndarray = None):
        self.design = design
        self.lambdas = lambdas
        self.y = y if y is not None else (design.A @ design.true_beta)
        self.G = design.gram

    @abstractmethod
    def objective(self, beta: np.ndarray) -> float:
        """Total objective F(beta)."""
        pass

    @abstractmethod
    def _get_active_set(self, beta: np.ndarray) -> any:
        """Identifies binding constraints (the face of the polyhedron)."""
        pass

    @abstractmethod
    def _compute_geometric_metrics(self, beta: np.ndarray) -> tuple[float, float, float]:
        """Returns the Hoffman-relevant components: (eigenvalues, interaction, dual margin)."""
        pass

    @abstractmethod
    def solve(self, n_iters: int) -> SolverProgress:
        """The optimization algorithm implementation."""
        pass
