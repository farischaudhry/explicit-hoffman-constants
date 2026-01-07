# Explicit Hoffman Constants for LASSO

## Installation

Requires Python 3.10 or higher

```bash
pip install uv
uv sync
```

This will install all required packages specified in `pyproject.toml`.

## Project Structure

```plaintext
explicit-hoffman-constants/
├── src/
│   └── hoffman/
│       ├── __init__.py
│       ├── designs/              # Design matrix generators
│       │   ├── design_matrix.py  # DesignMatrix dataclass
│       │   └── design_factory.py # Factory for various designs
│       ├── solvers/              # Sparse problem solvers
│       │   ├── base_sparse_solver.py  # Abstract base solver
│       │   └── lasso_solver.py        # Solvers for LASSO
│       ├── bounds/               # Hoffman constant bounds
│       │   ├── hoffman_bounds.py        # Singular value bounding techniques 
|       |   └── sensitivity_analysis.py  # Perturbion tests to measure solution sensitivity
│       ├── utils/                
│       │   ├── math_ops.py       # Mathematical operations
│       │   └── viz.py            # Plotting Config
├── experiments/                  # Experiment scripts  
└── results/                      # Generated figures and data
```

## Experiments

### 01: Spiked Model Convergence

Demonstrates that RE alone is insufficient to guarantee fast convergence. Tests the spiked covariance model with varying correlation $\rho \in [0, 0.95]$, showing that convergence can be made arbitrarily slow as $\rho \to 1$ even when RE conditions hold.

### 02: RIP vs Non-RIP Convergence

Empirical comparison of ISTA convergence rates across RIP designs (e.g., Gaussian, Rademacher) and non-RIP designs (e.g., Spiked, AR1).

Demonstrates that RIP designs tend to have fast convergence, while non-RIP designs do not.

### 03: ISTA Trajectory Cone Containment

Investigates whether ISTA iterates stay within the restricted cone $\mathcal{C}_{\alpha}(\hat{\mathcal{A}})$ tracking:

- Cone ratio: $\| \Delta_{\hat{\mathcal{A}}^c} \|_1 / \Delta_{\hat{\mathcal{A}}} \|_1$ over iterations.
- Active set cardinality evolution and Jaccard similarity with true support.
- Manifold identification times (first iteration when $\text{supp}(\beta^{(k)}) = \hat{\mathcal{A}}$).

Tests both RIP and Non-RIP designs to validate that cone-conditional Hoffman bounds are practically relevant.
