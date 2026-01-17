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
│       │   └── lasso_solver.py        # Solvers for LASSO (e.g., ISTA/FISTA)
│       ├── bounds/               # Hoffman constant bounds
│       │   ├── hoffman_bounds.py        # Singular value bounding techniques 
|       |   └── sensitivity_analysis.py  # Perturbation tests to measure solution sensitivity
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

### 04: Hoffman Bound Tightness

Empirical evaluation of how tight the Hoffman constant upper bounds are across different design matrices. Compares exact $H(A)$ (computed via SVD) against theoretical upper bounds.

### 05: Leverage Scores Predict Feature Selection Instability

Describes leverage scores as a diagnostic for feature selection stability on regression tasks:

- **MADELON** (n=500, d=500): Low leverage -> moderate stability (Jaccard ≈ 0.40)
- **ARCENE** (n=200, d=10,000): High leverage (132 features ≥0.95) -> severe instability (Jaccard ≈ 0.28)

Demonstrates that $h_j = \frac{1}{n}\|P_{\mathcal{A}} a_j\|^2$ for inactive features $j \notin \mathcal{A}$ predicts cross-validation stability. High leverage indicates features confounded with the active set, leading to inconsistent selection across folds.

If a feature has a low leverage, then it also cannot contribute much to the interaction norm. Hence these features are not causing any optimization pathologies.
