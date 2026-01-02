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
│       │   ├── __init__.py
│       │   ├── design_matrix.py  # DesignMatrix dataclass
│       │   └── design_factory.py # Factory for various designs
│       ├── solvers/              # Sparse problem solvers
│       │   ├── __init__.py
│       │   ├── base_sparse_solver.py  # Abstract base solver
│       │   └── lasso_solver.py        # Solvers for LASSO
│       ├── utils/                
│       │   ├── __init__.py
│       │   ├── math_ops.py       # Mathematical operations
│       │   └── viz.py            # Plotting Config
├── experiments/                  # Experiment scripts  
└── results/                      # Generated figures and data
```

## Experiments

TODO
