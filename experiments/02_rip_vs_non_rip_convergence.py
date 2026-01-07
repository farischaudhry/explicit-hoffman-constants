"""
Experiment 2: RIP vs. Non-RIP Design Convergence Rate
----------------------------------------------------------

Previous work has shown that RIP condition is sufficent
for linear convergence on a variety of solvers.
Indeed, we also find that RIP designs have bounded Hoffman constants
on sufficently sparse active sets. This experiment empirically demonstrates this fact.

We test multiple design types with varying geometric properties:
- Gaussian: Satisfies RIP with high probability
- Discrete Cosine Transform (DCT): Satisfies RIP with high probability
- Rademacher (Subgaussian): Satisfies RIP with high probability
- Uniform Subgaussian: Satisfies RIP with high probability
- Sparse Binary: Satisfies RIP with high probability

- Spiked Model: Poorly conditioned, does not satisfy RIP for high rho
- AR1: Poorly conditioned, does not satisfy RIP for high rho
- Power Law: Poorly conditioned, does not satisfy RIP for low decay
- Low Rank: The probability 0 case where RIP fails when sampling random matrices.
"""

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Lasso

from hoffman.designs.design_factory import DesignFactory
from hoffman.solvers.lasso_solver import ISTALassoSolver
from hoffman.utils.viz import set_plotting_style

# Output directories
OUTPUT_DIR = './results/02_rip_vs_non_rip_convergence/'
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Configuration
np.random.seed(0)
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
set_plotting_style()

# Experiment parameters
N, D, S = 300, 600, 15
LAMBDA = 0.2
MAX_ITERS = 2000  # Max iterations for our own ISTA/FISTA solvers

design_scenarios = {
    'Gaussian (RIP)': lambda: DesignFactory.gaussian(N, D, S),
    'Partial DCT (RIP)': lambda: DesignFactory.partial_dct(N, D, S),
    'Rademacher (RIP)': lambda: DesignFactory.rademacher(N, D, S),
    'Uniform Subgaussian (RIP)': lambda: DesignFactory.uniform_subgaussian(N, D, S),
    'Sparse Binary (RIP)': lambda: DesignFactory.sparse_binary(N, D, S, density=0.1),
    'Spiked $\\rho=0.9$ (Non-RIP)': lambda: DesignFactory.spiked(N, D, S, rho=0.9),
    'AR1 $\\rho=0.9$ (Non-RIP)': lambda: DesignFactory.ar1_stochastic(N, D, S, rho=0.9),
    'PowerLaw decay=1.0 (Non-RIP)': lambda: DesignFactory.power_law(N, D, S, decay=1.0),
    'Low Rank (Non-RIP)': lambda: DesignFactory.low_rank(N, D, S, rank=S),
}


results = {}
for name, gen_func in design_scenarios.items():
    design = gen_func()
    logger.info(f'Benchmarking {name}')

    # Calculate Ground Truth y and F*
    y = design.A @ design.true_beta + 0.05 * np.random.normal(0, 1, N)
    sk_lasso = Lasso(alpha=LAMBDA, fit_intercept=False, tol=1e-15, max_iter=50000).fit(design.A, y)
    
    resid = y - design.A @ sk_lasso.coef_
    f_star = (1/(2*N)) * np.sum(resid**2) + LAMBDA * np.linalg.norm(sk_lasso.coef_, 1)
    
    # Run our solver with geometric introspection
    solver = ISTALassoSolver(design, lam=LAMBDA, y=y)
    progress = solver.solve(n_iters=MAX_ITERS)
    
    results[name] = {
        'error': np.abs(progress.objective_path - f_star),
        'is_not_rip': 'Non-RIP' in name
    }
    
    # Save individual result
    safe_name = name.replace(' ', '_').replace('(', '').replace(')', '').replace('$', '').replace('\\', '').replace('=', '')
    np.savez_compressed(
        os.path.join(DATA_DIR, f'{safe_name}.npz'),
        error=results[name]['error'],
        f_star=f_star,
        design_name=name
    )

# Plotting
plt.figure(figsize=(10, 7))
colors_good = ['blue', 'cyan', 'teal', 'navy', 'royalblue']
colors_bad = ['red', 'orange', 'magenta', 'darkred']
g_idx, b_idx = 0, 0

for name, res in results.items():
    if res['is_not_rip']:
        color, style = colors_bad[b_idx % 4], '--'
        b_idx += 1
    else:
        color, style = colors_good[g_idx % 5], '-'
        g_idx += 1
    plt.semilogy(res['error'], linestyle=style, linewidth=2.5, color=color, label=name)

plt.title(r'LASSO Convergence: $F(\beta^{(k)}) - F^*$ on RIP vs. Non-RIP Designs')
plt.xlabel('Iteration $k$')
plt.ylabel('Estimator Suboptimality')
plt.grid(True, which='both', alpha=0.3)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'rip_vs_non_rip_ista_convergence.png'))
