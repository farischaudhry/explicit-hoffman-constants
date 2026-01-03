"""
Experiment 1: ISTA Convergence Rate on Spiked Model Designs
-----------------------------------------------

It was shown in the paper that designs generated from the spiked model
have RE -> 0 as rho -> 1, leading to large Hoffman constants.
Equivalently, \|G_AA^{-1}\|_2 = 1/1-rho grows large as rho -> 1.

This experiment demonstrates the empirical convergence rate of ISTA
on spiked model designs with varying rho values.
This shows that despite satisfying RE, the convergence rate can be arbitrarily slow
as rho -> 1 (and Hoffman constant grows large). That is to say, RE alone is not sufficent
for optimization guarantees.
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
OUTPUT_DIR = './results/01_spiked_model_convergence/'
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
MAX_ITERS = 10000  # Max iterations for our own ISTA/FISTA solvers
rhos = [0.0, 0.05, 0.1, 0.3, 0.5, 0.9, 0.95]

results = {}
for rho in rhos:
    name = f'Spiked rho={rho}'
    logger.info(f'Benchmarking {name}')    

    # Generate design
    design = DesignFactory.spiked(n=N, d=D, s=S, rho=rho)
    y = design.A @ design.true_beta

    # Compute ground truth reference solution using sklearn
    # We use very low tolerance + high max_iters to ensure we are at the manifold center
    sk_lasso = Lasso(alpha=LAMBDA, fit_intercept=False, tol=1e-15, max_iter=50000)
    sk_lasso.fit(design.A, y)

    resid = y - design.A @ sk_lasso.coef_
    f_star = (1/(2*N)) * np.sum(resid**2) + LAMBDA * np.linalg.norm(sk_lasso.coef_, 1)

    # Run custom solvers for geometric introspection
    ista_solver = ISTALassoSolver(design, lam=LAMBDA)

    ista_prog = ista_solver.solve(n_iters=MAX_ITERS)

    # Save results
    results[name] = {
        'ista_err': np.abs(ista_prog.objective_path - f_star),
        'f_star': f_star
    }
    np.savez_compressed(
        os.path.join(DATA_DIR, f'spiked_rho_{rho:.3f}.npz'), 
        ista_err=results[name]['ista_err'],
        f_star=results[name]['f_star']
    )
    

# Plotting 
plt.figure(figsize=(12, 6))
linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (5, 1))]
colors = plt.cm.plasma(np.linspace(0, 0.8, len(rhos)))

for i, rho in enumerate(rhos):
    res = results[f'Spiked rho={rho}']
    current_linestyle = linestyles[i % len(linestyles)]
    plt.semilogy(
        res['ista_err'], 
        color=colors[i], 
        linestyle=current_linestyle, 
        alpha=0.8, 
        linewidth=2,
        label=f'ISTA ($\\rho={rho}$)'
    )

plt.title(r'LASSO Convergence: $F(\beta^{(k)}) - F^*$ on Spiked Model Designs')
plt.xlabel('Iteration, $k$')
plt.ylabel('Error to Estimator Optimum')
plt.grid(True, which='both', alpha=0.3)
plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
plt.tight_layout()
plt.savefig(os.path.join(OUTPUT_DIR, 'spiked_model_ista_convergence.png'))
