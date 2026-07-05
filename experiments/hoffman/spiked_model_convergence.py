"""
This experiment demonstrates the empirical convergence rate of various 
ISTA-based solvers (LASSO, Elastic Net, Adaptive variants) on spiked 
model designs with varying rho values.

It shows that control on the primal curvature (e.g., RE) alone is not 
sufficient for optimization guarantees, and demonstrates how different 
regularization architectures alter the geometric conditioning.
"""

import logging 
import os
import numpy as np
import matplotlib.pyplot as plt

from hoffman.designs.design_factory import DesignFactory
from hoffman.solvers import (
    ISTALassoSolver, 
    ISTAElasticNetSolver, 
    ISTAAdaptiveLassoSolver, 
    ISTAAdaptiveElasticNetSolver
)
from hoffman.utils.viz import set_plotting_style

# Configuration
np.random.seed(0)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)
set_plotting_style()

# Experiment parameters
N, D, S = 300, 600, 15
LAMBDA_1 = 0.2
LAMBDA_2 = 0.05
MAX_ITERS = 10000
RHOS = [0.0, 0.05, 0.1, 0.3, 0.5, 0.9, 0.95]


def build_spiked_designs():
    """Generate one design per rho so every solver shares the same problems."""
    return {rho: DesignFactory.spiked(n=N, d=D, s=S, rho=rho) for rho in RHOS}


def run_spiked_convergence(solver_class, solver_name, out_dir, designs):
    """
    Runs the convergence benchmark across varying rho values for a given solver.
    """
    data_dir = os.path.join(out_dir, 'data')
    os.makedirs(data_dir, exist_ok=True)
    
    results = {}
    for rho in RHOS:
        name = f'Spiked rho={rho}'
        logger.info(f'Benchmarking {solver_name} on {name}')    

        # Reuse the same pre-generated design across all solvers for this rho.
        design = designs[rho]
        y = design.A @ design.true_beta
        
        # Configure solver arguments dynamically based on the class requirements
        kwargs = {'lam1': LAMBDA_1} if 'Elastic' in solver_name else {'lam': LAMBDA_1}
        
        if 'Elastic' in solver_name:
            kwargs['lam2'] = LAMBDA_2
        if hasattr(solver_class, 'compute_initial_weights'):
            kwargs['weights'] = solver_class.compute_initial_weights(design.A, y)

        # Initialize the solver
        solver = solver_class(design, y=y, **kwargs)

        total_iters = MAX_ITERS + 20000
        prog = solver.solve(n_iters=total_iters)

        # The ground truth is the last value reached
        f_star = prog.objective_path[-1]

        # Slice the trajectory to only keep the first MAX_ITERS iterations for the plot
        plot_trajectory = prog.objective_path[:MAX_ITERS]
        
        # Calculate the error 
        ista_err = np.maximum(plot_trajectory - f_star, 1e-16) 
        # Save results
        results[name] = {'ista_err': ista_err, 'f_star': f_star}
        
        np.savez_compressed(
            os.path.join(data_dir, f'{solver_name.lower()}_spiked_rho_{rho:.3f}.npz'), 
            ista_err=ista_err,
            f_star=f_star
        )

    # Plotting 
    plt.figure(figsize=(10, 6))
    linestyles = ['-', '--', '-.', ':', (0, (3, 5, 1, 5)), (0, (3, 1, 1, 1)), (0, (5, 1))]
    colors = plt.cm.plasma(np.linspace(0, 0.8, len(RHOS)))

    for i, rho in enumerate(RHOS):
        res = results[f'Spiked rho={rho}']
        plt.semilogy(
            res['ista_err'], 
            color=colors[i], 
            linestyle=linestyles[i % len(linestyles)], 
            alpha=0.8, 
            linewidth=2,
            label=f'ISTA ($\\rho={rho}$)'
        )

    plt.title(f'{solver_name} Convergence on Spiked Model')
    plt.xlabel('Iteration $k$')
    plt.ylabel('Estimator Suboptimality')
    plt.grid(True, which='both', alpha=0.3)
    plt.legend(bbox_to_anchor=(1, 1), loc='upper left')
    plt.tight_layout()
    plt.savefig(os.path.join(out_dir, f'spiked_model_{solver_name.lower()}_convergence.png'))
    plt.close()


if __name__ == "__main__":
    shared_designs = build_spiked_designs()

    EXPERIMENTS = {
        'LASSO': ISTALassoSolver,
        'Elastic Net': ISTAElasticNetSolver,
        'Adaptive LASSO': ISTAAdaptiveLassoSolver,
        'Adaptive Elastic Net': ISTAAdaptiveElasticNetSolver,
    }
    
    base_out = './results/hoffman/spiked_model_convergence/'
    for name, s_class in EXPERIMENTS.items():
            run_spiked_convergence(s_class, name, os.path.join(base_out, name), shared_designs)
