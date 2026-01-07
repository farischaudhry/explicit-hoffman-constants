"""
Experiment 3: ISTA Trajectory Cone Containment
--------------------------------------------------

This experiment investigates the key conjecture for trajectory-based Hoffman constants:
    Do ISTA iterates stay within the restricted cone C_α(Â)?

For the cone-restricted Hoffman constant to be meaningful, we need:
    ||Δ^(k)_Â^c||₁ ≤ α ||Δ^(k)_Â||₁  for all (or most) k
where Δ^(k) = β^(k) - β̂ is the error vector.
In this experiment, we run β^(0) = 0. 

We track:
1. Cone ratio: ||Δ_Â^c||₁ / ||Δ_Â||₁ over iterations
2. Active set trajectory: which active sets are visited
3. Manifold identification time: when does supp(β^(k)) = Â?

This validates the claim that a bound on the cone-conditional Hoffman constant
is relevant in practice.
"""

import logging
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from sklearn.linear_model import Lasso

from hoffman.designs.design_factory import DesignFactory
from hoffman.solvers.lasso_solver import ISTALassoSolver
from hoffman.utils.viz import set_plotting_style

# Output directories
OUTPUT_DIR = './results/03_trajectory_analysis/'
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Configuration
np.random.seed(0)
set_plotting_style()
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'trajectory_analysis.log'), mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)
# Suppress matplotlib's font substitution logging messages
logging.getLogger('matplotlib.mathtext').setLevel(logging.WARNING)
logging.getLogger('matplotlib.font_manager').setLevel(logging.WARNING)


# Experiment parameters
N, D, S = 200, 400, 10
LAMBDA = 0.15
MAX_ITERS = 2000


def track_trajectory(design, lam: float, max_iters: int):
    """Run ISTA and track detailed trajectory information."""
    
    # Get ground truth
    y = design.A @ design.true_beta
    sk_lasso = Lasso(alpha=lam, fit_intercept=False, tol=1e-15, max_iter=50000)
    sk_lasso.fit(design.A, y)
    beta_star = sk_lasso.coef_
    active_star = set(np.where(np.abs(beta_star) > 1e-8)[0])
    
    # Run ISTA solver with trajectory tracking
    solver = ISTALassoSolver(design, lam=lam, y=y, beta_hat=beta_star)
    progress = solver.solve(n_iters=max_iters)
    
    # Extract metrics from solver history
    cone_ratios = np.array([m.cone_ratio for m in progress.history])
    active_set_sizes = np.array([m.current_active_set_size for m in progress.history])
    
    # Compute Jaccard similarity for each iteration
    active_set_jaccard = []
    manifold_identified = False
    identification_iter = None
    
    for k, metrics in enumerate(progress.history):
        active_current = metrics.active_constraints
        
        # Jaccard similarity: |A ∩ Â| / |A ∪ Â|
        # If Jaccard = 1, perfect recovery
        if len(active_current) == 0:
            jaccard = 0.0
        else:
            intersection = len(active_current.intersection(active_star))
            union = len(active_current.union(active_star))
            jaccard = intersection / union if union > 0 else 0.0
        active_set_jaccard.append(jaccard)
        
        # Check manifold identification
        if not manifold_identified and active_current == active_star:
            manifold_identified = True
            identification_iter = k
            logger.info(f'Manifold identified at iteration {k}')
    
    return {
        'beta_star': beta_star,
        'active_star': list(active_star),
        'cone_ratios': cone_ratios,
        'active_set_sizes': active_set_sizes,
        'active_set_jaccard': np.array(active_set_jaccard),
        'identification_iter': identification_iter
    }


def plot_trajectory_analysis(results_dict, true_sparsity):
    """Plot trajectory properties."""
    
    fig = plt.figure(figsize=(18, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.35, wspace=0.3)
    
    designs = list(results_dict.keys())
    
    # Color RIP and Non-RIP differently
    colors_rip = ['blue', 'cyan', 'navy']
    colors_non_rip = ['red', 'orange', 'magenta', 'darkred', 'orangered', 'crimson']
    
    rip_idx = 0
    non_rip_idx = 0
    color_map = {}
    
    for name in designs:
        if '(RIP)' in name:
            color_map[name] = colors_rip[rip_idx % len(colors_rip)]
            rip_idx += 1
        else:
            color_map[name] = colors_non_rip[non_rip_idx % len(colors_non_rip)]
            non_rip_idx += 1
    
    # Plot cone ratio over time
    ax1 = fig.add_subplot(gs[0, :])
    for name, res in results_dict.items():
        ax1.semilogy(res['cone_ratios'], label=name, color=color_map[name], 
                    alpha=0.7, linewidth=2, linestyle='-' if '(RIP)' in name else '--')
    
    ax1.axhline(1.0, color='black', linestyle=':', linewidth=2, 
               label='Cone boundary (α=1)')
    ax1.axhline(3.0, color='gray', linestyle=':', linewidth=2, 
               label='Relaxed cone (α=3)')
    
    ax1.set_xlabel('Iteration k')
    ax1.set_ylabel('Cone Ratio: $\\|\\Delta_{\\hat{\\mathcal{A}}^c}\\|_1 / \\|\\Delta_{\\hat{\\mathcal{A}}}\\|_1$')
    ax1.set_title('Trajectory Cone Containment: Do ISTA iterates stay in $\\mathcal{C}_\\alpha(\\hat{\\mathcal{A}})$?')
    ax1.legend(loc='upper right', ncol=2, bbox_to_anchor=(1, 1))
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim([1e-6, 1e2])
    
    # Plot active set size evolution
    ax2 = fig.add_subplot(gs[1, 0])
    
    for name, res in results_dict.items():
        ax2.plot(res['active_set_sizes'], color=color_map[name], 
                alpha=0.7, linewidth=2)
    
    # True sparsity line
    ax2.axhline(true_sparsity, color='black', 
               linestyle='--', linewidth=2, label=f'True sparsity (s={true_sparsity})')
    
    ax2.set_xlabel('Iteration k')
    ax2.set_ylabel('Active Set Size: $|\\mathcal{A}^{(k)}|$')
    ax2.set_title('Active Set Cardinality Evolution')
    ax2.legend(loc='best')
    ax2.grid(True, alpha=0.3)
    
    # Plot Jaccard similarity
    ax3 = fig.add_subplot(gs[1, 1])
    
    for name, res in results_dict.items():
        ax3.plot(res['active_set_jaccard'], color=color_map[name],
                alpha=0.7, linewidth=2)
    
    ax3.set_xlabel('Iteration k')
    ax3.set_ylabel('Jaccard Sim.: $|\\mathcal{A}^{(k)} \\cap \\hat{\\mathcal{A}}| / |\\mathcal{A}^{(k)} \\cup \\hat{\\mathcal{A}}|$')
    ax3.set_title('Active Set Recovery Quality')
    ax3.grid(True, alpha=0.3)
    ax3.set_ylim([0, 1.1])
    
    plt.savefig(os.path.join(OUTPUT_DIR, 'trajectory_analysis.png'), dpi=150, bbox_inches='tight')
    plt.close()


def main():
    """Run trajectory analysis on specified designs."""
    designs_to_test = [
        # RIP Designs
        ('Gaussian (RIP)', DesignFactory.gaussian(N, D, S)),
        ('Rademacher (RIP)', DesignFactory.rademacher(N, D, S)),
        ('Uniform Subgaussian (RIP)', DesignFactory.uniform_subgaussian(N, D, S)),
        
        # Non-RIP Designs
        ('Spiked ρ=0.3 (Non-RIP)', DesignFactory.spiked(N, D, S, rho=0.3)),
        ('Spiked ρ=0.9 (Non-RIP)', DesignFactory.spiked(N, D, S, rho=0.9)),
        ('AR1 ρ=0.5 (Non-RIP)', DesignFactory.ar1_stochastic(N, D, S, rho=0.5)),
        ('AR1 ρ=0.9 (Non-RIP)', DesignFactory.ar1_stochastic(N, D, S, rho=0.9)),
        ('PowerLaw decay=1.0 (Non-RIP)', DesignFactory.power_law(N, D, S, decay=1.0)),
    ]
    
    results_dict = {}
    for name, design in designs_to_test:
        logger.info(f'\nAnalyzing: {name}')
        logger.info('-'*70)
        
        results = track_trajectory(design, LAMBDA, MAX_ITERS)
        results_dict[name] = results
        
        # Summary statistics
        final_cone_ratio = results['cone_ratios'][-1]
        mean_cone_ratio = np.mean(results['cone_ratios'][-1000:])
        max_cone_ratio = np.max(results['cone_ratios'])
        
        logger.info(f'Final cone ratio: {final_cone_ratio:.6f}')
        logger.info(f'Mean (last 1000): {mean_cone_ratio:.6f}')
        logger.info(f'Max over trajectory: {max_cone_ratio:.6f}')
        
        if results['identification_iter']:
            logger.info(f'Manifold identified: iteration {results["identification_iter"]}')
        else:
            logger.info(f'Manifold NOT identified within {MAX_ITERS} iterations')
        
        # Check cone containment
        pct_in_cone_1 = np.mean(results['cone_ratios'] <= 1.0) * 100
        pct_in_cone_3 = np.mean(results['cone_ratios'] <= 3.0) * 100
        
        logger.info(f'% iterations with cone ratio ≤ 1: {pct_in_cone_1:.1f}%')
        logger.info(f'% iterations with cone ratio ≤ 3: {pct_in_cone_3:.1f}%')
    
    # Create visualizations and plot manifold identification times
    plot_trajectory_analysis(results_dict, S)
    logger.info('\n' + '='*70)
    logger.info('MANIFOLD IDENTIFICATION TIMES:')
    logger.info('-'*70)
    for name, res in results_dict.items():
        if res['identification_iter'] is not None:
            logger.info(f'{name:40s}: iteration {res["identification_iter"]:5d}')
        else:
            logger.info(f'{name:40s}: NOT IDENTIFIED within {MAX_ITERS} iterations')
    logger.info('='*70)
    

if __name__ == '__main__':
    main()
