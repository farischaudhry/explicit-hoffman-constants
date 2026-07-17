"""
Evaluates how the exact Hoffman constants and their theoretical upper bounds 
(in l2, l1, and l_inf norms) scale as the ambient dimension d increases.

This is done my generating some s sparse solution for a d_initial dimensonal design matrix.
Then new unrelated columns are added to the design matrix, and the exact Hoffman constants 
and their bounds are computed for the same s-sparse solution.

Outputs a 1x2 plot for each design:
  - Left: Comparison of Exact vs Upper Bounds across l2, l1, and l_inf norms.
  - Right: Decomposition of H2 into Active Curvature and Interaction Norm.
"""

import logging 
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd

from hoffman.designs.design_factory import DesignFactory
from hoffman.bounds.hoffman_bounds import HoffmanBoundCalculator
from hoffman.utils.viz import set_plotting_style

# Output directories
OUTPUT_DIR = './results/hoffman/ambient_dimension_scaling/'
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Configuration
np.random.seed(0)
set_plotting_style()
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_exact_norms(A: np.ndarray, active_set: np.ndarray, lam1: float):
    """Computes exact matrix norms of the KKT inverse for ground truth."""
    n = A.shape[0]
    G = (A.T @ A) / n
    M_A = HoffmanBoundCalculator.build_kkt_matrix(G, active_set, lam1)
    try:
        M_inv = np.linalg.inv(M_A)
        return {
            'exact_h2': np.linalg.norm(M_inv, ord=2),
            'exact_h1': np.linalg.norm(M_inv, ord=1),
            'exact_hinf': np.linalg.norm(M_inv, ord=np.inf)
        }
    except np.linalg.LinAlgError:
        return {'exact_h2': np.inf, 'exact_h1': np.inf, 'exact_hinf': np.inf}


def compute_metrics_for_support(A: np.ndarray, active_set: np.ndarray, lambda_val: float):
    """Computes geometric and Hoffman metrics for a single active set."""
    bounds = HoffmanBoundCalculator.compute_all(A=A, active_set=active_set, lam1=lambda_val)
    exact_norms = get_exact_norms(A, active_set, lambda_val)
    active_curv_penalty = 1.0 / bounds.lambda_min_active if bounds.lambda_min_active > 0 else np.nan
    return {
        'exact_h2': exact_norms['exact_h2'],
        'upper_h2': bounds.dimension_free_bound_upper,
        'exact_h1': exact_norms['exact_h1'],
        'upper_h1': bounds.l_1_bound,
        'exact_hinf': exact_norms['exact_hinf'],
        'upper_hinf': bounds.l_inf_bound,
        'interaction_norm': bounds.interaction_norm,
        'active_curvature': active_curv_penalty
    }


def run_scaling_experiment(n=200, initial_d=400, max_d=10000, step=100, s=10, lambda_val=0.1):
    """Evaluates scaling across multiple designs, generating a 1x2 plot for each."""
    base_designs = {
        'Gaussian': DesignFactory.gaussian(n, initial_d, s),
    }
    
    d_steps = np.arange(initial_d, max_d + 1, step)
    all_results = []
    
    for name, base_design in base_designs.items():
        logger.info(f"Evaluating {name}.")
        true_support = base_design.true_support
        A_current = base_design.A.copy()
        
        design_results = []
        
        for current_d in d_steps:
            if current_d > A_current.shape[1]:
                n_add = current_d - A_current.shape[1]
                A_new = np.random.normal(0, 1, (n, n_add))
                
                # Normalize new columns
                norms = np.linalg.norm(A_new, axis=0)
                norms[norms == 0] = 1.0
                A_new = (A_new / norms) * np.sqrt(n)
                
                A_current = np.hstack([A_current, A_new])
                
            metrics = compute_metrics_for_support(A_current, true_support, lambda_val)
            
            res_dict = {
                'Design': name,
                'd': current_d,
                'exact_h2': metrics['exact_h2'],
                'upper_h2': metrics['upper_h2'],
                'exact_h1': metrics['exact_h1'],
                'upper_h1': metrics['upper_h1'],
                'exact_hinf': metrics['exact_hinf'],
                'upper_hinf': metrics['upper_hinf'],
                'interaction_norm': metrics['interaction_norm'],
                'active_curvature': metrics['active_curvature']
            }
            design_results.append(res_dict)
            all_results.append(res_dict)
            
        df_design = pd.DataFrame(design_results)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
        
        # Left Plot: Norm Comparisons
        ax1.plot(df_design['d'], df_design['exact_h2'], 'k-', marker='o', alpha=0.7, label='Exact $H^{2,2}$')
        ax1.plot(df_design['d'], df_design['upper_h2'], 'r--', marker='s', alpha=0.7, label='Upper $H^{2,2}$')
        
        ax1.plot(df_design['d'], df_design['exact_h1'], 'c-', marker='d', alpha=0.7, label='Exact $H^{1,1}$')
        ax1.plot(df_design['d'], df_design['upper_h1'], 'b--', marker='^', alpha=0.7, label='Upper $H^{1,1}$')
        
        ax1.plot(df_design['d'], df_design['exact_hinf'], 'y-', marker='x', alpha=0.7, label='Exact $H^{\infty, \infty}$')
        ax1.plot(df_design['d'], df_design['upper_hinf'], 'm--', marker='+', alpha=0.7, label='Upper $H^{\infty, \infty}$')
        
        ax1.set_yscale('log')
        ax1.set_xlabel('Ambient Dimension $d$')
        ax1.set_ylabel('Hoffman Constant or Bound (log scale)')
        ax1.set_title(f'Norm Comparisons - {name.replace("_", " ")}')
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc='upper left', fontsize='small')
        
        # Right Plot: H^2,2 Decomposition 
        ax2.plot(df_design['d'], df_design['exact_h2'], 'k-', marker='o', alpha=0.7, label='Exact $H^{2,2}$')
        ax2.plot(df_design['d'], df_design['upper_h2'], 'r--', marker='s', alpha=0.7, label='Upper $H^{2,2}$')
        ax2.plot(df_design['d'], df_design['interaction_norm'], 'b:', marker='v', linewidth=2, alpha=0.7, label='Active-Inactive Interaction')
        ax2.plot(df_design['d'], df_design['active_curvature'], 'c-.', marker='>', linewidth=2, alpha=0.7, label='Active Curvature')
        ax2.plot(df_design['d'], 1/lambda_val * np.ones_like(df_design['d']), 'g--', linewidth=2, alpha=0.7, label='1/$\lambda$')
        
        # ax2.set_yscale('log')
        # ax2.set_xscale('log')
        ax2.set_xlabel('Ambient Dimension $d$')
        ax2.set_ylabel('Value')
        clean_name = name.replace('_', ' ')
        ax2.set_title(f'$H^{{2,2}}$ Geometric Decomposition - {clean_name}')
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper left', fontsize='small')
        
        plt.tight_layout()
        safe_name = name.lower().replace(' ', '_')
        plt.savefig(os.path.join(FIG_DIR, f'dimensionality_scaling_{safe_name}.png'))
        plt.close()
        
    # Save all raw data
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(DATA_DIR, 'dimensionality_scaling_all.csv'), index=False)
    logger.info(f"Plots and data saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    run_scaling_experiment(n=200, initial_d=400, max_d=2000, step=50, s=10, lambda_val=0.1)
