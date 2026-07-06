"""
Experiment 4b: Ambient Dimensionality Scaling Across Norms
----------------------------------------------------------
Evaluates how the Hoffman constants and their theoretical upper bounds 
(in l2, l1, and l_inf norms) scale as the ambient dimension d increases.
Outputs a distinct dual-axis plot for each design matrix.
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
OUTPUT_DIR = './results/04_hoffman_bounds/'
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Configuration
np.random.seed(42)
set_plotting_style()
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def compute_metrics_for_support(A: np.ndarray, active_set: np.ndarray, lambda_val: float):
    """Computes geometric and Hoffman metrics for a single active set."""
    bounds = HoffmanBoundCalculator.compute_all(A=A, active_set=active_set, lam1=lambda_val)
    
    return {
        'exact_h2': bounds.exact_local_affine_hoffman,
        'upper_h2': bounds.dimension_free_bound_upper,
        'upper_h1': bounds.l_1_bound,
        'upper_hinf': bounds.l_inf_bound,
        'interaction_norm': bounds.interaction_norm
    }


def run_scaling_experiment(n=200, initial_d=50, max_d=1000, step=50, s=10, lambda_val=0.1):
    """Evaluates scaling across multiple designs, generating a plot for each."""
    logger.info("Running Dimensionality Scaling Across Norms...")
    
    base_designs = {
        'Gaussian': DesignFactory.gaussian(n, initial_d, s),
        'Partial_DCT': DesignFactory.partial_dct(n, initial_d, s),
        'Spiked_rho0.5': DesignFactory.spiked(n, initial_d, s, rho=0.5)
    }
    
    d_steps = np.arange(initial_d, max_d + 1, step)
    all_results = []
    
    for name, base_design in base_designs.items():
        logger.info(f"  Evaluating {name}...")
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
                'upper_h1': metrics['upper_h1'],
                'upper_hinf': metrics['upper_hinf'],
                'interaction_norm': metrics['interaction_norm']
            }
            design_results.append(res_dict)
            all_results.append(res_dict)
            
        # --- Create individual plot for this design ---
        df_design = pd.DataFrame(design_results)
        
        fig, ax1 = plt.subplots(figsize=(10, 6))
        
        # Left Axis: Hoffman Bounds (Using log scale since H1/H2 can blow up)
        ax1.plot(df_design['d'], df_design['exact_h2'], 'k-', marker='o', label='Exact H2')
        ax1.plot(df_design['d'], df_design['upper_h2'], 'r--', marker='s', label='Upper H2')
        ax1.plot(df_design['d'], df_design['upper_h1'], 'g-.', marker='d', label='Upper H1')
        ax1.plot(df_design['d'], df_design['upper_hinf'], 'm:', marker='x', label='Upper Hinf')
        
        ax1.set_yscale('log')
        ax1.set_xlabel('Ambient Dimension (d)')
        ax1.set_ylabel('Hoffman Constant / Bound (log scale)')
        ax1.set_title(f'Degradation of Local Geometry - {name.replace("_", " ")}')
        
        # Right Axis: Interaction Norm
        ax2 = ax1.twinx()
        ax2.plot(df_design['d'], df_design['interaction_norm'], 'b-', alpha=0.6, marker='^', label='Interaction ||G_AcA||_2')
        ax2.set_ylabel('Interaction Norm', color='b')
        ax2.tick_params(axis='y', labelcolor='b')
        
        # Combine legends
        lines_1, labels_1 = ax1.get_legend_handles_labels()
        lines_2, labels_2 = ax2.get_legend_handles_labels()
        ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
        
        plt.tight_layout()
        safe_name = name.lower().replace(' ', '_')
        plt.savefig(os.path.join(FIG_DIR, f'dimensionality_scaling_{safe_name}.png'))
        plt.close()
        
    # Save all raw data
    df_all = pd.DataFrame(all_results)
    df_all.to_csv(os.path.join(DATA_DIR, 'dimensionality_scaling_all.csv'), index=False)
    logger.info(f"Plots and data saved to {OUTPUT_DIR}")


if __name__ == '__main__':
    run_scaling_experiment(n=200, initial_d=200, max_d=1000, step=50, s=10, lambda_val=0.1)