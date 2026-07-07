"""
Experiment 4: Comprehensive Hoffman Bound Tightness and Scaling
---------------------------------------------------------------
1. Evaluates the tightness of explicit theoretical bounds on the Hoffman constant 
   across various problem configurations (n, d, s) using Monte Carlo support sampling.
2. Plots the empirical distribution of H(A) to contrast average vs worst-case geometry.
3. Analyzes how H(A) scales when pure noise features are iteratively appended to the design.
"""

import logging 
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tqdm import tqdm

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
        'lower_h2': bounds.dimension_free_bound_lower,
        'upper_h2': bounds.dimension_free_bound_upper,
        'weyl_h2': bounds.weyl_perturbation_bound,
        'upper_h1': bounds.l_1_bound,
        'upper_hinf': bounds.l_inf_bound,
        'interaction_norm': bounds.interaction_norm,
        'active_curvature': bounds.lambda_min_active
    }


def part_1_comprehensive_tables(n_list, d_list, s_list, n_mc_samples=50, lambda_val=0.1):
    """Generates table comparing exact H(A) to theoretical bounds over sampled supports."""
    logger.info("Running Part 1: Comprehensive Bound Tables...")
    
    results = []
    
    for n, d, s in zip(n_list, d_list, s_list):
        designs = {
            'Gaussian': DesignFactory.gaussian(n, d, s),
            'Partial DCT': DesignFactory.partial_dct(n, d, s),
            'Rademacher': DesignFactory.rademacher(n, d, s),
            'Spiked (rho=0.5)': DesignFactory.spiked(n, d, s, rho=0.5),
            'Spiked (rho=0.9)': DesignFactory.spiked(n, d, s, rho=0.9)
        }
        
        for name, design in designs.items():
            trial_metrics = []
            
            # Monte Carlo sampling of s-sparse active sets
            for _ in range(n_mc_samples):
                random_support = np.random.choice(d, s, replace=False)
                metrics = compute_metrics_for_support(design.A, random_support, lambda_val)
                if np.isfinite(metrics['exact_h2']):
                    trial_metrics.append(metrics)
            
            df_trials = pd.DataFrame(trial_metrics)
            
            if len(df_trials) == 0:
                continue
                
            mean_metrics = df_trials.mean().to_dict()
            worst_metrics = df_trials.max().to_dict()
            
            results.append({
                'Design': name,
                'n': n, 'd': d, 's': s,
                'Exact H2 (Mean)': mean_metrics['exact_h2'],
                'Exact H2 (Worst)': worst_metrics['exact_h2'],
                'Lower H2': mean_metrics['lower_h2'],
                'Upper H2': mean_metrics['upper_h2'],
                'Weyl H2': mean_metrics['weyl_h2'],
                'L1 Bound': mean_metrics['upper_h1'],
                'Linf Bound': mean_metrics['upper_hinf'],
                'Interaction ||G_AcA||_2': mean_metrics['interaction_norm'],
                'Upper/Lower Ratio': mean_metrics['upper_h2'] / mean_metrics['lower_h2'] if mean_metrics['lower_h2'] > 0 else np.nan,
                'Upper/Exact Ratio': mean_metrics['upper_h2'] / mean_metrics['exact_h2'] if mean_metrics['exact_h2'] > 0 else np.nan
            })
            
    df_results = pd.DataFrame(results)
    df_results.to_csv(os.path.join(DATA_DIR, 'comprehensive_bounds_table.csv'), index=False)
    logger.info(f"Saved comprehensive table to {DATA_DIR}\n")
    return df_results


if __name__ == '__main__':
    # Define grid of parameters for Table 1
    n_list = [200, 300, 400]
    d_list = [400, 600, 800]
    s_list = [10, 15, 20]
    
    part_1_comprehensive_tables(n_list, d_list, s_list)
    
    logger.info("Experiment 4 complete.")