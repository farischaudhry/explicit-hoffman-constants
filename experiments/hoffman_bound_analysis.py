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


def part_2_support_distribution(n=200, d=500, s=15, n_mc_samples=1000, lambda_val=0.1):
    """Plots the distribution of the exact local affine Hoffman constant."""
    logger.info("Running Part 2: H(A) Support Distribution Analysis...")
    
    # Compare a well-conditioned vs poorly-conditioned design
    designs = {
        'Gaussian (RIP)': DesignFactory.gaussian(n, d, s),
        'Spiked (rho=0.8, Non-RIP)': DesignFactory.spiked(n, d, s, rho=0.8)
    }
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for ax, (name, design) in zip(axes, designs.items()):
        h_values = []
        for _ in tqdm(range(n_mc_samples), desc=f"Sampling {name}"):
            support = np.random.choice(d, s, replace=False)
            h_val = compute_metrics_for_support(design.A, support, lambda_val)['exact_h2']
            if np.isfinite(h_val):
                h_values.append(h_val)
                
        ax.hist(h_values, bins=50, color='steelblue', edgecolor='black', alpha=0.7)
        ax.axvline(np.mean(h_values), color='red', linestyle='dashed', linewidth=2, label=f'Mean: {np.mean(h_values):.2f}')
        ax.axvline(np.max(h_values), color='darkorange', linestyle='solid', linewidth=2, label=f'Worst: {np.max(h_values):.2f}')
        
        ax.set_title(f'Distribution of H(A) - {name}')
        ax.set_xlabel('Exact Local Affine Hoffman Constant (l2)')
        ax.set_ylabel('Frequency')
        ax.legend()
        
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'hoffman_distribution.png'))
    plt.close()


def part_3_dimensionality_scaling(n=200, initial_d=50, s=10, lambda_val=0.1):
    """Evaluates how H(A) and interaction scale as dummy ambient features are added."""
    logger.info("Running Part 3: Ambient Dimensionality Scaling...")
    
    # We create a base design, and its true support
    base_design = DesignFactory.spiked(n, initial_d, s, rho=0.5)
    true_support = base_design.true_support
    
    # We will incrementally add features up to d = 1000
    d_steps = np.arange(initial_d, 1001, 50)
    
    A_current = base_design.A.copy()
    
    results = []
    
    for current_d in d_steps:
        # If we need to add features, append random Gaussian columns and normalize
        if current_d > A_current.shape[1]:
            n_add = current_d - A_current.shape[1]
            A_new = np.random.normal(0, 1, (n, n_add))
            
            # Normalize new columns
            norms = np.linalg.norm(A_new, axis=0)
            A_new = (A_new / norms) * np.sqrt(n)
            
            A_current = np.hstack([A_current, A_new])
            
        metrics = compute_metrics_for_support(A_current, true_support, lambda_val)
        
        results.append({
            'd': current_d,
            'exact_h2': metrics['exact_h2'],
            'upper_h2': metrics['upper_h2'],
            'interaction_norm': metrics['interaction_norm']
        })
        
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_DIR, 'dimensionality_scaling.csv'), index=False)
    
    # Plotting
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.plot(df['d'], df['exact_h2'], 'k-', marker='o', label='Exact H(A)')
    ax1.plot(df['d'], df['upper_h2'], 'r--', marker='s', label='Theoretical Upper Bound')
    ax1.set_xlabel('Ambient Dimension (d)')
    ax1.set_ylabel('Hoffman Constant (l2)')
    ax1.set_title('Degradation of Local Geometry as Ambient Noise Features Increase')
    
    ax2 = ax1.twinx()
    ax2.plot(df['d'], df['interaction_norm'], 'b:', marker='^', label='Interaction ||G_AcA||_2')
    ax2.set_ylabel('Interaction Norm', color='b')
    ax2.tick_params(axis='y', labelcolor='b')
    
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='upper left')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'dimensionality_scaling.png'))
    plt.close()


if __name__ == '__main__':
    # Define grid of parameters for Table 1
    n_list = [200, 300, 400]
    d_list = [400, 600, 800]
    s_list = [10, 15, 20]
    
    part_1_comprehensive_tables(n_list, d_list, s_list)
    part_2_support_distribution()
    part_3_dimensionality_scaling()
    
    logger.info("Experiment 4 complete.")