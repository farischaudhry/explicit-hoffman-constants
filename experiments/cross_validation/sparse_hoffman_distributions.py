"""
Evaluates the empirical distribution of the exact l2 Hoffman constant H(A) 
across randomly sampled s-sparse active sets, iterating over various 
sparsity levels (s). Outputs histogram + statistical metrics for distribution.
"""

import logging 
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from scipy import stats
from tqdm import tqdm

from hoffman.designs.design_factory import DesignFactory
from hoffman.bounds.hoffman_bounds import HoffmanBoundCalculator
from hoffman.utils.viz import set_plotting_style

# Output directories
OUTPUT_DIR = './results/cross_validation/sparse_hoffman_distributions/'
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Configuration
np.random.seed(0)
set_plotting_style()
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_exact_h2(A: np.ndarray, active_set: np.ndarray, lam1: float):
    """Computes exact l2 matrix norm of the KKT inverse for ground truth."""
    n = A.shape[0]
    G = (A.T @ A) / n
    M_A = HoffmanBoundCalculator.build_kkt_matrix(G, active_set, lam1)
    try:
        M_inv = np.linalg.inv(M_A)
        return np.linalg.norm(M_inv, ord=2)
    except np.linalg.LinAlgError:
        return np.nan


def run_distribution_analysis(n=300, d=600, s_list=[5, 10, 15, 20, 30], n_mc_samples=2000, lambda_val=0.1):
    # We generate the base ambient matrix A once per design type
    # (The s=1 argument is just a placeholder for the factory; we manually sample supports later)
    designs = [
        ('Gaussian', DesignFactory.gaussian(n, d, s=1), 'gaussian'),
        (r'Spiked ($\rho$=0.8)', DesignFactory.spiked(n, d, s=1, rho=0.8), 'spiked_rho0.8')
    ]
    
    all_stats = []
    for name, design, file_path in designs:
        logger.info(f"\nProcessing Design: {name}")
        A = design.A
        
        design_fig_dir = os.path.join(FIG_DIR, file_path)
        os.makedirs(design_fig_dir, exist_ok=True)
        
        for current_s in s_list:
            logger.info(f"Sampling s={current_s}.")
            
            h2_values = []
            for _ in tqdm(range(n_mc_samples), desc=f"MC Loop", leave=False):
                support = np.random.choice(d, current_s, replace=False)
                h2 = get_exact_h2(A, support, lambda_val)
                if np.isfinite(h2):
                    h2_values.append(h2)
                    
            if not h2_values:
                logger.warning(f"All singular for s={current_s}. Skipping.")
                continue
                
            h2_values = np.array(h2_values)
            
            # Calculate statistical metrics
            mean_h2 = np.mean(h2_values)
            median_h2 = np.median(h2_values)
            var_h2 = np.var(h2_values)
            skewness = stats.skew(h2_values)
            kurt = stats.kurtosis(h2_values)
            max_h2 = np.max(h2_values)
            p99_h2 = np.percentile(h2_values, 99)
            
            all_stats.append({
                'Design': name,
                's': current_s,
                'Samples': len(h2_values),
                'Mean': mean_h2,
                'Median': median_h2,
                'Variance': var_h2,
                'Skewness': skewness,
                'Kurtosis': kurt,
                'Max (Worst-Case)': max_h2,
                '99th Percentile': p99_h2
            })
            
            # --- Plotting ---
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.hist(h2_values, bins=50, density=True, color='steelblue', edgecolor='black', alpha=0.7)
            # KDE Overlay
            try:
                kde = stats.gaussian_kde(h2_values)
                x_grid = np.linspace(h2_values.min(), h2_values.max(), 500)
                ax.plot(x_grid, kde(x_grid), 'k-', lw=2, label='KDE Density')
            except np.linalg.LinAlgError:
                pass # Fallback if KDE fails due to singular covariance in edge cases
            # Markers
            ax.axvline(mean_h2, color='red', linestyle='dashed', linewidth=2, label=f'Mean: {mean_h2:.2f}')
            ax.axvline(p99_h2, color='orange', linestyle='dashdot', linewidth=2, label=f'99th %: {p99_h2:.2f}')
            ax.axvline(max_h2, color='darkred', linestyle='solid', linewidth=2, label=f'Max: {max_h2:.2f}')
            # Stats Text Box
            textstr = '\n'.join((
                r'$s=%d$' % (current_s, ),
                r'$\mu=%.3f$' % (mean_h2, ),
                r'$\sigma^2=%.3f$' % (var_h2, ),
                r'$\mathrm{skew}=%.3f$' % (skewness, ),
                r'$\mathrm{kurtosis}=%.3f$' % (kurt, )))
            props = dict(boxstyle='round', facecolor='wheat', alpha=0.5)
            ax.text(0.95, 0.50, textstr, transform=ax.transAxes, fontsize=12,
                    verticalalignment='center', horizontalalignment='right', bbox=props)
            ax.set_title(f'Empirical Distribution of $H^2$ - {name.replace("_", " ")} (s={current_s})')
            ax.set_xlabel('Exact Local Affine Hoffman Constant')
            ax.set_ylabel('Density')
            ax.legend(loc='upper right')
            
            plt.tight_layout()
            # Save into the specific design subdirectory
            plt.savefig(os.path.join(design_fig_dir, f'local_hoffman_s={current_s}.png'))
            plt.close()

    # Save stats to CSV
    df_stats = pd.DataFrame(all_stats)
    df_stats.to_csv(os.path.join(DATA_DIR, 'support_distribution_stats.csv'), index=False)
    
    logger.info("\nStatistical Summary across s values:")
    logger.info(df_stats[['Design', 's', 'Mean', 'Variance', 'Skewness', 'Kurtosis', 'Max (Worst-Case)']].to_string())
    logger.info(f"\nPlots saved to design subdirectories in {FIG_DIR}")


if __name__ == '__main__':
    # Sweep through varying sparsity levels
    s_list_to_test = [5]
    run_distribution_analysis(n=300, d=600, s_list=s_list_to_test, n_mc_samples=100, lambda_val=0.1)
