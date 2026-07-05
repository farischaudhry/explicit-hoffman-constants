"""
Experiment 4: Hoffman Bound Tightness Analysis
-----------------------------------------------
Evaluates the tightness of explicit theoretical bounds on the Hoffman constant 
across various problem configurations and design matrix types.
"""

import logging 
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import pandas as pd

from hoffman.designs.design_factory import DesignFactory
from hoffman.bounds.hoffman_bounds import HoffmanBoundCalculator
from hoffman.utils.viz import set_plotting_style

# Output directories
OUTPUT_DIR = './results/04_hoffman_bound_tightness/'
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Configuration
np.random.seed(0)
set_plotting_style()
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'hoffman_bound_tightness.log'), mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)


def compute_bounds_for_design(design_name: str, design, lambda_val: float, active_set: np.ndarray):
    """Compute all Hoffman bounds for a given design and active set."""
    n = design.A.shape[0]
    
    # API Call: A, active_set, lam1
    bounds = HoffmanBoundCalculator.compute_all(
        A=design.A,
        active_set=active_set, 
        lam1=lambda_val
    )
    
    # Supplemental metrics for visualization
    G = (design.A.T @ design.A) / n
    G_AA = G[np.ix_(active_set, active_set)] if len(active_set) > 0 else None
    cond_AA = np.linalg.cond(G_AA) if G_AA is not None else 1.0
    lam_max_AA = np.linalg.eigvalsh(G_AA).max() if G_AA is not None else 0.0

    return {
        'design': design_name,
        'n': n,
        'd': design.A.shape[1],
        's': len(active_set),
        'lambda': lambda_val,
        'exact_hoffman': bounds.exact_local_affine_hoffman,
        'lower_bound': bounds.dimension_free_bound_lower,
        'upper_bound': bounds.dimension_free_bound_upper,
        'weyl_bound': bounds.weyl_perturbation_bound,
        'l_inf_bound': bounds.l_inf_bound,
        'l_1_bound': bounds.l_1_bound,
        'lambda_min_AA': bounds.lambda_min_active,
        'condition_number_AA': cond_AA,
        'interaction_norm': bounds.interaction_norm,
        'cross_leverage_bound': bounds.cross_leverage_bound,
        'max_leverage': bounds.max_leverage_score,
        'inactive_leverage_sum': bounds.inactive_leverage_sum,
        'active_amplification': lam_max_AA,
        'lower_ratio': bounds.exact_local_affine_hoffman / bounds.dimension_free_bound_lower if bounds.dimension_free_bound_lower > 0 else np.nan,
        'upper_ratio': bounds.dimension_free_bound_upper / bounds.exact_local_affine_hoffman if bounds.exact_local_affine_hoffman > 0 else np.nan,
        'weyl_ratio': bounds.weyl_perturbation_bound / bounds.exact_local_affine_hoffman if bounds.exact_local_affine_hoffman > 0 else np.nan,
    }


def part_1_spiked_model_varying_rho():
    """Test bound tightness on spiked model with varying correlation rho."""
    logger.info('\n' + '='*80)
    logger.info('Part 1: Spiked Model - Varying Correlation rho')
    logger.info('='*80)
    
    n, d, s = 200, 400, 10
    lambda_val = 0.01 
    rho_values = np.linspace(0.0, 0.95, 10)
    n_trials = 10
    
    results = []
    for rho in rho_values:
        trial_results = []
        for _ in range(n_trials):
            design = DesignFactory.spiked(n=n, d=d, s=s, rho=rho)
            trial_results.append(compute_bounds_for_design(f'Spiked_{rho:.3f}', design, lambda_val, design.true_support))
        
        res_mean = {'rho': rho}
        for key in ['exact_hoffman', 'lower_bound', 'upper_bound', 'lower_ratio', 'upper_ratio', 'condition_number_AA', 'interaction_norm']:
            vals = [r[key] for r in trial_results if np.isfinite(r[key])]
            res_mean[key] = np.mean(vals)
            res_mean[f'{key}_ci_lower'] = np.percentile(vals, 2.5)
            res_mean[f'{key}_ci_upper'] = np.percentile(vals, 97.5)
        results.append(res_mean)
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_DIR, 'spiked_varying_rho.csv'), index=False)
    
    fig = plt.figure(figsize=(14, 8))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['rho'], df['exact_hoffman'], 'k-', lw=3, label='Exact H(A)')
    ax1.plot(df['rho'], df['lower_bound'], 'b--', label='Lower Bound')
    ax1.plot(df['rho'], df['upper_bound'], 'r--', label='Upper Bound')
    ax1.set_yscale('log')
    ax1.set_title('Hoffman Constant vs Correlation (Spiked Model)')
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df['rho'], df['lower_ratio'], 'b-', label='H / Lower')
    ax2.plot(df['rho'], df['upper_ratio'], 'r-', label='Upper / H')
    ax2.set_title('Bound Ratios (Closer to 1 is tighter)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = fig.add_subplot(gs[1, 1])
    ax3.plot(df['rho'], df['interaction_norm'], 'purple')
    ax3.set_title('Interaction Component Growth')
    ax3.set_ylabel('||Interaction||')
    ax3.grid(True, alpha=0.3)
    
    plt.savefig(os.path.join(FIG_DIR, 'spiked_varying_rho.png'))
    return df


def part_2_leverage_bound_tightness():
    """Test how well leverage bound approximates spectral norm."""
    logger.info('\n' + '='*80)
    logger.info('Part 2: Leverage-Based Interaction Bound Tightness')
    logger.info('='*80)
    
    n, d, s = 200, 400, 10
    lambda_val = 0.1
    
    scenarios = [
        ('Gaussian', lambda: DesignFactory.gaussian(n, d, s)),
        ('Partial DCT', lambda: DesignFactory.partial_dct(n, d, s)),
        ('Spiked ρ=0.3', lambda: DesignFactory.spiked(n, d, s, rho=0.3)),
        ('Spiked ρ=0.7', lambda: DesignFactory.spiked(n, d, s, rho=0.7)),
    ]
    
    results = []
    for name, gen in scenarios:
        design = gen()
        res = compute_bounds_for_design(name, design, lambda_val, design.true_support)
        results.append(res)
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_DIR, 'leverage_tightness.csv'), index=False)
    
    logger.info(f"{'Design':<15} | {'Interaction Norm':<18} | {'Leverage Bound':<15}")
    logger.info("-" * 55)
    for _, row in df.iterrows():
        logger.info(f"{row['design']:<15} | {row['interaction_norm']:<18.4f} | {row['cross_leverage_bound']:<15.4f}")
    
    return df


def part_3_many_design_comparison():
    """Comparison across many design types."""
    logger.info('\n' + '='*80)
    logger.info('Part 3: Many Design Comparison')
    logger.info('='*80)
    
    n, d, s = 300, 600, 15
    lambda_val = 0.2
    
    scenarios = {
        'Orthonormal': lambda: DesignFactory.orthonormal(n, 100, s),
        'Gaussian': lambda: DesignFactory.gaussian(n, d, s),
        'Rademacher': lambda: DesignFactory.rademacher(n, d, s),
        'Partial DCT': lambda: DesignFactory.partial_dct(n, d, s),
        'Spiked ρ=0.5': lambda: DesignFactory.spiked(n, d, s, rho=0.5),
        'Spiked ρ=0.9': lambda: DesignFactory.spiked(n, d, s, rho=0.9),
        'Toeplitz ρ=0.9': lambda: DesignFactory.strict_toeplitz(n, d, s, rho=0.9),
    }
    
    results = []
    for name, gen in scenarios.items():
        design = gen()
        results.append(compute_bounds_for_design(name, design, lambda_val, design.true_support))
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_DIR, 'many_design_comparison.csv'), index=False)
    
    logger.info(f"{'Design':<20} | {'Exact H(A)':<10} | {'Upper/Exact':<12} | {'Lower':<12} | {'Upper':<12}")
    logger.info("-" * 50)
    for _, row in df.sort_values('upper_ratio').iterrows():
        logger.info(f"{row['design']:<20} | {row['exact_hoffman']:<10.2f} | {row['upper_ratio']:<12.2f} | {row['lower_bound']:<12.2f} | {row['upper_bound']:<12.2f}")
    
    return df


def part_4_dimension_scaling():
    """Test scaling with ambient dimension d."""
    logger.info('\n' + '='*80)
    logger.info('Part 4: Dimension Scaling (fixed n, s)')
    logger.info('='*80)
    
    n, s = 200, 10
    lambda_val = 0.1
    d_values = [200, 400, 800, 1600]
    
    all_results = []
    for d in d_values:
        design = DesignFactory.gaussian(n, d, s)
        all_results.append(compute_bounds_for_design(f'Gaussian_d={d}', design, lambda_val, design.true_support))
        
        design_spiked = DesignFactory.spiked(n, d, s, rho=0.7)
        all_results.append(compute_bounds_for_design(f'Spiked_d={d}', design_spiked, lambda_val, design_spiked.true_support))

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(DATA_DIR, 'dimension_scaling.csv'), index=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for name in ['Gaussian', 'Spiked']:
        data = df[df['design'].str.contains(name)]
        # Extract d from design name string or use stored d
        ax.plot(data['d'], data['exact_hoffman'], marker='o', label=name)
    
    ax.set_yscale('log')
    ax.set_xlabel('Dimension d')
    ax.set_ylabel('H(A)')
    ax.set_title('Hoffman Scaling with Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, 'dimension_scaling.png'))
    
    return df


def part_5_sparsity_scaling():
    """Test scaling with sparsity s."""
    logger.info('\n' + '='*80)
    logger.info('Part 5: Sparsity Scaling (fixed n, d)')
    logger.info('='*80)
    
    n, d = 200, 400
    lambda_val = 0.1
    s_values = [5, 10, 20, 40, 80]
    
    all_results = []
    for s in s_values:
        design = DesignFactory.gaussian(n, d, s)
        all_results.append(compute_bounds_for_design(f'Gaussian_s={s}', design, lambda_val, design.true_support))

    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(DATA_DIR, 'sparsity_scaling.csv'), index=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(df['s'], df['exact_hoffman'], marker='s', color='C0')
    ax.set_yscale('log')
    ax.set_xlabel('Sparsity s')
    ax.set_ylabel('H(A)')
    ax.set_title('Hoffman Scaling with Sparsity')
    ax.grid(True, alpha=0.3)
    plt.savefig(os.path.join(FIG_DIR, 'sparsity_scaling.png'))
    
    return df


if __name__ == '__main__':
    part_1_spiked_model_varying_rho()
    part_2_leverage_bound_tightness()
    part_3_many_design_comparison()
    part_4_dimension_scaling()
    part_5_sparsity_scaling()
    logger.info("\nAll analysis parts complete. Results stored in ./results/04_hoffman_bound_tightness/")
