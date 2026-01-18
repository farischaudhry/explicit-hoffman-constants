"""
Experiment 4: Hoffman Bound Tightness Analysis
-----------------------------------------------

This experiment evaluates the tightness of different theoretical bounds on the 
Hoffman constant across various problem configurations and design matrix types.

We test:
1. How tight are the lower/upper bounds from Eq. (hoffman-bounds)?
2. How does the Weyl bound compare?
3. How well does the leverage score bound approximate ||G_AcA||_2?
4. How do these bounds scale with dimension and sparsity?

Metrics computed:
- Exact Hoffman constant H(A) = 1/sigma_min(M_A)
- Lower bound: 1/min(lambda_min(G_AA), lambda)
- Upper bound: max(||G_AA^{-1}||_2, (1/lambda)(1 + ||G_AcA G_AA^{-1}||_2))
- Weyl bound: 1/(min(lambda_min(G_AA), lambda) - ||G_AcA||_2)
- Leverage bound: ||G_AcA||_2 vs sqrt(lambda_max(G_AA)) x sqrt(sum h_j(A))
"""

import logging 
import os
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
from matplotlib.patches import Patch
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
    G = (design.A.T @ design.A) / n
    
    bounds = HoffmanBoundCalculator.compute_all(
        G=G, 
        active_set=active_set, 
        lam=lambda_val,
        A=design.A
    )
    
    return {
        'design': design_name,
        'n': n,
        'd': design.A.shape[1],
        's': len(active_set),
        'lambda': lambda_val,
        'exact_hoffman': bounds.exact_hoffman,
        'lower_bound': bounds.max_bound_lower,
        'upper_bound': bounds.max_bound_upper,
        'weyl_bound': bounds.weyl_bound_upper,
        'lambda_min_AA': bounds.lambda_min_AA,
        'condition_number_AA': bounds.condition_number_AA,
        'true_interaction_norm': bounds.true_interaction_norm,  # True ||G_AcA||_2
        'interaction_norm_bound': bounds.interaction_norm_bound,  # Leverage-based bound
        'max_leverage': bounds.max_leverage_score,
        'inactive_leverage_sum': bounds.inactive_leverage_sum,
        'active_amplification': bounds.lambda_max_AA,
        'lower_gap': bounds.exact_hoffman - bounds.max_bound_lower,
        'upper_gap': bounds.max_bound_upper - bounds.exact_hoffman,
        'weyl_gap': bounds.weyl_bound_upper - bounds.exact_hoffman,
        'lower_ratio': bounds.exact_hoffman / bounds.max_bound_lower if bounds.max_bound_lower > 0 else np.nan,
        'upper_ratio': bounds.max_bound_upper / bounds.exact_hoffman if bounds.exact_hoffman > 0 else np.nan,
        'weyl_ratio': bounds.weyl_bound_upper / bounds.exact_hoffman if bounds.exact_hoffman > 0 else np.nan,
    }


def part_1_spiked_model_varying_rho():
    """Test bound tightness on spiked model with varying correlation rho."""
    logger.info('\n' + '='*80)
    logger.info('Part 1: Spiked Model - Varying Correlation rho (with 95% CI)')
    logger.info('='*80)
    
    n, d, s = 200, 400, 10
    lambda_val = 0.01  # Small lambda to observe 'correct' growth in H(A)
    rho_values = np.linspace(0.0, 0.95, 20)
    n_trials = 20  # Multiple trials for CIs
    
    results = []
    
    for rho in rho_values:
        trial_results = []
        for _ in range(n_trials):
            design = DesignFactory.spiked(n=n, d=d, s=s, rho=rho)
            active_set = design.true_support
            
            result = compute_bounds_for_design(
                design_name=f'Spiked_rho={rho:.3f}',
                design=design,
                lambda_val=lambda_val,
                active_set=active_set
            )
            trial_results.append(result)
        
        # Compute mean and 95% CI across trials
        result_mean = {
            'design': f'Spiked_rho={rho:.3f}',
            'rho': rho,
            'n': n, 'd': d, 's': s, 'lambda': lambda_val,
        }
        
        for key in ['exact_hoffman', 'lower_bound', 'upper_bound', 'weyl_bound',
                    'lambda_min_AA', 'condition_number_AA', 'true_interaction_norm', 'interaction_norm_bound', 'max_leverage', 
                    'lower_ratio', 'upper_ratio', 'weyl_ratio', 'lower_gap', 'upper_gap', 'weyl_gap']:
            values = [r[key] for r in trial_results]
            # Filter out inf/nan for statistics
            finite_values = [v for v in values if np.isfinite(v)]
            if finite_values:
                result_mean[key] = np.mean(finite_values)
                result_mean[f'{key}_std'] = np.std(finite_values)
                result_mean[f'{key}_ci_lower'] = np.percentile(finite_values, 2.5)
                result_mean[f'{key}_ci_upper'] = np.percentile(finite_values, 97.5)
            else:
                result_mean[key] = np.nan
                result_mean[f'{key}_std'] = np.nan
                result_mean[f'{key}_ci_lower'] = np.nan
                result_mean[f'{key}_ci_upper'] = np.nan
        
        results.append(result_mean)
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_DIR, 'spiked_varying_rho.csv'), index=False)
    
    # Plot
    fig = plt.figure(figsize=(16, 10))
    gs = GridSpec(2, 2, figure=fig, hspace=0.3, wspace=0.3)
    
    # Plot 1: Hoffman constant and bounds vs rho (with confidence intervals)
    ax1 = fig.add_subplot(gs[0, :])
    ax1.plot(df['rho'], df['exact_hoffman'], 'k-', linewidth=3, label='Exact H(A)', zorder=5)
    
    # Add 95% confidence interval for exact Hoffman
    if 'exact_hoffman_ci_lower' in df.columns:
        ax1.fill_between(df['rho'], df['exact_hoffman_ci_lower'], df['exact_hoffman_ci_upper'],
                         alpha=0.2, color='black', zorder=4, label='95% CI')
    
    ax1.plot(df['rho'], df['lower_bound'], 'b-', linewidth=2, label='Lower Bound', alpha=0.8)
    ax1.plot(df['rho'], df['upper_bound'], 'r-', linewidth=2, label='Upper Bound', alpha=0.8)
    
    # Fill between lower and upper bounds
    ax1.fill_between(df['rho'], df['lower_bound'], df['upper_bound'], 
                     alpha=0.15, color='gray', label='Bound Range')
    
    ax1.set_xlabel(r'Correlation $\rho$')
    ax1.set_ylabel('Hoffman Constant H(A)')
    ax1.set_title('Hoffman Bounds vs Correlation (Spiked Model)', fontweight='bold')
    ax1.legend(loc='upper left')
    ax1.grid(True, alpha=0.3)
    ax1.set_yscale('log')
    
    # Plot 2: Bound ratios (tightness)
    ax2 = fig.add_subplot(gs[1, 0])
    ax2.plot(df['rho'], df['lower_ratio'], 'b-', linewidth=2, label='H / Lower')
    ax2.plot(df['rho'], df['upper_ratio'], 'r-', linewidth=2, label='Upper / H')
    ax2.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, linewidth=1)
    ax2.set_xlabel(r'Correlation $\rho$')
    ax2.set_ylabel('Bound Ratio')
    ax2.set_title('Bound Tightness (closer to 1 = tighter)')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    ax2.set_yscale('log')
    
    # Plot 3: Interaction term and condition number
    ax3 = fig.add_subplot(gs[1, 1])
    ax3_twin = ax3.twinx()
    l1 = ax3.plot(df['rho'], df['true_interaction_norm'], 'purple', linewidth=2, label=r'$\|G_{A^c A}\|_2$')
    l2 = ax3_twin.plot(df['rho'], df['condition_number_AA'], 'orange', linewidth=2, label=r'$\kappa(G_{AA})$')
    ax3.set_xlabel(r'Correlation $\rho$')
    ax3.set_ylabel(r'Interaction Norm $\|G_{A^c A}\|_2$', color='purple')
    ax3_twin.set_ylabel(r'Condition Number $\kappa(G_{AA})$', color='orange')
    ax3.set_title('Problem Geometry vs Correlation')
    ax3.tick_params(axis='y', labelcolor='purple')
    ax3_twin.tick_params(axis='y', labelcolor='orange')
    ax3.grid(True, alpha=0.3)
    
    # Combine legends
    lines = l1 + l2
    labels = [l.get_label() for l in lines]
    ax3.legend(lines, labels, loc='upper left')
    
    plt.savefig(os.path.join(FIG_DIR, 'spiked_varying_rho.png'), dpi=300, bbox_inches='tight')
    logger.info(f'Saved figure: spiked_varying_rho.png')
    
    # Print summary table 
    lower_valid = (df['exact_hoffman'] >= df['lower_bound'] - 1e-6).all()
    upper_valid = (df['exact_hoffman'] <= df['upper_bound'] + 1e-6).all()
    avg_tightness = df['upper_ratio'].mean() / df['lower_ratio'].mean()
    
    logger.info(f'  Lower bound valid for all rho: {lower_valid}')
    logger.info(f'  Upper bound valid for all rho: {upper_valid}')
    logger.info(f'  Average tightness (upper/lower): {avg_tightness:.2f}x')
    logger.info(f"  Tightest at rho={df.loc[df['upper_ratio'].idxmin(), 'rho']:.2f} (ratio={df['upper_ratio'].min():.2f}x)")
    logger.info(f"  Loosest at rho={df.loc[df['upper_ratio'].idxmax(), 'rho']:.2f} (ratio={df['upper_ratio'].max():.2f}x)")
    
    return df


def part_2_leverage_bound_tightness():
    """
    Test the leverage-based interaction bound from the paper:
        ||G_AcA||_2 <= sqrt(λ_max(G_AA)) * sqrt(Σ h_j)
    """
    logger.info('\n' + '='*80)
    logger.info('Part 2: Leverage-Based Interaction Bound Tightness (with 95% CI)')
    logger.info('='*80)
    
    n, d, s = 200, 400, 10
    lambda_val = 0.1
    n_trials = 30  # Multiple trials for confidence intervals
    
    # Test on different design types
    # RIP designs: Gaussian, Rademacher, Partial DCT
    # Non-RIP designs: Spiked (correlation-based)
    designs_to_test = [
        ('Gaussian', lambda: DesignFactory.gaussian(n=n, d=d, s=s)),
        ('Rademacher', lambda: DesignFactory.rademacher(n=n, d=d, s=s)),
        ('Partial DCT', lambda: DesignFactory.partial_dct(n=n, d=d, s=s)),
        ('Spiked $\\rho=0.3$', lambda: DesignFactory.spiked(n=n, d=d, s=s, rho=0.3)),
        ('Spiked $\\rho=0.7$', lambda: DesignFactory.spiked(n=n, d=d, s=s, rho=0.7)),
        ('Spiked $\\rho=0.9$', lambda: DesignFactory.spiked(n=n, d=d, s=s, rho=0.9)),
    ]
    
    results = []
    for design_name, gen_func in designs_to_test:
        trial_results = []
        for _ in range(n_trials):
            design = gen_func()
            active_set = design.true_support
            inactive_set = np.array([i for i in range(d) if i not in active_set])
            
            # Compute G matrix
            G = (design.A.T @ design.A) / n
            G_AA = G[np.ix_(active_set, active_set)]
            G_AcA = G[np.ix_(inactive_set, active_set)]
            
            # True spectral norm (what we want to bound)
            true_spectral_norm = np.linalg.norm(G_AcA, ord=2)
            
            # Frobenius norm (classic bound)
            frobenius_norm = np.linalg.norm(G_AcA, ord='fro')
            
            # Compute leverage scores h_j(A) = (1/n)||P_A a_j||^2
            leverage_scores = HoffmanBoundCalculator._compute_leverage_scores(design.A, active_set)
            inactive_leverage_scores = leverage_scores[inactive_set]
            
            # Three-factor decomposition components
            sum_leverage = np.sum(inactive_leverage_scores) 
            lambda_max_AA = np.linalg.eigvalsh(G_AA).max() 
            
            # Leverage-based bound: sqrt(λ_max(G_AA)) * sqrt(Σh_j)
            leverage_bound = np.sqrt(lambda_max_AA) * np.sqrt(sum_leverage)
            
            # Max leverage score
            max_leverage = np.max(inactive_leverage_scores)
            
            # RIP-based bound: δ_{2s} * α * (1 + √s)
            # Only compute for RIP-satisfying designs (Gaussian, not Spiked)
            if 'Spiked' in design_name:
                # Spiked designs do not satisfy RIP, so RIP bound is not applicable
                rip_constant = np.nan
                rip_bound = np.nan
                rip_ratio = np.nan
                rip_bound_valid = False
            else:
                # Estimate RIP constant δ_{2s} for sparsity 2s
                rip_constant = HoffmanBoundCalculator.compute_rip_constant(design.A, s=2*s, num_trials=50)
                # Restricted cone parameter α = 3
                alpha = 3.0
                # RIP-based interaction bound: δ_{2s} * α * (1 + √s)
                rip_bound = rip_constant * alpha * (1 + np.sqrt(s))
                rip_ratio = rip_bound / true_spectral_norm
                rip_bound_valid = rip_bound >= true_spectral_norm
            
            trial_results.append({
                'true_spectral_norm': true_spectral_norm,
                'frobenius_norm': frobenius_norm,
                'leverage_bound': leverage_bound,
                'rip_bound': rip_bound,
                'rip_constant': rip_constant,
                'lambda_max_AA': lambda_max_AA,
                'sum_leverage': sum_leverage,
                'max_leverage': max_leverage,
                'frobenius_ratio': frobenius_norm / true_spectral_norm,
                'leverage_ratio': leverage_bound / true_spectral_norm,
                'rip_ratio': rip_ratio,
                # Check if bounds are valid (should be >= 1 if bound holds)
                'leverage_bound_valid': leverage_bound >= true_spectral_norm,
                'rip_bound_valid': rip_bound_valid,
            })
        
        # Compute mean and 95% CI
        result_mean = {'design': design_name}
        for key in ['true_spectral_norm', 'frobenius_norm', 'leverage_bound', 'rip_bound',
                    'rip_constant', 'lambda_max_AA', 'sum_leverage', 'max_leverage', 
                    'frobenius_ratio', 'leverage_ratio', 'rip_ratio']:
            values = [r[key] for r in trial_results]
            # Handle NaN values for non-RIP designs
            values_clean = [v for v in values if not np.isnan(v)]
            if values_clean:
                result_mean[key] = np.mean(values_clean)
                result_mean[f'{key}_std'] = np.std(values_clean)
                result_mean[f'{key}_ci_lower'] = np.percentile(values_clean, 2.5)
                result_mean[f'{key}_ci_upper'] = np.percentile(values_clean, 97.5)
            else:
                result_mean[key] = np.nan
                result_mean[f'{key}_std'] = np.nan
                result_mean[f'{key}_ci_lower'] = np.nan
                result_mean[f'{key}_ci_upper'] = np.nan
        
        # Compute bound validity percentages
        leverage_valid_pct = np.mean([r['leverage_bound_valid'] for r in trial_results]) * 100
        result_mean['leverage_valid_pct'] = leverage_valid_pct
        
        rip_valid_trials = [r['rip_bound_valid'] for r in trial_results if not np.isnan(r['rip_bound'])]
        if rip_valid_trials:
            result_mean['rip_valid_pct'] = np.mean(rip_valid_trials) * 100
        else:
            result_mean['rip_valid_pct'] = np.nan
        
        results.append(result_mean)
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_DIR, 'leverage_bound_tightness.csv'), index=False)
    
    # Print summary with confidence intervals
    for _, row in df.iterrows():
        logger.info(f"\n{row['design']}:")
        logger.info(f"  ||G_AcA||_2 (true)             = {row['true_spectral_norm']:.4f} ± [{row['true_spectral_norm_ci_lower']:.4f}, {row['true_spectral_norm_ci_upper']:.4f}]")
        logger.info(f"  sqrt(λ_max) * sqrt(Σh_j)       = {row['leverage_bound']:.4f} ± [{row['leverage_bound_ci_lower']:.4f}, {row['leverage_bound_ci_upper']:.4f}]")
        if not np.isnan(row['rip_bound']):
            logger.info(f"  δ_{{2s}} * α * (1+√s) (RIP)    = {row['rip_bound']:.4f} ± [{row['rip_bound_ci_lower']:.4f}, {row['rip_bound_ci_upper']:.4f}]")
        else:
            logger.info(f"  δ_{{2s}} * α * (1+√s) (RIP)    = N/A (non-RIP design)")
        logger.info(f"  ||G_AcA||_F (Frobenius)        = {row['frobenius_norm']:.4f} ± [{row['frobenius_norm_ci_lower']:.4f}, {row['frobenius_norm_ci_upper']:.4f}]")
        logger.info(f"  Decomposition:")
        logger.info(f"    λ_max(G_AA)                  = {row['lambda_max_AA']:.4f} ± [{row['lambda_max_AA_ci_lower']:.4f}, {row['lambda_max_AA_ci_upper']:.4f}]")
        logger.info(f"    Σh_j (alignment)             = {row['sum_leverage']:.4f} ± [{row['sum_leverage_ci_lower']:.4f}, {row['sum_leverage_ci_upper']:.4f}]")
        logger.info(f"    Max h_j                      = {row['max_leverage']:.4f} ± [{row['max_leverage_ci_lower']:.4f}, {row['max_leverage_ci_upper']:.4f}]")
        if not np.isnan(row['rip_constant']):
            logger.info(f"    δ_{{2s}} (RIP constant)       = {row['rip_constant']:.4f} ± [{row['rip_constant_ci_lower']:.4f}, {row['rip_constant_ci_upper']:.4f}]")
        logger.info(f"  Tightness Ratios:")
        logger.info(f"    Leverage / True              = {row['leverage_ratio']:.4f} ± {row['leverage_ratio_std']:.4f}")
        if not np.isnan(row['rip_ratio']):
            logger.info(f"    RIP / True                   = {row['rip_ratio']:.4f} ± {row['rip_ratio_std']:.4f}")
        logger.info(f"    Frobenius / True             = {row['frobenius_ratio']:.4f} ± {row['frobenius_ratio_std']:.4f}")
        logger.info(f"  Bound validity:")
        logger.info(f"    Leverage: {row['leverage_valid_pct']:.1f}% of trials")
        if not np.isnan(row['rip_valid_pct']):
            logger.info(f"    RIP: {row['rip_valid_pct']:.1f}% of trials")
    
    # Plot with error bars
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Comparing bounds with error bars
    ax = axes[0]
    x = np.arange(len(df))
    width = 0.2
    
    ax.bar(x - 1.5*width, df['true_spectral_norm'], width, label='True $\\|G_{A^c A}\\|_2$', 
           color='black', alpha=0.8, yerr=df['true_spectral_norm_std'], capsize=3)
    ax.bar(x - 0.5*width, df['leverage_bound'], width, label='Leverage Bound', 
           color='red', alpha=0.7, yerr=df['leverage_bound_std'], capsize=3)
    
    # Only plot RIP bound for non-Spiked designs
    rip_values = df['rip_bound'].copy()
    rip_errors = df['rip_bound_std'].copy()
    for i, design in enumerate(df['design']):
        if 'Spiked' in design:
            rip_values.iloc[i] = 0
            rip_errors.iloc[i] = 0
    
    ax.bar(x + 0.5*width, rip_values, width, label='RIP Bound', 
           color='green', alpha=0.7, yerr=rip_errors, capsize=3)
    ax.bar(x + 1.5*width, df['frobenius_norm'], width, label='Frobenius Bound', 
           color='blue', alpha=0.7, yerr=df['frobenius_norm_std'], capsize=3)
    
    ax.set_xlabel('Design Type')
    ax.set_ylabel('Norm Value')
    ax.set_title(r'Bounding $\|G_{A^c A}\|_2$ via Leverage vs RIP')
    ax.set_xticks(x)
    ax.set_xticklabels(df['design'], rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 2: Tightness ratios
    ax = axes[1]
    x = np.arange(len(df))
    width = 0.25
    ax.bar(x - width, df['leverage_ratio'], width, label='Leverage / True', color='red', alpha=0.7)
    
    # Only show RIP ratios for non-Spiked designs
    rip_ratio_plot = df['rip_ratio'].copy()
    for i, design in enumerate(df['design']):
        if 'Spiked' in design:
            rip_ratio_plot.iloc[i] = 0
    ax.bar(x, rip_ratio_plot, width, label='RIP / True', color='green', alpha=0.7)
    
    ax.bar(x + width, df['frobenius_ratio'], width, label='Frobenius / True', color='blue', alpha=0.7)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5, linewidth=1.5, label='Perfect bound')
    
    ax.set_xlabel('Design Type')
    ax.set_ylabel('Ratio (≥1 if bound holds)')
    ax.set_title('Bound Tightness')
    ax.set_xticks(x)
    ax.set_xticklabels(df['design'], rotation=15, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'leverage_rip_bound_comparison.png'), dpi=300, bbox_inches='tight')
    logger.info(f'\n  Saved figure: leverage_rip_bound_comparison.png')
    
    return df


def part_3_many_design_comparison():
    """
    Test bound tightness across many design matrices.
    This addresses the observation that leverage bounds and Hoffman scaling
    vary significantly across different correlation structures.
    """
    logger.info('\n' + '='*80)
    logger.info('Part 3: Many Design Comparison')
    logger.info('='*80)
    
    n, d, s = 300, 600, 15
    lambda_val = 0.2
    
    # Define diverse set of design matrices
    scenarios = {
        'Orthonormal': lambda: DesignFactory.orthonormal(n, min(n, s*2), s),
        'Gaussian': lambda: DesignFactory.gaussian(n, d, s),
        'Rademacher': lambda: DesignFactory.rademacher(n, d, s),
        'Partial DCT': lambda: DesignFactory.partial_dct(n, d, s),
        'Spiked ρ=0.3': lambda: DesignFactory.spiked(n, d, s, rho=0.3),
        'Spiked ρ=0.5': lambda: DesignFactory.spiked(n, d, s, rho=0.5),
        'Spiked ρ=0.7': lambda: DesignFactory.spiked(n, d, s, rho=0.7),
        'Spiked ρ=0.9': lambda: DesignFactory.spiked(n, d, s, rho=0.9),
        'AR1-Stochastic ρ=0.5': lambda: DesignFactory.ar1_stochastic(n, d, s, rho=0.5),
        'AR1-Stochastic ρ=0.9': lambda: DesignFactory.ar1_stochastic(n, d, s, rho=0.9),
        'Toeplitz ρ=0.9': lambda: DesignFactory.strict_toeplitz(n, d, s, rho=0.9),
        'Block-Diagonal': lambda: DesignFactory.block_diagonal_ar1(n, d, s, block_size=20, rho=0.8),
    }
    
    results = []
    
    for name, gen_func in scenarios.items():
        design = gen_func()
        active_set = design.true_support
        
        result = compute_bounds_for_design(
            design_name=name,
            design=design,
            lambda_val=lambda_val,
            active_set=active_set
        )
        results.append(result)
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_DIR, 'many_design_comparison.csv'), index=False)
    
    # Plot bar charts for each design
    fig, axes = plt.subplots(4, 3, figsize=(18, 16))
    axes = axes.flatten()
    
    for idx, (_, row) in enumerate(df.iterrows()):
        if idx >= len(axes):
            break
            
        ax = axes[idx]
        
        # Bar chart showing bounds
        y_pos = [0, 1, 2]
        values = [row['lower_bound'], row['exact_hoffman'], row['upper_bound']]
        colors = ['#2ecc71', '#e74c3c', '#3498db']
        labels = ['Lower', 'H(A)', 'Upper']
        
        bars = ax.barh(y_pos, values, color=colors, alpha=0.7, edgecolor='black', linewidth=1.5)
        
        # Add value labels
        for i, (bar, val) in enumerate(zip(bars, values)):
            if np.isfinite(val) and val < 1e6:
                ax.text(val * 1.05, bar.get_y() + bar.get_height()/2, 
                       f'{val:.1f}', va='center', fontweight='bold')
        
        ax.set_yticks(y_pos)
        ax.set_yticklabels(labels)
        ax.set_xlabel('Value')
        ax.set_title(f'{row["design"]}\n(Upper/Lower: {row["upper_ratio"]:.1f}×)', 
                    fontweight='bold')
        ax.grid(axis='x', alpha=0.3)
        
        # Add stats
        textstr = f"$\|G_AcA\|$: {row['true_interaction_norm']:.2f}\n"
        textstr += f"$\lambda_\min$: {row['lambda_min_AA']:.2f}\n"
        textstr += f"|A|: {row['s']}"
        ax.text(0.99, 1.18, textstr, transform=ax.transAxes,
               verticalalignment='top', horizontalalignment='right', fontsize=9,
               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
        
        # Set reasonable x-axis limits
        max_val = max(v for v in values if np.isfinite(v) and v < 1e6)
        ax.set_xlim([0, max_val * 1.2])
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'many_design_bars.png'), dpi=300, bbox_inches='tight')
    logger.info(f"  Saved: many_design_bars.png")
    
    # Sort by tightness
    df_sorted = df.sort_values('upper_ratio')
    
    logger.info(f"\n{'Design':<25} | {'H(A)':<8} | {'Upper/H':<8} | {'Weyl/H':<8} | {'||G_AcA||':<10} | {'Leverage Bd':<12}")
    logger.info("-"*95)
    for _, row in df_sorted.iterrows():
        logger.info(f"{row['design']:<25} | {row['exact_hoffman']:<8.2f} | {row['upper_ratio']:<8.2f} | "
                   f"{row['weyl_ratio']:<8.2f} | {row['true_interaction_norm']:<10.2f} | {row['interaction_norm_bound']:<12.2f}")
    
    logger.info(f"\nTightest bounds: {df_sorted.iloc[0]['design']} (ratio={df_sorted.iloc[0]['upper_ratio']:.2f}×)")
    logger.info(f"Loosest bounds:  {df_sorted.iloc[-1]['design']} (ratio={df_sorted.iloc[-1]['upper_ratio']:.2f}×)")
    logger.info(f"Average tightness: {df['upper_ratio'].mean():.2f}×")
    logger.info(f"Median tightness:  {df['upper_ratio'].median():.2f}×")
    
    # Check leverage-based bound validity
    logger.info(f'\nLeverage-Based Interaction Bound Analysis:')
    n_valid = (df['interaction_norm_bound'] >= df['true_interaction_norm']).sum()
    logger.info(f'  Designs where leverage bound >= spectral norm: {n_valid}/{len(df)}')
    avg_slack = (df['interaction_norm_bound'] / df['true_interaction_norm']).mean()
    logger.info(f'  Average slack (leverage/spectral): {avg_slack:.3f}×')
    
    # # Show decomposition for leverage bound
    # logger.info(f'\nBound-Factor Decomposition:')
    # for design in scenarios.keys():
    #     if design in df['design'].values:
    #         row = df[df['design'] == design].iloc[0]
    #         logger.info(f'{design}:')
    #         logger.info(f"    True ||G_AcA||_2:            {row['true_interaction_norm']:.2f}")
    #         logger.info(f"    Leverage bound:              {row['interaction_norm_bound']:.2f}")
    #         logger.info(f"    Subspace Alignment (Σh_j):   {row['inactive_leverage_sum']:.2f}")
    #         logger.info(f"    Active Amplification (λ_max): {row['active_amplification']:.2f}")
    #         logger.info(f"    Slack ratio:                 {row['interaction_norm_bound'] / row['true_interaction_norm']:.3f}×")
    
    return df


def part_4_dimension_scaling():
    """
    Test how bounds scale with ambient dimension d (curse of dimensionality).
    Fixed n, s; varying d tests: How does H(A) degrade as we add features?
    """
    logger.info('\n' + '='*80)
    logger.info('Part 4: Dimension Scaling (fixed n, varying d)')
    logger.info('='*80)
    
    n = 200  # Fixed sample size
    s = 10   # Fixed sparsity
    lambda_val = 0.1
    d_values = [200, 400, 800, 1600, 3200]
    
    designs_to_test = [
        ('Gaussian', lambda n, d, s: DesignFactory.gaussian(n, d, s)),
        ('Rademacher', lambda n, d, s: DesignFactory.rademacher(n, d, s)),
        ('Partial DCT', lambda n, d, s: DesignFactory.partial_dct(n, d, s)),
        ('Spiked ρ=0.3', lambda n, d, s: DesignFactory.spiked(n, d, s, rho=0.3)),
        ('Spiked ρ=0.7', lambda n, d, s: DesignFactory.spiked(n, d, s, rho=0.7)),
        ('Spiked ρ=0.9', lambda n, d, s: DesignFactory.spiked(n, d, s, rho=0.9)),
    ]
    
    all_results = []
    
    for design_name, design_func in designs_to_test:
        for d in d_values:
            design = design_func(n, d, s)
            
            active_set = design.true_support
            result = compute_bounds_for_design(
                design_name=design_name,
                design=design,
                lambda_val=lambda_val,
                active_set=active_set
            )
            all_results.append(result)
    
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(DATA_DIR, 'dimension_scaling.csv'), index=False)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    design_types = df['design'].unique()
    colors = {'Gaussian': 'C0', 'Rademacher': 'C1', 'Partial DCT': 'C2', 
              'Spiked ρ=0.3': 'C3', 'Spiked ρ=0.7': 'C4', 'Spiked ρ=0.9': 'C5'}
    markers = {'Gaussian': 'o', 'Rademacher': 's', 'Partial DCT': '^',
               'Spiked ρ=0.3': 'D', 'Spiked ρ=0.7': 'v', 'Spiked ρ=0.9': 'X'}
    
    # Plot 1: Exact Hoffman vs dimension
    ax = axes[0, 0]
    for design_type in design_types:
        data = df[df['design'] == design_type]
        ax.plot(data['d'], data['exact_hoffman'], 
                marker=markers[design_type], color=colors[design_type],
                linewidth=2, markersize=8, label=design_type, alpha=0.8)
    ax.set_xlabel('Ambient Dimension d')
    ax.set_ylabel('Exact Hoffman H(A)')
    ax.set_title(f'Hoffman Constant vs Dimension (n={n}, s={s})')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Upper/Lower ratio
    ax = axes[0, 1]
    for design_type in design_types:
        data = df[df['design'] == design_type]
        ax.plot(data['d'], data['upper_ratio'], 
                marker=markers[design_type], color=colors[design_type],
                linewidth=2, markersize=8, label=design_type, alpha=0.8)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Ambient Dimension d')
    ax.set_ylabel('Upper Bound / Exact')
    ax.set_title('Upper Bound Tightness vs Dimension')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Interaction norm scaling
    ax = axes[1, 0]
    for design_type in design_types:
        data = df[df['design'] == design_type]
        ax.plot(data['d'], data['true_interaction_norm'], 
                marker=markers[design_type], color=colors[design_type],
                linewidth=2, markersize=8, label=design_type, alpha=0.8)
    ax.set_xlabel('Ambient Dimension d')
    ax.set_ylabel(r'$\|G_{A^c A}\|_2$')
    ax.set_title('Interaction Norm vs Dimension')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Condition number scaling
    ax = axes[1, 1]
    for design_type in design_types:
        data = df[df['design'] == design_type]
        ax.plot(data['d'], data['condition_number_AA'], 
                marker=markers[design_type], color=colors[design_type],
                linewidth=2, markersize=8, label=design_type, alpha=0.8)
    ax.set_xlabel('Ambient Dimension d')
    ax.set_ylabel(r'$\kappa(G_{AA})$')
    ax.set_title('Condition Number vs Dimension')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'dimension_scaling.png'), dpi=300, bbox_inches='tight')
    logger.info(f'  Saved figure: dimension_scaling.png')
    
    return df


def part_5_sparsity_scaling():
    """Test how bounds scale with active set size (sparsity)."""
    logger.info('\n' + '='*80)
    logger.info('Part 5: Sparsity Scaling')
    logger.info('='*80)
    
    n, d = 200, 400
    lambda_val = 0.1
    sparsity_values = [5, 10, 20, 30, 40, 50, 100, 200]
    
    designs_to_test = [
        ('Gaussian', lambda n, d, s: DesignFactory.gaussian(n, d, s)),
        ('Rademacher', lambda n, d, s: DesignFactory.rademacher(n, d, s)),
        ('Partial DCT', lambda n, d, s: DesignFactory.partial_dct(n, d, s)),
        ('Spiked ρ=0.3', lambda n, d, s: DesignFactory.spiked(n, d, s, rho=0.3)),
        ('Spiked ρ=0.7', lambda n, d, s: DesignFactory.spiked(n, d, s, rho=0.7)),
        ('Spiked ρ=0.9', lambda n, d, s: DesignFactory.spiked(n, d, s, rho=0.9)),
    ]
    
    all_results = []
    
    for design_name, design_func in designs_to_test:
        for s in sparsity_values:
            if s > n:
                continue  # Skip if sparsity > n
            
            design = design_func(n, d, s)
            
            active_set = design.true_support
            
            result = compute_bounds_for_design(
                design_name=design_name,
                design=design,
                lambda_val=lambda_val,
                active_set=active_set
            )
            all_results.append(result)
    
    df = pd.DataFrame(all_results)
    df.to_csv(os.path.join(DATA_DIR, 'sparsity_scaling.csv'), index=False)
    
    # Plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    design_types = df['design'].unique()
    colors = {'Gaussian': 'C0', 'Rademacher': 'C1', 'Partial DCT': 'C2',
              'Spiked ρ=0.3': 'C3', 'Spiked ρ=0.7': 'C4', 'Spiked ρ=0.9': 'C5'}
    markers = {'Gaussian': 'o', 'Rademacher': 's', 'Partial DCT': '^',
               'Spiked ρ=0.3': 'D', 'Spiked ρ=0.7': 'v', 'Spiked ρ=0.9': 'X'}
    
    # Plot 1: Exact Hoffman vs sparsity
    ax = axes[0, 0]
    for design_type in design_types:
        data = df[df['design'] == design_type]
        ax.plot(data['s'], data['exact_hoffman'], 
                marker=markers[design_type], color=colors[design_type],
                linewidth=2, markersize=8, label=design_type, alpha=0.8)
    ax.set_xlabel('Sparsity s')
    ax.set_ylabel('Exact Hoffman H(A)')
    ax.set_title('Hoffman Constant vs Sparsity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 2: Upper bound ratio
    ax = axes[0, 1]
    for design_type in design_types:
        data = df[df['design'] == design_type]
        ax.plot(data['s'], data['upper_ratio'], 
                marker=markers[design_type], color=colors[design_type],
                linewidth=2, markersize=8, label=design_type, alpha=0.8)
    ax.axhline(y=1.0, color='k', linestyle='--', alpha=0.5)
    ax.set_xlabel('Sparsity s')
    ax.set_ylabel('Upper Bound / Exact')
    ax.set_title('Upper Bound Tightness vs Sparsity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_yscale('log')
    
    # Plot 3: Interaction norm
    ax = axes[1, 0]
    for design_type in design_types:
        data = df[df['design'] == design_type]
        ax.plot(data['s'], data['true_interaction_norm'], 
                marker=markers[design_type], color=colors[design_type],
                linewidth=2, markersize=8, label=design_type, alpha=0.8)
    ax.set_xlabel('Sparsity s')
    ax.set_ylabel(r'$\|G_{A^c A}\|_2$')
    ax.set_title('Interaction Norm vs Sparsity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Max leverage score
    ax = axes[1, 1]
    for design_type in design_types:
        data = df[df['design'] == design_type]
        ax.plot(data['s'], data['max_leverage'], 
                marker=markers[design_type], color=colors[design_type],
                linewidth=2, markersize=8, label=design_type, alpha=0.8)
    ax.set_xlabel('Sparsity s')
    ax.set_ylabel('Max Leverage Score')
    ax.set_title('Max Inactive Leverage vs Sparsity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'sparsity_scaling.png'), dpi=300, bbox_inches='tight')
    logger.info(f'  Saved figure: sparsity_scaling.png')
    
    return df


if __name__ == '__main__':
    logger.info('='*80)
    logger.info('Hoffman Bound Tightness Analysis')
    logger.info('='*80)
    
    # Run all experiments
    df1 = part_1_spiked_model_varying_rho()
    df3 = part_2_leverage_bound_tightness()
    df5 = part_3_many_design_comparison()
    df2 = part_4_dimension_scaling()
    df4 = part_5_sparsity_scaling()
    
    logger.info('\n' + '='*80)
    logger.info(f'All experiments completed. Results saved to: {OUTPUT_DIR}')
    logger.info('='*80)
