"""
Experiment 8: Exact Support Recovery Phase Transition

This experiment maps the "Feasible Region" D in the (λ₁, λ₂) plane.
We test the hypothesis that support recovery is primarily determined by λ₁,
bounded by the Noise Floor (τ) and Signal Ceiling (β_min).

We perform a dense grid search to visualize the "Phase Transition" boundaries.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse

from hoffman.designs.design_factory import DesignFactory
from hoffman.solvers import ISTAElasticNetSolver
from hoffman.utils.viz import set_plotting_style

set_plotting_style()

def generate_sparse_problem(n: int, d: int, s: int, beta_min: float, beta_max: float, 
                            sigma: float, seed: int = 0) -> dict:
    np.random.seed(seed)
    design = DesignFactory.gaussian(n=n, d=d, s=s)
    
    beta_true = design.true_beta.copy()
    support = design.true_support
    
    # Uniform signal magnitude in [beta_min, beta_max]
    current_magnitudes = np.abs(beta_true[support])
    target_magnitudes = np.random.uniform(beta_min, beta_max, size=s)
    beta_true[support] = np.sign(beta_true[support]) * target_magnitudes
    
    y_clean = design.A @ beta_true
    noise = np.random.normal(0, sigma, size=n)
    y = y_clean + noise
    
    # Calculate Theoretical Noise Floor (Dual Norm supremum)
    # This is the exact value that λ₁ must exceed to kill all noise variables at start
    noise_floor_theoretical = np.max(np.abs(design.A.T @ noise)) / n
    
    return {
        'design': design,
        'y': y,
        'beta_true': beta_true,
        'support': support,
        'noise_floor': noise_floor_theoretical,
        'sigma': sigma
    }

def solve_and_check_recovery(problem, lam1, lam2, tol=1e-6):
    solver = ISTAElasticNetSolver(
        problem['design'], lam1=lam1, lam2=lam2, 
        y=problem['y'], beta_hat=problem['beta_true']
    )
    # Solve to high precision to ensure support is stable
    progress = solver.solve(n_iters=5000)
    beta_est = progress.history[-1].beta
    
    # Support check
    active_idx = np.where(np.abs(beta_est) > tol)[0]
    true_idx = problem['support']
    
    # Metrics
    TP = len(set(active_idx) & set(true_idx))
    FP = len(set(active_idx) - set(true_idx))
    FN = len(set(true_idx) - set(active_idx))
    
    # Strict Exact Recovery
    exact_recovery = (FP == 0) and (FN == 0)
    
    # Conditional Bias (only if we recovered at least one signal)
    if TP > 0:
        # Get values of true positives
        mask = np.isin(np.arange(len(beta_est)), list(set(active_idx) & set(true_idx)))
        bias = np.mean(np.abs(beta_est[mask] - problem['beta_true'][mask]) / np.abs(problem['beta_true'][mask]))
    else:
        bias = 1.0 # Max bias if signal lost
        
    return {
        'lambda1': lam1,
        'lambda2': lam2,
        'exact_recovery': int(exact_recovery),
        'TP': TP,
        'FP': FP,
        'FN': FN,
        'conditional_bias': bias,
        'iterations': len(progress.history) # Proxy for stability cost
    }


def compute_theoretical_bounds(problem, tau):
    """Compute theoretical boundary curves from the Elastic Net Hoffman constant analysis.

    Returns the geometric balance line (λ₁ = λ₂ + ‖G_{A^c A}‖) and the statistical
    coupling curve λ_stat(λ₂) derived in the Remark on coupled regularization.
    """
    design = problem['design']
    A = design.A
    n, d = A.shape
    support = problem['support']
    inactive = np.setdiff1d(np.arange(d), support)
    beta_true = problem['beta_true']
    y = problem['y']

    G = A.T @ A / n
    G_AA = G[np.ix_(support, support)]
    G_AcA = G[np.ix_(inactive, support)]
    beta_A = beta_true[support]
    epsilon = y - A @ beta_true  # approximate noise

    # ‖G_{A^c A}‖_2  (spectral norm of cross-interaction block)
    G_cross_norm = np.linalg.norm(G_AcA, ord=2)

    # λ_min(G_AA)  (for the tighter Hoffman bound)
    G_AA_min_eig = float(np.clip(np.linalg.eigvalsh(G_AA).min(), 0, None))

    # Noise floor restricted to inactive features
    inactive_noise = float(np.max(np.abs(A[:, inactive].T @ epsilon / n)))

    # Statistical coupling curve  (Remark on coupled regularization terms)
    # λ_stat(λ₂) ≳ ‖A_{A^c}^T ε/n‖_∞  +  λ₂ · ‖G_{A^c A}(G_AA + λ₂ I)^{-1} β_A^*‖_∞
    l2_curve = np.logspace(-3, 1, 300)
    stat_l1 = []
    for l2 in l2_curve:
        reg_inv = np.linalg.inv(G_AA + l2 * np.eye(len(support)))
        coupling = G_AcA @ reg_inv @ beta_A
        stat_l1.append(inactive_noise + l2 * float(np.max(np.abs(coupling))))

    print(f"  ‖G_{{A^c A}}‖_2          = {G_cross_norm:.4f}")
    print(f"  λ_min(G_AA)             = {G_AA_min_eig:.4f}")
    print(f"  Inactive noise floor    = {inactive_noise:.4f}  (τ = {tau:.4f})")

    return {
        'G_cross_norm': G_cross_norm,
        'G_AA_min_eig': G_AA_min_eig,
        'stat_l2_abs': l2_curve,
        'stat_l1_abs': np.array(stat_l1),
    }


def compute_optimal_paths(theo_bounds, tau, beta_min):
    """For a sweep of total budgets Λ = λ₁ + λ₂, compute the optimal split under:

    - Closed-form (Corollary/Theorem): λ_stat = τ is fixed (coupling ignored).
      λ₁^unc is the piecewise closed-form; λ₁* = max(λ₁^unc, τ).

    - Coupled/numerical (Remark): λ_stat(λ₂) grows with λ₂ due to shrinkage
      bias leaking into inactive variables. Optimal λ₂ is found by scanning
      feasible candidates and evaluating the Hoffman upper bound.

    Returns ratio-normalised arrays (divided by τ) for direct axis labelling.
    """
    G_cross = theo_bounds['G_cross_norm']
    stat_l2 = theo_bounds['stat_l2_abs']
    stat_l1 = theo_bounds['stat_l1_abs']
    lambda_stat_fixed = tau  # uncoupled (closed-form) statistical threshold

    # Total budget sweep that covers the grid range
    Lambda_grid = np.linspace(tau * 1.05, beta_min * 2.5, 500)

    cf_l1, cf_l2, cf_H, cf_Lambda = [], [], [], []
    cp_l1, cp_l2, cp_H, cp_Lambda = [], [], [], []

    for Lambda in Lambda_grid:
        # ── Closed-form optimal (Corollary / Theorem) ────────────────────────
        if G_cross <= Lambda / 3:
            l1_unc = (Lambda + G_cross) / 2.0
        else:
            disc = G_cross * (Lambda + G_cross)
            l1_unc = Lambda + G_cross - np.sqrt(max(disc, 0.0))
        l1_cf = max(l1_unc, lambda_stat_fixed)
        l2_cf = Lambda - l1_cf
        if l2_cf > 1e-9 and l1_cf > 1e-9:
            H = max(1.0 / l2_cf, (1.0 / l1_cf) * (1.0 + G_cross / l2_cf))
            cf_l1.append(l1_cf); cf_l2.append(l2_cf)
            cf_H.append(H);      cf_Lambda.append(Lambda)

        # ── Coupled / numerical optimal (Remark) ────────────────────────────
        # Scan λ₂ ∈ (0, Λ) in log space; check stat feasibility; minimise H bound.
        l2_scan = np.logspace(np.log10(max(1e-4, Lambda * 0.002)),
                              np.log10(Lambda * 0.998), 600)
        best_H, best_l1, best_l2 = np.inf, None, None
        for l2_try in l2_scan:
            l1_try = Lambda - l2_try
            if l1_try <= 1e-9:
                continue
            # Interpolate the coupled statistical threshold λ_stat(λ₂)
            thresh = float(np.interp(l2_try, stat_l2, stat_l1))
            if l1_try < thresh:
                continue  # strict dual feasibility violated
            H = max(1.0 / l2_try, (1.0 / l1_try) * (1.0 + G_cross / l2_try))
            if H < best_H:
                best_H = H; best_l1 = l1_try; best_l2 = l2_try
        if best_l1 is not None:
            cp_l1.append(best_l1); cp_l2.append(best_l2)
            cp_H.append(best_H);   cp_Lambda.append(Lambda)

    print(f"  Closed-form path: {len(cf_l1)} feasible budget levels")
    print(f"  Coupled path:     {len(cp_l1)} feasible budget levels")

    return {
        'cf_l1': np.array(cf_l1), 'cf_l2': np.array(cf_l2),
        'cf_H': np.array(cf_H),   'cf_Lambda': np.array(cf_Lambda),
        'cp_l1': np.array(cp_l1), 'cp_l2': np.array(cp_l2),
        'cp_H': np.array(cp_H),   'cp_Lambda': np.array(cp_Lambda),
    }


def experiment_phase_transition_boundary():
    print("\n" + "="*80)
    print("Experiment 7: Exact Support Recovery Phase Transition")
    print("="*80)
    
    # Setup: Medium SNR (Clear boundaries expected)
    n, d, s = 200, 100, 10
    sigma = 0.2
    beta_min = 1.5
    beta_max = 2.0
    
    problem = generate_sparse_problem(n, d, s, beta_min, beta_max, sigma)
    
    tau = problem['noise_floor']
    print(f"Problem Parameters:")
    print(f"  Signal Range: [{beta_min}, {beta_max}]")
    print(f"  Noise Floor (τ): {tau:.4f}")

    print("\nComputing theoretical boundaries...")
    theo_bounds = compute_theoretical_bounds(problem, tau)

    # Define Grid in absolute units; ratios are stored in results for plotting
    # λ₁: span [0.5τ, 1.5 β_min] to reveal both failure modes
    l1_min = tau * 0.5
    l1_max = beta_min * 1.5
    l1_grid = np.linspace(l1_min, l1_max, 40)

    # λ₂: logspace so the low-regularisation regime is well-resolved
    l2_grid = np.logspace(-3, 1, 30)  # 0.001 to 10.0

    results = []
    total = len(l1_grid) * len(l2_grid)
    idx = 0

    print(f"\nScanning {total} grid points...")

    for l2 in l2_grid:
        for l1 in l1_grid:
            idx += 1
            if idx % 100 == 0:
                print(f"  Progress: {idx}/{total}...")

            res = solve_and_check_recovery(problem, l1, l2)
            # Store normalised ratios (noise-floor units) for axis-independent plotting
            res['lambda1_ratio'] = l1 / tau
            res['lambda2_ratio'] = l2 / tau
            results.append(res)

    df = pd.DataFrame(results)

    # Save
    output_dir = Path('results/07_phase_transition')
    output_dir.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_dir / 'phase_transition_data.csv', index=False)

    print("\nComputing closed-form vs coupled optimal paths...")
    opt_paths = compute_optimal_paths(theo_bounds, tau, beta_min)

    # ── Persist all computed artefacts so plotting can run independently ──────
    output_dir = Path('results/07_phase_transition')
    output_dir.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_dir / 'computed_bounds.npz',
        tau=tau,
        beta_min=beta_min,
        G_cross_norm=theo_bounds['G_cross_norm'],
        G_AA_min_eig=theo_bounds['G_AA_min_eig'],
        stat_l2_abs=theo_bounds['stat_l2_abs'],
        stat_l1_abs=theo_bounds['stat_l1_abs'],
        cf_l1=opt_paths['cf_l1'], cf_l2=opt_paths['cf_l2'],
        cf_H=opt_paths['cf_H'],   cf_Lambda=opt_paths['cf_Lambda'],
        cp_l1=opt_paths['cp_l1'], cp_l2=opt_paths['cp_l2'],
        cp_H=opt_paths['cp_H'],   cp_Lambda=opt_paths['cp_Lambda'],
    )
    print(f"Bounds saved to {output_dir / 'computed_bounds.npz'}")

    # Visualization
    visualize_phase_transition(df, tau, beta_min, output_dir, theo_bounds, opt_paths)

def visualize_phase_transition(df, tau, beta_min, output_dir, theoretical_bounds, optimal_paths):
    """Three-panel figure:
      A – Support recovery heatmap (λ/τ axes) + both optimal paths + theory curves.
      B – Conditional bias heatmap + same overlays.
      C – Hoffman upper-bound comparison: closed-form vs coupled, as Λ/τ varies.
    """
    sns.set_style("white")

    # ── Pivot on ratio columns so axes carry physical meaning ──────────────────
    pivot_recovery = df.pivot_table(
        index='lambda2_ratio', columns='lambda1_ratio', values='exact_recovery'
    )
    pivot_bias = df.pivot_table(
        index='lambda2_ratio', columns='lambda1_ratio', values='conditional_bias'
    )

    l1_ratios = np.array(sorted(df['lambda1_ratio'].unique()))
    l2_ratios = np.array(sorted(df['lambda2_ratio'].unique()))

    # ── Cell edges for pcolormesh ───────────────────────────────────────────────
    # λ₁ grid is linear → linear midpoints
    dl1 = np.diff(l1_ratios)
    l1_edges = np.concatenate([
        [l1_ratios[0] - dl1[0] / 2],
        l1_ratios[:-1] + dl1 / 2,
        [l1_ratios[-1] + dl1[-1] / 2]
    ])
    # λ₂ grid is log-spaced → geometric midpoints in log space
    log_l2 = np.log(l2_ratios)
    dlog2 = np.diff(log_l2)
    l2_edges = np.exp(np.concatenate([
        [log_l2[0] - dlog2[0] / 2],
        log_l2[:-1] + dlog2 / 2,
        [log_l2[-1] + dlog2[-1] / 2]
    ]))

    X, Y = np.meshgrid(l1_edges, l2_edges)

    # ── Theoretical quantities (in τ-normalised units) ─────────────────────────
    G_cross = theoretical_bounds['G_cross_norm']
    stat_l2_ratio = theoretical_bounds['stat_l2_abs'] / tau
    stat_l1_ratio = theoretical_bounds['stat_l1_abs'] / tau

    beta_min_ratio = beta_min / tau
    l1_lo, l1_hi = l1_ratios.min(), l1_ratios.max()
    l2_lo, l2_hi = l2_ratios.min(), l2_ratios.max()

    # Geometric balance line:  λ₁ = λ₂ + ‖G_{A^c A}‖  (Corollary / Theorem)
    l1_geo = l2_ratios + G_cross / tau
    geo_mask = l1_geo <= l1_hi * 1.05

    # Statistical coupling curve  (Remark)
    stat_mask = (
        (stat_l2_ratio >= l2_lo * 0.8)
        & (stat_l2_ratio <= l2_hi * 1.25)
        & (stat_l1_ratio <= l1_hi * 1.05)
    )

    # ── Optimal paths in ratio units ───────────────────────────────────────────
    cf_l1r = optimal_paths['cf_l1'] / tau
    cf_l2r = optimal_paths['cf_l2'] / tau
    cf_Hr  = optimal_paths['cf_H']
    cf_Lr  = optimal_paths['cf_Lambda'] / tau
    cp_l1r = optimal_paths['cp_l1'] / tau
    cp_l2r = optimal_paths['cp_l2'] / tau
    cp_Hr  = optimal_paths['cp_H']
    cp_Lr  = optimal_paths['cp_Lambda'] / tau

    fig, axes = plt.subplots(1, 3, figsize=(30, 8))

    # ── Shared helper: overlay theory curves and optimal paths on a heatmap ax ─
    def _overlay_theory(ax, include_bias_budget=False):
        ax.set_yscale('log')
        ax.axvline(1.0, color='royalblue', linestyle='--', linewidth=2,
                   label=r'Noise Floor  $(\lambda_1/\tau=1)$')
        ax.axvline(beta_min_ratio, color='firebrick', linestyle='--', linewidth=2,
                   label=r'Signal Ceiling  $(\lambda_1=\beta_{\min})$')
        if geo_mask.any():
            ax.plot(l1_geo[geo_mask], l2_ratios[geo_mask],
                    color='deepskyblue', linestyle='-', linewidth=1.5,
                    label=r'Geom. Balance: $\lambda_1=\lambda_2+\|G_{\mathcal{A}^c\mathcal{A}}\|$')
        if stat_mask.any():
            ax.plot(stat_l1_ratio[stat_mask], stat_l2_ratio[stat_mask],
                    color='darkorange', linestyle='-', linewidth=1.5,
                    label=r'Coupled $\lambda_{\mathrm{stat}}(\lambda_2)$ boundary')
        # Optimal-path loci — clipped at signal ceiling; different color AND style
        path_l1_mask = (cf_l1r >= l1_lo) & (cf_l1r <= beta_min_ratio) & (cf_l2r >= l2_lo) & (cf_l2r <= l2_hi)
        if path_l1_mask.any():
            ax.plot(cf_l1r[path_l1_mask], cf_l2r[path_l1_mask],
                    color='lime', linestyle='-', linewidth=3, marker='',
                    label=r'Closed-form opt. path $(\lambda_1^*,\lambda_2^*)$')
        path_cp_mask = (cp_l1r >= l1_lo) & (cp_l1r <= beta_min_ratio) & (cp_l2r >= l2_lo) & (cp_l2r <= l2_hi)
        if path_cp_mask.any():
            ax.plot(cp_l1r[path_cp_mask], cp_l2r[path_cp_mask],
                    color='magenta', linestyle='--', linewidth=3, marker='',
                    label=r'Coupled opt. path $(\lambda_1^*,\lambda_2^*)$')
        if include_bias_budget:
            delta = 0.1
            ax.axhline(delta / tau, color='limegreen', linestyle='-.', linewidth=1.5,
                       label=rf'Bias Budget $(\delta={delta})$')
        ax.set_xlim(l1_lo * 0.95, l1_hi * 1.05)
        ax.set_ylim(l2_lo * 0.8, l2_hi * 1.25)
        ax.set_xlabel(r'$\lambda_1 / \tau$')
        ax.set_ylabel(r'$\lambda_2 / \tau$  (log scale)')
        ax.legend(loc='lower right', framealpha=0.9)

    # ── Plot A: Support Recovery ───────────────────────────────────────────────
    ax = axes[0]
    pc = ax.pcolormesh(X, Y, pivot_recovery.values,
                       cmap='Greys_r', vmin=0, vmax=1, shading='auto')
    plt.colorbar(pc, ax=ax, label='Exact Recovery  (white = yes,  black = no)')
    _overlay_theory(ax)
    ax.set_title('Exact Support Recovery', fontweight='bold')

    # ── Plot B: Conditional Bias ───────────────────────────────────────────────
    ax = axes[1]
    pc2 = ax.pcolormesh(X, Y, pivot_bias.values,
                        cmap='YlOrRd', vmin=0, vmax=0.5, shading='auto')
    plt.colorbar(pc2, ax=ax, label='Conditional Bias')
    _overlay_theory(ax, include_bias_budget=False)
    ax.set_title('Conditional Bias Landscape', fontweight='bold')

    # ── Plot C: Hoffman Bound Comparison ───────────────────────────────────────
    ax = axes[2]

    # Reference: unconstrained geometric optimum 2/Λ (no stat constraint, no coupling)
    Lambda_ref = np.linspace(cf_Lr.min() if len(cf_Lr) else 1.0,
                             max(cf_Lr.max() if len(cf_Lr) else 5.0,
                                 cp_Lr.max() if len(cp_Lr) else 5.0) * 1.05, 400)
    H_unconstrained = 2.0 / Lambda_ref  # equipartition bound (Corollary remark H*=2/Λ)

    ax.plot(Lambda_ref, H_unconstrained, color='gray', linestyle=':', linewidth=1.5,
            label=r'Unconstrained geometric optimum  $H^* = 2/\Lambda$')

    if len(cf_Lr):
        ax.plot(cf_Lr, cf_Hr, color='lime', linestyle='-', linewidth=2.5,
                label=r'Closed-form opt. $H_{\mathrm{EN}}$ (fixed $\lambda_{\mathrm{stat}}=\tau$)')

    if len(cp_Lr):
        ax.plot(cp_Lr, cp_Hr, color='magenta', linestyle='--', linewidth=2.5,
                label=r'Coupled opt. $H_{\mathrm{EN}}$ ($\lambda_{\mathrm{stat}}$ depends on $\lambda_2$)')

    # Mark where the constraint binds for closed-form (where l1* = τ, not l1_unc)
    cf_constrained = np.abs(optimal_paths['cf_l1'] - tau) < tau * 0.05
    if cf_constrained.any():
        ax.axvline(cf_Lr[cf_constrained][0], color='royalblue', linestyle='--', linewidth=1,
                   label=r'Budget where stat. constraint binds ($\lambda_1^*=\tau$)')

    # Gap annotation: shade region where coupled is worse
    if len(cf_Lr) and len(cp_Lr):
        # Interpolate coupled H onto closed-form Λ grid for direct comparison
        cp_H_interp = np.interp(cf_Lr, cp_Lr, cp_Hr,
                                left=np.nan, right=np.nan)
        valid = ~np.isnan(cp_H_interp)
        if valid.any():
            ax.fill_between(cf_Lr[valid], cf_Hr[valid], cp_H_interp[valid],
                            where=cp_H_interp[valid] > cf_Hr[valid],
                            alpha=0.18, color='red', label='Coupling penalty (gap)')

    ax.set_yscale('log')
    ax.set_xlabel(r'Total budget $\Lambda / \tau$')
    ax.set_ylabel(r'Hoffman upper bound  $H_{\mathrm{EN}}(\lambda_1^*, \lambda_2^*)$')
    ax.set_title('Hoffman Bound: Closed-Form vs Coupled Optimum\n', fontweight='bold')
    ax.legend(framealpha=0.9)
    ax.grid(True, which='both', linestyle=':', alpha=0.4)

    plt.tight_layout()
    plt.savefig(output_dir / 'phase_transition_map.png', dpi=300)
    print(f"Plot saved to {output_dir / 'phase_transition_map.png'}")


def load_saved_data(output_dir: Path):
    """Load all artefacts produced by experiment_phase_transition_boundary()."""
    csv_path = output_dir / 'phase_transition_data.csv'
    npz_path = output_dir / 'computed_bounds.npz'
    if not csv_path.exists():
        raise FileNotFoundError(f"Grid data not found: {csv_path}\nRun with --mode run first.")
    if not npz_path.exists():
        raise FileNotFoundError(f"Bounds data not found: {npz_path}\nRun with --mode run first.")

    df = pd.read_csv(csv_path)
    d = np.load(npz_path)

    tau      = float(d['tau'])
    beta_min = float(d['beta_min'])
    theo_bounds = {
        'G_cross_norm': float(d['G_cross_norm']),
        'G_AA_min_eig': float(d['G_AA_min_eig']),
        'stat_l2_abs':  d['stat_l2_abs'],
        'stat_l1_abs':  d['stat_l1_abs'],
    }
    opt_paths = {
        'cf_l1': d['cf_l1'], 'cf_l2': d['cf_l2'],
        'cf_H':  d['cf_H'],  'cf_Lambda': d['cf_Lambda'],
        'cp_l1': d['cp_l1'], 'cp_l2': d['cp_l2'],
        'cp_H':  d['cp_H'],  'cp_Lambda': d['cp_Lambda'],
    }
    print(f"Loaded {len(df)} grid points  (tau={tau:.4f}, beta_min={beta_min})")
    return df, tau, beta_min, theo_bounds, opt_paths


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Elastic Net Phase Transition Experiment")
    parser.add_argument(
        '--mode', choices=['run', 'plot', 'both'], default='both',
        help="'run' = grid scan + save data only;  "
             "'plot' = load saved data and plot only;  "
             "'both' = run then plot (default)"
    )
    parser.add_argument(
        '--output-dir', default='results/07_phase_transition',
        help='Directory for saved data and figures (default: results/07_phase_transition)'
    )
    args = parser.parse_args()
    output_dir = Path(args.output_dir)

    if args.mode in ('run', 'both'):
        experiment_phase_transition_boundary()
    else:
        # Plot-only: load saved artefacts and re-render the figure
        df, tau, beta_min, theo_bounds, opt_paths = load_saved_data(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        visualize_phase_transition(df, tau, beta_min, output_dir, theo_bounds, opt_paths)