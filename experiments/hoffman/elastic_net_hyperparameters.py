"""
Sweeps (lambda_1, lambda_2) hyperparameter grid for elastic net.
Plots exact recovery (binary map) and Hoffman constant map for grid.
The theoretical paths (naive, constrained, coupled) are also plotted with the noise floor and signal ceiling.
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import argparse
from tqdm import tqdm
import logging

from hoffman.designs.design_factory import DesignFactory
from hoffman.solvers import ISTAElasticNetSolver
from hoffman.bounds.hoffman_bounds import HoffmanBoundCalculator
from hoffman.utils.viz import set_plotting_style

set_plotting_style()

# Output directories
OUTPUT_DIR = Path('./results/hoffman/elastic_net_hyperparameters/')
DATA_DIR = OUTPUT_DIR / 'data'
DATA_DIR.mkdir(parents=True, exist_ok=True)

# Configuration
np.random.seed(0)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_l1_unc(Lambda_n, G_cross_norm):
    """Unconstrained lambda 1 split based on geometric balance."""
    if G_cross_norm <= Lambda_n / 3:
        return (Lambda_n + G_cross_norm) / 2.0
    else:
        return Lambda_n + G_cross_norm - np.sqrt(G_cross_norm * (Lambda_n + G_cross_norm))


def generate_sparse_problem(n, d, s, sigma, beta_min, beta_max):
    design = DesignFactory.gaussian(n=n, d=d, s=s)
    beta_true = design.true_beta.copy()
    support = design.true_support
    target_magnitudes = np.random.uniform(beta_min, beta_max, size=s)
    beta_true[support] = np.sign(beta_true[support]) * target_magnitudes

    noise = np.random.normal(0, sigma, size=n)
    y = design.A @ beta_true + noise
    
    inactive = np.setdiff1d(np.arange(d), support)
    # Ξ (Xi): environmental noise floor
    Xi = np.max(np.abs(design.A[:, inactive].T @ noise)) / n

    return {
        'design': design, 
        'beta_true': beta_true, 
        'support': support,
        'inactive': inactive, 
        'y': y, 
        'Xi': Xi, 
        'beta_min': beta_min
    }


def run_grid_scan(n, d, s, sigma, beta_min, beta_max, max_iters=10000):
    prob = generate_sparse_problem(n, d, s, sigma, beta_min, beta_max)
    G = prob['design'].gram
    Xi = prob['Xi']
    support = prob['support']
    inactive = prob['inactive']
    beta_A = prob['beta_true'][support]
    G_AA = G[np.ix_(support, support)]
    G_AcA = G[np.ix_(inactive, support)]
    G_cross = np.linalg.norm(G_AcA, ord=2)
    min_eig = np.linalg.eigvalsh(G_AA).min()
    true_beta_min = np.min(np.abs(beta_A))

    # Grid: Normalized by Xi
    l1_grid = np.linspace(Xi * 0.5, beta_min * 1.5, 35)
    l2_grid = np.logspace(-3, 1.5, 30)

    results = []
    logger.info(f"Scanning grid (Xi={Xi:.4f}, G_cross={G_cross:.4f}).")
    for l2 in tqdm(l2_grid):
        for l1 in l1_grid:
            logger.info(f"Solving for l1={l1:.4f}, l2={l2:.4f}...")
            solver = ISTAElasticNetSolver(prob['design'], lam1=l1, lam2=l2, y=prob['y'])
            res = solver.solve(n_iters=max_iters).history[-1]
            active_est = np.where(np.abs(res.beta) > 1e-6)[0]
            exact_rec = int(set(active_est) == set(support))
            # if exact_rec:
            #     print(f"Exact recovery at l1={l1:.4f}, l2={l2:.4f}")
            # else:
            #     print(f"Differences at l1={l1:.4f}, l2={l2:.4f}: {set(active_est) ^ set(support)}")
            
            try:
                M_A = HoffmanBoundCalculator.build_kkt_matrix(G, support, l1, l2)
                h_emp = np.linalg.norm(np.linalg.inv(M_A), ord=2)
            except: h_emp = np.nan
                
            results.append({'l1_r': l1/Xi, 'l2_r': l2/Xi, 'recovery': exact_rec, 'h_emp': h_emp, 'l1': l1, 'l2': l2})

    df = pd.DataFrame(results)
    df.to_csv(DATA_DIR / 'grid_results.csv', index=False)

    # Theoretical curves
    L_vals = np.linspace(Xi * 1.1, beta_min * 2.5, 500)
    l2_range = np.logspace(-2, 1.5, 500)
    # Uncoupled statistical boundary
    l1_stat_curve = [Xi + l2 * np.linalg.norm(G_AcA @ np.linalg.inv(G_AA + l2*np.eye(s)) @ beta_A, ord=np.inf) for l2 in l2_range]
    # Geometric Balance
    l1_geo_balance = l2_range + G_cross + min_eig

    # Paths across Budget Lambda_n
    paths = {'naive': [], 'constrained': [], 'coupled': []}
    for L in L_vals:  # Lambda budget
        l1_u = get_l1_unc(L, G_cross)
        paths['naive'].append((l1_u, L - l1_u))
        paths['constrained'].append((max(l1_u, Xi), L - max(l1_u, Xi)))
        # Coupled solved numerically
        l1_t = np.linspace(Xi, L * 0.99, 200)
        l2_t = L - l1_t
        l1_s = np.array([Xi + lt * np.linalg.norm(G_AcA @ np.linalg.inv(G_AA + lt*np.eye(s)) @ beta_A, ord=np.inf) for lt in l2_t])
        l1_c = max(l1_u, l1_t[np.argmin(np.abs(l1_t - l1_s))])
        paths['coupled'].append((l1_c, L - l1_c))

    np.savez(DATA_DIR / 'theory.npz', Xi=Xi, beta_min=beta_min, G_cross=G_cross, true_beta_min=true_beta_min,
             stat_l1=np.array(l1_stat_curve), l2_range=l2_range, geo_bal=l1_geo_balance,
             naive=np.array(paths['naive']), constrained=np.array(paths['constrained']), coupled=np.array(paths['coupled']))


def visualize():
    df = pd.read_csv(DATA_DIR / 'grid_results.csv')
    theo = np.load(DATA_DIR / 'theory.npz')
    Xi, true_beta_min = float(theo['Xi']), float(theo['true_beta_min'])
    l1_r, l2_r = sorted(df['l1_r'].unique()), sorted(df['l2_r'].unique())

    fig, axes = plt.subplots(1, 2, figsize=(22, 9))
    
    for i, ax in enumerate(axes):
        if i == 0:
            pivot = df.pivot(index='l2_r', columns='l1_r', values='recovery')
            ax.pcolormesh(l1_r, l2_r, pivot.values, cmap='Greys_r', shading='auto')
            ax.plot(theo['stat_l1']/Xi, theo['l2_range']/Xi, color='orange', linestyle='-', linewidth=2.5, label='Coupled $\lambda_{\mathrm{stat}}(\lambda_{{n, 2}})$ boundary')
        else:
            pivot = df.pivot(index='l2_r', columns='l1_r', values='h_emp')
            im = ax.pcolormesh(l1_r, l2_r, pivot.values, shading='auto', 
                               norm=plt.matplotlib.colors.LogNorm(vmin=df['h_emp'].min(), vmax=df['h_emp'].quantile(0.85)))
            plt.colorbar(im, ax=ax, label=r'Empirical $H(\mathcal{A}^*)$')

            ## Plot lambda budget contours (for testing purposes)
            # l_max = max(l1_r) + max(l2_r)
            # l_min = min(l1_r) + min(l2_r)
            # lambda_budgets = np.linspace(l_min * 0.1, l_max * 1.5, 20)
            # x_plot = np.linspace(min(l1_r), max(l1_r), 100)
            # for L in lambda_budgets:
            #     y_plot = L - x_plot
            #     valid = y_plot > min(l2_r)
            #     if np.any(valid):
            #         line, = ax.plot(x_plot[valid], y_plot[valid], color='white', 
            #                         alpha=0.3, linestyle='--', linewidth=1)
            #         # Label the contours
            #         lbl_x = x_plot[valid][len(x_plot[valid])//4]
            #         lbl_y = y_plot[valid][len(y_plot[valid])//4]
            #         ax.text(lbl_x, lbl_y, f'$\\Lambda_n/\\Xi={L:.1f}$', 
            #                 color='white', alpha=0.6, fontsize=9, rotation=-30,
            #                 ha='center', va='center')

        # Static reference lines
        ax.axvline(1.0, color='royalblue', linestyle='--', linewidth=2, label=r'Noise floor ($\Xi$)')
        ax.axvline(true_beta_min/Xi, color='firebrick', linestyle='-.', linewidth=2, label=r'Signal ceiling ($\beta_{\min}$)')
        # ax.axvline(theo['G_cross']/(Xi), color='darkgreen', linestyle='--', linewidth=2, label=r'Interaction term ($\|G_{A^c A}\|_2$)')
    
        # Regime boundary (Lambda_n = 3 * G_cross)
        c_trans = 3 * theo['G_cross'] / Xi
        x_vals = np.array(l1_r)
        y_vals = c_trans - x_vals
        ax.plot(x_vals[y_vals > 0], y_vals[y_vals > 0], color='gold', linestyle=':', linewidth=2, 
                label=r'Regime boundary ($\Lambda_n = 3\|G_{A^c A}\|_2$)')

        # Theoretical paths
        ax.plot(theo['naive'][:,0]/Xi, theo['naive'][:,1]/Xi, color='cyan', 
                linestyle=(0, (1, 1)), linewidth=2, label='Naive closed-form')
        ax.plot(theo['constrained'][:,0]/Xi, theo['constrained'][:,1]/Xi, color='lime', 
                linestyle=(0, (5, 5)), linewidth=3, label='Constrained closed-form')
        ax.plot(theo['coupled'][:,0]/Xi, theo['coupled'][:,1]/Xi, color='magenta', 
                linestyle='-', linewidth=1.5, label='Coupled')

        ax.set_yscale('log')
        ax.set_xlim(min(l1_r), max(l1_r))
        ax.set_ylim(min(l2_r), max(l2_r))
        ax.set_xlabel(r"$\lambda_{n,1} / \Xi$")
        ax.set_ylabel(r"$\lambda_{n,2} / \Xi$")
        ax.set_title(["Exact Support Recovery", "Empirical Hoffman"][i], fontweight='bold', fontsize=18)
        ax.legend(loc='lower right', framealpha=0.9, fontsize=12)

    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / 'phase_transition_map.png', dpi=300)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--mode', choices=['run', 'plot', 'both'], default='both')
    args = parser.parse_args()
    
    N, D, S = 300, 500, 5
    SIGMA, B_MIN, B_MAX = 0.05, 3.0, 5.0

    if args.mode in ['run', 'both']:
        run_grid_scan(N, D, S, SIGMA, B_MIN, B_MAX)
    if args.mode in ['plot', 'both']:
        visualize()
