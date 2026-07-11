"""
Demonstrates feature pruning using active-inactive cross-leverage.

Fits LASSO models on some datasets (e.g., ARCENE) with different
regularization strengths and computes cross-leverage scores for inactive features.
When cross-leverage score is high (e.g., h_j > 0.95), it indicates that the inactive feature is almost entirely
in the span of the active features, suggesting that it can be pruned without losing much information.

This can theoretically be used to improve the Hoffman constant (and thus stability under perturbation)
while mantaining the same solution via feature pruning as a post-hoc step after LASSO selection. We 
would also expect that the pruned model would converge faster due to the reduced dimensionality.
"""

import logging
import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import fetch_openml, make_regression
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso

from hoffman.bounds.hoffman_bounds import HoffmanBoundCalculator
from hoffman.utils.viz import set_plotting_style

# Output directories
OUTPUT_DIR = './results/hoffman/cross_leverage_diagnostic/'
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Configuration
np.random.seed(0)
set_plotting_style()
logging.basicConfig(filename=os.path.join(OUTPUT_DIR, 'cross_leverage_diagnostic.log'), level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
logger = logging.getLogger(__name__)
logger.addHandler(console_handler)


def compute_cross_leverage(A: np.ndarray, active_set: np.ndarray):
    """Computes the cross-leverage score h_j for all inactive features."""
    n, d = A.shape
    inactive_set = np.setdiff1d(np.arange(d), active_set)
    
    A_act = A[:, active_set]
    A_act_pinv = np.linalg.pinv(A_act)
    Pi_A = A_act @ A_act_pinv
    
    leverage_scores = {}
    for j in inactive_set:
        a_j = A[:, j]
        h_j = (1.0 / n) * np.linalg.norm(Pi_A @ a_j)**2
        leverage_scores[j] = h_j
        
    return leverage_scores


def get_exact_h2(A: np.ndarray, active_set: np.ndarray, lam1: float):
    n = A.shape[0]
    G = (A.T @ A) / n
    M_A = HoffmanBoundCalculator.build_kkt_matrix(G, active_set, lam1)
    try:
        return np.linalg.norm(np.linalg.inv(M_A), ord=2)
    except np.linalg.LinAlgError:
        return np.nan


def load_openml_dataset(dataset_name: str):
    if dataset_name == 'Synthetic':
        X, y = make_regression(n_samples=100, n_features=1000, n_informative=15, noise=1.0, random_state=0)
        y = StandardScaler().fit_transform(y.reshape(-1, 1)).flatten()
        return X, y

    openml_data_id_map = {
        'ARCENE': 1458,
        'DEXTER': 4136,
        'DOROTHEA': 4137,
        'LEUKEMIA': 1104,
    }

    if dataset_name not in openml_data_id_map:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    last_error = None
    for attempt in range(3):
        try:
            X, y = fetch_openml(data_id=openml_data_id_map[dataset_name], return_X_y=True, as_frame=False)
            break
        except Exception as exc:
            last_error = exc
            if attempt < 2:
                logger.warning(f"Retrying OpenML fetch for {dataset_name} (attempt {attempt + 2}/3)...")
                time.sleep(2)
    else:
        raise last_error

    is_sparse_input = hasattr(X, 'toarray')
    scaler = StandardScaler(with_mean=not is_sparse_input, with_std=True)
    X_scaled = scaler.fit_transform(X)

    if hasattr(X_scaled, 'toarray'):
        X_scaled = X_scaled.toarray()

    X_scaled = np.asarray(X_scaled, dtype=float) * np.sqrt(X_scaled.shape[0])
    y = LabelEncoder().fit_transform(y).astype(float)
    return X_scaled, y


def get_alpha_grid(dataset_name: str):
    if dataset_name == 'ARCENE':
        return [0.1, 0.5, 1.0]
    if dataset_name == 'DEXTER':
        return [1.0]
    if dataset_name == 'DOROTHEA':
        return [0.1]
    if dataset_name == 'LEUKEMIA':
        return [0.1, 0.5, 1.0]
    if dataset_name == 'Synthetic':
        return [0.01, 0.05, 0.1]
    raise ValueError(f"Unsupported dataset: {dataset_name}")


def evaluate_alpha_regime(X_scaled, y, alpha_sklearn, dataset_name):
    """Runs the baseline and pruning pipeline for a specific alpha."""
    n, d = X_scaled.shape
    lambda_val = alpha_sklearn * n
    
    logger.info(f"Alpha = {alpha_sklearn}")
    lasso = Lasso(alpha=alpha_sklearn, fit_intercept=False, max_iter=5000, tol=1e-4)
    lasso.fit(X_scaled, y)
    
    baseline_coef = lasso.coef_.copy()
    baseline_active_set = np.where(np.abs(baseline_coef) > 1e-7)[0]
    s_base = len(baseline_active_set)
    baseline_iters = int(lasso.n_iter_)
    base_norm_l2 = np.linalg.norm(baseline_coef)
    base_norm_linf = np.linalg.norm(baseline_coef, ord=np.inf)
    
    if s_base == 0:
        logger.warning("Alpha too high. All coefficients penalized to zero.")
        return None, None, None, None
        
    baseline_h2 = get_exact_h2(X_scaled, baseline_active_set, lambda_val)
    logger.info(f"Ambient Dim (d): {d} | Samples (n): {n}")
    logger.info(f"Baseline: s={s_base}, Exact H2={baseline_h2:.2f}")
    logger.info(f"Baseline Iterations to Converge: {baseline_iters}")
    logger.info(f"Baseline ||Beta||_2: {base_norm_l2:.4f} | ||Beta||_inf: {base_norm_linf:.4f}")
    
    leverage_scores = compute_cross_leverage(X_scaled, baseline_active_set)
    
    thresholds = [0.99, 0.95, 0.90, 0.80, 0.70, 0.60, 0.50, 0.40, 0.30, 0.20, 0.10]
    results = []

    logger.info(
        f"{'Tau':<5} | {'Drop':<5} | {'s':<5} | {'H2':<8} | {'Jac':<4} | "
        f"{'Diff_L2':<9} | {'Diff_Linf':<9} | {'Iters':<5} | {'Red%':<7} | {'||b||_2':<8} | {'||b||_inf':<8}"
    )
    logger.info("-" * 110)
    
    for tau in thresholds:
        features_to_drop = [j for j, h in leverage_scores.items() if h > tau]
        keep_indices = np.setdiff1d(np.arange(d), features_to_drop)
        X_pruned = X_scaled[:, keep_indices]
        
        lasso_pruned = Lasso(alpha=alpha_sklearn, fit_intercept=False, max_iter=5000, tol=1e-4)
        lasso_pruned.fit(X_pruned, y)
        pruned_iters = int(lasso_pruned.n_iter_)
        
        pruned_active_local = np.where(np.abs(lasso_pruned.coef_) > 1e-7)[0]
        pruned_active_orig = keep_indices[pruned_active_local]
        s_pruned = len(pruned_active_orig)
        
        union = len(np.union1d(baseline_active_set, pruned_active_orig))
        jaccard = len(np.intersect1d(baseline_active_set, pruned_active_orig)) / union if union > 0 else 0
        
        beta_pruned_full = np.zeros(d)
        beta_pruned_full[keep_indices] = lasso_pruned.coef_
        pruned_norm_l2 = np.linalg.norm(beta_pruned_full)
        pruned_norm_linf = np.linalg.norm(beta_pruned_full, ord=np.inf)
        diff_l2 = np.linalg.norm(baseline_coef - beta_pruned_full)
        diff_linf = np.linalg.norm(baseline_coef - beta_pruned_full, ord=np.inf)
        
        pruned_h2 = get_exact_h2(X_pruned, pruned_active_local, lambda_val)
        iter_reduction_pct = 100.0 * (1.0 - (pruned_iters / baseline_iters)) if baseline_iters > 0 else np.nan

        logger.info(
            f"{tau:<5.2f} | {len(features_to_drop):<5} | {s_pruned:<5} | {pruned_h2:<8.2f} | "
            f"{jaccard:<4.2f} | {diff_l2:<9.2e} | {diff_linf:<9.2e} | "
            f"{pruned_iters:<5d} | {iter_reduction_pct:<7.2f} | "
            f"{pruned_norm_l2:<8.4f} | {pruned_norm_linf:<8.4f}"
        )
        
        results.append({
            'Alpha': alpha_sklearn,
            'Threshold': tau,
            'Dropped': len(features_to_drop),
            'Dropped_Features': len(features_to_drop),
            'Sparsity_Base': s_base,
            'Sparsity_Pruned': s_pruned,
            'Exact_H2': pruned_h2,
            'Jaccard': jaccard,
            'Diff_L2': diff_l2,
            'Diff_Linf': diff_linf,
            'Baseline_Iters': baseline_iters,
            'Pruned_Iters': pruned_iters,
            'Iter_Reduction_Pct': iter_reduction_pct,
            'Norm_L2_Base': base_norm_l2,
            'Norm_L2_Pruned': pruned_norm_l2,
            'Norm_Linf_Base': base_norm_linf,
            'Norm_Linf_Pruned': pruned_norm_linf
        })
        
    return pd.DataFrame(results), leverage_scores, s_base, baseline_h2


def plot_pruning_trajectory(df, dataset_name, alpha, baseline_h2):
    """Plots the geometric improvement vs stability for a specific alpha."""
    fig, ax1 = plt.subplots(figsize=(10, 6))
    
    ax1.axhline(baseline_h2, color='k', linestyle='dashed', alpha=0.5, label='Baseline H2')
    ax1.plot(df['Threshold'], df['Exact_H2'], 'r-o', linewidth=2, label='Pruned Exact H2')
    ax1.set_xlabel(r'Cross-Leverage Threshold ($\tau$)')
    ax1.set_ylabel('Exact Affine Hoffman Constant ($H_2$)', color='r')
    ax1.tick_params(axis='y', labelcolor='r')
    ax1.invert_xaxis() 
    
    ax2 = ax1.twinx()
    ax2.plot(df['Threshold'], df['Jaccard'], 'g-s', linewidth=2, label='Support Stability (Jaccard)')
    ax2.set_ylabel('Support Jaccard Similarity', color='g')
    ax2.tick_params(axis='y', labelcolor='g')
    ax2.set_ylim(0, 1.1)
    
    plt.title(f'Pruning Trajectory ({dataset_name} | alpha={alpha})')
    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc='lower left')
    
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'{dataset_name}_pruning_alpha_{alpha}.png'))
    plt.close()


def plot_comparative_distribution(leverage_dict, s_dict, dataset_name):
    """Plots the distribution of cross-leverage scores for different alphas."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = {0.001: 'yellow', 0.005: 'lightblue', 0.01: 'darkblue', 0.05: 'darkgreen', 0.1: 'darkred', 0.5: 'steelblue', 1.0: 'orange'}
    bins = np.linspace(0.0, 1.0, 51)
    
    for alpha, scores in leverage_dict.items():
        score_vals = np.asarray(list(scores.values()), dtype=float)
        score_vals = score_vals[np.isfinite(score_vals)]
        score_vals = np.clip(score_vals, 0.0, 1.0)

        if score_vals.size == 0:
            logger.warning(f"No finite leverage scores available for alpha={alpha}; skipping plot.")
            continue

        s_base = s_dict[alpha]

        sns.histplot(
            score_vals,
            bins=bins,
            stat='probability',
            element='step',
            fill=True,
            alpha=0.25,
            color=colors[alpha],
            label=r'$\alpha=%.3f$ (Active $s=%d$)' % (alpha, s_base),
            ax=ax,
        )
        
        # Add a rug plot to show where the actual features sit
        sns.rugplot(score_vals, color=colors[alpha], alpha=0.1, ax=ax)

    ax.set_title(f'Distribution of Inactive Cross-Leverage Scores ({dataset_name})')
    ax.set_xlabel('Cross-Leverage Score')
    ax.set_ylabel('Probability')
    ax.set_xlim(0, 1.05)
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, f'{dataset_name}_leverage_distribution.png'))
    plt.close()


def run_full_experiment():
    datasets_to_test = ['Synthetic', 'ARCENE']
    all_results = []

    for dataset_name in datasets_to_test:
        logger.info(f"Loading {dataset_name}.")
        try:
            X_scaled, y = load_openml_dataset(dataset_name)
        except Exception as e:
            logger.error(f"Failed: {e}")
            continue

        leverage_collections = {}
        s_collections = {}
        alphas_to_test = get_alpha_grid(dataset_name)

        for alpha in alphas_to_test:
            df_res, lev_scores, s_base, base_h2 = evaluate_alpha_regime(X_scaled, y, alpha, dataset_name)

            if df_res is not None:
                df_res['Dataset'] = dataset_name
                all_results.append(df_res)
                leverage_collections[alpha] = lev_scores
                s_collections[alpha] = s_base

                # Generate individual pruning plots
                plot_pruning_trajectory(df_res, dataset_name, alpha, base_h2)

        # Generate the comparative distribution plot for the current dataset
        if len(leverage_collections) > 0:
            plot_comparative_distribution(leverage_collections, s_collections, dataset_name)

    return pd.concat(all_results) if all_results else None

if __name__ == '__main__':
    df_master = run_full_experiment()

    if df_master is not None:
        df_master.to_csv(os.path.join(DATA_DIR, 'cross_leverage_data.csv'), index=False)
