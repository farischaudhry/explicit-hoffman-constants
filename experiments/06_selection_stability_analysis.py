"""
Experiment 6: Hoffman Stability Bound Validation
-------------------------------------------------

Core Theorem (Robinson-Lipschitz Stability):
    ||β hat(y) - β hat(y')|| <= H(A hat) x ||(1/n)A^T(y - y')||₂
where y is the the original response vector, and A hat is it's true active set. 
The perturbed response y' can differ in various ways (noise addition, CV fold removal, etc).

The bound depends on the perturbation projected onto feature space.
Noise orthogonal to the column space of A has zero effect.

Experimental Design:
1. Fix design matrix A (Gaussian, Spiked, etc.)
2. Generate true response y = A x β_true + ε
3. Fit LASSO: β hat(y)
4. Perturb y in controlled ways:
   - Add Gaussian noise: y' = y + δ
   - CV folds: y' uses subset of samples (some entries zeroed)
   - Adversarial: perturb in worst-case direction (min eigvec of KKT matrix)
5. Fit LASSO: β hat(y')
6. Measure actual: ||β hat(y) - β hat(y')||
7. Compute bound: H x ||(1/n)A^T(y - y')||
8. Validate: actual ≤ bound

CV Stability Implication:
When H is large (correlated features), small changes in y from removing
a CV fold get amplified => unstable β hat => unreliable model selection.
"""

import logging 
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import Lasso
from sklearn.model_selection import KFold
from sklearn.datasets import fetch_california_housing
from sklearn.preprocessing import StandardScaler

from hoffman.designs.design_factory import DesignFactory
from hoffman.bounds.hoffman_bounds import HoffmanBoundCalculator
from hoffman.utils.viz import set_plotting_style

# Output directories
OUTPUT_DIR = './results/06_selection_stability/'
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
FIG_DIR = os.path.join(OUTPUT_DIR, 'figures')
os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIG_DIR, exist_ok=True)

# Configuration
np.random.seed(0)
logging.basicConfig(
    level=logging.INFO,
    format='%(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'local_hoffman_validation.log'), mode='w')
    ]
)
logger = logging.getLogger(__name__)
set_plotting_style()


def get_active_set(beta, threshold=1e-10):
    """Extract active set from coefficient vector."""
    return np.where(np.abs(beta) > threshold)[0]


def compute_perturbation_bound(A, y1, y2):
    """
    Compute the perturbation term in Hoffman stability bound:
        ||(1/n)A^T(y1 - y2)||_2.
    This is the projected perturbation onto the feature space.
    """
    n = len(y1)
    y_diff = y1 - y2
    projected_pert = (A.T @ y_diff) / n
    return np.linalg.norm(projected_pert)


def part_1_controlled_noise_validation():
    """
    Test the Hoffman bound with controlled noise perturbations.
    
    Setup:
    - Fix design matrix A
    - Generate y = A x β_true + ε₁
    - Add varying noise: y' = y + ε₂
    - Verify: ||β hat(y) - β hat(y')|| ≤ H(A hat) x ||(1/n)A^T(y - y')||
    
    This is the cleanest test: we control perturbation magnitude exactly.
    """
    logger.info("\n" + "="*80)
    logger.info("Part 1: Controlled Noise Validation")
    logger.info("="*80)
    
    n, d, s = 200, 400, 15
    lambda_val = 0.1
    noise_levels = [0.001, 0.005, 0.01, 0.05, 0.1, 0.2, 0.5]
    
    # Test on different design types
    designs_to_test = [
        ('Gaussian RIP', DesignFactory.gaussian(n=n, d=d, s=s)),
        ('Spiked rho=0.3', DesignFactory.spiked(n=n, d=d, s=s, rho=0.3)),
        ('Spiked rho=0.7', DesignFactory.spiked(n=n, d=d, s=s, rho=0.7)),
    ]
    
    results = []
    
    for design_name, design in designs_to_test:
        logger.info(f"\n{design_name}:")
        logger.info(f"  n={n}, d={d}, s={s}, lambda={lambda_val}")
        A = design.A
        y_true = A @ design.true_beta + 0.01 * np.random.randn(n)
        
        # Solve LASSO on true data
        lasso_true = Lasso(alpha=lambda_val, fit_intercept=False, max_iter=10000, tol=1e-8)
        lasso_true.fit(A, y_true)
        beta_true = lasso_true.coef_.copy()
        active_true = get_active_set(beta_true)
        
        # Compute Hoffman constant using ESTIMATOR's active set (theoretically correct)
        G = (A.T @ A) / n
        bounds_estimator = HoffmanBoundCalculator.compute_all(G, active_true, lambda_val, A)
        H_estimator = bounds_estimator.exact_hoffman
        
        # Get active submatrix for adversarial test
        A_active = A[:, active_true]
        G_active = A_active.T @ A_active / n
        
        logger.info(f"  Active set size (estimator): {len(active_true)}")
        logger.info(f"  Hoffman constant H (estimator): {H_estimator:.6f}")
        
        # ==== ADVERSARIAL PERTURBATION TEST ====
        # TODO: COnstruct an adversarial example to demonstrate that worst-case bound is achievable
        # logger.info(f"\n  Testing ADVERSARIAL perturbation:")
        # try:
        #     # Theoretical worst case: perturb in direction of minimum eigenvector
        #     M = G_active + lambda_val * np.eye(len(active_true))
        #     eigenvalues, eigenvectors = np.linalg.eigh(M)
        #     min_idx = np.argmin(eigenvalues)
        #     sigma_min = eigenvalues[min_idx]
        #     v_worst = eigenvectors[:, min_idx]
            
        #     # Theoretical prediction:
        #     # If we perturb beta directly: delta_beta = epsilon * v_worst
        #     # And ||delta_beta|| = epsilon
        #     # The KKT system says: M @ delta_beta ≈ (1/n)A^T @ delta_y
        #     # So: ||delta_y|| ≈ n * ||M @ delta_beta|| / ||A^T||
        #     # Hoffman bound: ||delta_beta|| <= H * ||(1/n)A^T delta_y||
        #     #              = (1/sigma_min) * ||M @ delta_beta||
        #     #              = (1/sigma_min) * sigma_min * ||v_worst|| = 1.0 (TIGHT!)
            
        #     logger.info(f"    THEORETICAL worst-case:")
        #     logger.info(f"      Minimum eigenvalue of M: {sigma_min:.6f}")
        #     logger.info(f"      Hoffman constant H: {H_estimator:.6f}")
        #     logger.info(f"      Theoretical slack (adversarial): 1.0 (by definition)")
        #     logger.info(f"      -> This is why H = 1/sigma_min!")
            
        #     # Practical test: perturb y in worst direction
        #     # Problem: LASSO may change active set, making bound not applicable
        #     delta_y_adv = A_active @ v_worst * 0.001
        #     y_adv = y_true + delta_y_adv
            
        #     lasso_adv = Lasso(alpha=lambda_val, fit_intercept=False, max_iter=10000, tol=1e-8)
        #     lasso_adv.fit(A, y_adv)
        #     beta_adv = lasso_adv.coef_.copy()
        #     active_adv = get_active_set(beta_adv)
        #     same_active_adv = np.array_equal(np.sort(active_true), np.sort(active_adv))
            
        #     actual_pert_adv = np.linalg.norm(beta_true - beta_adv)
        #     projected_pert_adv = compute_perturbation_bound(A, y_true, y_adv)
        #     predicted_bound_adv = H_estimator * projected_pert_adv
        #     slack_adv = predicted_bound_adv / actual_pert_adv if actual_pert_adv > 1e-12 else np.inf
            
        #     logger.info(f"    EMPIRICAL (through LASSO):")
        #     logger.info(f"      Same active set: {same_active_adv}")
        #     logger.info(f"      Slack: {slack_adv:.3f}x")
        #     logger.info(f"      -> {'TIGHT!' if slack_adv < 2 and same_active_adv else 'Loose because ' + ('active set changed' if not same_active_adv else 'LASSO nonlinearity')}")
            
        #     # Add adversarial result to data
        #     results.append({
        #         'design': design_name,
        #         'noise_level': 0.001,  # Very small scale for adversarial
        #         'same_active_set': same_active_adv,
        #         'hoffman_estimator': H_estimator,
        #         'actual_perturbation': actual_pert_adv,
        #         'predicted_bound_estimator': predicted_bound_adv,
        #         'projected_perturbation': projected_pert_adv,
        #         'naive_bound': H_estimator * np.linalg.norm(delta_y_adv) / np.sqrt(n),
        #         'naive_perturbation': np.linalg.norm(delta_y_adv) / np.sqrt(n),
        #         'bound_satisfied': actual_pert_adv <= predicted_bound_adv * (1 + 1e-6),
        #         'slack_ratio_estimator': slack_adv,
        #         'perturbation_type': 'adversarial',
        #         'active_size_estimator': len(active_true),
        #         'active_size_pert': len(active_adv),
        #     })
        # except Exception as e:
        #     logger.info(f"    Could not construct adversarial: {e}")
        
        # ==== RANDOM NOISE PERTURBATION TEST ====
        for noise_level in noise_levels:
            # Perturb response: y' = y + noise
            noise = np.random.randn(n) * noise_level
            y_perturbed = y_true + noise
            
            # Solve LASSO on perturbed data
            lasso_pert = Lasso(alpha=lambda_val, fit_intercept=False, max_iter=10000, tol=1e-8)
            lasso_pert.fit(A, y_perturbed)  # Same A, different y
            beta_pert = lasso_pert.coef_.copy()
            active_pert = get_active_set(beta_pert)
            
            # Check if active sets match
            same_active = np.array_equal(np.sort(active_true), np.sort(active_pert))
            
            # Measure ACTUAL perturbation
            actual_pert = np.linalg.norm(beta_true - beta_pert)
            
            # Compute PREDICTED bound: H × ||(1/n)A^T(y - y')||
            projected_pert = compute_perturbation_bound(A, y_true, y_perturbed)
            predicted_bound_estimator = H_estimator * projected_pert
            
            # Verify bound (using estimator's H)
            bound_satisfied = actual_pert <= predicted_bound_estimator * (1 + 1e-6)
            slack_estimator = predicted_bound_estimator / actual_pert if actual_pert > 1e-12 else np.inf
            
            # Also compute naive bound (not projected)
            naive_pert = np.linalg.norm(y_true - y_perturbed) / np.sqrt(n)
            naive_bound = H_estimator * naive_pert
            
            results.append({
                'design': design_name,
                'noise_level': noise_level,
                'same_active_set': same_active,
                'hoffman_estimator': H_estimator,
                'actual_perturbation': actual_pert,
                'predicted_bound_estimator': predicted_bound_estimator,
                'projected_perturbation': projected_pert,
                'naive_bound': naive_bound,
                'naive_perturbation': naive_pert,
                'bound_satisfied': bound_satisfied,
                'slack_ratio_estimator': slack_estimator,
                'perturbation_type': 'random',
                'active_size_estimator': len(active_true),
                'active_size_pert': len(active_pert),
            })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_DIR, 'controlled_noise_validation.csv'), index=False)
    
    # Print summary
    logger.info('\n' + '-'*80)
    logger.info('Summary:')
    logger.info('-'*80)
    for design_name in df['design'].unique():
        subset = df[df['design'] == design_name]
        bound_satisfied = subset['bound_satisfied'].sum()
        avg_slack_estimator = subset['slack_ratio_estimator'].mean()
        
        logger.info(f"\n{design_name}:")
        logger.info(f"  Bound satisfied: {bound_satisfied}/{len(subset)} trials ({100*bound_satisfied/len(subset):.1f}%)")
        logger.info(f"  Average slack (estimator H): {avg_slack_estimator:.3f}x")

    return df


def part_2_cv_fold_perturbations():
    """
    Validate Hoffman bound using cross-validation splits.
    
    Each CV fold creates a natural perturbation:
    - Different training subsets
    - Solutions should have similar (but not identical) active sets
    - Bound: ||β_fold_i - β_fold_j||_2 ≤ H(A) × (data difference)
    
    This tests if the bound holds for realistic perturbations.
    """
    logger.info('\n' + '='*80)
    logger.info('Part 2: Cross-Validation Fold Perturbations')
    logger.info('='*80)
    
    n, d, s = 300, 500, 20
    lambda_val = 0.05
    n_folds = 5
    
    designs_to_test = [
        ('Gaussian', DesignFactory.gaussian(n=n, d=d, s=s)),
        ('Spiked rho=0.5', DesignFactory.spiked(n=n, d=d, s=s, rho=0.5)),
        ('Spiked rho=0.8', DesignFactory.spiked(n=n, d=d, s=s, rho=0.8)),
    ]
    
    results = []
    
    for design_name, design in designs_to_test:
        logger.info(f'\n{design_name}:')
        
        # Generate full data
        y = design.A @ design.true_beta + 0.05 * np.random.randn(n)
        
        # K-fold CV
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        fold_solutions = []
        fold_active_sets = []
        fold_indices = []
        
        for train_idx, _ in kf.split(design.A):
            A_train = design.A[train_idx]
            y_train = y[train_idx]
            
            lasso = Lasso(alpha=lambda_val, fit_intercept=False, max_iter=10000, tol=1e-8)
            lasso.fit(A_train, y_train)
            
            fold_solutions.append(lasso.coef_.copy())
            fold_active_sets.append(get_active_set(lasso.coef_))
            fold_indices.append(train_idx)
        
        # Compute full solution (for Hoffman constant)
        lasso_full = Lasso(alpha=lambda_val, fit_intercept=False, max_iter=10000, tol=1e-8)
        lasso_full.fit(design.A, y)
        active_full = get_active_set(lasso_full.coef_)
        
        G = (design.A.T @ design.A) / n
        bounds = HoffmanBoundCalculator.compute_all(G, active_full, lambda_val, design.A)
        H_exact = bounds.exact_hoffman
        
        logger.info(f"  Full active set size: {len(active_full)}")
        logger.info(f"  Hoffman constant: {H_exact:.6f}")
        
        # Compare all pairs of folds
        for i in range(n_folds):
            for j in range(i+1, n_folds):
                beta_i = fold_solutions[i]
                beta_j = fold_solutions[j]
                active_i = fold_active_sets[i]
                active_j = fold_active_sets[j]
                
                # Check if same active set
                same_active = np.array_equal(np.sort(active_i), np.sort(active_j))
                jaccard = len(np.intersect1d(active_i, active_j)) / len(np.union1d(active_i, active_j)) if len(np.union1d(active_i, active_j)) > 0 else 1.0
                
                # Actual perturbation
                actual_pert = np.linalg.norm(beta_i - beta_j)
                
                # Data perturbation (approximate - different training sets)
                # Use difference in predictions on full data as proxy
                pred_diff = np.linalg.norm(design.A @ beta_i - design.A @ beta_j) / np.sqrt(n)
                
                # Predicted bound (using full H(A) as approximation)
                predicted_bound = H_exact * pred_diff
                
                bound_satisfied = actual_pert <= predicted_bound * (1 + 1e-5)
                slack = predicted_bound / actual_pert if actual_pert > 1e-12 else np.inf
                
                results.append({
                    'design': design_name,
                    'fold_pair': f'{i+1}-{j+1}',
                    'same_active_set': same_active,
                    'jaccard_similarity': jaccard,
                    'hoffman_constant': H_exact,
                    'actual_perturbation': actual_pert,
                    'predicted_bound': predicted_bound,
                    'bound_satisfied': bound_satisfied,
                    'slack_ratio': slack,
                })
    
    df = pd.DataFrame(results)
    df.to_csv(os.path.join(DATA_DIR, 'cv_fold_perturbations.csv'), index=False)
    
    # Summary
    logger.info('\n' + '-'*80)
    logger.info('Summary:')
    logger.info('-'*80)
    for design_name in df['design'].unique():
        subset = df[df['design'] == design_name]
        same_active_pct = 100 * subset['same_active_set'].mean()
        bound_satisfied_pct = 100 * subset['bound_satisfied'].mean()
        avg_jaccard = subset['jaccard_similarity'].mean()
        avg_slack_same = subset[subset['same_active_set']]['slack_ratio'].mean() if len(subset[subset['same_active_set']]) > 0 else 0
        
        design_clean = design_name.replace('ρ', 'rho')
        logger.info(f"\n{design_clean}:")
        logger.info(f"  Same active set: {same_active_pct:.1f}%")
        logger.info(f"  Avg Jaccard: {avg_jaccard:.3f}")
        logger.info(f"  Bound satisfied: {bound_satisfied_pct:.1f}%")
        if avg_slack_same > 0:
            logger.info(f"  Avg slack (same active): {avg_slack_same:.3f}x")
    
    # Visualization
    fig, ax = plt.subplots(1, 1, figsize=(7, 6))
    
    # Actual vs Predicted by active set similarity
    colors_same = {True: 'green', False: 'red'}
    for design_name in df['design'].unique():
        for same_active in [True, False]:
            subset = df[(df['design'] == design_name) & (df['same_active_set'] == same_active)]
            if len(subset) > 0:
                label = f"{design_name} ({'Same' if same_active else 'Diff'} active)"
                ax.scatter(subset['predicted_bound'], subset['actual_perturbation'],
                          s=60, alpha=0.5, label=label, 
                          color=colors_same[same_active], 
                          marker='o' if 'Gaussian' in design_name else ('s' if '0.5' in design_name else '^'))
    
    max_val = max(df['predicted_bound'].max(), df['actual_perturbation'].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5)
    ax.set_xlabel('Predicted: H(A) x perturbation')
    ax.set_ylabel('Actual: $\\|\\beta_i - \\beta_j\\|$')
    ax.set_title('CV Fold Comparison', fontweight='bold')
    ax.legend(ncol=2)
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_val])
    ax.set_ylim([0, max_val])
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'cv_fold_perturbations.png'), dpi=300, bbox_inches='tight')
    logger.info(f'\nSaved: cv_fold_perturbations.png')
    
    return df


def part_3_real_data_validation():
    """
    Uses California Housing dataset to test:
    1. CV fold stability (realistic perturbations)
    2. Bootstrap stability (resampling)
    """
    logger.info('\n' + '='*80)
    logger.info('Part 3: Real Data Validation')
    logger.info('='*80)
    
    # Load dataset
    data = fetch_california_housing()
    X_raw, y_raw = data.data, data.target
    
    # Standardize
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    A = scaler_X.fit_transform(X_raw)
    y = scaler_y.fit_transform(y_raw.reshape(-1, 1)).ravel()
    
    # Subsample for speed
    np.random.seed(0)
    idx = np.random.choice(len(y), size=1000, replace=False)
    A = A[idx]
    y = y[idx]
    
    n, d = A.shape
    lambda_val = 0.1
    
    logger.info(f'\nDataset: California Housing')
    logger.info(f'  Samples n: {n}')
    logger.info(f'  Features d: {d}')
    logger.info(f'  Lambda: {lambda_val}')
    
    # Fit full model
    lasso_full = Lasso(alpha=lambda_val, fit_intercept=False, max_iter=10000, tol=1e-8)
    lasso_full.fit(A, y)
    beta_full = lasso_full.coef_.copy()
    active_full = get_active_set(beta_full)
    
    logger.info(f"  Active set size: {len(active_full)}")
    
    # Compute Hoffman constant
    G = (A.T @ A) / n
    bounds = HoffmanBoundCalculator.compute_all(G, active_full, lambda_val, A)
    H = bounds.exact_hoffman
    logger.info(f'  Hoffman constant H: {H:.6f}')
    
    results = []
    
    # Test 1: CV fold stability
    logger.info(f'\n  CV Fold Stability:')
    kf = KFold(n_splits=5, shuffle=True, random_state=0)
    fold_betas = []
    fold_idx_list = []
    
    for train_idx, _ in kf.split(A):
        A_train = A[train_idx]
        y_train = y[train_idx]
        
        lasso = Lasso(alpha=lambda_val, fit_intercept=False, max_iter=10000, tol=1e-8)
        lasso.fit(A_train, y_train)
        fold_betas.append(lasso.coef_.copy())
        fold_idx_list.append(train_idx)
    
    # Compare fold pairs
    for i in range(len(fold_betas)):
        for j in range(i+1, len(fold_betas)):
            beta_i = fold_betas[i]
            beta_j = fold_betas[j]
            
            actual_diff = np.linalg.norm(beta_i - beta_j)
            
            # Approximate data perturbation (different training sets)
            pred_diff_i = A @ beta_i
            pred_diff_j = A @ beta_j
            data_pert = np.linalg.norm(pred_diff_i - pred_diff_j) / np.sqrt(n)
            
            bound = H * data_pert
            
            results.append({
                'dataset': 'California Housing',
                'perturbation_type': 'CV fold',
                'actual_diff': actual_diff,
                'bound': bound,
                'slack': bound / actual_diff if actual_diff > 1e-12 else np.inf,
            })
    
    df_cv = pd.DataFrame(results)
    df_cv.to_csv(os.path.join(DATA_DIR, 'real_data_validation.csv'), index=False)
    
    logger.info(f'    CV pairs tested: {len(df_cv)}')
    logger.info(f'    Avg slack: {df_cv["slack"].mean():.2f}x')
    logger.info(f'    Bound satisfied: {(df_cv["actual_diff"] <= df_cv["bound"]).sum()}/{len(df_cv)}')
    
    # Test 2: Bootstrap stability
    logger.info(f'\n  Bootstrap Stability:')
    n_bootstrap = 10
    bootstrap_betas = []
    
    for _ in range(n_bootstrap):
        boot_idx = np.random.choice(n, size=n, replace=True)
        A_boot = A[boot_idx]
        y_boot = y[boot_idx]
        
        lasso_boot = Lasso(alpha=lambda_val, fit_intercept=False, max_iter=10000, tol=1e-8)
        lasso_boot.fit(A_boot, y_boot)
        bootstrap_betas.append(lasso_boot.coef_.copy())
    
    # Compare bootstrap pairs
    boot_results = []
    for i in range(n_bootstrap):
        for j in range(i+1, n_bootstrap):
            beta_i = bootstrap_betas[i]
            beta_j = bootstrap_betas[j]
            
            actual_diff = np.linalg.norm(beta_i - beta_j)
            pred_diff = np.linalg.norm(A @ beta_i - A @ beta_j) / np.sqrt(n)
            
            bound = H * pred_diff
            
            boot_results.append({
                'actual': actual_diff,
                'bound': bound,
                'slack': bound / actual_diff if actual_diff > 1e-12 else np.inf,
            })
    
    df_boot = pd.DataFrame(boot_results)
    
    logger.info(f'    Bootstrap pairs: {len(df_boot)}')
    logger.info(f'    Avg slack: {df_boot["slack"].mean():.2f}x')
    
    # Visualization
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    # Plot 1: CV fold comparison
    ax = axes[0]
    ax.scatter(df_cv['bound'], df_cv['actual_diff'], 
              s=80, alpha=0.6, label='Hoffman Bound', color='steelblue')
    
    max_val = max(df_cv['bound'].max(), df_cv['actual_diff'].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5, label='y=x (tight)')
    
    ax.set_xlabel('Predicted Bound')
    ax.set_ylabel('Actual $\\|\\beta_i - \\beta_j\\|$')
    ax.set_title('CV Fold Stability', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_val])
    ax.set_ylim([0, max_val])
    
    # Plot 2: Bootstrap comparison
    ax = axes[1]
    ax.scatter(df_boot['bound'], df_boot['actual'],
              s=80, alpha=0.6, color='coral')
    
    max_val = max(df_boot['bound'].max(), df_boot['actual'].max()) * 1.1
    ax.plot([0, max_val], [0, max_val], 'k--', linewidth=2, alpha=0.5, label='y=x (tight)')
    ax.set_xlabel('Predicted Bound')
    ax.set_ylabel('Actual $\\|\\beta_i - \\beta_j\\|$')
    ax.set_title('Bootstrap Stability', fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, max_val])
    ax.set_ylim([0, max_val])
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'real_data_validation.png'), dpi=300, bbox_inches='tight')
    logger.info(f'\nSaved: real_data_validation.png')
    
    return df_cv


if __name__ == '__main__':
    logger.info('='*80)
    logger.info('Hoffman Stability Bound Validation')
    logger.info('='*80)
    
    # Run experiments
    df1 = part_1_controlled_noise_validation()
    df2 = part_2_cv_fold_perturbations()
    df3 = part_3_real_data_validation()
    
    logger.info('\n' + '='*80)
    logger.info('All experiments completed!')
    logger.info(f'Results saved to: {OUTPUT_DIR}')
    logger.info('='*80)
    
    # Final summary
    logger.info('\n' + '='*80)
    logger.info('KEY FINDINGS:')
    logger.info('='*80)
    
    # From Part 1
    exp1_same = df1[df1['same_active_set']]
    exp1_bound_rate = 100 * exp1_same['bound_satisfied'].mean()
    exp1_avg_slack_estimator = exp1_same['slack_ratio_estimator'].mean()
    
    logger.info(f'\nPart 1 (Controlled Noise):')
    logger.info(f'  When active sets match:')
    logger.info(f'    - Bound satisfied: {exp1_bound_rate:.1f}% of trials')
    logger.info(f'    - Average slack: {exp1_avg_slack_estimator:.2f}x')
    
    # From Part 2
    exp2_same = df2[df2['same_active_set']]
    if len(exp2_same) > 0:
        exp2_bound_rate = 100 * exp2_same['bound_satisfied'].mean()
        exp2_avg_slack = exp2_same['slack_ratio'].mean()
        
        logger.info(f'\nPart 2 (CV Folds):')
        logger.info(f'  When active sets match:')
        logger.info(f'    - Bound satisfied: {exp2_bound_rate:.1f}% of trials')
        logger.info(f'    - Average slack: {exp2_avg_slack:.2f}x')
    
    # From Part 3
    logger.info(f'\nPart 3 (CV Folds on Real Data):')
    logger.info(f'  CV fold stability:')
    logger.info(f'    - Avg slack: {df3["slack"].mean():.2f}x')
    logger.info(f'  Bootstrap stability:')
    