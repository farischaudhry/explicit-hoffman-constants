"""
Experiment 5: Leverage Scores for Model Interpretability
--------------------------------------------------------

This experiment demonstrates how leverage scores can be used as an interpretability
tool for understanding which inactive features are highly correlated with the selected active set.

Key interpretability insights:
1. Feature Confusion Matrix: Which inactive features look most like the active ones?
2. Multicollinearity Detection: High leverage = potential false positive risk
3. Model Stability Prediction: Max leverage predicts CV/bootstrap instability  
4. Redundancy Analysis: Identify which inactive features are 'explained away' by actives

Use cases:
- Post-hoc model validation: 'Why wasn't feature j selected?'
- Debugging unstable feature selection across CV folds
- Identifying when adding dimensions hurts (inactive noise with high leverage)

Datasets are found at and can be added from: https://www.openml.org/
"""

import logging 
import os
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.datasets import fetch_openml
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Lasso, LassoCV
from sklearn.model_selection import KFold

from hoffman.bounds.hoffman_bounds import HoffmanBoundCalculator
from hoffman.utils.viz import set_plotting_style

# Output directories
OUTPUT_DIR = './results/05_leverage_interpretability/'
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
        logging.FileHandler(os.path.join(OUTPUT_DIR, 'leverage_interpretability.log'), mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger(__name__)

# Experiment parameters
N_FOLDS = 5 

# Datasets to test (name, max_samples, task_type)
DATASETS = [
    ('madelon', 50, 'classification'),  # n=50, d=500 (moderate-dim)
    ('arcene', 200, 'classification'),  # n=200, d=10000 (very high-dim)
]


def load_dataset(dataset_name: str, n_samples: int = None, task_type: str = 'classification'):
    """
    Load dataset for supervised learning task.
    
    Supports both classification and regression from OpenML.
    """
    logger.info(f'Loading {dataset_name}')
    
    try:
        data = fetch_openml(dataset_name, version=1, parser='auto')
        X = data.data.values if hasattr(data.data, 'values') else data.data
        y = data.target.values if hasattr(data.target, 'values') else data.target
    except Exception as e:
        logger.warning(f'  Failed to load {dataset_name}: {e}')
        return None
    
    # Handle classification vs regression
    if task_type == 'classification':
        # Convert to binary 0/1
        if y.dtype == 'object' or len(np.unique(y)) <= 10:
            # For multi-class, use one-vs-rest for first class
            y = (y == np.unique(y)[0]).astype(float)
    else:
        # For regression, convert to float
        try:
            y = y.astype(float)
        except:
            logger.warning(f'  Could not convert target to float')
            return None
    
    # Remove NaN/inf from both X and y
    try:
        X = np.array(X, dtype=float)
        y = np.array(y, dtype=float)
        valid_mask = np.isfinite(X).all(axis=1) & np.isfinite(y)
        X, y = X[valid_mask], y[valid_mask]
        
        if len(X) < 50:
            logger.warning(f'  Too few valid samples: {len(X)}')
            return None
    except Exception as e:
        logger.warning(f'  Data conversion failed: {e}')
        return None
    
    # Subsample if requested
    if n_samples and len(X) > n_samples:
        idx = np.random.choice(len(X), n_samples, replace=False)
        X, y = X[idx], y[idx]
    
    # Standardize features
    X = StandardScaler().fit_transform(X)
    
    # Center and scale target for better LASSO performance
    y = (y - np.mean(y)) / np.std(y)
    
    class Dataset:
        def __init__(self, A, y, task):
            self.A = A
            self.y = y
            self.task = task
    
    n, d = X.shape
    logger.info(f'  Loaded: n={n}, d={d}')
    logger.info(f'  Task: {task_type}')
    
    return Dataset(X, y, task_type)


def stability_validation(n_folds: int = 5):
    """
    Test hypothesis: High leverage => severe instability.
    
    Compare on different datasets and task types.
    """
    logger.info('\n' + '='*80)
    logger.info('Feature Selection Stability Analysis')
    logger.info('='*80)
    
    def jaccard_similarity(set1, set2):
        if len(set1) == 0 and len(set2) == 0:
            return 1.0
        intersection = len(set1.intersection(set2))
        union = len(set1.union(set2))
        return intersection / union if union > 0 else 0.0
    
    def pairwise_stability(active_sets):
        similarities = []
        for i in range(len(active_sets)):
            for j in range(i + 1, len(active_sets)):
                similarities.append(jaccard_similarity(active_sets[i], active_sets[j]))
        return np.mean(similarities) if similarities else 0.0
    
    def selection_frequency(active_sets, d):
        counts = np.zeros(d)
        for active in active_sets:
            for feat in active:
                counts[feat] += 1
        return counts / len(active_sets)
    
    def test_stability(dataset_name, design, n_folds=5):
        n, d = design.A.shape
        
        logger.info(f'\n  {dataset_name.upper()}')
        logger.info(f'    n={n}, d={d}')
        
        # Find optimal lambda
        model_cv = LassoCV(cv=n_folds, fit_intercept=False, max_iter=10000, tol=1e-3, n_jobs=-1)
        model_cv.fit(design.A, design.y)
        optimal_lambda = model_cv.alpha_
        
        # Test stability with CV
        kf = KFold(n_splits=n_folds, shuffle=True, random_state=0)
        active_sets = []
        
        for _, (train_idx, _) in enumerate(kf.split(design.A)):
            model = Lasso(alpha=optimal_lambda, fit_intercept=False, max_iter=10000, tol=1e-3)
            model.fit(design.A[train_idx], design.y[train_idx])
            active = np.where(np.abs(model.coef_) > 1e-6)[0]
            active_sets.append(set(active))
        
        # Metrics
        stability = pairwise_stability(active_sets)
        freq = selection_frequency(active_sets, d)
        consistent = np.sum(freq >= 0.8)
        
        # Leverage analysis
        model_full = Lasso(alpha=optimal_lambda, fit_intercept=False, max_iter=10000, tol=1e-3)
        model_full.fit(design.A, design.y)
        full_active = np.where(np.abs(model_full.coef_) > 1e-6)[0]
        
        leverage = HoffmanBoundCalculator._compute_leverage_scores(design.A, full_active)
        inactive_set = np.array([i for i in range(d) if i not in full_active])
        
        if len(inactive_set) > 0:
            max_lev = np.max(leverage[inactive_set])
            mean_lev = np.mean(leverage[inactive_set])
            num_high = np.sum(leverage[inactive_set] >= 0.95)
        else:
            max_lev = mean_lev = num_high = 0
        
        logger.info(f'    Jaccard stability:         {stability:.3f}')
        logger.info(f'    Consistent features:       {consistent}/{d}')
        logger.info(f'    Mean inactive leverage:    {mean_lev:.3f}')
        logger.info(f'    Max inactive leverage:     {max_lev:.3f}')
        logger.info(f'    High leverage (≥0.95):     {num_high} ({100*num_high/len(inactive_set):.1f}%)')
        
        return {
            'dataset': dataset_name.upper(),
            'n': n, 'd': d,
            'stability': stability,
            'consistent': consistent,
            'max_leverage': max_lev,
            'mean_leverage': mean_lev,
            'high_leverage_count': num_high,
            'freq': freq,
            'leverage_dist': leverage[inactive_set]
        }
    
    # Test both datasets
    results = []
    
    for dataset_name, max_samples, task_type in DATASETS:
        try:
            design = load_dataset(dataset_name, n_samples=max_samples, task_type=task_type)
            if design is not None:
                results.append(test_stability(dataset_name, design, n_folds=n_folds))
        except Exception as e:
            logger.warning(f'Failed {dataset_name}: {e}')
    
    df = pd.DataFrame([{
        'Dataset': r['dataset'],
        'n': r['n'],
        'd': r['d'],
        'Mean Leverage': r['mean_leverage'],
        'High Lev Count': r['high_leverage_count'],
        'Jaccard': r['stability'],
        'Consistent': r['consistent']
    } for r in results])
    
    # Plots
    fig = plt.figure(figsize=(16, 10))
    for idx, result in enumerate(results[:2]):  # Limit to first 2 datasets for comparison
        # Top: Selection frequency
        ax_freq = plt.subplot(2, 2, idx + 1)
        freq = result['freq']
        
        sorted_idx = np.argsort(freq)[::-1]
        x = np.arange(min(100, result['d']))
        bars = ax_freq.bar(x, freq[sorted_idx[:100]], width=1.0)
        
        # Color by consistency
        for i, (bar, f) in enumerate(zip(bars, freq[sorted_idx[:100]])):
            if f >= 0.8:
                bar.set_color('darkgreen')
            elif f >= 0.6:
                bar.set_color('orange')
            else:
                bar.set_color('darkred')
        
        ax_freq.axhline(0.8, color='green', linestyle='--', linewidth=2, label='Consistent (80%)')
        ax_freq.set_xlabel('Feature Rank', fontsize=11)
        ax_freq.set_ylabel('Selection Frequency', fontsize=11)
        ax_freq.set_title(f"{result['dataset']} - Selection Stability\n"
                        f"Jaccard: {result['stability']:.3f}", fontsize=12)
        ax_freq.legend(fontsize=9)
        ax_freq.grid(True, alpha=0.3, axis='y')
        ax_freq.set_ylim([0, 1.05])
        
        # Bottom: Leverage distribution
        ax_lev = plt.subplot(2, 2, idx + 3)
        
        leverage = result['leverage_dist']
        
        if len(leverage) > 0:
            ax_lev.hist(leverage, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
            ax_lev.axvline(result['mean_leverage'], color='orange', linestyle='--', 
                          linewidth=2, label=f"Mean: {result['mean_leverage']:.3f}")
            ax_lev.axvline(result['max_leverage'], color='darkred', linestyle='-', 
                          linewidth=2.5, label=f"Max: {result['max_leverage']:.3f}")
            ax_lev.axvline(0.95, color='red', linestyle=':', linewidth=1.5, alpha=0.7,
                          label=f"High (≥0.95): {result['high_leverage_count']}")
        
        ax_lev.set_xlabel('Leverage Score', fontsize=11)
        ax_lev.set_ylabel('Count', fontsize=11)
        ax_lev.set_title(f"{result['dataset']} - Leverage Distribution\n"
                       f"High-leverage features: {result['high_leverage_count']}", fontsize=12)
        ax_lev.legend(fontsize=9)
        ax_lev.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIG_DIR, 'stability_validation.png'), dpi=300, bbox_inches='tight')
    
    # Save results
    df.to_csv(os.path.join(DATA_DIR, 'stability_validation.csv'), index=False)


if __name__ == '__main__':
    logger.info('='*80)
    logger.info('Leverage Scores Predict Feature Selection Instability')
    logger.info('='*80)
    
    stability_validation(n_folds=N_FOLDS)
    
    logger.info('\n' + '='*80)
    logger.info(f'Results saved to: {OUTPUT_DIR}')
    logger.info('='*80)
