"""
Generates a grid of (n, d, s) configurations across different 
design matrices to evaluate the tightness of explicit affine Hoffman bounds.

For each configuration, it instantiates MULTIPLE distinct 
design matrices, calculates exact KKT norms and bounds separately, 
and reports the robust summary statistics (Median, Min, Max) of the 
tightness ratios. Also exports a csv file with all results.
"""

import logging 
import os
import numpy as np
import pandas as pd
from tqdm import tqdm

from hoffman.designs.design_factory import DesignFactory
from hoffman.bounds.hoffman_bounds import HoffmanBoundCalculator

# Output directories
OUTPUT_DIR = './results/hoffman/hoffman_bound_tightness/'
DATA_DIR = os.path.join(OUTPUT_DIR, 'data')
os.makedirs(DATA_DIR, exist_ok=True)

# Configuration
np.random.seed(0)
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)


def get_exact_norms(A: np.ndarray, active_set: np.ndarray, lam1: float):
    """Computes exact matrix norms of the KKT inverse for ground truth."""
    n = A.shape[0]
    G = (A.T @ A) / n
    M_A = HoffmanBoundCalculator.build_kkt_matrix(G, active_set, lam1)
    try:
        M_inv = np.linalg.inv(M_A)
        return {
            'exact_h2': np.linalg.norm(M_inv, ord=2),
            'exact_h1': np.linalg.norm(M_inv, ord=1),
            'exact_hinf': np.linalg.norm(M_inv, ord=np.inf)
        }
    except np.linalg.LinAlgError:
        return {'exact_h2': np.nan, 'exact_h1': np.nan, 'exact_hinf': np.nan}


def compute_metrics(A: np.ndarray, active_set: np.ndarray, lambda_val: float):
    """Computes theoretical bounds, exact norms, and their tightness ratios."""
    bounds = HoffmanBoundCalculator.compute_all(A=A, active_set=active_set, lam1=lambda_val)
    exact = get_exact_norms(A, active_set, lambda_val)
    
    # Calculate ratios only if KKT was invertible (exact norm is finite)
    if not np.isfinite(exact['exact_h2']):
        return None
        
    ratios = {
        'h2_ue': bounds.dimension_free_bound_upper / exact['exact_h2'],
        'h2_ul': bounds.dimension_free_bound_upper / bounds.dimension_free_bound_lower if bounds.dimension_free_bound_lower > 0 else np.nan,
        'h1_ue': bounds.l_1_bound / exact['exact_h1'],
        'hinf_ue': bounds.l_inf_bound / exact['exact_hinf']
    }
    
    return ratios


def generate_design_matrix(name: str, n: int, d: int, s: int):
    """
    Wrapper to generate a fresh design matrix for each MC trial.
    Must be done like this since we don't know (n, d, s) until runtime.
    """
    if name == 'Gaussian':
        return DesignFactory.gaussian(n, d, s).A
    elif name == 'Spiked_rho0.3':
        return DesignFactory.spiked(n, d, s, rho=0.3).A
    elif name == 'Spiked_rho0.5':
        return DesignFactory.spiked(n, d, s, rho=0.5).A
    elif name == 'Spiked_rho0.8':
        return DesignFactory.spiked(n, d, s, rho=0.8).A
    else:
        raise ValueError(f"Unknown design {name}")


def run_comprehensive_sweep(n_vals, d_vals, s_vals, design_names, n_mc=20, lambda_val=0.1):
    """Sweeps over hyperparameter grids, sampling multiple matrices per cell."""
    logger.info("Starting Comprehensive Bound Sweep...")
    
    all_results = []
    
    for n in n_vals:
        for d in d_vals:
            if n >= d: continue  # High-dimensional regime only
            
            for s in s_vals:
                if s >= n / 2: continue  # Sparse regime only
                
                for design_name in design_names:
                    logger.info(f"Evaluating: {design_name} | n={n}, d={d}, s={s}")
                    
                    trial_ratios = {'h2_ue': [], 'h2_ul': [], 'h1_ue': [], 'hinf_ue': []}
                    
                    for _ in tqdm(range(n_mc), desc="MC Trials", leave=False):
                        # Sample an entirely new design matrix
                        A = generate_design_matrix(design_name, n, d, s)
                        
                        # Sample a random active set
                        support = np.random.choice(d, s, replace=False)
                        
                        # Compute bounds and ratios
                        res = compute_metrics(A, support, lambda_val)
                        if res is not None:
                            for key in trial_ratios.keys():
                                if np.isfinite(res[key]):
                                    trial_ratios[key].append(res[key])
                                    
                    # If all matrices were singula then skip.
                    # This can happen in extreme spiked cases due to precision issues.
                    if len(trial_ratios['h2_ue']) == 0:
                        continue
                        
                    # Aggregate statistics (Median, Min, Max)
                    row = {
                        'Design': design_name,
                        'n': n, 'd': d, 's': s,
                        'MC_Successes': len(trial_ratios['h2_ue'])
                    }
                    
                    # Add stats for each ratio
                    for metric_key, col_name in [
                        ('h2_ue', 'H2_Upper/Exact'), 
                        ('h2_ul', 'H2_Upper/Lower'), 
                        ('h1_ue', 'H1_Upper/Exact'), 
                        ('hinf_ue', 'Hinf_Upper/Exact')
                    ]:
                        arr = np.array(trial_ratios[metric_key])
                        row[f'{col_name}_Median'] = np.median(arr)
                        row[f'{col_name}_Min'] = np.min(arr)
                        row[f'{col_name}_Max'] = np.max(arr)
                        
                    all_results.append(row)
                    
    # Convert to DataFrame
    df_final = pd.DataFrame(all_results)
    
    # Save raw CSV
    out_path = os.path.join(DATA_DIR, 'bound_tightness_ratios.csv')
    df_final.to_csv(out_path, index=False)
    logger.info(f"\nSweep complete. Raw data saved to {out_path}")
    
    # Save summary tables to log file
    log_file_path = os.path.join(DATA_DIR, 'bound_tightness_summary.txt')
    with open(log_file_path, 'w') as f:
        f.write("Bound Tightness Summary (Median [Min, Max])\n")
        for _, row in df_final.iterrows():
            h2_str = f"{row['H2_Upper/Exact_Median']:<15} [{row['H2_Upper/Exact_Min']:<15}, {row['H2_Upper/Exact_Max']:<15}]"
            hinf_str = f"{row['Hinf_Upper/Exact_Median']:<15} [{row['Hinf_Upper/Exact_Min']:<15}, {row['Hinf_Upper/Exact_Max']:<15}]"
            f.write(f"{row['Design']:<15} | n={row['n']:<3} d={row['d']:<4} s={row['s']:<2} | H2 U/E: {h2_str} | Hinf U/E: {hinf_str}\n")

    return df_final


if __name__ == '__main__':
    # Define grid. Keep MC samples relatively high to get true min/max behaviors.
    n_values = [50, 100, 200, 400, 800, 1600, 3200]
    d_values = [100, 200, 400, 800, 1600, 3200, 6400, 12800]
    s_values = [5, 10, 20, 40, 80, 160]
    design_names = ['Gaussian', 'Spiked_rho0.3', 'Spiked_rho0.5', 'Spiked_rho0.8']
    
    run_comprehensive_sweep(n_values, d_values, s_values, design_names, n_mc=3)