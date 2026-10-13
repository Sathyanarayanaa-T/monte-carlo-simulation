"""Leaky-design demonstration (SPEC.md Section 7).

Quantifies the consequence of placing Commitment Intention items inside the
Individual Resilience Index instead of keeping them purely as the outcome:
- Correct design: C regressed on the contextual block, then the Individual Resilience Index (A items only).
- Leaky design: the Individual Resilience Index includes both A items and C items.
- Evaluated under the H1-null scenario (where true A weights are zero).
- Reports mean Delta R^2 and the false-positive rate of the H1 test for each design.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import time
from typing import Any

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import statsmodels.api as sm

from src.dgp import build_scenario, generate


CONTEXTUAL_COLS = [
    "score_B_i",
    "score_B_ii",
    "score_B_iii",
    "semester",
    "covar_prototype",
    "covar_hackathon",
    "covar_internship",
]


def evaluate_single_replication(
    df: pd.DataFrame,
    rep: int,
) -> dict[str, Any]:
    """Evaluate one replication for both correct and leaky designs."""
    a_cols = [c for c in df.columns if c.startswith("A_tech_") or c.startswith("A_mkt_")]
    c_cols = [c for c in df.columns if c.startswith("C_")]

    idx_correct = df[a_cols].mean(axis=1)
    idx_leaky = df[a_cols + c_cols].mean(axis=1)

    y = df["score_C"].to_numpy(dtype=np.float64)
    X_block1 = sm.add_constant(df[CONTEXTUAL_COLS], has_constant="add")

    # Block 1 OLS
    fit1 = sm.OLS(y, X_block1).fit(cov_type="HC3")
    r2_1 = float(fit1.rsquared)

    # Block 2 Correct Design
    X_block2_corr = X_block1.copy()
    X_block2_corr["index_individual"] = idx_correct
    fit2_corr = sm.OLS(y, X_block2_corr).fit(cov_type="HC3")
    r2_2_corr = float(fit2_corr.rsquared)
    delta_r2_corr = max(0.0, r2_2_corr - r2_1)
    p_corr = float(fit2_corr.pvalues["index_individual"])
    t_corr = float(fit2_corr.tvalues["index_individual"])

    # Block 2 Leaky Design
    X_block2_leaky = X_block1.copy()
    X_block2_leaky["index_individual"] = idx_leaky
    fit2_leaky = sm.OLS(y, X_block2_leaky).fit(cov_type="HC3")
    r2_2_leaky = float(fit2_leaky.rsquared)
    delta_r2_leaky = max(0.0, r2_2_leaky - r2_1)
    p_leaky = float(fit2_leaky.pvalues["index_individual"])
    t_leaky = float(fit2_leaky.tvalues["index_individual"])

    return {
        "replication": rep,
        "delta_r2_correct": delta_r2_corr,
        "pvalue_correct": p_corr,
        "tstat_correct": t_corr,
        "significant_correct": int(p_corr < 0.05),
        "delta_r2_leaky": delta_r2_leaky,
        "pvalue_leaky": p_leaky,
        "tstat_leaky": t_leaky,
        "significant_leaky": int(p_leaky < 0.05),
    }


def run_leaky_demonstration(
    n: int = 150,
    n_replications: int = 1000,
    master_seed: int = 20261013,
    n_jobs: int = 1,
    verbose: bool = True,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Run the leaky-design simulation experiment (SPEC.md Section 7).

    Parameters:
    - n: sample size (default: 150).
    - n_replications: number of replications (spec default: 1000).
    - master_seed: master random seed.
    - n_jobs: parallel jobs (default: 1).
    - verbose: print progress and summary table.

    Returns:
    - results_df: per-replication DataFrame.
    - summary: summary dictionary with mean Delta R^2 and false-positive rates.
    """
    sc = build_scenario({
        "name": f"h1_null_n{n}",
        "n": n,
        "linear_weights": {
            "A_tech": 0.0,
            "A_mkt": 0.0,
            "B_i": 0.20,
            "B_ii": 0.10,
            "B_iii": 0.15,
        },
        "non_linear_form": "none",
        "psi": 0.0,
    })

    master_seq = np.random.SeedSequence(master_seed)
    child_seeds = master_seq.spawn(n_replications)

    def _worker(rep: int, s: np.random.SeedSequence) -> dict[str, Any]:
        data = generate(sc, seed=s)
        return evaluate_single_replication(data, rep)

    t0 = time.time()
    if verbose:
        print(f"=== Starting Leaky-Design Demonstration: n = {n}, {n_replications} replications (n_jobs = {n_jobs}) ===")

    if n_jobs == 1:
        rows = [_worker(rep, child_seeds[rep]) for rep in range(n_replications)]
    else:
        rows = Parallel(n_jobs=n_jobs)(
            delayed(_worker)(rep, child_seeds[rep]) for rep in range(n_replications)
        )

    df_results = pd.DataFrame(rows).sort_values("replication").reset_index(drop=True)

    # Compute summaries
    r = len(df_results)
    mean_d_corr = float(df_results["delta_r2_correct"].mean())
    std_d_corr = float(df_results["delta_r2_correct"].std(ddof=1)) if r > 1 else 0.0
    se_d_corr = std_d_corr / np.sqrt(r) if r > 1 else 0.0

    mean_d_leaky = float(df_results["delta_r2_leaky"].mean())
    std_d_leaky = float(df_results["delta_r2_leaky"].std(ddof=1)) if r > 1 else 0.0
    se_d_leaky = std_d_leaky / np.sqrt(r) if r > 1 else 0.0

    fpr_corr = float(df_results["significant_correct"].mean())
    mcse_fpr_corr = np.sqrt(fpr_corr * (1.0 - fpr_corr) / r) if r > 1 else 0.0

    fpr_leaky = float(df_results["significant_leaky"].mean())
    mcse_fpr_leaky = np.sqrt(fpr_leaky * (1.0 - fpr_leaky) / r) if r > 1 else 0.0

    summary = {
        "n": float(n),
        "replications": float(r),
        "mean_delta_r2_correct": mean_d_corr,
        "se_delta_r2_correct": se_d_corr,
        "fpr_correct": fpr_corr,
        "mcse_fpr_correct": mcse_fpr_corr,
        "mean_delta_r2_leaky": mean_d_leaky,
        "se_delta_r2_leaky": se_d_leaky,
        "fpr_leaky": fpr_leaky,
        "mcse_fpr_leaky": mcse_fpr_leaky,
        "delta_r2_inflation_ratio": mean_d_leaky / mean_d_corr if mean_d_corr > 1e-6 else float("nan"),
    }

    if verbose:
        elapsed = time.time() - t0
        print(f"\nCompleted in {elapsed:.2f}s")
        print("\n--- Leaky-Design Demonstration Results (SPEC.md Section 7) ---")
        print(f"Correct Design:  Mean Delta R^2 = {mean_d_corr:.4f} (SE: {se_d_corr:.4f}), FPR = {fpr_corr:.3f} (MCSE: {mcse_fpr_corr:.3f})")
        print(f"Leaky Design:    Mean Delta R^2 = {mean_d_leaky:.4f} (SE: {se_d_leaky:.4f}), FPR = {fpr_leaky:.3f} (MCSE: {mcse_fpr_leaky:.3f})")
        print(f"Delta R^2 Inflation Ratio: {summary['delta_r2_inflation_ratio']:.2f}x")

    return df_results, summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run SPEC.md Section 7 Leaky-Design demonstration.")
    parser.add_argument("--n", type=int, default=150, help="Sample size (default: 150)")
    parser.add_argument("--n-reps", type=int, default=1000, help="Number of replications (default: 1000)")
    parser.add_argument("--seed", type=int, default=20261013, help="Master random seed")
    parser.add_argument("--n-jobs", type=int, default=4, help="Parallel jobs (default: 4)")
    args = parser.parse_args()

    run_leaky_demonstration(
        n=args.n,
        n_replications=args.n_reps,
        master_seed=args.seed,
        n_jobs=args.n_jobs,
        verbose=True,
    )
