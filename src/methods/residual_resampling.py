"""Residual-resampling interaction detection test for non-linear models.

SPEC.md Section 5 (lines 97-102):
- Statistic T = largest mean absolute SHAP interaction value over predictor pairs
  (TreeSHAP for M4 and M5; largest pairwise term importance for M6).
- Null distribution by residual resampling, not label permutation:
  1. Fit additive OLS model (main effects plus covariates).
  2. Build B null datasets as y* = fitted values + randomly permuted residuals.
  3. Refit the non-linear model and recompute null statistic T*.
- Empirical p-value:
  p = (1 + count(null T* >= observed T)) / (B + 1).
  Detect if p < 0.05.
- Why not permute y directly: that destroys the additive main effects too.
  The null must keep the additive signal and remove only the interaction.
"""

from __future__ import annotations

import itertools
from typing import Any, Callable
import numpy as np
import pandas as pd
import shap
import statsmodels.api as sm

SUBSTANTIVE_COLS_DEFAULT = ["score_A_tech", "score_A_mkt", "score_B_i", "score_B_ii", "score_B_iii"]
ALL_SUBSTANTIVE_PAIRS = list(itertools.combinations(SUBSTANTIVE_COLS_DEFAULT, 2))


def extract_shap_interaction_stats(
    model: Any,
    X: pd.DataFrame,
    substantive_cols: list[str] = SUBSTANTIVE_COLS_DEFAULT,
) -> tuple[float, dict[tuple[str, str], float], tuple[str, str]]:
    """Compute TreeSHAP pairwise interaction statistics and maximum pair statistic T."""
    col_names = list(X.columns)
    sub_indices = [col_names.index(c) for c in substantive_cols if c in col_names]
    sub_names = [col_names[idx] for idx in sub_indices]
    pairs = list(itertools.combinations(sub_names, 2))

    explainer = shap.TreeExplainer(model)
    # shap_interaction_values shape: (n_samples, n_features, n_features)
    inter_vals = explainer.shap_interaction_values(X.values)
    mean_abs_inter = np.mean(np.abs(inter_vals), axis=0)

    pairwise_stats: dict[tuple[str, str], float] = {}
    best_t = -1.0
    best_pair = pairs[0] if len(pairs) > 0 else (substantive_cols[0], substantive_cols[1])

    for v1, v2 in pairs:
        idx1 = col_names.index(v1)
        idx2 = col_names.index(v2)
        stat = float(mean_abs_inter[idx1, idx2])
        pairwise_stats[(v1, v2)] = stat
        if stat > best_t:
            best_t = stat
            best_pair = (v1, v2)

    t_obs = max(0.0, best_t)
    return t_obs, pairwise_stats, best_pair


def extract_ebm_interaction_stats(
    model: Any,
    X: pd.DataFrame,
    substantive_cols: list[str] = SUBSTANTIVE_COLS_DEFAULT,
) -> tuple[float, dict[tuple[str, str], float], tuple[str, str]]:
    """Extract largest pairwise interaction term importance from fitted EBM."""
    pairs = list(itertools.combinations(substantive_cols, 2))
    pairwise_stats: dict[tuple[str, str], float] = {p: 0.0 for p in pairs}

    term_names = getattr(model, "term_names_", [])
    term_importances = model.term_importances()

    best_t = -1.0
    best_pair = pairs[0] if len(pairs) > 0 else (substantive_cols[0], substantive_cols[1])

    for name, imp in zip(term_names, term_importances):
        if " & " in name:
            parts = [p.strip() for p in name.split(" & ")]
            if len(parts) == 2:
                v1, v2 = parts[0], parts[1]
                pair_key = (v1, v2) if (v1, v2) in pairwise_stats else ((v2, v1) if (v2, v1) in pairwise_stats else None)
                if pair_key is not None:
                    val = float(imp)
                    pairwise_stats[pair_key] = val
                    if val > best_t:
                        best_t = val
                        best_pair = pair_key

    t_obs = max(0.0, best_t)
    return t_obs, pairwise_stats, best_pair


def run_residual_resampling_test(
    fit_and_score_fn: Callable[[pd.DataFrame, np.ndarray, int], tuple[float, dict[tuple[str, str], float], tuple[str, str]]],
    X: pd.DataFrame,
    y: np.ndarray,
    b_null_refits: int = 100,
    alpha: float = 0.05,
    seed: int = 42,
) -> dict[str, Any]:
    """Execute the residual-resampling interaction test (SPEC.md Section 5 lines 99-100).

    Parameters:
    - fit_and_score_fn: function (X, y, seed) -> (T, pairwise_stats, top1_pair)
    - X: feature matrix (main effects + covariates)
    - y: observed target vector
    - b_null_refits: number of null datasets B (default: 100)
    - alpha: significance level (default: 0.05)
    - seed: random seed

    Returns:
    - dict with p_value, detected, t_observed, null_t_stats, pairwise_interactions, top1_pair
    """
    n = len(y)
    rng = np.random.default_rng(seed)

    # 1. Observed test statistic T_obs
    t_obs, pairwise_stats, top1_pair = fit_and_score_fn(X, y, seed)

    # 2. Fit additive OLS model on (X, y)
    X_const = sm.add_constant(X, has_constant="add")
    ols_fit = sm.OLS(y, X_const).fit()
    y_fitted = np.asarray(ols_fit.fittedvalues, dtype=np.float64)
    residuals = np.asarray(ols_fit.resid, dtype=np.float64)

    # 3. Simulate B null datasets: y* = y_fitted + permuted(residuals)
    null_t_stats: list[float] = []
    count_ge = 0

    for b in range(b_null_refits):
        perm_resid = rng.permutation(residuals)
        y_null = y_fitted + perm_resid

        b_seed = int(rng.integers(1, 2_000_000))
        t_null, _, _ = fit_and_score_fn(X, y_null, b_seed)
        null_t_stats.append(t_null)

        if t_null >= t_obs:
            count_ge += 1

    # 4. Empirical p-value: p = (1 + count(T* >= T_obs)) / (B + 1)
    p_value = float((1 + count_ge) / (b_null_refits + 1))
    detected = bool(p_value < alpha)

    return {
        "p_value": p_value,
        "detected": detected,
        "t_observed": t_obs,
        "null_t_stats": null_t_stats,
        "pairwise_interactions": pairwise_stats,
        "top1_pair": top1_pair,
        "b_null_refits": b_null_refits,
    }
