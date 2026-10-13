"""Tests for Milestone 5: Machine Learning Methods (M4, M5, M6) and Residual Resampling.

SPEC.md Section 5 (lines 97-102):
- Fast unit test: Null datasets keep the additive signal, while label permutation destroys it.
- Model unit tests for M4, M5, M6 fitting and interaction extraction.
- Null-calibration test on additive-only data (marked slow).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import statsmodels.api as sm

import src.dgp as dgp
from src.methods.m1_hierarchical import prepare_features
from src.methods.m4_random_forest import M4RandomForest
from src.methods.m5_gradient_boosting import M5GradientBoosting
from src.methods.m6_ebm import M6ExplainableBoosting
from src.methods.residual_resampling import run_residual_resampling_test


def test_null_datasets_preserve_additive_signal():
    """Fast unit test: Residual-resampled null datasets preserve the additive signal.

    SPEC.md line 101:
    'Why not permute y directly: that destroys the main effects too. A forest on main-effect-only
    data would then look as if it had found an interaction compared with pure noise. The null must
    keep the additive signal and remove only the interaction.'
    """
    rng = np.random.default_rng(20261013)
    n = 200
    p = 5

    # 1. Generate data with strong linear additive signal and no interaction
    X = rng.normal(size=(n, p))
    beta_true = np.array([0.4, 0.3, 0.25, 0.15, 0.10])
    noise = rng.normal(scale=0.5, size=n)
    y = X @ beta_true + noise

    # 2. Fit additive OLS model
    X_const = sm.add_constant(X, has_constant="add")
    ols_fit = sm.OLS(y, X_const).fit()
    y_fitted = ols_fit.fittedvalues
    residuals = ols_fit.resid
    r2_orig = ols_fit.rsquared

    # 3. Build null dataset via residual resampling: y* = y_fitted + permuted(residuals)
    permuted_residuals = rng.permutation(residuals)
    y_null = y_fitted + permuted_residuals

    # 4. Refit OLS on null dataset
    ols_null = sm.OLS(y_null, X_const).fit()
    r2_null = ols_null.rsquared

    # Check: Additive signal is fully preserved in the null dataset
    assert np.isclose(r2_null, r2_orig, atol=0.08), (
        f"Residual resampling should preserve additive R2: orig={r2_orig:.3f}, null={r2_null:.3f}"
    )
    # Estimated coefficients match original additive coefficients
    assert np.allclose(ols_null.params[1:], ols_fit.params[1:], atol=0.10), (
        "Null dataset coefficients should match original additive coefficients"
    )

    # 5. Contrast with naive outcome permutation (which destroys the additive signal)
    y_permuted_naive = rng.permutation(y)
    ols_naive = sm.OLS(y_permuted_naive, X_const).fit()
    r2_naive = ols_naive.rsquared

    # Naive label permutation destroys all main effects (R2 ~ 0)
    assert r2_naive <= 0.08, f"Label permutation should destroy main effects, got R2={r2_naive:.3f}"


def test_m4_random_forest_components():
    """Fast unit test: M4 Random Forest fits, tunes OOB, and extracts TreeSHAP interactions."""
    sc = dgp.build_scenario({"n": 60, "non_linear_form": "interaction", "psi": 0.08})
    df = dgp.generate(sc, seed=42)
    _, X2 = prepare_features(df)
    y = df["score_C"].values

    m4 = M4RandomForest(n_estimators=50, min_samples_leaf_grid=[3, 5], random_state=42)
    m4.fit(X2, y, tune_oob=True)

    assert m4.best_min_samples_leaf_ in [3, 5]
    assert len(m4.feature_importances_) == len(X2.columns)

    t_obs, pairs, top1 = m4.compute_interaction_stats(X2)
    assert t_obs >= 0.0
    assert len(pairs) == 10
    assert top1 in pairs


def test_m5_gradient_boosting_components():
    """Fast unit test: M5 Gradient Boosting fits, tunes inner CV, and extracts TreeSHAP interactions."""
    sc = dgp.build_scenario({"n": 60, "non_linear_form": "interaction", "psi": 0.08})
    df = dgp.generate(sc, seed=42)
    _, X2 = prepare_features(df)
    y = df["score_C"].values

    m5 = M5GradientBoosting(
        max_depth_grid=[1, 2],
        n_estimators_grid=[50],
        random_state=42,
    )
    m5.fit(X2, y, tune_cv=True)

    assert m5.best_max_depth_ in [1, 2]
    t_obs, pairs, top1 = m5.compute_interaction_stats(X2)
    assert t_obs >= 0.0
    assert len(pairs) == 10
    assert top1 in pairs


def test_m6_ebm_components():
    """Fast unit test: M6 EBM fits and extracts pairwise interaction term statistics."""
    sc = dgp.build_scenario({"n": 60, "non_linear_form": "interaction", "psi": 0.08})
    df = dgp.generate(sc, seed=42)
    _, X2 = prepare_features(df)
    y = df["score_C"].values

    m6 = M6ExplainableBoosting(interactions=10, outer_bags=2, random_state=42)
    m6.fit(X2, y)

    t_obs, pairs, top1 = m6.compute_interaction_stats(X2)
    assert t_obs >= 0.0
    assert len(pairs) == 10


@pytest.mark.slow
def test_null_calibration_additive_data_slow():
    """Slow test: Null-calibration of residual-resampling test on additive-only data (psi = 0).

    Verifies that under purely additive data, the residual-resampling test rejects at
    approximately nominal alpha (5%).
    """
    sc = dgp.build_scenario({"n": 100, "non_linear_form": "none", "psi": 0.0})
    seeds = np.random.SeedSequence(12345).spawn(20)

    rejections = 0
    for s in seeds:
        df = dgp.generate(sc, seed=s)
        _, X2 = prepare_features(df)
        y = df["score_C"].values

        m4 = M4RandomForest(n_estimators=100, min_samples_leaf_grid=[5], random_state=42)
        m4.fit(X2, y, tune_oob=False)
        res = m4.run_interaction_test(X2, y, b_null_refits=20, alpha=0.05)
        if res["detected"]:
            rejections += 1

    rejection_rate = rejections / len(seeds)
    # On 20 replications, rejection rate should stay small (<= 20%)
    assert rejection_rate <= 0.20, f"False rejection rate {rejection_rate:.2%} was too high on additive data"
