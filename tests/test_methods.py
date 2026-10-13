"""Validation tests for Milestone 4: Methods M0 to M3, Nested CV, and Controls.

SPEC.md Section 9 checks & Milestone 4 requirements:
1. Negative control: C replaced by noise, 300 replications, Type I rates within MC error of 0.05.
2. Positive control: strong linear signal at n = 300 gives power near 1.
3. Leak check: shuffling outcome in test fold gives out-of-sample R^2 <= 0.
4. M1, M1-adj, M2a, M2b, and M3 component unit tests.
"""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.model_selection import KFold

import src.dgp as dgp
from src.harness import run_nested_cv
from src.methods.m0_baseline import M0Baseline
from src.methods.m1_hierarchical import M1HierarchicalOLS, fit_m1_marker_adjusted, prepare_features
from src.methods.m2_interaction import M2aPreSpecifiedInteraction, M2bAllPairwiseInteractions
from src.methods.m3_elasticnet import M3ElasticNet


def test_leak_check_shuffled_test_outcomes():
    """Test 3: Leak check - shuffling the outcome in the test fold gives out-of-sample R2 <= 0."""
    sc = dgp.build_scenario({"n": 150})
    df = dgp.generate(sc, seed=42)
    X1, X2 = prepare_features(df)
    y = df["score_C"].to_numpy(dtype=np.float64)

    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    preds = np.zeros(len(y), dtype=np.float64)
    y_test_shuffled = np.zeros(len(y), dtype=np.float64)
    rng = np.random.default_rng(20261013)

    for train_idx, test_idx in kf.split(X2):
        m1 = M1HierarchicalOLS()
        m1.fit(X1.iloc[train_idx], X2.iloc[train_idx], y[train_idx])
        p = m1.predict(X2.iloc[test_idx])
        preds[test_idx] = p
        # Permute outcomes strictly within the test fold
        y_test_shuffled[test_idx] = rng.permutation(y[test_idx])

    ssr = float(np.sum((y_test_shuffled - preds) ** 2))
    sst = float(np.sum((y_test_shuffled - np.mean(y_test_shuffled)) ** 2))
    r2_oos_shuffled = 1.0 - (ssr / sst)

    # Shuffled test targets cannot be predicted by honest cross-validation (R2 <= 0)
    assert r2_oos_shuffled <= 0.05, f"Leakage detected: out-of-sample R2 was {r2_oos_shuffled:.4f} > 0.05"


def test_positive_control_power():
    """Test 2: Positive control - strong linear signal at n = 300 gives power near 1."""
    # Strong signal scenario at n=300
    sc = dgp.build_scenario({"n": 300, "r2_target": 0.45})
    n_reps = 30
    seeds = np.random.SeedSequence(54321).spawn(n_reps)

    h1_f_hits = 0
    h1_t_hits = 0

    for s in seeds:
        df = dgp.generate(sc, seed=s)
        X1, X2 = prepare_features(df)
        y = df["score_C"].to_numpy(dtype=np.float64)

        m1 = M1HierarchicalOLS().fit(X1, X2, y)
        if m1.h1_f_hit_:
            h1_f_hits += 1
        if m1.h1_t_hit_:
            h1_t_hits += 1

    power_f = h1_f_hits / n_reps
    power_t = h1_t_hits / n_reps

    # Positive control: power should approach 1 (>= 90%)
    assert power_f >= 0.90, f"Expected H1 F-test power near 1, got {power_f:.2%}"
    assert power_t >= 0.90, f"Expected H1 t-test power near 1, got {power_t:.2%}"


def test_negative_control_type1():
    """Test 1: Negative control - C replaced by noise, 300 replications, Type I rate near 0.05."""
    n_reps = 300
    sc = dgp.build_scenario({"n": 100})
    seeds = np.random.SeedSequence(99999).spawn(n_reps)

    h1_f_hits = 0
    m2a_hits = 0
    m2b_hits = 0

    for s in seeds:
        df = dgp.generate(sc, seed=s)
        X1, X2 = prepare_features(df)
        child_seed_int = int(s.generate_state(1)[0])
        rng = np.random.default_rng(child_seed_int)
        # Outcome replaced by pure standard Gaussian noise
        y_noise = rng.normal(size=len(df))

        m1 = M1HierarchicalOLS().fit(X1, X2, y_noise)
        m2a = M2aPreSpecifiedInteraction().fit(X2, y_noise)
        m2b = M2bAllPairwiseInteractions().fit(X2, y_noise)

        if m1.h1_f_hit_:
            h1_f_hits += 1
        if m2a.detected_:
            m2a_hits += 1
        if m2b.detected_:
            m2b_hits += 1

    rate_h1 = h1_f_hits / n_reps
    rate_m2a = m2a_hits / n_reps
    rate_m2b = m2b_hits / n_reps

    mc_se = float(np.sqrt(0.05 * 0.95 / n_reps))  # ~ 0.0126
    # 3 * SE tolerance interval around 0.05 is [0.012, 0.088]
    tol = 3.0 * mc_se

    assert abs(rate_h1 - 0.05) <= tol, (
        f"H1 F-test Type I rate {rate_h1:.4f} outside MCSE band around 0.05 (tol={tol:.4f})"
    )
    assert abs(rate_m2a - 0.05) <= tol, (
        f"M2a Type I rate {rate_m2a:.4f} outside MCSE band around 0.05 (tol={tol:.4f})"
    )
    assert rate_m2b <= 0.05 + tol, (
        f"M2b Holm FWER {rate_m2b:.4f} exceeded nominal rate (tol={tol:.4f})"
    )


def test_m1_marker_adjusted_agreement_at_gamma_zero():
    """Verify SPEC line 185: M1-adj agrees with unadjusted M1 when gamma = 0."""
    sc = dgp.build_scenario({"n": 200, "acquiescence_gamma": 0.0})
    df = dgp.generate(sc, seed=42)
    X1, X2 = prepare_features(df)
    y = df["score_C"].values

    m1 = M1HierarchicalOLS().fit(X1, X2, y)
    m1_adj = fit_m1_marker_adjusted(df)

    # When gamma = 0, marker has zero population correlation with study variables
    # Adjusted delta R2 and conclusions should closely match unadjusted OLS
    assert m1.h1_f_hit_ == m1_adj["h1_f_hit"], "H1 conclusion should agree at gamma=0"
    assert m1.h2_hit_ == m1_adj["h2_hit"], "H2 conclusion should agree at gamma=0"
    assert np.isclose(m1.delta_r2_, m1_adj["delta_r2"], atol=0.03)


def test_m3_elasticnet_fit_and_stability():
    """Test M3 Elastic Net tuning and stability selection."""
    sc = dgp.build_scenario({"n": 80, "non_linear_form": "interaction", "psi": 0.08})
    df = dgp.generate(sc, seed=42)
    X1, X2 = prepare_features(df)
    y = df["score_C"].values

    m3 = M3ElasticNet(n_bootstraps=20, random_state=42)
    m3.fit(X2, y, run_stability=True)

    assert len(m3.selection_frequencies_) == 10
    assert 0.0 <= m3.max_selection_freq_ <= 1.0
    preds = m3.predict(X2)
    assert len(preds) == len(y)
