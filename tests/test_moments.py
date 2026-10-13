"""Validation check 1: Latent factor moments at n = 200,000 (SPEC.md Section 9).

Checks:
1. Contextual factors B-i, B-ii, B-iii correlations match targets (0.35, 0.30, 0.25)
   and have unit variance within Monte Carlo error.
2. Individual factors A_tech and A_mkt have unit variance and residual correlation 0.30.
3. Sensitivity contextual correlations (0.10 and 0.50) match specification.
4. Latent G correlation with C matches target ~0.50, with unit variance.
5. Covariates: Semester is categorical {6, 7, 8} and binary flags have ~40% prevalence.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.dgp as dgp


def test_base_contextual_factor_moments(large_n_base_df):
    """Test that B-i, B-ii, B-iii variances and correlations match SPEC.md Section 3.1."""
    df = large_n_base_df
    b1 = df["latent_B_i"].values
    b2 = df["latent_B_ii"].values
    b3 = df["latent_B_iii"].values

    # Unit variance
    assert np.var(b1) == pytest.approx(1.0, abs=0.015)
    assert np.var(b2) == pytest.approx(1.0, abs=0.015)
    assert np.var(b3) == pytest.approx(1.0, abs=0.015)

    # Contextual correlations
    corr_12 = np.corrcoef(b1, b2)[0, 1]
    corr_13 = np.corrcoef(b1, b3)[0, 1]
    corr_23 = np.corrcoef(b2, b3)[0, 1]

    assert corr_12 == pytest.approx(0.35, abs=0.01)
    assert corr_13 == pytest.approx(0.30, abs=0.01)
    assert corr_23 == pytest.approx(0.25, abs=0.01)


def test_individual_factor_variances(large_n_base_df):
    """Test that A_tech and A_mkt have unit variance and match residual correlation (Section 3.2)."""
    df = large_n_base_df
    a_tech = df["latent_A_tech"].values
    a_mkt = df["latent_A_mkt"].values
    b1 = df["latent_B_i"].values
    b2 = df["latent_B_ii"].values
    b3 = df["latent_B_iii"].values

    # Unit variance
    assert np.var(a_tech) == pytest.approx(1.0, abs=0.015)
    assert np.var(a_mkt) == pytest.approx(1.0, abs=0.015)

    # Check residual correlation: u1 = A_tech - 0.35*B_i - 0.15*B_iii
    u1 = a_tech - (0.35 * b1 + 0.15 * b3)
    u2 = a_mkt - (0.35 * b2 + 0.15 * b3)
    corr_u = np.corrcoef(u1, u2)[0, 1]
    assert corr_u == pytest.approx(0.30, abs=0.015)


def test_sensitivity_contextual_correlations():
    """Test contextual correlations under low (0.10) and high (0.50) sensitivity settings."""
    # Low setting (all 0.10)
    sc_low = dgp.build_scenario({
        "n": 200_000,
        "latent_correlations": {
            "B_i_B_ii": 0.10,
            "B_i_B_iii": 0.10,
            "B_ii_B_iii": 0.10,
        },
    })
    df_low = dgp.generate(sc_low, seed=101)
    b1_l = df_low["latent_B_i"].values
    b2_l = df_low["latent_B_ii"].values
    b3_l = df_low["latent_B_iii"].values

    assert np.corrcoef(b1_l, b2_l)[0, 1] == pytest.approx(0.10, abs=0.01)
    assert np.corrcoef(b1_l, b3_l)[0, 1] == pytest.approx(0.10, abs=0.01)
    assert np.corrcoef(b2_l, b3_l)[0, 1] == pytest.approx(0.10, abs=0.01)
    assert np.var(df_low["latent_A_tech"]) == pytest.approx(1.0, abs=0.015)
    assert np.var(df_low["latent_A_mkt"]) == pytest.approx(1.0, abs=0.015)

    # High setting (all 0.50)
    sc_high = dgp.build_scenario({
        "n": 200_000,
        "latent_correlations": {
            "B_i_B_ii": 0.50,
            "B_i_B_iii": 0.50,
            "B_ii_B_iii": 0.50,
        },
    })
    df_high = dgp.generate(sc_high, seed=102)
    b1_h = df_high["latent_B_i"].values
    b2_h = df_high["latent_B_ii"].values
    b3_h = df_high["latent_B_iii"].values

    assert np.corrcoef(b1_h, b2_h)[0, 1] == pytest.approx(0.50, abs=0.01)
    assert np.corrcoef(b1_h, b3_h)[0, 1] == pytest.approx(0.50, abs=0.01)
    assert np.corrcoef(b2_h, b3_h)[0, 1] == pytest.approx(0.50, abs=0.01)
    assert np.var(df_high["latent_A_tech"]) == pytest.approx(1.0, abs=0.015)
    assert np.var(df_high["latent_A_mkt"]) == pytest.approx(1.0, abs=0.015)


def test_latent_g_numerical_calibration(large_n_base_df):
    """Test that latent G achieves target correlation ~0.50 with C and unit variance."""
    df = large_n_base_df
    g = df["latent_G"].values
    c = df["latent_C"].values

    # Achieved correlation with C
    corr_gc = np.corrcoef(g, c)[0, 1]
    assert corr_gc == pytest.approx(0.50, abs=0.015)

    # Unit variance
    assert np.var(g) == pytest.approx(1.0, abs=0.015)


def test_covariates_moments(large_n_base_df):
    """Test semester distribution and prior-exposure flags prevalence."""
    df = large_n_base_df
    sem = df["semester"].values
    proto = df["covar_prototype"].values
    hack = df["covar_hackathon"].values
    intern = df["covar_internship"].values

    # Semester: values in {6, 7, 8}, each roughly 33.3%
    assert set(np.unique(sem)) == {6, 7, 8}
    for val in [6, 7, 8]:
        prop = np.mean(sem == val)
        assert prop == pytest.approx(1.0 / 3.0, abs=0.01)

    # Binary flags: prevalence roughly 40% (target_prevalence: 0.40)
    assert np.mean(proto) == pytest.approx(0.40, abs=0.015)
    assert np.mean(hack) == pytest.approx(0.40, abs=0.015)
    assert np.mean(intern) == pytest.approx(0.40, abs=0.015)
