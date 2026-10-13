"""Unit tests for measurement module (src/measurement.py).

Hand-computable validation checks:
1. HTMT is ~1 for two constructs measuring the same latent, and ~0 for independent ones.
2. Parallel analysis recovers 2 factors on simulated 2-factor data, and 0 on pure noise.
3. Lindell-Whitney marker adjustment with marker correlation 0 returns the input unchanged.
4. Harman's first factor variance share exceeds 0.50 on single-factor data.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.measurement import (
    harman_first_factor_share,
    htmt,
    lindell_whitney,
    parallel_analysis,
)


def test_htmt_hand_computable():
    """Hand-computable check: HTMT ~ 1 for identical latent, ~ 0 for independent latents."""
    rng = np.random.default_rng(42)
    n = 20_000

    # 1. Two constructs measuring the SAME latent z:
    # Items: x_1, x_2 for construct 1; y_1, y_2 for construct 2
    # All load with high loading 0.90 on common latent z:
    z = rng.normal(size=n)
    x1 = 0.90 * z + np.sqrt(1 - 0.90**2) * rng.normal(size=n)
    x2 = 0.90 * z + np.sqrt(1 - 0.90**2) * rng.normal(size=n)
    y1 = 0.90 * z + np.sqrt(1 - 0.90**2) * rng.normal(size=n)
    y2 = 0.90 * z + np.sqrt(1 - 0.90**2) * rng.normal(size=n)

    # Monotrait r_xx ~ 0.81, r_yy ~ 0.81. Heterotrait r_xy ~ 0.81.
    # HTMT = 0.81 / sqrt(0.81 * 0.81) = 1.00 exactly.
    df_same = pd.DataFrame({"x1": x1, "x2": x2, "y1": y1, "y2": y2})
    res_same = htmt(df_same, {"c1": ["x1", "x2"], "c2": ["y1", "y2"]})
    htmt_same = float(res_same[("c1", "c2")])
    assert np.isclose(htmt_same, 1.0, atol=0.03), f"Expected HTMT ~ 1.0, got {htmt_same:.4f}"

    # 2. Two INDEPENDENT constructs (z1 indep of z2):
    z1 = rng.normal(size=n)
    z2 = rng.normal(size=n)
    u1 = 0.80 * z1 + np.sqrt(1 - 0.80**2) * rng.normal(size=n)
    u2 = 0.80 * z1 + np.sqrt(1 - 0.80**2) * rng.normal(size=n)
    v1 = 0.80 * z2 + np.sqrt(1 - 0.80**2) * rng.normal(size=n)
    v2 = 0.80 * z2 + np.sqrt(1 - 0.80**2) * rng.normal(size=n)

    # Monotrait r_uu ~ 0.64, r_vv ~ 0.64. Heterotrait r_uv ~ 0.00.
    # HTMT = 0.00 / 0.64 = 0.00.
    df_indep = pd.DataFrame({"u1": u1, "u2": u2, "v1": v1, "v2": v2})
    res_indep = htmt(df_indep, {"c1": ["u1", "u2"], "c2": ["v1", "v2"]})
    htmt_indep = float(res_indep[("c1", "c2")])
    assert np.isclose(htmt_indep, 0.0, atol=0.03), f"Expected HTMT ~ 0.0, got {htmt_indep:.4f}"


def test_parallel_analysis_hand_computable():
    """Hand-computable check: parallel analysis recovers 2 factors on 2-factor data and 0 on noise."""
    rng = np.random.default_rng(12345)
    n = 500
    p = 6

    # 1. Two-factor data: 3 items for factor 1, 3 items for factor 2
    f1 = rng.normal(size=(n, 1))
    f2 = rng.normal(size=(n, 1))
    # High loadings (0.90): lambda_1 ~ 2.6, lambda_2 ~ 2.6 >> 95th percentile null eigenvalues (~1.2)
    items_f1 = 0.90 * f1 + np.sqrt(1 - 0.90**2) * rng.normal(size=(n, 3))
    items_f2 = 0.90 * f2 + np.sqrt(1 - 0.90**2) * rng.normal(size=(n, 3))
    X_2factor = np.hstack([items_f1, items_f2])

    n_factors_2 = parallel_analysis(X_2factor, n_iter=500, percentile=95.0, seed=100)
    assert n_factors_2 == 2, f"Expected 2 factors for 2-factor data, got {n_factors_2}"

    # 2. Pure noise data: 6 independent Gaussian noise variables
    # At the 95th percentile, sample eigenvalues of null noise should not exceed null threshold
    X_noise = rng.normal(size=(n, p))
    n_factors_noise = parallel_analysis(X_noise, n_iter=500, percentile=95.0, seed=42)
    assert n_factors_noise == 0, f"Expected 0 factors for pure noise data, got {n_factors_noise}"


def test_lindell_whitney_hand_computable():
    """Hand-computable check: Lindell-Whitney with marker_corr=0 returns input unchanged."""
    # 1. Scalar check: (0.45 - 0) / (1 - 0) = 0.45
    r_scalar = 0.45
    r_adj_0 = lindell_whitney(r_scalar, marker_corr=0.0)
    assert np.isclose(r_adj_0, r_scalar), f"Expected {r_scalar}, got {r_adj_0}"

    # Hand-computable with non-zero marker: (0.40 - 0.10) / (1.0 - 0.10) = 0.30 / 0.90 = 1/3
    r_adj_nonzero = lindell_whitney(0.40, marker_corr=0.10)
    assert np.isclose(r_adj_nonzero, 1.0 / 3.0), f"Expected 1/3, got {r_adj_nonzero}"

    # 2. Matrix check: 2x2 correlation matrix with marker_corr=0 returns matrix unchanged
    R = np.array([[1.0, 0.35], [0.35, 1.0]])
    R_adj_0 = lindell_whitney(R, marker_corr=0.0)
    assert np.allclose(R_adj_0, R), "Correlation matrix should be unchanged when marker_corr=0"

    # 3. DataFrame check: maintains index and columns
    df_R = pd.DataFrame(R, index=["A", "B"], columns=["A", "B"])
    df_adj = lindell_whitney(df_R, marker_corr=0.0)
    assert isinstance(df_adj, pd.DataFrame)
    assert np.allclose(df_adj.values, R)


def test_harman_first_factor_share_hand_computable():
    """Hand-computable check: Harman's first factor share exceeds 0.50 on single-factor data."""
    rng = np.random.default_rng(42)
    n = 1000
    p = 4

    # Single-factor data: 4 items measuring latent factor z with loading 0.90
    # Expected correlation between any two items is ~ 0.90^2 = 0.81
    # Expected first eigenvalue: lambda_1 = 1 + (p - 1) * 0.81 = 1 + 3 * 0.81 = 3.43
    # Expected variance share: lambda_1 / p = 3.43 / 4 = 0.8575 > 0.50
    z = rng.normal(size=(n, 1))
    items_1factor = 0.90 * z + np.sqrt(1 - 0.90**2) * rng.normal(size=(n, p))

    share = harman_first_factor_share(items_1factor)
    assert share > 0.50, f"Harman variance share should exceed 0.50, got {share:.4f}"
    assert np.isclose(share, 0.8575, atol=0.05), f"Expected share ~ 0.8575, got {share:.4f}"
