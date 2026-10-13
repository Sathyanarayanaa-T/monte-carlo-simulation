"""Shared fixtures and metric functions for DGP validation tests."""

from __future__ import annotations

import numpy as np
import pytest
from sklearn.decomposition import FactorAnalysis

import src.dgp as dgp


@pytest.fixture(scope="session")
def base_scenario():
    """Build base scenario from config."""
    return dgp.build_scenario()


@pytest.fixture(scope="session")
def large_n_base_df():
    """Generate a single large n (200,000) dataset under base scenario for moment tests."""
    sc = dgp.build_scenario({"n": 200_000})
    return dgp.generate(sc, seed=42)


def calc_cronbach_alpha(items: np.ndarray) -> float:
    """Compute empirical Cronbach's alpha for item columns."""
    k = items.shape[1]
    if k <= 1:
        return 1.0
    item_vars = np.var(items, axis=0, ddof=1)
    tot_var = np.var(np.sum(items, axis=1), ddof=1)
    if tot_var <= 0:
        return 0.0
    return float((k / (k - 1)) * (1.0 - np.sum(item_vars) / tot_var))


def calc_mcdonald_omega(items: np.ndarray) -> float:
    """Compute empirical McDonald's omega using single-factor decomposition."""
    k = items.shape[1]
    if k <= 1:
        return 1.0
    fa = FactorAnalysis(n_components=1, random_state=42)
    fa.fit(items)
    loadings = fa.components_[0]
    uniquenesses = fa.noise_variance_
    sum_loadings = np.sum(np.abs(loadings))
    numerator = sum_loadings**2
    denominator = numerator + np.sum(uniquenesses)
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def calc_htmt(df, cols_x: list[str], cols_y: list[str]) -> float:
    """Compute Heterotrait-Monotrait Ratio of Correlations (HTMT)."""
    p = len(cols_x)
    q = len(cols_y)
    mat_x = df[cols_x].values
    mat_y = df[cols_y].values

    # Heterotrait cross-correlations (absolute)
    corr_full = np.corrcoef(mat_x.T, mat_y.T)
    r_xy = np.abs(corr_full[:p, p:])
    mean_r_xy = np.mean(r_xy)

    # Monotrait intra-correlations X
    r_xx = np.abs(corr_full[:p, :p])
    triu_x = np.triu_indices(p, k=1)
    mean_r_xx = np.mean(r_xx[triu_x]) if p > 1 else 1.0

    # Monotrait intra-correlations Y
    r_yy = np.abs(corr_full[p:, p:])
    triu_y = np.triu_indices(q, k=1)
    mean_r_yy = np.mean(r_yy[triu_y]) if q > 1 else 1.0

    denom = np.sqrt(mean_r_xx * mean_r_yy)
    if denom <= 0:
        return 0.0
    return float(mean_r_xy / denom)
