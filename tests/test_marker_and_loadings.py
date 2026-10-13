"""Validation of marker items, loading persistence, and scoring rules (SPEC.md Section 3.4).

Checks:
1. Marker items load only on the acquiescence factor (gamma > 0), with no substantive loading.
2. Loadings are drawn once per scenario, not per replication.
3. Reverse-scoring and index computation rules match Section 3.4.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.dgp as dgp


def test_marker_items_load_only_on_acquiescence():
    """Test that marker variable items load only on acquiescence factor and have 0 substantive loading."""
    # When gamma = 0: marker items are pure noise, uninformative
    sc_gamma0 = dgp.build_scenario({"n": 200_000, "acquiescence_gamma": 0.0})
    df0 = dgp.generate(sc_gamma0, seed=42)

    a0 = df0["latent_acquiescence"].values
    c0 = df0["latent_C"].values
    m1_0 = df0["M_1"].values

    assert np.corrcoef(m1_0, a0)[0, 1] == pytest.approx(0.0, abs=0.015)
    assert np.corrcoef(m1_0, c0)[0, 1] == pytest.approx(0.0, abs=0.015)

    # When gamma = 0.3: marker items load on acquiescence factor with weight gamma
    sc_gamma3 = dgp.build_scenario({"n": 200_000, "acquiescence_gamma": 0.30})
    df3 = dgp.generate(sc_gamma3, seed=42)

    a3 = df3["latent_acquiescence"].values
    c3 = df3["latent_C"].values
    b1_3 = df3["latent_B_i"].values

    for col in ["M_1", "M_2", "M_3", "M_4"]:
        m_vals = df3[col].values
        # Correlates with acquiescence factor
        corr_a = np.corrcoef(m_vals, a3)[0, 1]
        assert corr_a > 0.20, f"Marker item {col} should correlate with acquiescence factor, got {corr_a:.3f}"

        # Substantive correlations with C, B_i remain zero within Monte Carlo error
        corr_c = np.corrcoef(m_vals, c3)[0, 1]
        corr_b = np.corrcoef(m_vals, b1_3)[0, 1]
        assert corr_c == pytest.approx(0.0, abs=0.015)
        assert corr_b == pytest.approx(0.0, abs=0.015)


def test_loadings_drawn_once_per_scenario_fixed_across_replications():
    """Test that loadings are fixed across replications within a scenario."""
    sc = dgp.build_scenario({"n": 50})

    # Draw loadings explicitly or via first call
    loadings_initial = dgp.draw_scenario_loadings(sc)
    loadings_snapshot = {k: v.copy() for k, v in loadings_initial.items()}

    # Run multiple replications with different seeds
    for seed in [1, 2, 3, 100, 2026]:
        _ = dgp.generate(sc, seed=seed)
        # Ensure loadings in the scenario dictionary remain strictly identical
        for k, v in loadings_snapshot.items():
            assert np.array_equal(sc["loadings"][k], v), f"Loadings for {k} mutated on seed {seed}!"


def test_scoring_and_index_computations(large_n_base_df):
    """Test sub-construct means and block indices match exact item means after reverse-scoring."""
    df = large_n_base_df

    # Sub-construct score matches item mean
    for construct, count in [("A_tech", 5), ("A_mkt", 4), ("B_i", 5), ("B_ii", 4), ("B_iii", 6), ("C", 4)]:
        cols = [f"{construct}_{j + 1}" for j in range(count)]
        expected_score = df[cols].mean(axis=1)
        actual_score = df[f"score_{construct}"]
        assert np.allclose(actual_score, expected_score), f"Score mismatch for {construct}"

    # Individual Resilience Index = mean of A-i and A-ii items (9 items total)
    a_cols = [f"A_tech_{j + 1}" for j in range(5)] + [f"A_mkt_{j + 1}" for j in range(4)]
    expected_ind = df[a_cols].mean(axis=1)
    assert np.allclose(df["index_individual"], expected_ind), "Individual index mismatch"

    # Contextual Exposure Index = mean of Part B items (15 items total)
    b_cols = (
        [f"B_i_{j + 1}" for j in range(5)]
        + [f"B_ii_{j + 1}" for j in range(4)]
        + [f"B_iii_{j + 1}" for j in range(6)]
    )
    expected_ctx = df[b_cols].mean(axis=1)
    assert np.allclose(df["index_contextual"], expected_ctx), "Contextual index mismatch"
