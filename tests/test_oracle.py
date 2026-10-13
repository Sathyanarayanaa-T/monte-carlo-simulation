"""Unit tests for oracle feature importance (src/oracle.py).

Hand-computable validation check:
In a purely linear scenario, oracle permutation importance ranks features identically
to the absolute standardised regression weights.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.oracle import FEATURE_NAMES, oracle_importance


def test_oracle_importance_purely_linear():
    """Hand-computable check: In purely linear scenario, oracle importance ranks features by |w|."""
    # Define scenario with strictly ordered linear weights and zero non-linear signal
    # Weights: A_tech (0.35) > B_i (0.25) > A_mkt (0.20) > B_ii (0.15) > B_iii (0.10)
    linear_weights = {
        "A_tech": 0.35,
        "A_mkt": 0.20,
        "B_i": 0.25,
        "B_ii": 0.15,
        "B_iii": 0.10,
    }

    scenario_override = {
        "linear_weights": linear_weights,
        "non_linear_form": "none",
        "psi": 0.0,
    }

    # Run oracle importance on population sample (n_pop=200,000 per SPEC.md Section 6.C)
    importances = oracle_importance(scenario_override, n_pop=200_000, seed=42, n_repeats=3)

    # Sort features by permutation importance (descending)
    ranked_by_importance = sorted(FEATURE_NAMES, key=lambda f: importances[f], reverse=True)

    # Sort features by absolute linear weights (descending)
    ranked_by_weights = sorted(FEATURE_NAMES, key=lambda f: abs(linear_weights[f]), reverse=True)

    expected_ranking = ["A_tech", "B_i", "A_mkt", "B_ii", "B_iii"]
    assert ranked_by_weights == expected_ranking
    assert ranked_by_importance == expected_ranking, (
        f"Oracle ranking mismatch: got {ranked_by_importance}, expected {expected_ranking}."
    )

    # Hand-computable property check:
    # Under linear DGP f(X) = k_lin * sum w_j X_j, permutation importance ~ 2 * (k_lin * w_j)^2.
    # Therefore, ratio of importances between any two features should approximately equal (w_a / w_b)^2.
    ratio_imp = importances["A_tech"] / importances["B_iii"]
    ratio_w_sq = (0.35 / 0.10) ** 2  # 3.5^2 = 12.25
    assert np.isclose(ratio_imp, ratio_w_sq, rtol=0.10), (
        f"Importance ratio {ratio_imp:.2f} should be close to weight-squared ratio {ratio_w_sq:.2f}"
    )


def test_top_k_recovery_relative_gap_h2_null():
    """Verify D13 relative-gap rule (v_k - v_{k+1}) / v_k >= 0.10 on h2_null scenario."""
    import src.dgp as dgp

    catalog = dgp.get_scenario_catalog()
    sc = catalog["h2_null_n100"]

    # Compute oracle importance
    imps = oracle_importance(sc, n_pop=50_000, seed=42, n_repeats=2)
    sorted_imps = sorted(imps.values(), reverse=True)

    v1, v2, v3 = sorted_imps[0], sorted_imps[1], sorted_imps[2]

    # In h2_null, A_tech weight = 0.30, while A_mkt, B_i, B_ii, B_iii weights = 0.15.
    # Top-1 is well-separated: (v1 - v2) / v1 >= 0.10
    rel_gap_1 = (v1 - v2) / v1
    top1_defined = bool(v1 > 0 and rel_gap_1 >= 0.10)
    assert top1_defined is True, f"Expected top1 to be defined, got rel_gap={rel_gap_1:.4f}"

    # Top-2 is tied at 0.15: (v2 - v3) / v2 < 0.10, so top-2 is undefined
    rel_gap_2 = (v2 - v3) / v2
    top2_defined = bool(v2 > 0 and rel_gap_2 >= 0.10)
    assert top2_defined is False, f"Expected top2 to be undefined (tied), got rel_gap={rel_gap_2:.4f}"
