"""Validation of added HTMT false-flag check (SPEC.md Section 9 line 185 & Section 5).

Checks:
1. With kappa = 0 and cross-loading = 0, the HTMT false-flag rate (share > 0.85)
   should stay low (<= 5%) across replications at n = 150.
2. Population HTMT values for (A-i, B-i), (A-ii, B-ii), and (C, G) are safely below 0.85.
3. Sensitivity settings (cross-loading 0.3 and contaminated kappa 0.4) show elevated HTMT.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.dgp as dgp
from tests.conftest import calc_htmt


def test_htmt_false_flag_rate_base_dgp():
    """Test that HTMT false-flag rate (share > 0.85) stays <= 5% in base setting at n=150."""
    sc = dgp.build_scenario({"n": 150, "cross_loading": 0.0, "kappa": 0.0})

    # Spawn 100 independent child seeds (Rule 9)
    seeds = np.random.SeedSequence(54321).spawn(100)

    cols_a_tech = [f"A_tech_{j + 1}" for j in range(5)]
    cols_b_i    = [f"B_i_{j + 1}" for j in range(5)]
    cols_a_mkt  = [f"A_mkt_{j + 1}" for j in range(4)]
    cols_b_ii   = [f"B_ii_{j + 1}" for j in range(4)]
    cols_c      = [f"C_{j + 1}" for j in range(4)]
    cols_g      = [f"G_{j + 1}" for j in range(4)]

    flags_a_tech_b_i = 0
    flags_a_mkt_b_ii = 0
    flags_c_g        = 0

    htmt_values_ab: list[float] = []
    htmt_values_cg: list[float] = []

    for s in seeds:
        df = dgp.generate(sc, seed=s)
        h_ab = calc_htmt(df, cols_a_tech, cols_b_i)
        h_am = calc_htmt(df, cols_a_mkt, cols_b_ii)
        h_cg = calc_htmt(df, cols_c, cols_g)

        htmt_values_ab.append(h_ab)
        htmt_values_cg.append(h_cg)

        if h_ab > 0.85:
            flags_a_tech_b_i += 1
        if h_am > 0.85:
            flags_a_mkt_b_ii += 1
        if h_cg > 0.85:
            flags_c_g += 1

    rate_ab = flags_a_tech_b_i / len(seeds)
    rate_am = flags_a_mkt_b_ii / len(seeds)
    rate_cg = flags_c_g / len(seeds)

    # SPEC line 185: "HTMT false-flag rate should stay low (about 5 percent or less)"
    assert rate_ab <= 0.05, f"HTMT false-flag rate for (A-i, B-i) was {rate_ab:.2%} > 5%"
    assert rate_am <= 0.05, f"HTMT false-flag rate for (A-ii, B-ii) was {rate_am:.2%} > 5%"
    assert rate_cg <= 0.05, f"HTMT false-flag rate for (C, G) was {rate_cg:.2%} > 5%"


def test_htmt_sensitivity_shifts(large_n_base_df):
    """Test that cross-loading (0.3) and contamination (kappa=0.4) noticeably elevate HTMT."""
    df_base = large_n_base_df

    cols_a_tech = [f"A_tech_{j + 1}" for j in range(5)]
    cols_b_i    = [f"B_i_{j + 1}" for j in range(5)]
    cols_c      = [f"C_{j + 1}" for j in range(4)]
    cols_g      = [f"G_{j + 1}" for j in range(4)]

    htmt_base_ab = calc_htmt(df_base, cols_a_tech, cols_b_i)
    htmt_base_cg = calc_htmt(df_base, cols_c, cols_g)

    # Cross-loading sensitivity
    sc_cross = dgp.build_scenario({"n": 200_000, "cross_loading": 0.30})
    df_cross = dgp.generate(sc_cross, seed=42)
    htmt_cross_ab = calc_htmt(df_cross, cols_a_tech, cols_b_i)

    # Outcome contamination sensitivity
    sc_contam = dgp.build_scenario({"n": 200_000, "kappa": 0.40})
    df_contam = dgp.generate(sc_contam, seed=42)
    htmt_contam_cg = calc_htmt(df_contam, cols_c, cols_g)

    assert htmt_cross_ab > htmt_base_ab + 0.15, (
        f"Cross-loading should elevate HTMT: base={htmt_base_ab:.3f}, cross={htmt_cross_ab:.3f}"
    )
    assert htmt_contam_cg > htmt_base_cg + 0.15, (
        f"Contamination should elevate HTMT: base={htmt_base_cg:.3f}, contam={htmt_contam_cg:.3f}"
    )
