"""Validation check 3: Variance shares and target R^2 scaling (SPEC.md Section 9 & Section 3.3).

Verifies the mathematical variance scaling:
- Total variance of C is 1.0 (standardized).
- Signal (linear + non-linear) explains target R^2.
- The non-linear part carries variance share psi.
- Noise carries variance share 1 - R^2.
Tested across all non-linear forms (none, interaction, threshold) and effect sizes.
"""

from __future__ import annotations

import numpy as np
import pytest

import src.dgp as dgp


@pytest.mark.parametrize(
    "nl_form,psi,r2_target",
    [
        ("none", 0.0, 0.35),
        ("interaction", 0.03, 0.35),
        ("interaction", 0.08, 0.35),
        ("threshold", 0.03, 0.35),
        ("threshold", 0.08, 0.35),
        ("none", 0.0, 0.20),
        ("interaction", 0.08, 0.20),
        ("threshold", 0.08, 0.20),
    ],
)
def test_variance_scaling_and_psi(nl_form: str, psi: float, r2_target: float):
    """Test that Var(Signal)/Var(C) == R^2 and Var(NonLinear)/Var(C) == psi."""
    sc = dgp.build_scenario({
        "n": 200_000,
        "non_linear_form": nl_form,
        "psi": psi,
        "r2_target": r2_target,
    })
    df = dgp.generate(sc, seed=42)

    c = df["latent_C"].values
    lin = df["latent_C_linear"].values
    nl = df["latent_C_nonlinear"].values
    noise = df["latent_C_noise"].values
    signal = lin + nl

    var_c = float(np.var(c))
    var_signal = float(np.var(signal))
    var_nl = float(np.var(nl))
    var_noise = float(np.var(noise))

    # 1. Total variance of C should be 1.0
    assert var_c == pytest.approx(1.0, abs=0.015)

    # 2. Signal variance share should equal target R^2
    signal_share = var_signal / var_c
    assert signal_share == pytest.approx(r2_target, abs=0.01)

    # 3. Non-linear variance share should equal target psi
    nl_share = var_nl / var_c
    assert nl_share == pytest.approx(psi, abs=0.005)

    # 4. Noise variance share should equal 1 - R^2
    noise_share = var_noise / var_c
    assert noise_share == pytest.approx(1.0 - r2_target, abs=0.01)


def test_null_settings_variance_scaling():
    """Test H1-null and H2-null linear weight configurations achieve target R^2."""
    config = dgp.load_config()
    h1_weights = config["grid_presets"]["linear_weights"]["h1_null"]
    h2_weights = config["grid_presets"]["linear_weights"]["h2_null"]

    for name, weights in [("h1_null", h1_weights), ("h2_null", h2_weights)]:
        sc = dgp.build_scenario({
            "n": 200_000,
            "linear_weights": weights,
            "r2_target": 0.35,
            "psi": 0.0,
            "non_linear_form": "none",
        })
        df = dgp.generate(sc, seed=123)

        c = df["latent_C"].values
        lin = df["latent_C_linear"].values
        signal_share = np.var(lin) / np.var(c)
        assert signal_share == pytest.approx(0.35, abs=0.01), f"Failed for {name}"
