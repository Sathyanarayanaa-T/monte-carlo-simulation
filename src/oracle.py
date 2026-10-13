"""Oracle feature importance evaluation on the noise-free data generating function.

Implements SPEC.md Section 6.C (lines 125-126):
- Oracle importance: Computes true feature importance from the noise-free DGP function
  f(A_tech, A_mkt, B_i, B_ii, B_iii) on a large population sample (n = 200,000) using
  permutation importance (Breiman, 2001).
- Provides ground truth feature importances and rankings against which empirical methods
  (M1, M3, M4, M5, M6) are benchmarked via Kendall's tau and top-1 recovery.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd

from src.dgp import build_scenario, calibrate_scenario, generate


FEATURE_NAMES = ["A_tech", "A_mkt", "B_i", "B_ii", "B_iii"]


def evaluate_noise_free_target(
    X: np.ndarray,
    scenario_dict: dict[str, Any],
    cal: dict[str, Any],
) -> np.ndarray:
    """Evaluate the noise-free outcome function f(X) for latent predictors.

    Formula (SPEC.md Section 3.3):
    ------------------------------
    f(X) = k_lin * L_0(X) + k_nl * N_0(X)

    where:
    - X has columns [A_tech, A_mkt, B_i, B_ii, B_iii]
    - L_0 = sum_{j} beta_j * X_j
    - N_0 is the interaction or threshold functional form (orthogonalised if configured).
    """
    lin_w = scenario_dict["linear_weights"]
    w = np.array([
        float(lin_w["A_tech"]),
        float(lin_w["A_mkt"]),
        float(lin_w["B_i"]),
        float(lin_w["B_ii"]),
        float(lin_w["B_iii"]),
    ], dtype=np.float64)

    L0 = X @ w
    k_lin = float(cal["k_lin"])
    k_nl = float(cal["k_nl"])
    linear_part = k_lin * L0

    nl_form = str(scenario_dict.get("non_linear_form", "none"))
    psi = float(scenario_dict.get("psi", 0.0))
    orth_nl = bool(scenario_dict.get("orthogonalise_nonlinear", True))

    if nl_form == "none" or psi <= 0.0 or k_nl == 0.0:
        non_linear_part = np.zeros(X.shape[0], dtype=np.float64)
    elif orth_nl and cal.get("nl_orth_coefs") is not None:
        if nl_form == "interaction":
            raw_nl = X[:, 0] * X[:, 2]  # A_tech * B_i
        elif nl_form == "threshold":
            c_thresh = float(scenario_dict["threshold_cutoff"])
            raw_nl = X[:, 4] * (X[:, 0] < c_thresh)  # B_iii * 1[A_tech < c_thresh]
        else:
            raise ValueError(f"Unknown non_linear_form: {nl_form}")

        X_const = np.column_stack([np.ones(X.shape[0]), X])
        non_linear_part = k_nl * (raw_nl - X_const @ cal["nl_orth_coefs"])
    elif nl_form == "interaction":
        # Unorthogonalised z-scored interaction
        P = X[:, 0] * X[:, 2]
        E_N0 = float(cal["E_N0"])
        var_N0 = float(cal["var_N0"])
        non_linear_part = k_nl * ((P - E_N0) / np.sqrt(var_N0))
    elif nl_form == "threshold":
        # Unorthogonalised centered threshold
        c_thresh = float(scenario_dict["threshold_cutoff"])
        T = X[:, 4] * (X[:, 0] < c_thresh)
        E_N0 = float(cal["E_N0"])
        non_linear_part = k_nl * (T - E_N0)
    else:
        raise ValueError(f"Unknown non_linear_form: {nl_form}")

    return linear_part + non_linear_part


def oracle_importance(
    scenario: dict[str, Any] | str | None = None,
    n_pop: int = 200_000,
    seed: int = 42,
    n_repeats: int = 5,
) -> dict[str, float]:
    """Compute permutation feature importance of the noise-free true DGP function.

    Formula & Procedure (Breiman, 2001; SPEC.md Section 6.C line 125):
    ------------------------------------------------------------------
    1. Sample population latent factors X ~ P(A_tech, A_mkt, B_i, B_ii, B_iii) of size n_pop.
    2. Compute the noise-free true outcome y = f(X).
    3. For each predictor j in {A_tech, A_mkt, B_i, B_ii, B_iii}:
       a. For r = 1 .. n_repeats, randomly permute the j-th column: X^(pi_j, r).
       b. Compute permuted prediction: y_hat^(pi_j, r) = f(X^(pi_j, r)).
       c. Compute mean squared prediction error increase:
          Delta_MSE_j,r = (1 / n_pop) * sum_{i=1}^{n_pop} (y_i - y_hat^(pi_j, r)_i)^2
       d. Average Delta_MSE_j across repeats:
          Importance_j = (1 / n_repeats) * sum_{r=1}^{n_repeats} Delta_MSE_j,r

    Analytical property:
    In a purely linear model f(X) = k_lin * sum_j w_j X_j with Var(X_j) = 1:
        E[(X_j - X_j^pi)^2] = 2 * Var(X_j) = 2
        Importance_j = 2 * (k_lin * w_j)^2
    Hence, Importance_j is strictly monotonic in |w_j|, ranking features identically
    to the absolute standardised regression weights.

    Parameters:
    - scenario: scenario dictionary, name, or None (defaults to base scenario).
    - n_pop: population sample size (default: 200,000 per SPEC.md).
    - seed: random seed for population generation and permutation shuffling.
    - n_repeats: number of permutation shuffle repeats per feature (default: 5).

    Returns:
    - dict[str, float]: mapping feature name -> permutation importance score (MSE loss increase).
    """
    if scenario is None:
        sc = build_scenario({"n": n_pop})
    elif isinstance(scenario, str):
        sc = build_scenario({"name": scenario, "n": n_pop})
    else:
        sc = build_scenario({**scenario, "n": n_pop})

    cal = calibrate_scenario(sc)

    # Generate population sample via DGP
    pop_df = generate(sc, seed=seed)
    latent_cols = [f"latent_{col}" for col in FEATURE_NAMES]
    X_pop = pop_df[latent_cols].to_numpy(dtype=np.float64)

    # True noise-free outcome
    y_true = evaluate_noise_free_target(X_pop, sc, cal)

    # Shuffling RNG
    perm_rng = np.random.default_rng(seed + 1000)

    importances: dict[str, float] = {}

    for j, name in enumerate(FEATURE_NAMES):
        losses: list[float] = []
        for _ in range(n_repeats):
            X_perm = X_pop.copy()
            X_perm[:, j] = perm_rng.permutation(X_perm[:, j])
            y_perm = evaluate_noise_free_target(X_perm, sc, cal)
            loss = float(np.mean((y_true - y_perm) ** 2))
            losses.append(loss)
        importances[name] = float(np.mean(losses))

    return importances
