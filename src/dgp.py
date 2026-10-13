"""Data-generating process (DGP) for the FEDRS Monte Carlo study.

Implements SPEC.md Section 3:
- 3.1 Latent contextual factors (B-i, B-ii, B-iii)
- 3.2 Latent individual factors (A_tech, A_mkt) with correlated residuals
- 3.3 Outcome (Commitment Intention C) with linear + non-linear + noise scaling,
      and General Startup Intent latent G
- 3.4 Measurement layer (congeneric item models, factor loadings, reverse keying,
      acquiescence common-method factor, ceiling shift, Likert discretisation,
      cross-loading, outcome contamination, and marker variable M)
- 3.5 Covariates (semester, prior-exposure binary flags)
"""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml


# =============================================================================
# Configuration Loading & Helpers
# =============================================================================

def load_config(config_path: str | Path = "config/scenarios.yaml") -> dict[str, Any]:
    """Load scenarios.yaml configuration."""
    path = Path(config_path)
    if not path.is_absolute():
        # Search relative to repo root if possible
        repo_root = Path(__file__).resolve().parent.parent
        candidate = repo_root / path
        if candidate.exists():
            path = candidate
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def solve_logistic_intercept(slope: float, target_prevalence: float) -> float:
    """Solve for logistic intercept alpha such that E[sigma(slope * Z + alpha)] = target_prevalence for Z ~ N(0, 1)."""
    import scipy.integrate as integrate
    import scipy.optimize as optimize
    import scipy.stats as stats

    def obj(alpha: float) -> float:
        val, _ = integrate.quad(
            lambda z: (1.0 / (1.0 + np.exp(-(slope * z + alpha)))) * stats.norm.pdf(z),
            -8.0, 8.0,
        )
        return val - target_prevalence

    res = optimize.root_scalar(obj, bracket=[-10.0, 10.0], method="brentq")
    return float(res.root)


def build_scenario(
    scenario_override: dict[str, Any] | None = None,
    config_path: str | Path = "config/scenarios.yaml",
) -> dict[str, Any]:
    """Build a complete scenario dictionary by deep-merging overrides into base_parameters.

    Parameters live in config/scenarios.yaml to eliminate magic numbers (Rule 8).
    """
    raw_config = load_config(config_path)
    base_params = copy.deepcopy(raw_config["base_parameters"])

    if scenario_override is None:
        scenario = base_params
    elif isinstance(scenario_override, str):
        catalog = get_scenario_catalog(config_path)
        if scenario_override in catalog:
            return copy.deepcopy(catalog[scenario_override])
        scenario = copy.deepcopy(base_params)
        scenario["name"] = scenario_override
    else:
        scenario = copy.deepcopy(base_params)
        _deep_update(scenario, scenario_override)

    # Solve logistic intercept if not explicitly provided in scenario
    pe_cfg = scenario.get("covariates", {}).get("prior_exposure", {})
    if "slope" in pe_cfg and "target_prevalence" in pe_cfg:
        if "intercept" not in pe_cfg or pe_cfg["intercept"] is None:
            pe_cfg["intercept"] = solve_logistic_intercept(
                float(pe_cfg["slope"]), float(pe_cfg["target_prevalence"])
            )

    return scenario


def get_scenario_catalog(config_path: str | Path = "config/scenarios.yaml") -> dict[str, dict[str, Any]]:
    """Build the catalog of all 60 scenarios from SPEC.md Section 4.

    Returns:
    - dict[scenario_name, scenario_dict]

    Core grid: 28 scenarios (4 sample sizes x (5 non-linearity cells + 2 nulls)).
    Sensitivity grid: 32 scenarios (2 sample sizes x 2 non-linearity cells x 8 sensitivity factors).
    """
    catalog: dict[str, dict[str, Any]] = {}
    scenario_idx = 0

    # 1. Core grid (28 scenarios)
    sample_sizes = [50, 100, 150, 300]
    nl_cells = {
        "none": {"non_linear_form": "none", "psi": 0.0},
        "inter_small": {"non_linear_form": "interaction", "psi": 0.03},
        "inter_med": {"non_linear_form": "interaction", "psi": 0.08},
        "thresh_small": {"non_linear_form": "threshold", "psi": 0.03},
        "thresh_med": {"non_linear_form": "threshold", "psi": 0.08},
    }

    h1_null_weights = {"A_tech": 0.0, "A_mkt": 0.0, "B_i": 0.20, "B_ii": 0.10, "B_iii": 0.15}
    h2_null_weights = {"A_tech": 0.30, "A_mkt": 0.15, "B_i": 0.15, "B_ii": 0.15, "B_iii": 0.15}

    for n in sample_sizes:
        for cell_name, cell_overrides in nl_cells.items():
            sc_name = f"base_n{n}_{cell_name}"
            overrides = {"name": sc_name, "scenario_idx": scenario_idx, "n": n, **cell_overrides}
            catalog[sc_name] = build_scenario(overrides, config_path=config_path)
            scenario_idx += 1

        # H1-null
        sc_name = f"h1_null_n{n}"
        overrides = {
            "name": sc_name,
            "scenario_idx": scenario_idx,
            "n": n,
            "linear_weights": h1_null_weights,
            "non_linear_form": "none",
            "psi": 0.0,
        }
        catalog[sc_name] = build_scenario(overrides, config_path=config_path)
        scenario_idx += 1

        # H2-null
        sc_name = f"h2_null_n{n}"
        overrides = {
            "name": sc_name,
            "scenario_idx": scenario_idx,
            "n": n,
            "linear_weights": h2_null_weights,
            "non_linear_form": "none",
            "psi": 0.0,
        }
        catalog[sc_name] = build_scenario(overrides, config_path=config_path)
        scenario_idx += 1

    # 2. Sensitivity grid (32 scenarios)
    sens_sample_sizes = [100, 150]
    sens_nl_cells = {
        "none": {"non_linear_form": "none", "psi": 0.0},
        "inter_med": {"non_linear_form": "interaction", "psi": 0.08},
    }
    sens_factors = {
        "corr_high": {"latent_correlations": {"B_i_B_ii": 0.50, "B_i_B_iii": 0.50, "B_ii_B_iii": 0.50}},
        "corr_low": {"latent_correlations": {"B_i_B_ii": 0.10, "B_i_B_iii": 0.10, "B_ii_B_iii": 0.10}},
        "low_reliability": {"loadings_range": {"min": 0.40, "max": 0.65}},
        "r2_low": {"r2_target": 0.20},
        "acquiescence": {"acquiescence_gamma": 0.30},
        "ceiling_shift": {"ceiling_shift": 0.60},
        "cross_loading": {"cross_loading": 0.30},
        "contaminated": {"kappa": 0.40},
    }

    for factor_name, factor_overrides in sens_factors.items():
        for n in sens_sample_sizes:
            for nl_name, nl_overrides in sens_nl_cells.items():
                sc_name = f"sens_{factor_name}_n{n}_{nl_name}"
                overrides = {
                    "name": sc_name,
                    "scenario_idx": scenario_idx,
                    "n": n,
                    **factor_overrides,
                    **nl_overrides,
                }
                catalog[sc_name] = build_scenario(overrides, config_path=config_path)
                scenario_idx += 1

    # Convenient aliases
    catalog["base"] = catalog["base_n150_none"]
    catalog["base_scenario"] = catalog["base_n150_none"]

    return catalog


def _deep_update(base: dict[str, Any], override: dict[str, Any]) -> None:
    """Recursively update a dictionary."""
    for key, value in override.items():
        if isinstance(value, dict) and key in base and isinstance(base[key], dict):
            _deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)



# =============================================================================
# Loadings Management (Drawn once per scenario, fixed across replications)
# =============================================================================

def draw_scenario_loadings(
    scenario: dict[str, Any],
    seed: int | np.random.SeedSequence | None = None,
) -> dict[str, np.ndarray]:
    """Draw factor loadings for all items once per scenario from Uniform(min, max).

    SPEC.md Section 3.4:
    - Loadings: lambda_j ~ Uniform(0.55, 0.85) (or low_reliability Uniform(0.40, 0.65)).
    - Fixed across replications.
    - Marker items load only on the acquiescence factor (lambda = 0 for substantive factor).
    """
    if "loadings" in scenario and scenario["loadings"] is not None:
        return scenario["loadings"]

    if seed is None:
        # Use a deterministic seed derived from scenario name or seedsequence
        scenario_seed = scenario.get("loadings_seed", 42)
        rng = np.random.default_rng(scenario_seed)
    else:
        rng = np.random.default_rng(seed)

    loadings_range = scenario["loadings_range"]
    lam_min = float(loadings_range["min"])
    lam_max = float(loadings_range["max"])
    item_counts = scenario["item_counts"]
    draw_mode = str(scenario.get("loading_draw", "equispaced")).lower()

    loadings: dict[str, np.ndarray] = {}
    for construct in ["A_tech", "A_mkt", "B_i", "B_ii", "B_iii", "C", "G"]:
        count = int(item_counts[construct])
        if draw_mode == "equispaced":
            loadings[construct] = np.linspace(lam_min, lam_max, count)
        elif draw_mode == "random":
            loadings[construct] = rng.uniform(lam_min, lam_max, size=count)
        else:
            raise ValueError(f"Unknown loading_draw mode: {draw_mode}. Must be 'equispaced' or 'random'.")

    # Marker variable items load ONLY on the acquiescence factor, no substantive factor
    count_m = int(item_counts["M"])
    loadings["M"] = np.zeros(count_m, dtype=np.float64)

    scenario["loadings"] = loadings
    return loadings


# =============================================================================
# Analytical & Numerical Scenario Calibration
# =============================================================================
#
# ALGEBRA OF VARIANCE SCALING:
# ----------------------------
# Latent outcome equation (SPEC.md Section 3.3):
#   C_latent = linear_part + non_linear_part + noise
#
# Requirements:
# 1. Total variance of standardized C is Var(C) = 1.0.
# 2. Linear and non-linear parts together explain target R^2:
#      Var(linear_part + non_linear_part) = R^2
# 3. The non-linear part takes a variance share psi of C:
#      Var(non_linear_part) = psi
# 4. The noise term takes the remaining variance:
#      Var(noise) = 1.0 - R^2, with noise ~ N(0, 1 - R^2) independent of predictors.
#
# Derivation:
# Let L_0 be the raw weighted linear sum of standardized predictors:
#   L_0 = sum_k beta_k * X_k
# Let N_0 be the non-linear feature (interaction or threshold), centered so E[N_0] = 0.
#
# We set:
#   linear_part     = k_lin * L_0
#   non_linear_part = k_nl  * N_0
#
# Condition 3: Var(k_nl * N_0) = psi
#   k_nl^2 * Var(N_0) = psi  ==>  k_nl = sqrt(psi / Var(N_0))   (if psi > 0, else 0)
#
# Condition 2: Var(k_lin * L_0 + k_nl * N_0) = R^2
# Expanding the variance:
#   Var(k_lin * L_0 + k_nl * N_0)
#     = k_lin^2 * Var(L_0) + 2 * k_lin * k_nl * Cov(L_0, N_0) + k_nl^2 * Var(N_0)
#     = Var(L_0) * k_lin^2 + [2 * k_nl * Cov(L_0, N_0)] * k_lin + psi
#     = R^2
#
# This yields the quadratic equation in k_lin:
#   a * k_lin^2 + b * k_lin + c = 0
# where:
#   a = Var(L_0) > 0
#   b = 2 * k_nl * Cov(L_0, N_0)
#   c = psi - R^2 < 0  (since R^2 > psi)
#
# Since a > 0 and c < 0, the discriminant:
#   Delta = b^2 - 4 * a * c = b^2 + 4 * a * (R^2 - psi) > 0
# is strictly positive, and the product of the two roots is c / a < 0.
# Hence there exists a UNIQUE strictly positive root:
#   k_lin = (-b + sqrt(Delta)) / (2 * a)
#
# Note:
# - For "none" (psi = 0): k_nl = 0, b = 0 ==> k_lin = sqrt(R^2 / Var(L_0)).
# - For "interaction": Since predictors are zero-mean multivariate normal,
#   odd moments vanish, so Cov(L_0, N_0) = 0 ==> b = 0 ==> k_lin = sqrt((R^2 - psi) / Var(L_0)).
# - For "threshold": Cov(L_0, N_0) != 0, so the quadratic formula exact root is used.
#
# =============================================================================
#
# NUMERICAL CALIBRATION OF LATENT G (SPEC.md Section 3.3):
# --------------------------------------------------------
# Outcome specificity requires a separate latent G (general startup intent):
#   - Generated from B_ii (weight 0.35), B_i (weight 0.15), and C
#   - Calibrated so that Corr(G, C) is about 0.50 and Var(G) = 1.0.
#
# Let G_sub = 0.35 * B_ii + 0.15 * B_i.
# We set G = G_sub + w_C * C + u_G, with u_G ~ N(0, sigma_uG^2) independent of (B, C).
# Then:
#   Cov(G, C) = Cov(G_sub, C) + w_C * Var(C) = Cov(G_sub, C) + w_C * 1.0.
# Requiring Corr(G, C) = target_corr (~ 0.50) with Var(G) = 1.0 and Var(C) = 1.0:
#   Cov(G, C) = target_corr ==> w_C = target_corr - Cov(G_sub, C).
#
# Requiring Var(G) = 1.0:
#   Var(G) = Var(G_sub + w_C * C) + sigma_uG^2 = 1.0
#   ==> sigma_uG^2 = 1.0 - Var(G_sub + w_C * C).
#
# =============================================================================

def calibrate_scenario(scenario: dict[str, Any]) -> dict[str, Any]:
    """Calibrate scenario variance scaling parameters and latent G parameters.

    Returns a dict with:
    - Sigma_B, var_u1, var_u2, cov_u1_u2 (Section 3.1 & 3.2)
    - k_lin, k_nl, E_N0, std_N0 (Section 3.3)
    - w_C, var_uG (Section 3.3)
    Cached inside scenario['calibrated_params'].
    """
    if "calibrated_params" in scenario and scenario["calibrated_params"] is not None:
        return scenario["calibrated_params"]

    # 1. Latent Contextual covariance Sigma_B (Section 3.1)
    corrs = scenario["latent_correlations"]
    r_12 = float(corrs["B_i_B_ii"])
    r_13 = float(corrs["B_i_B_iii"])
    r_23 = float(corrs["B_ii_B_iii"])
    Sigma_B = np.array([
        [1.0, r_12, r_13],
        [r_12, 1.0, r_23],
        [r_13, r_23, 1.0],
    ], dtype=np.float64)

    # 2. Individual factor residual variances (Section 3.2)
    ind = scenario["individual_factors"]
    w_A_tech = np.array([float(ind["A_tech"]["weight_B_i"]), 0.0, float(ind["A_tech"]["weight_B_iii"])])
    w_A_mkt  = np.array([0.0, float(ind["A_mkt"]["weight_B_ii"]), float(ind["A_mkt"]["weight_B_iii"])])

    var_expl_A_tech = float(w_A_tech @ Sigma_B @ w_A_tech)
    var_expl_A_mkt  = float(w_A_mkt @ Sigma_B @ w_A_mkt)

    var_u1 = 1.0 - var_expl_A_tech
    var_u2 = 1.0 - var_expl_A_mkt
    rho_u = float(ind["residual_correlation"])
    cov_u1_u2 = rho_u * np.sqrt(var_u1 * var_u2)

    # 3. Covariance matrix of all 5 predictors: [A_tech, A_mkt, B_i, B_ii, B_iii]
    Sigma_5 = np.zeros((5, 5), dtype=np.float64)
    Sigma_5[0, 0] = 1.0
    Sigma_5[1, 1] = 1.0
    cov_A_tech_A_mkt = float(w_A_tech @ Sigma_B @ w_A_mkt) + cov_u1_u2
    Sigma_5[0, 1] = Sigma_5[1, 0] = cov_A_tech_A_mkt
    cov_A_tech_B = w_A_tech @ Sigma_B
    cov_A_mkt_B  = w_A_mkt @ Sigma_B
    Sigma_5[0, 2:] = Sigma_5[2:, 0] = cov_A_tech_B
    Sigma_5[1, 2:] = Sigma_5[2:, 1] = cov_A_mkt_B
    Sigma_5[2:, 2:] = Sigma_B

    # 4. Linear part raw variance
    lin_w = scenario["linear_weights"]
    beta_lin = np.array([
        float(lin_w["A_tech"]),
        float(lin_w["A_mkt"]),
        float(lin_w["B_i"]),
        float(lin_w["B_ii"]),
        float(lin_w["B_iii"]),
    ], dtype=np.float64)
    var_L0 = float(beta_lin @ Sigma_5 @ beta_lin)

    # 5. Non-linear scaling calibration
    r2_target = float(scenario["r2_target"])
    psi = float(scenario["psi"])
    nl_form = str(scenario["non_linear_form"])
    orth_nl = bool(scenario.get("orthogonalise_nonlinear", True))

    cal_rng = np.random.default_rng(scenario.get("cal_seed", 99999))
    n_cal = 200_000
    B_cal = cal_rng.multivariate_normal([0.0, 0.0, 0.0], Sigma_B, size=n_cal)
    Sigma_u = np.array([[var_u1, cov_u1_u2], [cov_u1_u2, var_u2]])
    u_cal = cal_rng.multivariate_normal([0.0, 0.0], Sigma_u, size=n_cal)
    A_tech_cal = w_A_tech[0] * B_cal[:, 0] + w_A_tech[2] * B_cal[:, 2] + u_cal[:, 0]
    A_mkt_cal  = w_A_mkt[1] * B_cal[:, 1] + w_A_mkt[2] * B_cal[:, 2] + u_cal[:, 1]
    L0_cal = (
        beta_lin[0] * A_tech_cal
        + beta_lin[1] * A_mkt_cal
        + beta_lin[2] * B_cal[:, 0]
        + beta_lin[3] * B_cal[:, 1]
        + beta_lin[4] * B_cal[:, 2]
    )
    X_cal = np.column_stack([A_tech_cal, A_mkt_cal, B_cal[:, 0], B_cal[:, 1], B_cal[:, 2]])
    X_cal_const = np.column_stack([np.ones(n_cal), X_cal])

    nl_orth_coefs = None
    N0_cal = None

    if nl_form == "none" or psi <= 0.0:
        k_nl = 0.0
        E_N0 = 0.0
        var_N0 = 0.0
        k_lin = np.sqrt(r2_target / var_L0)
    elif orth_nl:
        if nl_form == "interaction":
            raw_nl_cal = A_tech_cal * B_cal[:, 0]
        elif nl_form == "threshold":
            c_thresh = float(scenario["threshold_cutoff"])
            raw_nl_cal = B_cal[:, 2] * (A_tech_cal < c_thresh)
        else:
            raise ValueError(f"Unknown non_linear_form: {nl_form}")

        # Residualise non-linear term on linear predictors (including intercept)
        nl_orth_coefs = np.linalg.lstsq(X_cal_const, raw_nl_cal, rcond=None)[0]
        N0_cal = raw_nl_cal - X_cal_const @ nl_orth_coefs
        E_N0 = 0.0
        var_N0 = float(np.var(N0_cal))
        k_nl = np.sqrt(psi / var_N0)
        # N0_cal is strictly orthogonal to linear predictors, so Cov(L0, N0) = 0
        k_lin = np.sqrt((r2_target - psi) / var_L0)
    elif nl_form == "interaction":
        # Unorthogonalised interaction: z-scored product A_tech * B_i
        cov_A_tech_B_i = Sigma_5[0, 2]
        E_N0 = cov_A_tech_B_i
        var_N0 = 1.0 + cov_A_tech_B_i**2
        k_nl = np.sqrt(psi)
        k_lin = np.sqrt((r2_target - psi) / var_L0)
    elif nl_form == "threshold":
        # Unorthogonalised threshold: centered threshold term with full quadratic scaling
        c_thresh = float(scenario["threshold_cutoff"])
        T_cal = B_cal[:, 2] * (A_tech_cal < c_thresh)
        E_N0 = float(np.mean(T_cal))
        var_N0 = float(np.var(T_cal))
        cov_L0_N0 = float(np.cov(L0_cal, T_cal - E_N0)[0, 1])

        k_nl = np.sqrt(psi / var_N0)
        a_quad = var_L0
        b_quad = 2.0 * k_nl * cov_L0_N0
        c_quad = psi - r2_target
        disc = b_quad**2 - 4.0 * a_quad * c_quad
        k_lin = (-b_quad + np.sqrt(disc)) / (2.0 * a_quad)
    else:
        raise ValueError(f"Unknown non_linear_form: {nl_form}")

    # 6. Latent G Calibration (Section 3.3)
    if nl_form == "none" or psi <= 0.0:
        nl_cal = np.zeros(n_cal)
    elif orth_nl and nl_orth_coefs is not None and N0_cal is not None:
        nl_cal = k_nl * N0_cal
    elif nl_form == "interaction":
        P_cal = A_tech_cal * B_cal[:, 0]
        nl_cal = k_nl * ((P_cal - E_N0) / np.sqrt(var_N0))
    elif nl_form == "threshold":
        c_thresh = float(scenario["threshold_cutoff"])
        T_cal = B_cal[:, 2] * (A_tech_cal < c_thresh)
        nl_cal = k_nl * (T_cal - E_N0)

    signal_cal = k_lin * L0_cal + nl_cal
    noise_cal = cal_rng.normal(0.0, np.sqrt(1.0 - r2_target), size=n_cal)
    C_cal = signal_cal + noise_cal

    g_cfg = scenario["general_intent_G"]
    w_G_B_ii = float(g_cfg["weight_B_ii"])
    w_G_B_i  = float(g_cfg["weight_B_i"])
    target_corr_G_C = float(g_cfg["target_correlation_C"])

    G_sub_cal = w_G_B_ii * B_cal[:, 1] + w_G_B_i * B_cal[:, 0]
    cov_Gsub_C = float(np.cov(G_sub_cal, C_cal)[0, 1])
    w_C = target_corr_G_C - cov_Gsub_C
    var_det_G = float(np.var(G_sub_cal + w_C * C_cal))
    var_uG = max(0.0, 1.0 - var_det_G)

    calibrated = {
        "Sigma_B": Sigma_B,
        "w_A_tech": w_A_tech,
        "w_A_mkt": w_A_mkt,
        "var_u1": var_u1,
        "var_u2": var_u2,
        "cov_u1_u2": cov_u1_u2,
        "Sigma_u": np.array([[var_u1, cov_u1_u2], [cov_u1_u2, var_u2]], dtype=np.float64),
        "Sigma_5": Sigma_5,
        "var_L0": var_L0,
        "k_lin": k_lin,
        "k_nl": k_nl,
        "E_N0": E_N0,
        "var_N0": var_N0,
        "orthogonalise_nonlinear": orth_nl,
        "nl_orth_coefs": nl_orth_coefs,
        "w_G_B_ii": w_G_B_ii,
        "w_G_B_i": w_G_B_i,
        "w_C": w_C,
        "var_uG": var_uG,
    }
    scenario["calibrated_params"] = calibrated
    return calibrated


# =============================================================================
# Core Generator
# =============================================================================

def generate(
    scenario: dict[str, Any] | str,
    seed: int | np.random.SeedSequence | np.random.Generator,
) -> pd.DataFrame:
    """Generate one synthetic dataset according to SPEC.md Section 3.

    Returns a pandas.DataFrame containing:
    - Likert item responses for all constructs (with reverse-scored items re-coded as 6 - x)
    - Latent factor scores (kept for validation and diagnostics)
    - Scored sub-construct means and block indices
    - Covariates (semester and 3 binary prior-exposure flags)

    Parameters:
    - scenario: scenario dictionary (or scenario name).
    - seed: child seed from np.random.SeedSequence.spawn.
    """
    if isinstance(scenario, str):
        raw_cfg = load_config()
        if scenario in raw_cfg.get("scenarios", {}):
            scenario_dict = build_scenario(raw_cfg["scenarios"][scenario])
        else:
            scenario_dict = build_scenario({"name": scenario})
    else:
        scenario_dict = scenario

    # Ensure loadings are drawn once per scenario and fixed across replications
    loadings = draw_scenario_loadings(scenario_dict)
    cal = calibrate_scenario(scenario_dict)

    # Initialize child RNG without touching global numpy RNG (Rule 9)
    if isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    n = int(scenario_dict["n"])

    # -------------------------------------------------------------------------
    # 3.1 Latent Contextual Factors: (B-i, B-ii, B-iii)
    # -------------------------------------------------------------------------
    B = rng.multivariate_normal([0.0, 0.0, 0.0], cal["Sigma_B"], size=n)
    B_i = B[:, 0]
    B_ii = B[:, 1]
    B_iii = B[:, 2]

    # -------------------------------------------------------------------------
    # 3.2 Latent Individual Factors: A_tech, A_mkt
    # -------------------------------------------------------------------------
    u = rng.multivariate_normal([0.0, 0.0], cal["Sigma_u"], size=n)
    w_A_tech = cal["w_A_tech"]
    w_A_mkt  = cal["w_A_mkt"]
    A_tech = w_A_tech[0] * B_i + w_A_tech[2] * B_iii + u[:, 0]
    A_mkt  = w_A_mkt[1] * B_ii + w_A_mkt[2] * B_iii + u[:, 1]

    # -------------------------------------------------------------------------
    # 3.3 Outcome (Commitment Intention, C)
    # -------------------------------------------------------------------------
    lin_w = scenario_dict["linear_weights"]
    L_0 = (
        float(lin_w["A_tech"]) * A_tech
        + float(lin_w["A_mkt"]) * A_mkt
        + float(lin_w["B_i"]) * B_i
        + float(lin_w["B_ii"]) * B_ii
        + float(lin_w["B_iii"]) * B_iii
    )

    k_lin = cal["k_lin"]
    k_nl  = cal["k_nl"]
    E_N0  = cal["E_N0"]
    var_N0 = cal["var_N0"]
    nl_form = str(scenario_dict["non_linear_form"])

    if nl_form == "none" or float(scenario_dict["psi"]) <= 0.0:
        non_linear_part = np.zeros(n, dtype=np.float64)
    elif cal.get("nl_orth_coefs") is not None:
        X_preds = np.column_stack([A_tech, A_mkt, B_i, B_ii, B_iii])
        X_const = np.column_stack([np.ones(n), X_preds])
        if nl_form == "interaction":
            raw_nl = A_tech * B_i
        elif nl_form == "threshold":
            c_thresh = float(scenario_dict["threshold_cutoff"])
            raw_nl = B_iii * (A_tech < c_thresh)
        else:
            raise ValueError(f"Unknown non_linear_form: {nl_form}")
        non_linear_part = k_nl * (raw_nl - X_const @ cal["nl_orth_coefs"])
    elif nl_form == "interaction":
        # z-scored product A_tech * B_i (unorthogonalised)
        P = A_tech * B_i
        non_linear_part = k_nl * ((P - E_N0) / np.sqrt(var_N0))
    elif nl_form == "threshold":
        # Centered threshold (unorthogonalised)
        c_thresh = float(scenario_dict["threshold_cutoff"])
        T = B_iii * (A_tech < c_thresh)
        non_linear_part = k_nl * (T - E_N0)
    else:
        raise ValueError(f"Unknown non_linear_form: {nl_form}")

    linear_part = k_lin * L_0
    r2_target = float(scenario_dict["r2_target"])
    noise = rng.normal(0.0, np.sqrt(1.0 - r2_target), size=n)
    C = linear_part + non_linear_part + noise

    # Latent G (general startup intent)
    w_G_B_ii = cal["w_G_B_ii"]
    w_G_B_i  = cal["w_G_B_i"]
    w_C      = cal["w_C"]
    var_uG   = cal["var_uG"]
    u_G = rng.normal(0.0, np.sqrt(var_uG), size=n)
    G = w_G_B_ii * B_ii + w_G_B_i * B_i + w_C * C + u_G

    # -------------------------------------------------------------------------
    # 3.4 Measurement Layer
    # -------------------------------------------------------------------------
    # Common acquiescence factor a_i ~ N(0, 1) shared by all items
    gamma = float(scenario_dict["acquiescence_gamma"])
    a = rng.normal(0.0, 1.0, size=n)

    shift = float(scenario_dict["ceiling_shift"])
    cut_points = np.array(scenario_dict["discretisation_cut_points"], dtype=np.float64)
    rev_indices_cfg = scenario_dict["reverse_keyed_indices"]
    cross_loading = float(scenario_dict["cross_loading"])
    kappa = float(scenario_dict["kappa"])
    rev_score_max = int(scenario_dict["reverse_score_max"])

    # Latent target for contaminated C outcome: sqrt(1 - kappa^2)*C + kappa*G
    if kappa > 0.0:
        theta_C_effective = np.sqrt(1.0 - kappa**2) * C + kappa * G
    else:
        theta_C_effective = C

    # Generate items per construct
    construct_latents: dict[str, np.ndarray] = {
        "A_tech": A_tech,
        "A_mkt": A_mkt,
        "B_i": B_i,
        "B_ii": B_ii,
        "B_iii": B_iii,
        "C": theta_C_effective,
        "G": G,
        "M": np.zeros(n, dtype=np.float64),  # marker has no substantive latent
    }

    item_columns: dict[str, np.ndarray] = {}
    scored_items_by_construct: dict[str, list[str]] = {k: [] for k in construct_latents}

    for construct, theta in construct_latents.items():
        lam = loadings[construct]
        n_items = len(lam)
        rev_indices = set(rev_indices_cfg.get(construct, []))

        for j in range(n_items):
            lam_j = lam[j]
            is_rev = j in rev_indices

            # Effective substantive loading: sign flipped if reverse-keyed
            eff_lam_j = -lam_j if is_rev else lam_j

            # Substantive signal
            if construct == "B_i" and cross_loading > 0.0:
                # B-i cross-loads on A_tech
                eff_cross = -cross_loading if is_rev else cross_loading
                subst_signal = eff_lam_j * theta + eff_cross * A_tech
            else:
                subst_signal = eff_lam_j * theta

            # Idiosyncratic error e_ij ~ N(0, 1 - lam_j^2)
            var_e = max(0.0, 1.0 - lam_j**2)
            e_j = rng.normal(0.0, np.sqrt(var_e), size=n)

            # Acquiescence factor adds with weight gamma (positive direction regardless of keying)
            x_star = subst_signal + gamma * a + e_j

            # Standardise x_star so that mean=0, std=1 across the sample
            std_x = np.std(x_star)
            if std_x > 0:
                x_star_std = (x_star - np.mean(x_star)) / std_x
            else:
                x_star_std = x_star

            # Discretise with ceiling shift: cut at [-1.5, -0.5, 0.5, 1.5]
            # Resulting in categories 1, 2, 3, 4, 5
            raw_category = np.digitize(x_star_std + shift, cut_points) + 1

            # Reverse-score: re-coded as (6 - x) if reverse-keyed
            if is_rev:
                scored_value = rev_score_max - raw_category
            else:
                scored_value = raw_category

            col_name = f"{construct}_{j + 1}"
            item_columns[col_name] = scored_value
            scored_items_by_construct[construct].append(col_name)

    # -------------------------------------------------------------------------
    # 3.5 Covariates
    # -------------------------------------------------------------------------
    cov_cfg = scenario_dict["covariates"]

    # Semester: categorical {6, 7, 8}
    sem_cfg = cov_cfg["semester"]
    sem_cats = np.array(sem_cfg["categories"], dtype=np.int64)
    sem_probs = np.array(sem_cfg["probabilities"], dtype=np.float64)
    semester = rng.choice(sem_cats, size=n, p=sem_probs)

    # Prior-exposure binary flags (logistic function of related latent factor)
    pe_cfg = cov_cfg["prior_exposure"]
    slope = float(pe_cfg["slope"])
    intercept = float(pe_cfg["intercept"])

    # Prototype: from B-i
    p_proto = 1.0 / (1.0 + np.exp(-(slope * B_i + intercept)))
    covar_prototype = (rng.uniform(0.0, 1.0, size=n) < p_proto).astype(np.int64)

    # Hackathon: from B-ii
    p_hack = 1.0 / (1.0 + np.exp(-(slope * B_ii + intercept)))
    covar_hackathon = (rng.uniform(0.0, 1.0, size=n) < p_hack).astype(np.int64)

    # Research-lab internship: from B-i
    p_intern = 1.0 / (1.0 + np.exp(-(slope * B_i + intercept)))
    covar_internship = (rng.uniform(0.0, 1.0, size=n) < p_intern).astype(np.int64)

    # -------------------------------------------------------------------------
    # Sub-construct Scores & Block Indices Scoring
    # -------------------------------------------------------------------------
    # Sub-construct score = mean of its items after reverse-scoring
    scores: dict[str, np.ndarray] = {}
    for construct, cols in scored_items_by_construct.items():
        mat = np.column_stack([item_columns[c] for c in cols])
        scores[f"score_{construct}"] = np.mean(mat, axis=1)

    # Individual Resilience Index = mean of A-i and A-ii items
    a_items = scored_items_by_construct["A_tech"] + scored_items_by_construct["A_mkt"]
    mat_a = np.column_stack([item_columns[c] for c in a_items])
    index_individual = np.mean(mat_a, axis=1)

    # Contextual Exposure Index = mean of Part B items (B-i, B-ii, B-iii)
    b_items = (
        scored_items_by_construct["B_i"]
        + scored_items_by_construct["B_ii"]
        + scored_items_by_construct["B_iii"]
    )
    mat_b = np.column_stack([item_columns[c] for c in b_items])
    index_contextual = np.mean(mat_b, axis=1)

    # -------------------------------------------------------------------------
    # Assemble DataFrame
    # -------------------------------------------------------------------------
    data: dict[str, np.ndarray] = {}

    # 1. Item responses (after reverse-coding as 6 - x)
    data.update(item_columns)

    # 2. Latent scores (kept for testing and diagnostics)
    data["latent_A_tech"] = A_tech
    data["latent_A_mkt"] = A_mkt
    data["latent_B_i"] = B_i
    data["latent_B_ii"] = B_ii
    data["latent_B_iii"] = B_iii
    data["latent_C"] = C
    data["latent_G"] = G
    data["latent_acquiescence"] = a
    data["latent_C_linear"] = linear_part
    data["latent_C_nonlinear"] = non_linear_part
    data["latent_C_noise"] = noise

    # 3. Covariates
    data["semester"] = semester
    data["covar_prototype"] = covar_prototype
    data["covar_hackathon"] = covar_hackathon
    data["covar_internship"] = covar_internship

    # 4. Scored sub-construct means and block indices
    data.update(scores)
    data["index_individual"] = index_individual
    data["index_contextual"] = index_contextual

    return pd.DataFrame(data)


# =============================================================================
# Helper for Replication Seed Management (Rule 9)
# =============================================================================

def spawn_seeds(
    master_seed: int,
    n_scenarios: int,
    n_replications: int,
) -> list[list[np.random.SeedSequence]]:
    """Spawn independent child seeds per (scenario, replication).

    Follows Rule 9: One master seed, np.random.SeedSequence.spawn for one child seed
    per (scenario, replication). Never uses the global numpy RNG.
    """
    master_ss = np.random.SeedSequence(master_seed)
    scenario_ss = master_ss.spawn(n_scenarios)
    return [s_ss.spawn(n_replications) for s_ss in scenario_ss]
