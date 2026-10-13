"""Evaluation metrics computation for simulation results.

SPEC.md Section 6:
- Metric A: Inferential performance (H1 block power, H1 t-power, H2 power, Delta R^2).
- Metric B: Interaction detection (M2a, M2b, M3, M4, M5, M6 detection flags and statistics).
- Metric C: Importance and stability (ranking accuracy vs oracle, bootstrap stability, H3 check).
- Metric D: Prediction (out-of-sample R^2, RMSE, difference vs M3 benchmark).
- Metric E: Estimation quality (M1 bias, RMSE, 95% CI coverage, attenuation ratio).
- Metric H: Method-bias gate (M1 vs M1-adj comparison).
"""

from __future__ import annotations

import time
from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from src.measurement import harman_first_factor_share, htmt, parallel_analysis
from src.methods.residual_resampling import SUBSTANTIVE_COLS_DEFAULT
from src.oracle import FEATURE_NAMES, oracle_importance


def compute_bootstrap_ranking_stability(
    model_name: str,
    X: pd.DataFrame,
    y: np.ndarray,
    substantive_cols: list[str] = SUBSTANTIVE_COLS_DEFAULT,
    n_bootstraps: int = 50,
    seed: int = 42,
    fitted_model: Any = None,
) -> float:
    """Compute mean Kendall's tau between original and bootstrap feature rankings (SPEC Section 6.C).

    Milestone 6.7 Part C: Tuning-conditional bootstrap.
    Reuses hyperparameters selected on observed data (M4: min_samples_leaf; M5: depth and n_estimators).
    """
    n = len(y)
    rng = np.random.default_rng(seed)

    def get_ranking_for_fit(X_data: pd.DataFrame, y_data: np.ndarray, s: int) -> list[str]:
        if model_name == "M1":
            X_c = sm.add_constant(X_data, has_constant="add")
            ols = sm.OLS(y_data, X_c).fit()
            coefs = {col: abs(float(ols.params[col])) for col in substantive_cols if col in ols.params}
            return sorted(substantive_cols, key=lambda c: coefs.get(c, 0.0), reverse=True)
        elif model_name == "M4":
            leaf = getattr(fitted_model, "best_min_samples_leaf_", 5) if fitted_model is not None else 5
            n_est = getattr(fitted_model, "n_estimators", 100) if fitted_model is not None else 100
            rf = RandomForestRegressor(n_estimators=n_est, min_samples_leaf=leaf, bootstrap=True, random_state=s, n_jobs=1)
            rf.fit(X_data.values, y_data)
            cols = list(X_data.columns)
            imps = {col: float(rf.feature_importances_[cols.index(col)]) for col in substantive_cols if col in cols}
            return sorted(substantive_cols, key=lambda c: imps.get(c, 0.0), reverse=True)
        elif model_name == "M5":
            depth = getattr(fitted_model, "best_max_depth_", 2) if fitted_model is not None else 2
            n_est = getattr(fitted_model, "best_n_estimators_", 100) if fitted_model is not None else 100
            gb = GradientBoostingRegressor(learning_rate=0.03, subsample=0.7, min_samples_leaf=5, max_depth=depth, n_estimators=n_est, random_state=s)
            gb.fit(X_data.values, y_data)
            cols = list(X_data.columns)
            imps = {col: float(gb.feature_importances_[cols.index(col)]) for col in substantive_cols if col in cols}
            return sorted(substantive_cols, key=lambda c: imps.get(c, 0.0), reverse=True)
        else:
            X_c = sm.add_constant(X_data, has_constant="add")
            ols = sm.OLS(y_data, X_c).fit()
            coefs = {col: abs(float(ols.params[col])) for col in substantive_cols if col in ols.params}
            return sorted(substantive_cols, key=lambda c: coefs.get(c, 0.0), reverse=True)

    orig_ranking = get_ranking_for_fit(X, y, seed)
    orig_ranks = [orig_ranking.index(c) for c in substantive_cols]

    taus: list[float] = []
    for b in range(n_bootstraps):
        boot_idx = rng.choice(n, size=n, replace=True)
        b_seed = int(rng.integers(1, 2_000_000))
        b_ranking = get_ranking_for_fit(X.iloc[boot_idx], y[boot_idx], b_seed)
        b_ranks = [b_ranking.index(c) for c in substantive_cols]
        tau = stats.kendalltau(orig_ranks, b_ranks).statistic
        if not np.isnan(tau):
            taus.append(float(tau))
        else:
            taus.append(1.0)

    return float(np.mean(taus)) if len(taus) > 0 else 0.0


def compute_replication_metrics(
    cv_output: dict[str, Any],
    scenario_dict: dict[str, Any],
    oracle_ranking: list[str] | None = None,
    run_stability: bool = False,
    n_bootstraps: int = 50,
    data: pd.DataFrame | None = None,
) -> dict[str, Any]:
    """Extract metrics A, B, C, D, E, H, and measurement checks from a single replication fit."""
    oos = cv_output["oos_metrics"]
    X_block2 = cv_output.get("X_block2")
    y = cv_output.get("y")

    # True latent weights from scenario
    lin_weights = scenario_dict.get("linear_weights", {})
    true_weights = {
        "score_A_tech": float(lin_weights.get("A_tech", 0.30)),
        "score_A_mkt": float(lin_weights.get("A_mkt", 0.15)),
        "score_B_i": float(lin_weights.get("B_i", 0.20)),
        "score_B_ii": float(lin_weights.get("B_ii", 0.10)),
        "score_B_iii": float(lin_weights.get("B_iii", 0.15)),
    }

    # Oracle ground truth ranking and per-k tie detection (Milestone 6.6 Part B)
    top1_defined = True
    top2_defined = True
    if oracle_ranking is None:
        try:
            oracle_imps = oracle_importance(scenario_dict, n_pop=50_000, seed=42, n_repeats=1)
            sorted_imps = sorted(oracle_imps.values(), reverse=True)
            # Top-1 is defined iff Rank 1 exceeds Rank 2 by at least 10% relative
            if len(sorted_imps) >= 2:
                v1, v2 = sorted_imps[0], sorted_imps[1]
                top1_defined = bool(v1 > 0 and ((v1 - v2) / v1) >= 0.10)
            # Top-2 is defined iff Rank 2 exceeds Rank 3 by at least 10% relative
            if len(sorted_imps) >= 3:
                v2, v3 = sorted_imps[1], sorted_imps[2]
                top2_defined = bool(v2 > 0 and ((v2 - v3) / v2) >= 0.10)
            oracle_sub_ranking = sorted(
                SUBSTANTIVE_COLS_DEFAULT,
                key=lambda f: oracle_imps.get(f.replace("score_", ""), 0.0),
                reverse=True,
            )
        except Exception:
            oracle_sub_ranking = sorted(
                SUBSTANTIVE_COLS_DEFAULT,
                key=lambda f: abs(true_weights.get(f, 0.0)),
                reverse=True,
            )
    else:
        oracle_sub_ranking = oracle_ranking

    # -------------------------------------------------------------------------
    # Metric A: Inferential Performance (M1)
    # -------------------------------------------------------------------------
    metric_a = {}
    m1 = cv_output.get("m1")
    if m1 is not None:
        metric_a = {
            "h1_f_hit": int(m1.h1_f_hit_),
            "h1_t_hit": int(m1.h1_t_hit_),
            "h2_hit": int(m1.h2_hit_),
            "f_stat": float(m1.f_stat_),
            "f_pvalue": float(m1.f_pvalue_),
            "a_tech_coef": float(m1.a_tech_coef_),
            "a_tech_pvalue": float(m1.a_tech_pvalue_),
            "contrast_diff": float(m1.contrast_diff_),
            "contrast_pvalue": float(m1.contrast_pvalue_),
            "delta_r2": float(m1.delta_r2_),
            "r2_block1": float(m1.r2_block1_),
            "r2_block2": float(m1.r2_block2_),
        }

    # -------------------------------------------------------------------------
    # Metric E: Estimation Quality (M1)
    # -------------------------------------------------------------------------
    metric_e = {}
    if m1 is not None and m1.fit_block2_hc3_ is not None:
        cols = list(m1.fit_block2_hc3_.model.exog_names)
        for var_name, true_w in true_weights.items():
            if var_name in cols:
                idx = cols.index(var_name)
                est_b = float(m1.fit_block2_hc3_.params[idx])
                se_b = float(m1.fit_block2_hc3_.bse[idx])
                bias = est_b - true_w
                ci_lower = est_b - 1.96 * se_b
                ci_upper = est_b + 1.96 * se_b
                covered = int(ci_lower <= true_w <= ci_upper)
                attenuation = est_b / true_w if abs(true_w) > 1e-6 else 1.0

                metric_e[f"beta_{var_name}"] = est_b
                metric_e[f"se_{var_name}"] = se_b
                metric_e[f"bias_{var_name}"] = bias
                metric_e[f"coverage_{var_name}"] = covered
                metric_e[f"attenuation_{var_name}"] = attenuation

    # -------------------------------------------------------------------------
    # Metric B: Interaction Detection (M2a, M2b, M3, M4, M5, M6)
    # -------------------------------------------------------------------------
    metric_b = {}
    m2a = cv_output.get("m2a")
    if m2a is not None:
        metric_b.update({
            "m2a_detected": int(m2a.detected_),
            "m2a_pvalue": float(m2a.interaction_pvalue_),
            "m2a_coef": float(m2a.interaction_coef_),
            "m2a_tstat": float(m2a.interaction_t_),
        })

    m2b = cv_output.get("m2b")
    if m2b is not None:
        metric_b.update({
            "m2b_detected": int(m2b.detected_),
            "m2b_min_raw_p": float(m2b.min_raw_p_),
            "m2b_min_corrected_p": float(m2b.min_corrected_p_),
            "m2b_top1_pair": f"{m2b.top1_pair_[0]}_x_{m2b.top1_pair_[1]}",
        })

    m3 = cv_output.get("m3")
    if m3 is not None:
        metric_b.update({
            "m3_detected": int(m3.detected_),
            "m3_max_freq": float(m3.max_selection_freq_),
            "m3_top1_pair": f"{m3.top1_pair_[0]}_x_{m3.top1_pair_[1]}",
        })

    m4 = cv_output.get("m4")
    if m4 is not None:
        metric_b.update({
            "m4_detected": int(m4.detected_),
            "m4_pvalue": float(m4.p_value_),
            "m4_tstat": float(m4.t_observed_),
            "m4_top1_pair": f"{m4.top1_pair_[0]}_x_{m4.top1_pair_[1]}",
        })

    m5 = cv_output.get("m5")
    if m5 is not None:
        metric_b.update({
            "m5_detected": int(m5.detected_),
            "m5_pvalue": float(m5.p_value_),
            "m5_tstat": float(m5.t_observed_),
            "m5_top1_pair": f"{m5.top1_pair_[0]}_x_{m5.top1_pair_[1]}",
        })

    m6 = cv_output.get("m6")
    if m6 is not None:
        metric_b.update({
            "m6_detected": int(m6.detected_),
            "m6_pvalue": float(m6.p_value_),
            "m6_tstat": float(m6.t_observed_),
            "m6_top1_pair": f"{m6.top1_pair_[0]}_x_{m6.top1_pair_[1]}",
        })

    # -------------------------------------------------------------------------
    # Metric C: Importance and Stability (M1, M3, M4, M5, M6)
    # -------------------------------------------------------------------------
    metric_c = {}
    oracle_ranks = [oracle_sub_ranking.index(c) for c in SUBSTANTIVE_COLS_DEFAULT]

    def evaluate_ranking(ranking: list[str], prefix: str):
        method_ranks = [ranking.index(c) for c in SUBSTANTIVE_COLS_DEFAULT]
        res = stats.kendalltau(oracle_ranks, method_ranks, variant="b")
        tau = float(res.statistic) if hasattr(res, "statistic") else float(res[0])
        top1 = float(int(ranking[0] == oracle_sub_ranking[0])) if top1_defined else float("nan")
        top2 = float(int(set(ranking[:2]) == set(oracle_sub_ranking[:2]))) if top2_defined else float("nan")
        metric_c[f"{prefix}_kendall_tau"] = tau
        metric_c[f"{prefix}_top1_recovery"] = top1
        metric_c[f"{prefix}_top2_recovery"] = top2

    if m1 is not None and m1.fit_block2_hc3_ is not None:
        cols = list(m1.fit_block2_hc3_.model.exog_names)
        m1_coefs = {c: abs(float(m1.fit_block2_hc3_.params[cols.index(c)])) for c in SUBSTANTIVE_COLS_DEFAULT if c in cols}
        ranking_m1 = sorted(SUBSTANTIVE_COLS_DEFAULT, key=lambda c: m1_coefs.get(c, 0.0), reverse=True)
        evaluate_ranking(ranking_m1, "m1")

    if m4 is not None and len(m4.feature_importances_) > 0:
        ranking_m4 = sorted(SUBSTANTIVE_COLS_DEFAULT, key=lambda c: m4.feature_importances_.get(c, 0.0), reverse=True)
        evaluate_ranking(ranking_m4, "m4")

    if m5 is not None and len(m5.feature_importances_) > 0:
        ranking_m5 = sorted(SUBSTANTIVE_COLS_DEFAULT, key=lambda c: m5.feature_importances_.get(c, 0.0), reverse=True)
        evaluate_ranking(ranking_m5, "m5")

    if m6 is not None and len(m6.feature_importances_) > 0:
        ranking_m6 = sorted(SUBSTANTIVE_COLS_DEFAULT, key=lambda c: m6.feature_importances_.get(c, 0.0), reverse=True)
        evaluate_ranking(ranking_m6, "m6")

    # Bootstrap stability (Milestone 6.7: Tuning-conditional bootstrap)
    time_bootstrap: dict[str, float] = {"m3": 0.0, "m4": 0.0, "m5": 0.0, "m6": 0.0}
    m3_obj = cv_output.get("m3")
    if m3_obj is not None:
        time_bootstrap["m3"] = float(getattr(m3_obj, "time_stability_", 0.0))

    if run_stability and X_block2 is not None and y is not None:
        metric_c["m1_stability"] = compute_bootstrap_ranking_stability("M1", X_block2, y, fitted_model=m1, n_bootstraps=n_bootstraps)
        if m4 is not None:
            t0 = time.perf_counter()
            metric_c["m4_stability"] = compute_bootstrap_ranking_stability("M4", X_block2, y, fitted_model=m4, n_bootstraps=n_bootstraps)
            time_bootstrap["m4"] = time.perf_counter() - t0
        if m5 is not None:
            t0 = time.perf_counter()
            metric_c["m5_stability"] = compute_bootstrap_ranking_stability("M5", X_block2, y, fitted_model=m5, n_bootstraps=n_bootstraps)
            time_bootstrap["m5"] = time.perf_counter() - t0

    # Component timings for M3 to M6 (Milestone 6.7 Part C)
    metric_timing = {}
    comp_timings = cv_output.get("component_timings", {})
    t_nested_cv = comp_timings.get("time_nested_cv", {})
    t_full_fit = comp_timings.get("time_full_fit", {})
    t_null_test = comp_timings.get("time_null_test", {})

    for m in ["m3", "m4", "m5", "m6"]:
        t_ncv = float(t_nested_cv.get(m, 0.0))
        t_ff = float(t_full_fit.get(m, 0.0))
        t_null = float(t_null_test.get(m, 0.0))
        t_boot = float(time_bootstrap.get(m, 0.0))
        t_tot = t_ncv + t_ff + t_null + t_boot
        metric_timing[f"{m}_time_nested_cv"] = t_ncv
        metric_timing[f"{m}_time_full_fit"] = t_ff
        metric_timing[f"{m}_time_null_test"] = t_null
        metric_timing[f"{m}_time_bootstrap"] = t_boot
        metric_timing[f"{m}_t_total"] = t_tot

    # H3 Check (SPEC line 128: at least one interaction surfaces)
    any_interaction_detected = int(
        bool(metric_b.get("m2a_detected", 0))
        or bool(metric_b.get("m2b_detected", 0))
        or bool(metric_b.get("m3_detected", 0))
        or bool(metric_b.get("m4_detected", 0))
        or bool(metric_b.get("m5_detected", 0))
        or bool(metric_b.get("m6_detected", 0))
    )
    metric_c["h3_any_interaction_detected"] = any_interaction_detected

    # -------------------------------------------------------------------------
    # Metric D: Out-of-Sample Prediction
    # -------------------------------------------------------------------------
    metric_d = {}
    m3_r2 = oos.get("M3", {}).get("r2_oos", 0.0)

    for m_key, metrics in oos.items():
        m_lower = m_key.lower()
        r2_val = metrics["r2_oos"]
        rmse_val = metrics["rmse_oos"]
        metric_d[f"{m_lower}_r2_oos"] = r2_val
        metric_d[f"{m_lower}_rmse_oos"] = rmse_val
        metric_d[f"{m_lower}_delta_r2_vs_m3"] = r2_val - m3_r2

    # -------------------------------------------------------------------------
    # Metric H: Method Bias Gate (M1 vs M1-adj)
    # -------------------------------------------------------------------------
    metric_h = {}
    m1_adj = cv_output.get("m1_adj")
    if m1 is not None and m1_adj is not None:
        metric_h = {
            "m1_adj_h1_f_hit": int(m1_adj["h1_f_hit"]),
            "m1_adj_h1_t_hit": int(m1_adj["h1_t_hit"]),
            "m1_adj_h2_hit": int(m1_adj["h2_hit"]),
            "m1_adj_delta_r2": float(m1_adj["delta_r2"]),
            "m1_adj_r_M": float(m1_adj["r_M"]),
            "h1_conclusion_changed": int(m1.h1_f_hit_ != m1_adj["h1_f_hit"]),
            "h2_conclusion_changed": int(m1.h2_hit_ != m1_adj["h2_hit"]),
        }

    # -------------------------------------------------------------------------
    # Measurement Checks (SPEC.md Section 5 & 10, Milestone 6.5 Part B)
    # -------------------------------------------------------------------------
    metric_meas = {}
    survey_data = data if data is not None else cv_output.get("data")
    if survey_data is not None:
        cols_a_tech = [c for c in survey_data.columns if c.startswith("A_tech_") and c[len("A_tech_"):].isdigit()]
        cols_b_i = [c for c in survey_data.columns if c.startswith("B_i_") and c[len("B_i_"):].isdigit()]
        cols_a_mkt = [c for c in survey_data.columns if c.startswith("A_mkt_") and c[len("A_mkt_"):].isdigit()]
        cols_b_ii = [c for c in survey_data.columns if c.startswith("B_ii_") and c[len("B_ii_"):].isdigit()]
        cols_c = [c for c in survey_data.columns if c.startswith("C_") and c[len("C_"):].isdigit()]
        cols_g = [c for c in survey_data.columns if c.startswith("G_") and c[len("G_"):].isdigit()]

        pa_iter = int(scenario_dict.get("pa_n_iter", 100))

        if cols_a_tech and cols_b_i:
            metric_meas["htmt_A_tech_B_i"] = float(htmt(survey_data[cols_a_tech + cols_b_i], {"A_tech": cols_a_tech, "B_i": cols_b_i}))
            metric_meas["pa_factors_A_tech_B_i"] = int(parallel_analysis(survey_data[cols_a_tech + cols_b_i].to_numpy(dtype=np.float64), n_iter=pa_iter))

        if cols_a_mkt and cols_b_ii:
            metric_meas["htmt_A_mkt_B_ii"] = float(htmt(survey_data[cols_a_mkt + cols_b_ii], {"A_mkt": cols_a_mkt, "B_ii": cols_b_ii}))
            metric_meas["pa_factors_A_mkt_B_ii"] = int(parallel_analysis(survey_data[cols_a_mkt + cols_b_ii].to_numpy(dtype=np.float64), n_iter=pa_iter))

        if cols_c and cols_g:
            metric_meas["htmt_C_G"] = float(htmt(survey_data[cols_c + cols_g], {"C": cols_c, "G": cols_g}))
            metric_meas["pa_factors_C_G"] = int(parallel_analysis(survey_data[cols_c + cols_g].to_numpy(dtype=np.float64), n_iter=pa_iter))

        all_study_items = [
            c for c in survey_data.columns
            if any(c.startswith(f"{p}_") and c[len(p)+1:].isdigit() for p in ["A_tech", "A_mkt", "B_i", "B_ii", "B_iii", "C"])
        ]
        if all_study_items:
            metric_meas["harman_first_factor_share"] = float(harman_first_factor_share(survey_data[all_study_items]))

        if m1_adj is not None and "r_M" in m1_adj:
            metric_meas["marker_r_M"] = float(m1_adj["r_M"])
        elif "score_M" in survey_data.columns:
            m_vals = survey_data["score_M"].to_numpy(dtype=np.float64)
            subst_scores = ["score_A_tech", "score_A_mkt", "score_B_i", "score_B_ii", "score_B_iii", "score_C"]
            metric_meas["marker_r_M"] = float(np.mean([
                abs(np.corrcoef(m_vals, survey_data[col].to_numpy(dtype=np.float64))[0, 1])
                for col in subst_scores if col in survey_data.columns
            ]))
        else:
            metric_meas["marker_r_M"] = 0.0

    # Combine into single flat dictionary
    row = {}
    row.update(metric_a)
    row.update(metric_b)
    row.update(metric_c)
    row.update(metric_d)
    row.update(metric_e)
    row.update(metric_h)
    row.update(metric_meas)
    row.update(metric_timing)
    return row
