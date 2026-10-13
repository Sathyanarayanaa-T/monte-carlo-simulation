"""Evaluation metrics computation for simulation results.

SPEC.md Section 6:
- Metric A: Inferential performance (H1 block power, H1 t-power, H2 power, Delta R^2).
- Metric B: Interaction detection (M2a, M2b, M3 detection flags and statistics).
- Metric E: Estimation quality (M1 bias, RMSE, 95% CI coverage, attenuation ratio).
- Metric H: Method-bias gate (M1 vs M1-adj comparison).
"""

from __future__ import annotations

from typing import Any
import numpy as np


def compute_replication_metrics(
    cv_output: dict[str, Any],
    scenario_dict: dict[str, Any],
) -> dict[str, Any]:
    """Extract metrics A, B, E, and H from a single simulation replication fit."""
    m1 = cv_output["m1"]
    m1_adj = cv_output.get("m1_adj")
    m2a = cv_output["m2a"]
    m2b = cv_output["m2b"]
    m3 = cv_output["m3"]
    oos = cv_output["oos_metrics"]

    # True latent weights from scenario
    lin_weights = scenario_dict.get("linear_weights", {})
    true_weights = {
        "score_A_tech": float(lin_weights.get("A_tech", 0.30)),
        "score_A_mkt": float(lin_weights.get("A_mkt", 0.15)),
        "score_B_i": float(lin_weights.get("B_i", 0.20)),
        "score_B_ii": float(lin_weights.get("B_ii", 0.10)),
        "score_B_iii": float(lin_weights.get("B_iii", 0.15)),
    }

    # -------------------------------------------------------------------------
    # Metric A: Inferential Performance (M1)
    # -------------------------------------------------------------------------
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
    if m1.fit_block2_hc3_ is not None:
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
    # Metric B: Interaction Detection (M2a, M2b, M3)
    # -------------------------------------------------------------------------
    metric_b = {
        # M2a
        "m2a_detected": int(m2a.detected_),
        "m2a_pvalue": float(m2a.interaction_pvalue_),
        "m2a_coef": float(m2a.interaction_coef_),
        "m2a_tstat": float(m2a.interaction_t_),
        # M2b
        "m2b_detected": int(m2b.detected_),
        "m2b_min_raw_p": float(m2b.min_raw_p_),
        "m2b_min_corrected_p": float(m2b.min_corrected_p_),
        "m2b_top1_pair": f"{m2b.top1_pair_[0]}_x_{m2b.top1_pair_[1]}",
        # M3
        "m3_detected": int(m3.detected_),
        "m3_max_freq": float(m3.max_selection_freq_),
        "m3_top1_pair": f"{m3.top1_pair_[0]}_x_{m3.top1_pair_[1]}",
    }

    # -------------------------------------------------------------------------
    # Metric D: Out-of-Sample Prediction
    # -------------------------------------------------------------------------
    metric_d = {
        "m0_r2_oos": oos["M0"]["r2_oos"],
        "m0_rmse_oos": oos["M0"]["rmse_oos"],
        "m1_r2_oos": oos["M1"]["r2_oos"],
        "m1_rmse_oos": oos["M1"]["rmse_oos"],
        "m2a_r2_oos": oos["M2a"]["r2_oos"],
        "m2a_rmse_oos": oos["M2a"]["rmse_oos"],
        "m2b_r2_oos": oos["M2b"]["r2_oos"],
        "m2b_rmse_oos": oos["M2b"]["rmse_oos"],
        "m3_r2_oos": oos["M3"]["r2_oos"],
        "m3_rmse_oos": oos["M3"]["rmse_oos"],
    }

    # -------------------------------------------------------------------------
    # Metric H: Method Bias Gate (M1 vs M1-adj)
    # -------------------------------------------------------------------------
    metric_h = {}
    if m1_adj is not None:
        metric_h = {
            "m1_adj_h1_f_hit": int(m1_adj["h1_f_hit"]),
            "m1_adj_h1_t_hit": int(m1_adj["h1_t_hit"]),
            "m1_adj_h2_hit": int(m1_adj["h2_hit"]),
            "m1_adj_delta_r2": float(m1_adj["delta_r2"]),
            "m1_adj_r_M": float(m1_adj["r_M"]),
            "h1_conclusion_changed": int(m1.h1_f_hit_ != m1_adj["h1_f_hit"]),
            "h2_conclusion_changed": int(m1.h2_hit_ != m1_adj["h2_hit"]),
        }

    # Combine into single flat dictionary
    row = {}
    row.update(metric_a)
    row.update(metric_b)
    row.update(metric_d)
    row.update(metric_e)
    row.update(metric_h)
    return row
