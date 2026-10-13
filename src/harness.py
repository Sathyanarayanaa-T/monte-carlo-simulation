"""Nested Cross-Validation Harness.

SPEC.md Section 5 (line 96):
- Outer CV: 5-fold, repeated twice (10 splits total).
- All fitting (scaling, tuning) occurs strictly inside training folds.
- Pooled out-of-sample predictions used to compute out-of-sample R^2 and RMSE.
- Full dataset fits compute hypothesis tests, parameter estimates, and interaction statistics.
"""

from __future__ import annotations

import time
from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from src.methods.m0_baseline import M0Baseline
from src.methods.m1_hierarchical import M1HierarchicalOLS, fit_m1_marker_adjusted, prepare_features
from src.methods.m2_interaction import M2aPreSpecifiedInteraction, M2bAllPairwiseInteractions
from src.methods.m3_elasticnet import M3ElasticNet
from src.methods.m4_random_forest import M4RandomForest
from src.methods.m5_gradient_boosting import M5GradientBoosting
from src.methods.m6_ebm import M6ExplainableBoosting


ALL_METHODS = ["M0", "M1", "M2a", "M2b", "M3", "M4", "M5", "M6"]


def run_nested_cv(
    data: pd.DataFrame,
    scenario_dict: dict[str, Any] | None = None,
    methods: list[str] | None = None,
    seed: int = 42,
    n_splits: int = 5,
    n_repeats: int = 2,
    run_stability_selection: bool = True,
    run_ml_interaction_tests: bool = True,
    b_null_refits: int = 50,
) -> dict[str, Any]:
    """Execute nested CV harness across specified models.

    Parameters:
    - data: survey DataFrame containing scores and covariates
    - scenario_dict: scenario parameters
    - methods: list of methods to fit (defaults to ALL_METHODS)
    - seed: random seed
    - n_splits: outer CV folds (default 5)
    - n_repeats: outer CV repetitions (default 2 -> 10 splits)
    - run_stability_selection: run M3 100-bootstrap stability selection
    - run_ml_interaction_tests: run residual-resampling test for M4, M5, M6
    - b_null_refits: number of null datasets for residual resampling (default 50)
    """
    if scenario_dict is None:
        scenario_dict = {}

    active_methods = [m.upper() for m in (methods if methods is not None else ALL_METHODS)]

    nl_form = str(scenario_dict.get("non_linear_form", "interaction"))
    # Pre-specified pair for M2a: (A_tech x B-i for interaction; A_tech x B-iii for threshold)
    if nl_form == "threshold":
        m2a_pair = ("score_A_tech", "score_B_iii")
    else:
        m2a_pair = ("score_A_tech", "score_B_i")

    X_block1, X_block2 = prepare_features(data)
    y = data["score_C"].to_numpy(dtype=np.float64)
    n = len(y)

    # -------------------------------------------------------------------------
    # 1. Outer Repeated Cross-Validation
    # -------------------------------------------------------------------------
    rkf = RepeatedKFold(n_splits=n_splits, n_repeats=n_repeats, random_state=seed)

    preds_by_method: dict[str, list[float]] = {m: [] for m in active_methods}
    y_test_pooled: list[float] = []

    time_nested_cv: dict[str, float] = {"m3": 0.0, "m4": 0.0, "m5": 0.0, "m6": 0.0}
    time_full_fit: dict[str, float] = {"m3": 0.0, "m4": 0.0, "m5": 0.0, "m6": 0.0}
    time_null_test: dict[str, float] = {"m3": 0.0, "m4": 0.0, "m5": 0.0, "m6": 0.0}

    split_seed = seed
    for train_idx, test_idx in rkf.split(X_block2):
        split_seed += 1
        X1_train = X_block1.iloc[train_idx]
        X2_train = X_block2.iloc[train_idx]
        y_train = y[train_idx]

        X1_test = X_block1.iloc[test_idx]
        X2_test = X_block2.iloc[test_idx]
        y_test = y[test_idx]

        y_test_pooled.extend(y_test)

        # M0: Baseline mean
        if "M0" in active_methods:
            m0 = M0Baseline().fit(X2_train, y_train)
            preds_by_method["M0"].extend(m0.predict(X2_test))

        # M1: Hierarchical OLS
        if "M1" in active_methods:
            m1 = M1HierarchicalOLS().fit(X1_train, X2_train, y_train)
            preds_by_method["M1"].extend(m1.predict(X2_test))

        # M2a: OLS + pre-specified interaction
        if "M2A" in active_methods:
            m2a = M2aPreSpecifiedInteraction(interaction_pair=m2a_pair).fit(X2_train, y_train)
            preds_by_method["M2A"].extend(m2a.predict(X2_test))

        # M2b: OLS + all 10 pairwise products
        if "M2B" in active_methods:
            m2b = M2bAllPairwiseInteractions().fit(X2_train, y_train)
            preds_by_method["M2B"].extend(m2b.predict(X2_test))

        # M3: ElasticNet
        if "M3" in active_methods:
            t0 = time.perf_counter()
            m3 = M3ElasticNet(random_state=split_seed).fit(X2_train, y_train, run_stability=False)
            preds_by_method["M3"].extend(m3.predict(X2_test))
            time_nested_cv["m3"] += time.perf_counter() - t0

        # M4: Random Forest (tuned by OOB R2)
        if "M4" in active_methods:
            t0 = time.perf_counter()
            m4 = M4RandomForest(random_state=split_seed).fit(X2_train, y_train, tune_oob=True)
            preds_by_method["M4"].extend(m4.predict(X2_test))
            time_nested_cv["m4"] += time.perf_counter() - t0

        # M5: Gradient Boosting (tuned by inner CV)
        if "M5" in active_methods:
            t0 = time.perf_counter()
            m5 = M5GradientBoosting(random_state=split_seed).fit(X2_train, y_train, tune_cv=True)
            preds_by_method["M5"].extend(m5.predict(X2_test))
            time_nested_cv["m5"] += time.perf_counter() - t0

        # M6: EBM
        if "M6" in active_methods:
            t0 = time.perf_counter()
            m6 = M6ExplainableBoosting(random_state=split_seed).fit(X2_train, y_train)
            preds_by_method["M6"].extend(m6.predict(X2_test))
            time_nested_cv["m6"] += time.perf_counter() - t0

    y_arr = np.array(y_test_pooled)
    total_var = float(np.sum((y_arr - np.mean(y_arr)) ** 2))

    def calc_oos_metrics(preds: list[float]) -> dict[str, float]:
        if len(preds) == 0:
            return {"r2_oos": 0.0, "rmse_oos": 0.0}
        p_arr = np.array(preds)
        ssr = float(np.sum((y_arr - p_arr) ** 2))
        r2 = 1.0 - (ssr / total_var) if total_var > 1e-12 else 0.0
        rmse = float(np.sqrt(np.mean((y_arr - p_arr) ** 2)))
        return {"r2_oos": float(r2), "rmse_oos": float(rmse)}

    oos_metrics = {m: calc_oos_metrics(preds_by_method[m]) for m in active_methods}

    # -------------------------------------------------------------------------
    # 2. Full-Dataset Fits for Inference & Interaction Detection
    # -------------------------------------------------------------------------
    full_fits: dict[str, Any] = {}

    if "M0" in active_methods:
        full_fits["m0"] = M0Baseline().fit(X_block2, y)
    if "M1" in active_methods:
        full_fits["m1"] = M1HierarchicalOLS().fit(X_block1, X_block2, y)
        full_fits["m1_adj"] = fit_m1_marker_adjusted(data) if "score_M" in data.columns else None
    if "M2A" in active_methods:
        full_fits["m2a"] = M2aPreSpecifiedInteraction(interaction_pair=m2a_pair).fit(X_block2, y)
    if "M2B" in active_methods:
        full_fits["m2b"] = M2bAllPairwiseInteractions().fit(X_block2, y)
    if "M3" in active_methods:
        m3_fitted = M3ElasticNet(random_state=seed).fit(
            X_block2, y, run_stability=run_stability_selection
        )
        full_fits["m3"] = m3_fitted
        time_full_fit["m3"] = getattr(m3_fitted, "time_full_fit_", 0.0)

    if "M4" in active_methods:
        t0 = time.perf_counter()
        m4 = M4RandomForest(random_state=seed).fit(X_block2, y, tune_oob=True)
        time_full_fit["m4"] = time.perf_counter() - t0
        if run_ml_interaction_tests:
            t0 = time.perf_counter()
            m4.run_interaction_test(X_block2, y, b_null_refits=b_null_refits)
            time_null_test["m4"] = time.perf_counter() - t0
        else:
            t_obs, pairs, top1 = m4.compute_interaction_stats(X_block2)
            m4.t_observed_ = t_obs
            m4.pairwise_interactions_ = pairs
            m4.top1_pair_ = top1
        full_fits["m4"] = m4

    if "M5" in active_methods:
        t0 = time.perf_counter()
        m5 = M5GradientBoosting(random_state=seed).fit(X_block2, y, tune_cv=True)
        time_full_fit["m5"] = time.perf_counter() - t0
        if run_ml_interaction_tests:
            t0 = time.perf_counter()
            m5.run_interaction_test(X_block2, y, b_null_refits=b_null_refits)
            time_null_test["m5"] = time.perf_counter() - t0
        else:
            t_obs, pairs, top1 = m5.compute_interaction_stats(X_block2)
            m5.t_observed_ = t_obs
            m5.pairwise_interactions_ = pairs
            m5.top1_pair_ = top1
        full_fits["m5"] = m5

    if "M6" in active_methods:
        t0 = time.perf_counter()
        m6 = M6ExplainableBoosting(random_state=seed).fit(X_block2, y)
        time_full_fit["m6"] = time.perf_counter() - t0
        if run_ml_interaction_tests:
            t0 = time.perf_counter()
            m6.run_interaction_test(X_block2, y, b_null_refits=b_null_refits)
            time_null_test["m6"] = time.perf_counter() - t0
        else:
            t_obs, pairs, top1 = m6.compute_interaction_stats(X_block2)
            m6.t_observed_ = t_obs
            m6.pairwise_interactions_ = pairs
            m6.top1_pair_ = top1
        full_fits["m6"] = m6

    output = {
        "oos_metrics": oos_metrics,
        "n_samples": n,
        "active_methods": active_methods,
        "X_block2": X_block2,
        "y": y,
        "data": data,
        "component_timings": {
            "time_nested_cv": time_nested_cv,
            "time_full_fit": time_full_fit,
            "time_null_test": time_null_test,
        },
    }
    output.update(full_fits)
    return output
