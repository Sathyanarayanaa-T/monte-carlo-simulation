"""Nested Cross-Validation Harness.

SPEC.md Section 5 (line 96):
- Outer CV: 5-fold, repeated twice (10 splits total).
- All fitting (scaling, tuning) occurs strictly inside training folds.
- Pooled out-of-sample predictions used to compute out-of-sample R^2 and RMSE.
- Full dataset fits compute hypothesis tests, parameter estimates, and interaction statistics.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.model_selection import RepeatedKFold

from src.methods.m0_baseline import M0Baseline
from src.methods.m1_hierarchical import M1HierarchicalOLS, fit_m1_marker_adjusted, prepare_features
from src.methods.m2_interaction import M2aPreSpecifiedInteraction, M2bAllPairwiseInteractions
from src.methods.m3_elasticnet import M3ElasticNet


def run_nested_cv(
    data: pd.DataFrame,
    scenario_dict: dict[str, Any] | None = None,
    seed: int = 42,
    n_splits: int = 5,
    n_repeats: int = 2,
    run_stability_selection: bool = True,
) -> dict[str, Any]:
    """Execute nested CV harness for M0, M1, M2a, M2b, and M3.

    Returns a dict containing:
    - out-of-sample predictions and metrics (R2, RMSE) for each method
    - full-dataset fit objects and hypothesis test results
    - marker-adjusted M1 results (M1-adj)
    """
    if scenario_dict is None:
        scenario_dict = {}

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

    preds_m0: list[float] = []
    preds_m1: list[float] = []
    preds_m2a: list[float] = []
    preds_m2b: list[float] = []
    preds_m3: list[float] = []
    y_test_pooled: list[float] = []

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
        m0 = M0Baseline()
        m0.fit(X2_train, y_train)
        preds_m0.extend(m0.predict(X2_test))

        # M1: Hierarchical OLS (Block 2 full model prediction)
        m1 = M1HierarchicalOLS()
        m1.fit(X1_train, X2_train, y_train)
        preds_m1.extend(m1.predict(X2_test))

        # M2a: OLS + pre-specified interaction
        m2a = M2aPreSpecifiedInteraction(interaction_pair=m2a_pair)
        m2a.fit(X2_train, y_train)
        preds_m2a.extend(m2a.predict(X2_test))

        # M2b: OLS + all 10 pairwise products
        m2b = M2bAllPairwiseInteractions()
        m2b.fit(X2_train, y_train)
        preds_m2b.extend(m2b.predict(X2_test))

        # M3: ElasticNet (tune inside training fold, stability selection skipped in folds)
        m3 = M3ElasticNet(random_state=split_seed)
        m3.fit(X2_train, y_train, run_stability=False)
        preds_m3.extend(m3.predict(X2_test))

    y_arr = np.array(y_test_pooled)
    total_var = float(np.sum((y_arr - np.mean(y_arr)) ** 2))

    def calc_oos_metrics(preds: list[float]) -> dict[str, float]:
        p_arr = np.array(preds)
        ssr = float(np.sum((y_arr - p_arr) ** 2))
        r2 = 1.0 - (ssr / total_var) if total_var > 1e-12 else 0.0
        rmse = float(np.sqrt(np.mean((y_arr - p_arr) ** 2)))
        return {"r2_oos": float(r2), "rmse_oos": float(rmse)}

    oos_metrics = {
        "M0": calc_oos_metrics(preds_m0),
        "M1": calc_oos_metrics(preds_m1),
        "M2a": calc_oos_metrics(preds_m2a),
        "M2b": calc_oos_metrics(preds_m2b),
        "M3": calc_oos_metrics(preds_m3),
    }

    # -------------------------------------------------------------------------
    # 2. Full-Dataset Fits for Inference & Detection
    # -------------------------------------------------------------------------
    full_m1 = M1HierarchicalOLS().fit(X_block1, X_block2, y)
    full_m2a = M2aPreSpecifiedInteraction(interaction_pair=m2a_pair).fit(X_block2, y)
    full_m2b = M2bAllPairwiseInteractions().fit(X_block2, y)
    full_m3 = M3ElasticNet(random_state=seed).fit(
        X_block2, y, run_stability=run_stability_selection
    )

    # M1-adj marker adjustment
    m1_adj_res = fit_m1_marker_adjusted(data) if "score_M" in data.columns else None

    return {
        "oos_metrics": oos_metrics,
        "m1": full_m1,
        "m1_adj": m1_adj_res,
        "m2a": full_m2a,
        "m2b": full_m2b,
        "m3": full_m3,
        "n_samples": n,
    }
