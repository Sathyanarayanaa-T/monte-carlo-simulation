"""Method M4: Random Forest Regressor.

SPEC.md Section 5 (lines 93, 97-101):
- 300 trees, bootstrap=True.
- Tuned by out-of-bag R^2 over min_samples_leaf in {3, 5, 10}.
- TreeSHAP for feature importance and pairwise interaction values.
- Shared residual-resampling interaction test.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
import shap

from src.methods.residual_resampling import (
    SUBSTANTIVE_COLS_DEFAULT,
    extract_shap_interaction_stats,
    run_residual_resampling_test,
)


class M4RandomForest:
    """Random Forest regressor tuned by out-of-bag R2."""

    def __init__(
        self,
        n_estimators: int = 300,
        min_samples_leaf_grid: list[int] | None = None,
        random_state: int = 42,
    ) -> None:
        self.n_estimators = n_estimators
        self.min_samples_leaf_grid = min_samples_leaf_grid if min_samples_leaf_grid is not None else [3, 5, 10]
        self.random_state = random_state

        self.model_: RandomForestRegressor | None = None
        self.best_min_samples_leaf_: int = 5
        self.feature_names_: list[str] = []
        self.feature_importances_: dict[str, float] = {}

        # Interaction testing outputs
        self.t_observed_: float = 0.0
        self.pairwise_interactions_: dict[tuple[str, str], float] = {}
        self.top1_pair_: tuple[str, str] = (SUBSTANTIVE_COLS_DEFAULT[0], SUBSTANTIVE_COLS_DEFAULT[1])
        self.detected_: bool = False
        self.p_value_: float = 1.0

    def fit(
        self,
        X: pd.DataFrame,
        y: pd.Series | np.ndarray,
        tune_oob: bool = True,
    ) -> M4RandomForest:
        """Fit Random Forest model, tuning min_samples_leaf via out-of-bag R2."""
        y_arr = np.asarray(y, dtype=np.float64)
        self.feature_names_ = list(X.columns)

        if tune_oob and len(self.min_samples_leaf_grid) > 1:
            best_score = -float("inf")
            best_leaf = self.min_samples_leaf_grid[0]
            best_rf = None

            for leaf in self.min_samples_leaf_grid:
                rf = RandomForestRegressor(
                    n_estimators=self.n_estimators,
                    min_samples_leaf=leaf,
                    bootstrap=True,
                    oob_score=True,
                    random_state=self.random_state,
                    n_jobs=1,
                )
                rf.fit(X.values, y_arr)
                score = float(rf.oob_score_)
                if score > best_score:
                    best_score = score
                    best_leaf = leaf
                    best_rf = rf

            self.best_min_samples_leaf_ = best_leaf
            self.model_ = best_rf
        else:
            self.best_min_samples_leaf_ = self.min_samples_leaf_grid[0]
            self.model_ = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.best_min_samples_leaf_,
                bootstrap=True,
                oob_score=True,
                random_state=self.random_state,
                n_jobs=1,
            )
            self.model_.fit(X.values, y_arr)

        # Feature importances for substantive predictors
        for name, imp in zip(self.feature_names_, self.model_.feature_importances_):
            self.feature_importances_[name] = float(imp)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict outcomes for new data."""
        return np.asarray(self.model_.predict(X.values), dtype=np.float64)

    def compute_interaction_stats(
        self,
        X: pd.DataFrame,
        substantive_cols: list[str] = SUBSTANTIVE_COLS_DEFAULT,
    ) -> tuple[float, dict[tuple[str, str], float], tuple[str, str]]:
        """Compute TreeSHAP interaction statistics."""
        return extract_shap_interaction_stats(self.model_, X, substantive_cols)

    def run_interaction_test(
        self,
        X: pd.DataFrame,
        y: np.ndarray,
        b_null_refits: int = 100,
        alpha: float = 0.05,
        substantive_cols: list[str] = SUBSTANTIVE_COLS_DEFAULT,
    ) -> dict[str, Any]:
        """Run residual-resampling interaction test."""
        def fit_and_score(X_df: pd.DataFrame, y_vec: np.ndarray, s: int):
            m = RandomForestRegressor(
                n_estimators=self.n_estimators,
                min_samples_leaf=self.best_min_samples_leaf_,
                bootstrap=True,
                random_state=s,
                n_jobs=1,
            )
            m.fit(X_df.values, y_vec)
            return extract_shap_interaction_stats(m, X_df, substantive_cols)

        res = run_residual_resampling_test(
            fit_and_score_fn=fit_and_score,
            X=X,
            y=y,
            b_null_refits=b_null_refits,
            alpha=alpha,
            seed=self.random_state,
        )
        self.t_observed_ = res["t_observed"]
        self.pairwise_interactions_ = res["pairwise_interactions"]
        self.top1_pair_ = res["top1_pair"]
        self.detected_ = res["detected"]
        self.p_value_ = res["p_value"]
        return res
