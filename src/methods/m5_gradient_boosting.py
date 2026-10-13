"""Method M5: Gradient Boosting Regressor.

SPEC.md Section 5 (lines 94, 97-101):
- Learning rate 0.03, subsample 0.7, min_samples_leaf 5.
- Inner CV grid: max_depth in {1, 2, 3}, n_estimators in {100, 200, 400}.
- TreeSHAP for feature importance and pairwise interaction values.
- Shared residual-resampling interaction test.
"""

from __future__ import annotations

import itertools
from typing import Any
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import KFold

from src.methods.residual_resampling import (
    SUBSTANTIVE_COLS_DEFAULT,
    extract_shap_interaction_stats,
    run_residual_resampling_test,
)


class M5GradientBoosting:
    """Gradient Boosting regressor tuned by inner CV grid search."""

    def __init__(
        self,
        learning_rate: float = 0.03,
        subsample: float = 0.7,
        min_samples_leaf: int = 5,
        max_depth_grid: list[int] | None = None,
        n_estimators_grid: list[int] | None = None,
        inner_cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.learning_rate = learning_rate
        self.subsample = subsample
        self.min_samples_leaf = min_samples_leaf
        self.max_depth_grid = max_depth_grid if max_depth_grid is not None else [1, 2, 3]
        self.n_estimators_grid = n_estimators_grid if n_estimators_grid is not None else [100, 200, 400]
        self.inner_cv_folds = inner_cv_folds
        self.random_state = random_state

        self.model_: GradientBoostingRegressor | None = None
        self.best_max_depth_: int = 2
        self.best_n_estimators_: int = 100
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
        tune_cv: bool = True,
    ) -> M5GradientBoosting:
        """Fit Gradient Boosting model, tuning hyperparameters via inner CV."""
        y_arr = np.asarray(y, dtype=np.float64)
        n = len(y_arr)
        self.feature_names_ = list(X.columns)

        if tune_cv and (len(self.max_depth_grid) > 1 or len(self.n_estimators_grid) > 1):
            k_folds = min(3 if n <= 50 else self.inner_cv_folds, n)
            cv = KFold(n_splits=k_folds, shuffle=True, random_state=self.random_state)

            best_mse = float("inf")
            best_depth = self.max_depth_grid[0]
            best_n_est = self.n_estimators_grid[0]

            for depth in self.max_depth_grid:
                for n_est in self.n_estimators_grid:
                    fold_mses = []
                    for train_idx, val_idx in cv.split(X):
                        m = GradientBoostingRegressor(
                            learning_rate=self.learning_rate,
                            subsample=self.subsample,
                            min_samples_leaf=self.min_samples_leaf,
                            max_depth=depth,
                            n_estimators=n_est,
                            random_state=self.random_state,
                        )
                        m.fit(X.iloc[train_idx].values, y_arr[train_idx])
                        preds = m.predict(X.iloc[val_idx].values)
                        fold_mses.append(float(np.mean((y_arr[val_idx] - preds) ** 2)))

                    mean_mse = float(np.mean(fold_mses))
                    if mean_mse < best_mse:
                        best_mse = mean_mse
                        best_depth = depth
                        best_n_est = n_est

            self.best_max_depth_ = best_depth
            self.best_n_estimators_ = best_n_est
        else:
            self.best_max_depth_ = self.max_depth_grid[0]
            self.best_n_estimators_ = self.n_estimators_grid[0]

        # Fit final model on full training set
        self.model_ = GradientBoostingRegressor(
            learning_rate=self.learning_rate,
            subsample=self.subsample,
            min_samples_leaf=self.min_samples_leaf,
            max_depth=self.best_max_depth_,
            n_estimators=self.best_n_estimators_,
            random_state=self.random_state,
        )
        self.model_.fit(X.values, y_arr)

        # Feature importances
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
            m = GradientBoostingRegressor(
                learning_rate=self.learning_rate,
                subsample=self.subsample,
                min_samples_leaf=self.min_samples_leaf,
                max_depth=self.best_max_depth_,
                n_estimators=self.best_n_estimators_,
                random_state=s,
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
