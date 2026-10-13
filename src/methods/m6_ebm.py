"""Method M6: Explainable Boosting Machine (EBM).

SPEC.md Section 5 (lines 95, 97-101):
- interpret's ExplainableBoostingRegressor with:
  * interactions = 10
  * max_bins = 32
  * learning_rate = 0.01
  * outer_bags = 8
  * min_samples_leaf = 3
- Used without inner tuning to keep compute manageable.
- Pairwise term importance used for interaction detection.
- Shared residual-resampling interaction test.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from interpret.glassbox import ExplainableBoostingRegressor

from src.methods.residual_resampling import (
    SUBSTANTIVE_COLS_DEFAULT,
    extract_ebm_interaction_stats,
    run_residual_resampling_test,
)


class M6ExplainableBoosting:
    """Explainable Boosting Machine (EBM) with pairwise interaction detection."""

    def __init__(
        self,
        interactions: int = 10,
        max_bins: int = 32,
        learning_rate: float = 0.01,
        outer_bags: int = 8,
        min_samples_leaf: int = 3,
        random_state: int = 42,
    ) -> None:
        self.interactions = interactions
        self.max_bins = max_bins
        self.learning_rate = learning_rate
        self.outer_bags = outer_bags
        self.min_samples_leaf = min_samples_leaf
        self.random_state = random_state

        self.model_: ExplainableBoostingRegressor | None = None
        self.feature_names_: list[str] = []
        self.feature_importances_: dict[str, float] = {}

        # Interaction testing outputs
        self.t_observed_: float = 0.0
        self.pairwise_interactions_: dict[tuple[str, str], float] = {}
        self.top1_pair_: tuple[str, str] = (SUBSTANTIVE_COLS_DEFAULT[0], SUBSTANTIVE_COLS_DEFAULT[1])
        self.detected_: bool = False
        self.p_value_: float = 1.0

    def fit(self, X: pd.DataFrame, y: pd.Series | np.ndarray) -> M6ExplainableBoosting:
        """Fit Explainable Boosting Machine."""
        y_arr = np.asarray(y, dtype=np.float64)
        self.feature_names_ = list(X.columns)

        self.model_ = ExplainableBoostingRegressor(
            interactions=self.interactions,
            max_bins=self.max_bins,
            learning_rate=self.learning_rate,
            outer_bags=self.outer_bags,
            min_samples_leaf=self.min_samples_leaf,
            random_state=self.random_state,
        )
        self.model_.fit(X, y_arr)

        # Main effect term importances
        term_names = getattr(self.model_, "term_names_", [])
        term_importances = self.model_.term_importances()
        for name, imp in zip(term_names, term_importances):
            if " & " not in name and name in self.feature_names_:
                self.feature_importances_[name] = float(imp)

        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        """Predict outcomes for new data."""
        return np.asarray(self.model_.predict(X), dtype=np.float64)

    def compute_interaction_stats(
        self,
        X: pd.DataFrame,
        substantive_cols: list[str] = SUBSTANTIVE_COLS_DEFAULT,
    ) -> tuple[float, dict[tuple[str, str], float], tuple[str, str]]:
        """Extract EBM pairwise interaction term statistics."""
        return extract_ebm_interaction_stats(self.model_, X, substantive_cols)

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
            m = ExplainableBoostingRegressor(
                interactions=self.interactions,
                max_bins=self.max_bins,
                learning_rate=self.learning_rate,
                outer_bags=self.outer_bags,
                min_samples_leaf=self.min_samples_leaf,
                random_state=s,
            )
            m.fit(X_df, y_vec)
            return extract_ebm_interaction_stats(m, X_df, substantive_cols)

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
