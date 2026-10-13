"""Method M3: Elastic Net with Pairwise Products & Stability Selection.

SPEC.md Section 5 (line 92):
- Predictors: 5 main effects, 10 pairwise products, covariates.
- Hyperparameter tuning: ElasticNetCV with l1_ratio in {0.5, 0.9, 1.0},
  inner CV (5-fold default; 3-fold at n <= 50), 1-SE rule.
- Interaction detection via stability selection:
  100 bootstrap refits; an interaction is detected if any product term is
  selected in at least 70% of them.
"""

from __future__ import annotations

import itertools
import time
from typing import Any
import numpy as np
import pandas as pd
from sklearn.linear_model import ElasticNet, ElasticNetCV
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

from src.methods.m2_interaction import ALL_10_PAIRS, generate_interaction_products


class M3ElasticNet:
    """Elastic Net regression with 1-SE rule and bootstrap stability selection."""

    def __init__(
        self,
        l1_ratios: list[float] | None = None,
        use_one_se_rule: bool = True,
        n_bootstraps: int = 100,
        stability_threshold: float = 0.70,
        inner_cv_folds: int = 5,
        random_state: int = 42,
    ) -> None:
        self.l1_ratios = l1_ratios if l1_ratios is not None else [0.5, 0.9, 1.0]
        self.use_one_se_rule = use_one_se_rule
        self.n_bootstraps = n_bootstraps
        self.stability_threshold = stability_threshold
        self.inner_cv_folds = inner_cv_folds
        self.random_state = random_state

        self.scaler_: StandardScaler | None = None
        self.model_: ElasticNet | None = None
        self.best_l1_ratio_: float = 1.0
        self.best_alpha_: float = 0.1
        self.feature_names_: list[str] = []
        self.product_cols_: list[str] = [f"{p[0]}_x_{p[1]}" for p in ALL_10_PAIRS]

        # Stability selection outputs
        self.selection_frequencies_: dict[tuple[str, str], float] = {}
        self.detected_: bool = False
        self.max_selection_freq_: float = 0.0
        self.top1_pair_: tuple[str, str] = ALL_10_PAIRS[0]

    def _prepare_full_design(self, X_block2: pd.DataFrame) -> pd.DataFrame:
        """Create design matrix with main effects, covariates, and 10 pairwise products."""
        prod_df = generate_interaction_products(X_block2, ALL_10_PAIRS)
        return pd.concat([X_block2, prod_df], axis=1)

    def _tune_with_one_se_rule(
        self,
        X_scaled: np.ndarray,
        y: np.ndarray,
        cv_splits: int,
        rng_seed: int,
    ) -> tuple[float, float]:
        """Tune l1_ratio and alpha using ElasticNetCV with the 1-SE rule."""
        n = len(y)
        k_folds = min(cv_splits, n)
        cv = KFold(n_splits=k_folds, shuffle=True, random_state=rng_seed)

        encv = ElasticNetCV(
            l1_ratio=self.l1_ratios,
            cv=cv,
            alphas=30,
            max_iter=3000,
            random_state=rng_seed,
            n_jobs=1,
        )
        encv.fit(X_scaled, y)

        # mse_path_ shape: (n_l1, n_alphas, n_folds)
        mse_path = encv.mse_path_
        if mse_path.ndim == 2:
            mse_path = mse_path[np.newaxis, :, :]

        mean_mse = np.mean(mse_path, axis=2)
        se_mse = np.std(mse_path, axis=2) / np.sqrt(k_folds)

        min_idx = np.unravel_index(np.argmin(mean_mse), mean_mse.shape)
        min_mse = mean_mse[min_idx]
        min_se = se_mse[min_idx]
        threshold = min_mse + min_se

        if not self.use_one_se_rule:
            best_l1 = float(self.l1_ratios[min_idx[0]])
            best_alpha = float(encv.alphas_[min_idx[0]][min_idx[1]])
            return best_l1, best_alpha

        # 1-SE rule: among all (l1, alpha) within 1-SE threshold, choose largest alpha
        eligible = np.where(mean_mse <= threshold)
        best_alpha = -1.0
        best_l1 = self.l1_ratios[0]
        for l1_idx, a_idx in zip(eligible[0], eligible[1]):
            alpha_val = float(encv.alphas_[l1_idx][a_idx])
            if alpha_val > best_alpha:
                best_alpha = alpha_val
                best_l1 = float(self.l1_ratios[l1_idx])

        return best_l1, best_alpha

    def fit(
        self,
        X_block2: pd.DataFrame,
        y: pd.Series | np.ndarray,
        run_stability: bool = True,
    ) -> M3ElasticNet:
        """Fit ElasticNet model and optionally run stability selection."""
        y_arr = np.asarray(y, dtype=np.float64)
        n = len(y_arr)

        X_full = self._prepare_full_design(X_block2)
        self.feature_names_ = list(X_full.columns)

        # Scale features
        self.scaler_ = StandardScaler()
        X_scaled = self.scaler_.fit_transform(X_full)

        # Adjust inner CV folds for small n (SPEC Section 5: 3-fold at n <= 50)
        cv_folds = 3 if n <= 50 else self.inner_cv_folds

        t_start_fit = time.perf_counter()
        # Tune hyperparameters
        self.best_l1_ratio_, self.best_alpha_ = self._tune_with_one_se_rule(
            X_scaled, y_arr, cv_folds, self.random_state
        )

        # Fit final model on full training set
        self.model_ = ElasticNet(
            l1_ratio=self.best_l1_ratio_,
            alpha=self.best_alpha_,
            max_iter=3000,
            random_state=self.random_state,
        )
        self.model_.fit(X_scaled, y_arr)
        self.time_full_fit_ = time.perf_counter() - t_start_fit
        self.time_stability_ = 0.0

        if not run_stability:
            return self

        # ---------------------------------------------------------------------
        # Stability Selection (100 bootstrap refits)
        # ---------------------------------------------------------------------
        t_start_stab = time.perf_counter()
        rng = np.random.default_rng(self.random_state + 500)
        prod_indices = [self.feature_names_.index(p_col) for p_col in self.product_cols_]

        selection_counts = np.zeros(len(ALL_10_PAIRS), dtype=np.int64)

        for b in range(self.n_bootstraps):
            boot_idx = rng.choice(n, size=n, replace=True)
            X_boot = X_scaled[boot_idx]
            y_boot = y_arr[boot_idx]

            boot_model = ElasticNet(
                l1_ratio=self.best_l1_ratio_,
                alpha=self.best_alpha_,
                max_iter=3000,
                random_state=rng.integers(1, 1_000_000),
            )
            boot_model.fit(X_boot, y_boot)

            # Check non-zero coefficients for each interaction product
            for i, p_idx in enumerate(prod_indices):
                if abs(boot_model.coef_[p_idx]) > 1e-6:
                    selection_counts[i] += 1

        freqs = selection_counts / self.n_bootstraps
        self.selection_frequencies_ = {pair: float(f) for pair, f in zip(ALL_10_PAIRS, freqs)}

        self.max_selection_freq_ = float(np.max(freqs))
        self.detected_ = bool(self.max_selection_freq_ >= self.stability_threshold)

        best_idx = int(np.argmax(freqs))
        self.top1_pair_ = ALL_10_PAIRS[best_idx]
        self.time_stability_ = time.perf_counter() - t_start_stab

        return self

    def predict(self, X_block2: pd.DataFrame) -> np.ndarray:
        """Predict outcomes for new data using the fitted ElasticNet model."""
        X_full = self._prepare_full_design(X_block2)
        X_scaled = self.scaler_.transform(X_full)
        return np.asarray(self.model_.predict(X_scaled), dtype=np.float64)
