"""Method M2: Moderated OLS Regressions (M2a and M2b).

SPEC.md Section 5 (lines 90-91):
- M2a: M1 full model + one pre-specified interaction matching the DGP:
  * A_tech x B-i for the interaction DGP
  * A_tech x B-iii for the threshold DGP
  * Evaluated via t-test on the interaction coefficient (alpha = 0.05).
- M2b: M1 full model + all 10 pairwise products of the five predictors:
  * Holm-Bonferroni correction across the 10 product terms.
  * Detection if any corrected p < 0.05.
  * Top-1 selected pair based on smallest p-value / largest |t|.
"""

from __future__ import annotations

import itertools
from typing import Any
import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests

from src.methods.m1_hierarchical import prepare_features


PREDICTOR_NAMES = ["score_A_tech", "score_A_mkt", "score_B_i", "score_B_ii", "score_B_iii"]
ALL_10_PAIRS = list(itertools.combinations(PREDICTOR_NAMES, 2))


def generate_interaction_products(
    X_main: pd.DataFrame,
    pairs: list[tuple[str, str]],
) -> pd.DataFrame:
    """Generate product terms for specified pairs of predictors."""
    prod_df = pd.DataFrame(index=X_main.index)
    for v1, v2 in pairs:
        col_name = f"{v1}_x_{v2}"
        prod_df[col_name] = X_main[v1] * X_main[v2]
    return prod_df


class M2aPreSpecifiedInteraction:
    """M2a: OLS with one pre-specified interaction term."""

    def __init__(self, interaction_pair: tuple[str, str] = ("score_A_tech", "score_B_i"), alpha: float = 0.05) -> None:
        self.interaction_pair = interaction_pair
        self.alpha = alpha
        self.fit_ = None
        self.fit_hc3_ = None
        self.interaction_col_ = f"{interaction_pair[0]}_x_{interaction_pair[1]}"
        self.interaction_coef_: float = 0.0
        self.interaction_se_: float = 0.0
        self.interaction_t_: float = 0.0
        self.interaction_pvalue_: float = 1.0
        self.detected_: bool = False
        self.feature_names_: list[str] = []

    def fit(self, X_block2: pd.DataFrame, y: pd.Series | np.ndarray) -> M2aPreSpecifiedInteraction:
        """Fit model with main effects, covariates, and the pre-specified product term."""
        y_arr = np.asarray(y, dtype=np.float64)

        X_full = X_block2.copy()
        X_full[self.interaction_col_] = X_full[self.interaction_pair[0]] * X_full[self.interaction_pair[1]]
        self.feature_names_ = list(X_full.columns)

        X_const = sm.add_constant(X_full, has_constant="add")
        self.fit_ = sm.OLS(y_arr, X_const).fit()
        self.fit_hc3_ = self.fit_.get_robustcov_results(cov_type="HC3")

        cols = list(X_const.columns)
        idx = cols.index(self.interaction_col_)

        self.interaction_coef_ = float(self.fit_hc3_.params[idx])
        self.interaction_se_ = float(self.fit_hc3_.bse[idx])
        self.interaction_t_ = float(self.fit_hc3_.tvalues[idx])
        self.interaction_pvalue_ = float(self.fit_hc3_.pvalues[idx])
        self.detected_ = bool(self.interaction_pvalue_ < self.alpha)

        return self

    def predict(self, X_block2: pd.DataFrame) -> np.ndarray:
        """Predict outcomes for new data."""
        X_full = X_block2.copy()
        X_full[self.interaction_col_] = X_full[self.interaction_pair[0]] * X_full[self.interaction_pair[1]]
        X_const = sm.add_constant(X_full, has_constant="add")
        return np.asarray(self.fit_.predict(X_const), dtype=np.float64)


class M2bAllPairwiseInteractions:
    """M2b: OLS with all 10 pairwise products and Holm-Bonferroni correction."""

    def __init__(self, alpha: float = 0.05, pairs: list[tuple[str, str]] | None = None) -> None:
        self.alpha = alpha
        self.pairs = pairs if pairs is not None else ALL_10_PAIRS
        self.fit_ = None
        self.fit_hc3_ = None
        self.product_cols_ = [f"{p[0]}_x_{p[1]}" for p in self.pairs]
        self.raw_pvalues_: list[float] = []
        self.corrected_pvalues_: list[float] = []
        self.detected_: bool = False
        self.top1_pair_: tuple[str, str] = self.pairs[0]
        self.min_raw_p_: float = 1.0
        self.min_corrected_p_: float = 1.0
        self.feature_names_: list[str] = []

    def fit(self, X_block2: pd.DataFrame, y: pd.Series | np.ndarray) -> M2bAllPairwiseInteractions:
        """Fit model with all 10 product terms and apply Holm correction."""
        y_arr = np.asarray(y, dtype=np.float64)

        # Build product terms
        prod_df = generate_interaction_products(X_block2, self.pairs)
        X_full = pd.concat([X_block2, prod_df], axis=1)
        self.feature_names_ = list(X_full.columns)

        X_const = sm.add_constant(X_full, has_constant="add")
        self.fit_ = sm.OLS(y_arr, X_const).fit()
        self.fit_hc3_ = self.fit_.get_robustcov_results(cov_type="HC3")

        cols = list(X_const.columns)
        pvals = []
        tvals = []
        for p_col in self.product_cols_:
            idx = cols.index(p_col)
            pvals.append(float(self.fit_hc3_.pvalues[idx]))
            tvals.append(abs(float(self.fit_hc3_.tvalues[idx])))

        self.raw_pvalues_ = pvals
        # Holm-Bonferroni correction
        reject, corr_p, _, _ = multipletests(pvals, alpha=self.alpha, method="holm")
        self.corrected_pvalues_ = [float(p) for p in corr_p]

        self.detected_ = bool(np.any(reject))
        self.min_raw_p_ = float(np.min(pvals))
        self.min_corrected_p_ = float(np.min(self.corrected_pvalues_))

        # Top-1 pair: smallest p-value / largest |t|
        best_idx = int(np.argmin(pvals))
        self.top1_pair_ = self.pairs[best_idx]

        return self

    def predict(self, X_block2: pd.DataFrame) -> np.ndarray:
        """Predict outcomes for new data."""
        prod_df = generate_interaction_products(X_block2, self.pairs)
        X_full = pd.concat([X_block2, prod_df], axis=1)
        X_const = sm.add_constant(X_full, has_constant="add")
        return np.asarray(self.fit_.predict(X_const), dtype=np.float64)
