"""Method M1: Hierarchical Ordinary Least Squares Regression & M1-adj.

SPEC.md Section 5 (lines 89, 106) & Section 6:
- Block 1: Contextual predictors (B-i, B-ii, B-iii) + Covariates.
- Block 2: Full model adding Individual predictors (A_tech, A_mkt).
- Tests:
  * H1 Block power: F-test on Delta R^2 for Block 2 addition (alpha = 0.05).
  * H1 Coefficient power: t-test on A_tech in full model with HC3 robust SE (alpha = 0.05).
  * H2 Power: Linear contrast test of beta(B-i) - beta(B-ii) = 0 with HC3 robust SE,
    with correct-sign rule (B-i > B-ii).
- M1-adj: Marker-adjusted refit via Lindell-Whitney correlation correction.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd
from scipy import stats
import statsmodels.api as sm

from src.measurement import lindell_whitney


def prepare_features(
    df: pd.DataFrame,
    contextual_cols: list[str] | None = None,
    individual_cols: list[str] | None = None,
    covariate_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Prepare Block 1 and Block 2 feature matrices with dummy-coded semester."""
    if contextual_cols is None:
        contextual_cols = ["score_B_i", "score_B_ii", "score_B_iii"]
    if individual_cols is None:
        individual_cols = ["score_A_tech", "score_A_mkt"]
    if covariate_cols is None:
        covariate_cols = ["semester", "covar_prototype", "covar_hackathon", "covar_internship"]

    # Copy and process covariates
    cov_df = pd.DataFrame(index=df.index)
    for col in covariate_cols:
        if col == "semester":
            # Semester {6, 7, 8}: dummy code with semester 6 as reference
            s = df["semester"]
            cov_df["semester_7"] = (s == 7).astype(np.float64)
            cov_df["semester_8"] = (s == 8).astype(np.float64)
        else:
            cov_df[col] = df[col].astype(np.float64)

    # Block 1 features: Contextual + Covariates
    ctx_df = df[contextual_cols].astype(np.float64)
    X_block1 = pd.concat([ctx_df, cov_df], axis=1)

    # Block 2 features: Block 1 + Individual
    ind_df = df[individual_cols].astype(np.float64)
    X_block2 = pd.concat([ctx_df, ind_df, cov_df], axis=1)

    return X_block1, X_block2


class M1HierarchicalOLS:
    """Hierarchical OLS regression model with HC3 robust standard errors."""

    def __init__(self, alpha: float = 0.05) -> None:
        self.alpha = alpha
        self.fit_block1_ = None
        self.fit_block2_ = None
        self.fit_block2_hc3_ = None
        self.r2_block1_: float = 0.0
        self.r2_block2_: float = 0.0
        self.delta_r2_: float = 0.0
        self.f_stat_: float = 0.0
        self.f_pvalue_: float = 1.0
        self.h1_f_hit_: bool = False
        self.a_tech_coef_: float = 0.0
        self.a_tech_se_: float = 0.0
        self.a_tech_pvalue_: float = 1.0
        self.h1_t_hit_: bool = False
        self.contrast_diff_: float = 0.0
        self.contrast_se_: float = 0.0
        self.contrast_pvalue_: float = 1.0
        self.h2_hit_: bool = False
        self.feature_names_: list[str] = []

    def fit(
        self,
        X_block1: pd.DataFrame | np.ndarray,
        X_block2: pd.DataFrame | np.ndarray,
        y: pd.Series | np.ndarray,
    ) -> M1HierarchicalOLS:
        """Fit Block 1 and Block 2 OLS regressions and run hypothesis tests."""
        y_arr = np.asarray(y, dtype=np.float64)
        n = len(y_arr)

        # Standardize continuous predictors in X
        if isinstance(X_block1, pd.DataFrame):
            X1_df = X_block1.copy()
            X2_df = X_block2.copy()
            self.feature_names_ = list(X2_df.columns)
        else:
            X1_df = pd.DataFrame(X_block1)
            X2_df = pd.DataFrame(X_block2)
            self.feature_names_ = [f"x_{i}" for i in range(X2_df.shape[1])]

        # Add constant
        X1_const = sm.add_constant(X1_df, has_constant="add")
        X2_const = sm.add_constant(X2_df, has_constant="add")

        # Fit models
        self.fit_block1_ = sm.OLS(y_arr, X1_const).fit()
        self.fit_block2_ = sm.OLS(y_arr, X2_const).fit()
        # HC3 robust covariance fit for full model
        self.fit_block2_hc3_ = self.fit_block2_.get_robustcov_results(cov_type="HC3")

        self.r2_block1_ = float(self.fit_block1_.rsquared)
        self.r2_block2_ = float(self.fit_block2_.rsquared)
        self.delta_r2_ = max(0.0, self.r2_block2_ - self.r2_block1_)

        # Classical F-test for Delta R^2 (q added variables)
        p1 = X1_df.shape[1]
        p2 = X2_df.shape[1]
        q = p2 - p1
        df_denom = n - p2 - 1

        if q > 0 and df_denom > 0 and (1.0 - self.r2_block2_) > 1e-12:
            ssr1 = float(self.fit_block1_.ssr)
            ssr2 = float(self.fit_block2_.ssr)
            self.f_stat_ = float(((ssr1 - ssr2) / q) / (ssr2 / df_denom))
            self.f_stat_ = max(0.0, self.f_stat_)
            self.f_pvalue_ = float(1.0 - stats.f.cdf(self.f_stat_, q, df_denom))
        else:
            self.f_stat_ = 0.0
            self.f_pvalue_ = 1.0

        self.h1_f_hit_ = bool(self.f_pvalue_ < self.alpha)

        # H1 Coefficient test on A_tech (HC3)
        a_tech_name = "score_A_tech" if "score_A_tech" in X2_const.columns else None
        if a_tech_name is not None:
            idx = list(X2_const.columns).index(a_tech_name)
            self.a_tech_coef_ = float(self.fit_block2_hc3_.params[idx])
            self.a_tech_se_ = float(self.fit_block2_hc3_.bse[idx])
            self.a_tech_pvalue_ = float(self.fit_block2_hc3_.pvalues[idx])
            # H1 coefficient power: p < 0.05 with positive sign
            self.h1_t_hit_ = bool(self.a_tech_pvalue_ < self.alpha and self.a_tech_coef_ > 0.0)

        # H2 Contrast test: beta(B-i) - beta(B-ii) = 0 with HC3 robust SE
        b1_name = "score_B_i" if "score_B_i" in X2_const.columns else None
        b2_name = "score_B_ii" if "score_B_ii" in X2_const.columns else None
        if b1_name is not None and b2_name is not None:
            cols = list(X2_const.columns)
            idx_b1 = cols.index(b1_name)
            idx_b2 = cols.index(b2_name)

            beta_b1 = float(self.fit_block2_hc3_.params[idx_b1])
            beta_b2 = float(self.fit_block2_hc3_.params[idx_b2])
            self.contrast_diff_ = beta_b1 - beta_b2

            # Robust variance of contrast: Var(b1) + Var(b2) - 2 Cov(b1, b2)
            cov_mat = self.fit_block2_hc3_.cov_params()
            var_contrast = float(
                cov_mat[idx_b1, idx_b1] + cov_mat[idx_b2, idx_b2] - 2.0 * cov_mat[idx_b1, idx_b2]
            )
            self.contrast_se_ = float(np.sqrt(max(1e-12, var_contrast)))

            t_val = self.contrast_diff_ / self.contrast_se_
            self.contrast_pvalue_ = float(2.0 * (1.0 - stats.t.cdf(abs(t_val), df=df_denom)))

            # Correct-sign rule: rejection p < alpha AND correct direction (B-i > B-ii)
            self.h2_hit_ = bool(self.contrast_pvalue_ < self.alpha and self.contrast_diff_ > 0.0)

        return self

    def predict(self, X_block2: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict outcomes from Block 2 (full) model."""
        if isinstance(X_block2, pd.DataFrame):
            X2_const = sm.add_constant(X_block2, has_constant="add")
        else:
            X2_const = sm.add_constant(pd.DataFrame(X_block2), has_constant="add")
        return np.asarray(self.fit_block2_.predict(X2_const), dtype=np.float64)


def fit_m1_marker_adjusted(
    df: pd.DataFrame,
    contextual_cols: list[str] | None = None,
    individual_cols: list[str] | None = None,
    covariate_cols: list[str] | None = None,
    marker_col: str = "score_M",
    target_col: str = "score_C",
    alpha: float = 0.05,
) -> dict[str, Any]:
    """Fit marker-adjusted M1 (M1-adj) using Lindell-Whitney correlation correction."""
    X_block1, X_block2 = prepare_features(df, contextual_cols, individual_cols, covariate_cols)
    y = df[target_col].to_numpy(dtype=np.float64)
    n = len(y)

    study_vars = pd.concat([X_block2, pd.Series(y, name=target_col, index=df.index)], axis=1)
    marker = df[marker_col].to_numpy(dtype=np.float64)

    # Compute correlation between marker and each study variable
    r_marker_vals = [
        float(np.abs(np.corrcoef(marker, study_vars[col])[0, 1])) for col in study_vars.columns
    ]
    r_M = float(np.mean(r_marker_vals))

    # Unadjusted correlation matrix of study variables
    R_raw = np.corrcoef(study_vars.values, rowvar=False)
    # Apply Lindell-Whitney adjustment
    R_adj = lindell_whitney(R_raw, marker_corr=r_M)

    cols1 = list(X_block1.columns)
    cols2 = list(X_block2.columns)
    idx1 = [study_vars.columns.get_loc(c) for c in cols1]
    idx2 = [study_vars.columns.get_loc(c) for c in cols2]
    idx_y = study_vars.columns.get_loc(target_col)

    p1 = len(idx1)
    p2 = len(idx2)

    R_xx_1 = R_adj[np.ix_(idx1, idx1)]
    R_xy_1 = R_adj[idx1, idx_y]

    R_xx_2 = R_adj[np.ix_(idx2, idx2)]
    R_xy_2 = R_adj[idx2, idx_y]

    # Solve standardized coefficients
    beta1 = np.linalg.solve(R_xx_1 + 1e-8 * np.eye(p1), R_xy_1)
    beta2 = np.linalg.solve(R_xx_2 + 1e-8 * np.eye(p2), R_xy_2)

    r2_1 = float(np.clip(beta1 @ R_xy_1, 0.0, 1.0))
    r2_2 = float(np.clip(beta2 @ R_xy_2, 0.0, 1.0))
    delta_r2 = max(0.0, r2_2 - r2_1)

    # F-test on delta R2
    q = p2 - p1
    df_denom = n - p2 - 1
    f_stat = ((delta_r2) / q) / ((1.0 - r2_2) / df_denom) if (1.0 - r2_2) > 1e-12 else 0.0
    f_pval = float(1.0 - stats.f.cdf(max(0.0, f_stat), q, df_denom))

    # A_tech test
    cols2 = list(X_block2.columns)
    idx_a = cols2.index("score_A_tech") if "score_A_tech" in cols2 else None
    a_coef = float(beta2[idx_a]) if idx_a is not None else 0.0
    var_a = (1.0 - r2_2) / df_denom * np.linalg.inv(R_xx_2)[idx_a, idx_a] if idx_a is not None else 1.0
    se_a = float(np.sqrt(max(1e-12, var_a))) if idx_a is not None else 1.0
    t_a = a_coef / se_a if se_a > 0 else 0.0
    p_a = float(2.0 * (1.0 - stats.t.cdf(abs(t_a), df_denom)))

    # H2 contrast
    idx_b1 = cols2.index("score_B_i") if "score_B_i" in cols2 else None
    idx_b2 = cols2.index("score_B_ii") if "score_B_ii" in cols2 else None
    diff_h2 = float(beta2[idx_b1] - beta2[idx_b2]) if (idx_b1 is not None and idx_b2 is not None) else 0.0
    inv_R2 = np.linalg.inv(R_xx_2)
    var_diff = (1.0 - r2_2) / df_denom * (inv_R2[idx_b1, idx_b1] + inv_R2[idx_b2, idx_b2] - 2 * inv_R2[idx_b1, idx_b2])
    se_diff = float(np.sqrt(max(1e-12, var_diff)))
    t_h2 = diff_h2 / se_diff if se_diff > 0 else 0.0
    p_h2 = float(2.0 * (1.0 - stats.t.cdf(abs(t_h2), df_denom)))

    return {
        "r2_block1": r2_1,
        "r2_block2": r2_2,
        "delta_r2": delta_r2,
        "f_stat": f_stat,
        "f_pvalue": f_pval,
        "h1_f_hit": bool(f_pval < alpha),
        "a_tech_coef": a_coef,
        "a_tech_pvalue": p_a,
        "h1_t_hit": bool(p_a < alpha and a_coef > 0.0),
        "contrast_diff": diff_h2,
        "contrast_pvalue": p_h2,
        "h2_hit": bool(p_h2 < alpha and diff_h2 > 0.0),
        "r_M": r_M,
        "beta2": beta2,
    }
