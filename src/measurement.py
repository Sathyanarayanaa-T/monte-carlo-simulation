"""Measurement layer validation, factor analysis, and method-bias diagnostics.

Implements procedures defined in SPEC.md Section 5 (lines 103-107):
- Heterotrait-Monotrait Ratio of Correlations (HTMT) (Henseler et al., 2015)
- Horn's Parallel Analysis for factor retention (Horn, 1965)
- Lindell-Whitney marker variable adjustment (Lindell & Whitney, 2001)
- Harman's single factor test (Harman, 1976; Podsakoff et al., 2003)

Note: All calculations use Pearson product-moment correlations on numeric item scores.
"""

from __future__ import annotations

from typing import Any
import numpy as np
import pandas as pd


class HTMTResult(dict[tuple[str, str], float]):
    """Dictionary of pairwise HTMT values with numeric convenience methods."""

    def __float__(self) -> float:
        """Allow single-pair HTMT result to be cast directly to float."""
        if len(self) == 1:
            return float(next(iter(self.values())))
        raise ValueError("Cannot cast multi-pair HTMT result to a single float.")

    def __eq__(self, other: Any) -> bool:
        if len(self) == 1 and isinstance(other, (int, float, np.floating, np.integer)):
            return float(self) == float(other)
        return super().__eq__(other)

    def __lt__(self, other: Any) -> bool:
        if len(self) == 1 and isinstance(other, (int, float, np.floating, np.integer)):
            return float(self) < float(other)
        return NotImplemented

    def __le__(self, other: Any) -> bool:
        if len(self) == 1 and isinstance(other, (int, float, np.floating, np.integer)):
            return float(self) <= float(other)
        return NotImplemented

    def __gt__(self, other: Any) -> bool:
        if len(self) == 1 and isinstance(other, (int, float, np.floating, np.integer)):
            return float(self) > float(other)
        return NotImplemented

    def __ge__(self, other: Any) -> bool:
        if len(self) == 1 and isinstance(other, (int, float, np.floating, np.integer)):
            return float(self) >= float(other)
        return NotImplemented


def _extract_item_matrix(items: pd.DataFrame | np.ndarray, cols: list[str] | list[int]) -> np.ndarray:
    """Extract 2D numpy array for specified columns."""
    if isinstance(items, pd.DataFrame):
        if len(cols) > 0 and isinstance(cols[0], str):
            return items[cols].to_numpy(dtype=np.float64)
        return items.iloc[:, cols].to_numpy(dtype=np.float64)
    mat = np.asarray(items, dtype=np.float64)
    return mat[:, cols]


def htmt(
    items: pd.DataFrame | np.ndarray,
    construct_map: dict[str, list[str] | list[int]],
) -> HTMTResult:
    """Compute Heterotrait-Monotrait Ratio of Correlations (HTMT).

    Formula (Henseler, Ringle, & Sarstedt, 2015; SPEC.md Section 5 line 104):
    -------------------------------------------------------------------------
    For two constructs i and j with item sets {X_ig} (g=1..K_i) and {X_jh} (h=1..K_j):

                      (1 / (K_i * K_j)) * sum_{g=1}^{K_i} sum_{h=1}^{K_j} |r(X_ig, X_jh)|
        HTMT_ij = -----------------------------------------------------------------------------
                  sqrt( [ (2 / (K_i*(K_i-1))) * sum_{g<h} |r(X_ig, X_ih)| ] *
                        [ (2 / (K_j*(K_j-1))) * sum_{g<h} |r(X_jg, X_jh)| ] )

    where r(., .) denotes the Pearson product-moment correlation on item scores.
    If K_i == 1, the monotrait correlation for construct i is set to 1.0.

    Parameters:
    - items: DataFrame or 2D array of item scores.
    - construct_map: mapping from construct name to list of column names or indices.

    Returns:
    - HTMTResult: dictionary mapping (construct_i, construct_j) -> HTMT value.
      When exactly one pair is computed, supports numeric comparison operators.
    """
    constructs = list(construct_map.keys())
    result = HTMTResult()

    # Pre-extract item matrices per construct
    mats: dict[str, np.ndarray] = {
        c: _extract_item_matrix(items, construct_map[c]) for c in constructs
    }

    # Pre-compute monotrait correlations for each construct
    # Mono-trait correlation is the average of absolute off-diagonal correlations
    mono_corrs: dict[str, float] = {}
    for c, mat in mats.items():
        k = mat.shape[1]
        if k <= 1:
            mono_corrs[c] = 1.0
        else:
            r_mat = np.corrcoef(mat, rowvar=False)
            triu_idx = np.triu_indices(k, k=1)
            off_diags = np.abs(r_mat[triu_idx])
            # Handle potential NaNs from zero-variance columns
            off_diags = np.nan_to_num(off_diags, nan=0.0)
            mean_mono = float(np.mean(off_diags)) if len(off_diags) > 0 else 1.0
            mono_corrs[c] = max(mean_mono, 1e-12)

    # Compute pairwise HTMT
    for i in range(len(constructs)):
        for j in range(i + 1, len(constructs)):
            c1 = constructs[i]
            c2 = constructs[j]
            mat1 = mats[c1]
            mat2 = mats[c2]
            k1 = mat1.shape[1]
            k2 = mat2.shape[1]

            # Cross-correlation matrix between items of c1 and items of c2
            mat_combined = np.column_stack([mat1, mat2])
            corr_full = np.corrcoef(mat_combined, rowvar=False)
            r_hetero = np.abs(corr_full[:k1, k1:])
            r_hetero = np.nan_to_num(r_hetero, nan=0.0)
            mean_hetero = float(np.mean(r_hetero))

            denom = np.sqrt(mono_corrs[c1] * mono_corrs[c2])
            htmt_val = float(mean_hetero / denom) if denom > 0 else 0.0
            result[(c1, c2)] = htmt_val

    return result


_PARALLEL_ANALYSIS_THRESHOLDS_CACHE: dict[tuple[int, int, int, float], np.ndarray] = {}


def get_parallel_analysis_thresholds(
    n_obs: int,
    n_items: int,
    n_iter: int = 500,
    percentile: float = 95.0,
    use_cache: bool = True,
    seed: int | np.random.Generator | None = None,
) -> np.ndarray:
    """Calculate and cache Horn's null eigenvalue percentile thresholds.

    Parameters:
    - n_obs: sample size n.
    - n_items: number of items p.
    - n_iter: number of Monte Carlo null datasets (default: 500).
    - percentile: percentile cutoff (default: 95.0).
    - use_cache: whether to look up/store in process-level cache (default: True).
    - seed: optional random seed. If None and use_cache is True, seed is derived
      deterministically from (n_obs, n_items) to ensure invariance to replication seeds.

    Returns:
    - np.ndarray of shape (n_items,) with critical eigenvalue cutoffs.
    """
    cache_key = (n_obs, n_items, n_iter, percentile)
    if use_cache and seed is None and cache_key in _PARALLEL_ANALYSIS_THRESHOLDS_CACHE:
        return _PARALLEL_ANALYSIS_THRESHOLDS_CACHE[cache_key].copy()

    if seed is None:
        # Fixed deterministic seed derived solely from (n_obs, n_items)
        seed_val = int(np.random.SeedSequence([20261013, n_obs, n_items]).generate_state(1)[0])
        rng = np.random.default_rng(seed_val)
    elif isinstance(seed, np.random.Generator):
        rng = seed
    else:
        rng = np.random.default_rng(seed)

    null_eigs = np.zeros((n_iter, n_items), dtype=np.float64)
    for b in range(n_iter):
        W = rng.normal(0.0, 1.0, size=(n_obs, n_items))
        R_null = np.corrcoef(W, rowvar=False)
        null_eigs[b] = np.sort(np.linalg.eigvalsh(R_null))[::-1]

    crit_eigs = np.percentile(null_eigs, percentile, axis=0)

    if use_cache and seed is None:
        _PARALLEL_ANALYSIS_THRESHOLDS_CACHE[cache_key] = crit_eigs

    return crit_eigs.copy()


def parallel_analysis(
    items: pd.DataFrame | np.ndarray,
    n_iter: int = 500,
    percentile: float = 95.0,
    seed: int | np.random.Generator | None = None,
    use_cache: bool = True,
) -> int:
    """Determine number of retained factors via Horn's Parallel Analysis.

    Formula & Procedure (Horn, 1965; SPEC.md Section 5 line 105):
    --------------------------------------------------------------
    1. Compute sample Pearson correlation matrix R of items (n x p).
    2. Extract sample eigenvalues in descending order:
       lambda_1 >= lambda_2 >= ... >= lambda_p
    3. Null benchmark eigenvalues extracted from get_parallel_analysis_thresholds
       (cached per (n_obs, n_items) with deterministic seed derived from dimensions).
    4. Count retained factors where sample eigenvalue exceeds null threshold:
       n_factors = sum_{k=1}^p 1[lambda_k > lambda_crit_k]
       Sequential retention rule stops at the first failure.

    Parameters:
    - items: DataFrame or 2D array of item responses (n observations x p items).
    - n_iter: number of random null datasets to simulate (default: 500).
    - percentile: percentile cutoff for eigenvalue comparison (default: 95.0).
    - seed: optional random seed or numpy Generator.
    - use_cache: whether to use cached null eigenvalue thresholds (default: True).

    Returns:
    - int: number of retained factors.
    """
    if isinstance(items, pd.DataFrame):
        X = items.to_numpy(dtype=np.float64)
    else:
        X = np.asarray(items, dtype=np.float64)

    n, p = X.shape
    if p <= 1:
        return p

    # Sample Pearson correlation matrix
    R_sample = np.corrcoef(X, rowvar=False)
    # eigvalsh returns eigenvalues in ascending order
    sample_eigs = np.sort(np.linalg.eigvalsh(R_sample))[::-1]

    # Obtain critical eigenvalue cutoffs (cached or computed)
    crit_eigs = get_parallel_analysis_thresholds(
        n_obs=n,
        n_items=p,
        n_iter=n_iter,
        percentile=percentile,
        use_cache=use_cache,
        seed=seed,
    )

    # Sequential retention rule (Horn, 1965; Hayton et al., 2004):
    # Retain factors up to the first component where sample eigenvalue <= null threshold
    n_retained = 0
    for k in range(p):
        if sample_eigs[k] > crit_eigs[k]:
            n_retained += 1
        else:
            break

    return n_retained


def lindell_whitney(
    corr: np.ndarray | pd.DataFrame | float,
    marker_corr: np.ndarray | float,
) -> np.ndarray | pd.DataFrame | float:
    """Apply Lindell-Whitney marker variable adjustment to correlation(s).

    Formula (Lindell & Whitney, 2001; SPEC.md Section 5 line 106):
    --------------------------------------------------------------
    For an unadjusted correlation r and marker correlation r_M:

        r_adj = (r - r_M) / (1 - r_M)

    For a full correlation matrix R:
        R_adj = (R - r_M) / (1 - r_M), with diagonal elements set to 1.0.

    When marker_corr == 0, r_adj = (r - 0) / (1 - 0) = r (input unchanged).

    Parameters:
    - corr: scalar correlation, 1D array, or 2D correlation matrix (numpy or DataFrame).
    - marker_corr: scalar marker correlation r_M (e.g., mean absolute marker correlation).

    Returns:
    - Adjusted correlation(s) with identical shape and container type as input.
    """
    r_m = float(marker_corr)
    if np.isclose(r_m, 1.0):
        raise ValueError("Marker correlation cannot be 1.0 (division by zero).")

    denom = 1.0 - r_m

    if isinstance(corr, (int, float, np.floating, np.integer)):
        return float((corr - r_m) / denom)

    if isinstance(corr, pd.DataFrame):
        arr = corr.to_numpy(dtype=np.float64)
        is_square = (arr.shape[0] == arr.shape[1])
        adj_arr = (arr - r_m) / denom
        if is_square:
            np.fill_diagonal(adj_arr, 1.0)
        return pd.DataFrame(adj_arr, index=corr.index, columns=corr.columns)

    arr = np.asarray(corr, dtype=np.float64)
    adj_arr = (arr - r_m) / denom
    if arr.ndim == 2 and arr.shape[0] == arr.shape[1]:
        np.fill_diagonal(adj_arr, 1.0)
    return adj_arr


def harman_first_factor_share(items: pd.DataFrame | np.ndarray) -> float:
    """Compute the variance share explained by the first unrotated factor (Harman's test).

    Formula (Harman, 1976; Podsakoff et al., 2003; SPEC.md Section 5 line 107):
    --------------------------------------------------------------------------
    1. Compute Pearson correlation matrix R of items (p x p).
    2. Extract eigenvalues of R: lambda_1 >= lambda_2 >= ... >= lambda_p.
    3. The variance share explained by the first unrotated principal component / factor is:

        share = lambda_1 / sum_{j=1}^p lambda_j = lambda_1 / p

    Parameters:
    - items: DataFrame or 2D array of item responses (n x p).

    Returns:
    - float: variance share in [0, 1] explained by the first unrotated factor.
    """
    if isinstance(items, pd.DataFrame):
        X = items.to_numpy(dtype=np.float64)
    else:
        X = np.asarray(items, dtype=np.float64)

    n, p = X.shape
    if p <= 1:
        return 1.0

    R = np.corrcoef(X, rowvar=False)
    eigs = np.linalg.eigvalsh(R)
    max_eig = float(np.max(eigs))
    return float(max_eig / p)
