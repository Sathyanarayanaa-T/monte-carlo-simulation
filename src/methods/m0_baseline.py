"""Method M0: Training-fold Mean Baseline.

SPEC.md Section 5 (line 81):
- Predicts the training-fold mean for all observations.
- Serves as the baseline for out-of-sample R^2 (where R^2_cv ~ 0).
"""

from __future__ import annotations

import numpy as np
import pandas as pd


class M0Baseline:
    """Baseline regression model that predicts the training-fold target mean."""

    def __init__(self) -> None:
        self.mean_y_: float = 0.0

    def fit(self, X: pd.DataFrame | np.ndarray, y: pd.Series | np.ndarray) -> M0Baseline:
        """Fit model by calculating the sample mean of y."""
        y_arr = np.asarray(y, dtype=np.float64)
        self.mean_y_ = float(np.mean(y_arr))
        return self

    def predict(self, X: pd.DataFrame | np.ndarray) -> np.ndarray:
        """Predict constant training-fold mean."""
        n = len(X)
        return np.full(n, self.mean_y_, dtype=np.float64)
