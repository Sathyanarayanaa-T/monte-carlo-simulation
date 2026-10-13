"""Timing pilot script: 5 replications per method at n = 100 and n = 150.

SPEC.md Section 5 (line 102):
- Run a timing pilot of about 5 replications before the full run.
- Prints seconds per replication per method.
- Evaluates computational budget for large-scale Monte Carlo execution.
"""

from __future__ import annotations

import sys
from pathlib import Path

# Add repo root to sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
import pandas as pd

from src.dgp import build_scenario, generate
from src.methods.m0_baseline import M0Baseline
from src.methods.m1_hierarchical import M1HierarchicalOLS, prepare_features
from src.methods.m2_interaction import M2aPreSpecifiedInteraction, M2bAllPairwiseInteractions
from src.methods.m3_elasticnet import M3ElasticNet
from src.methods.m4_random_forest import M4RandomForest
from src.methods.m5_gradient_boosting import M5GradientBoosting
from src.methods.m6_ebm import M6ExplainableBoosting


METHODS = ["M0", "M1", "M2a", "M2b", "M3", "M4", "M5", "M6"]


def time_single_method(method_name: str, X1: pd.DataFrame, X2: pd.DataFrame, y: np.ndarray, seed: int) -> float:
    """Measure execution time (in seconds) for a single replication of a method."""
    t0 = time.time()
    if method_name == "M0":
        M0Baseline().fit(X2, y).predict(X2)
    elif method_name == "M1":
        M1HierarchicalOLS().fit(X1, X2, y).predict(X2)
    elif method_name == "M2a":
        M2aPreSpecifiedInteraction().fit(X2, y).predict(X2)
    elif method_name == "M2b":
        M2bAllPairwiseInteractions().fit(X2, y).predict(X2)
    elif method_name == "M3":
        M3ElasticNet(random_state=seed).fit(X2, y, run_stability=False).predict(X2)
    elif method_name == "M4":
        m4 = M4RandomForest(random_state=seed).fit(X2, y, tune_oob=True)
        m4.compute_interaction_stats(X2)
        m4.predict(X2)
    elif method_name == "M5":
        m5 = M5GradientBoosting(random_state=seed).fit(X2, y, tune_cv=True)
        m5.compute_interaction_stats(X2)
        m5.predict(X2)
    elif method_name == "M6":
        m6 = M6ExplainableBoosting(random_state=seed).fit(X2, y)
        m6.compute_interaction_stats(X2)
        m6.predict(X2)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    return time.time() - t0


def run_timing_pilot(n_reps: int = 5, n_values: list[int] | None = None) -> pd.DataFrame:
    """Run timing pilot across 5 replications for n = 100 and n = 150."""
    if n_values is None:
        n_values = [100, 150]

    master_seq = np.random.SeedSequence(20261013)
    results = []

    print(f"=== Timing Pilot: {n_reps} replications per method across n in {n_values} ===")

    for n_val in n_values:
        print(f"\n--- Testing n = {n_val} ---")
        sc = build_scenario({"n": n_val})
        seeds = master_seq.spawn(n_reps)

        for rep, s in enumerate(seeds):
            child_seed_int = int(s.generate_state(1)[0])
            df = generate(sc, seed=s)
            X1, X2 = prepare_features(df)
            y = df["score_C"].to_numpy(dtype=np.float64)

            for method in METHODS:
                duration = time_single_method(method, X1, X2, y, child_seed_int)
                results.append({
                    "n": n_val,
                    "replication": rep + 1,
                    "method": method,
                    "seconds": duration,
                })

    df_results = pd.DataFrame(results)

    # Summary table
    summary = (
        df_results.groupby(["n", "method"])["seconds"]
        .agg(["mean", "std", "min", "max"])
        .reset_index()
    )

    print("\n=== Timing Pilot Summary Table (Seconds per Replication) ===")
    print(summary.to_string(index=False))
    return summary


if __name__ == "__main__":
    summary = run_timing_pilot(n_reps=5, n_values=[100, 150])
