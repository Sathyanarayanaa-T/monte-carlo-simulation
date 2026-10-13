"""Simulation experiment runner and smoke test execution.

SPEC.md Section 8 & Milestone 4:
- Executes nested CV and metrics collection across replications.
- Uses np.random.SeedSequence.spawn for independent child seeds (Rule 8).
- Provides a smoke runner for one scenario with 20 replications.
"""

from __future__ import annotations

import time
from typing import Any
import numpy as np
import pandas as pd

from src.dgp import build_scenario, generate
from src.harness import run_nested_cv
from src.metrics import compute_replication_metrics


def run_replications(
    scenario: dict[str, Any] | str | None = None,
    n_replications: int = 20,
    master_seed: int = 20261013,
    run_stability: bool = True,
    verbose: bool = True,
) -> pd.DataFrame:
    """Run simulation replications for a given scenario.

    Parameters:
    - scenario: scenario dict or name (default: base scenario).
    - n_replications: number of replications (default: 20).
    - master_seed: master random seed.
    - run_stability: whether to run 100 bootstrap stability selection in M3.
    - verbose: print progress.

    Returns:
    - pd.DataFrame containing metrics for all replications.
    """
    if scenario is None:
        sc = build_scenario()
    elif isinstance(scenario, str):
        sc = build_scenario({"name": scenario})
    else:
        sc = build_scenario(scenario)

    seeds = np.random.SeedSequence(master_seed).spawn(n_replications)
    rows: list[dict[str, Any]] = []

    t0 = time.time()
    for rep, s in enumerate(seeds):
        # Generate dataset
        data = generate(sc, seed=s)

        # Run nested CV harness
        child_seed_int = int(s.generate_state(1)[0])
        cv_out = run_nested_cv(
            data,
            scenario_dict=sc,
            seed=child_seed_int,
            run_stability_selection=run_stability,
        )

        # Extract replication metrics
        metrics_row = compute_replication_metrics(cv_out, sc)
        metrics_row["replication"] = rep
        rows.append(metrics_row)

        if verbose and (rep + 1) % 5 == 0:
            elapsed = time.time() - t0
            print(f"Completed replication {rep + 1}/{n_replications} ({elapsed:.1f}s)")

    results_df = pd.DataFrame(rows)
    return results_df


def summarize_results(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean and Monte Carlo standard error (MCSE) for metrics."""
    n_reps = len(df)
    summary_data = []

    for col in df.columns:
        if col == "replication" or not pd.api.types.is_numeric_dtype(df[col]):
            continue
        vals = df[col].to_numpy(dtype=np.float64)
        m = float(np.mean(vals))
        s = float(np.std(vals, ddof=1)) if n_reps > 1 else 0.0
        se = float(s / np.sqrt(n_reps))
        summary_data.append({"metric": col, "mean": m, "std": s, "mc_se": se})

    return pd.DataFrame(summary_data)


def run_smoke(n_replications: int = 20, master_seed: int = 20261013) -> pd.DataFrame:
    """Run smoke test for base scenario with 20 replications."""
    print(f"=== Starting Smoke Runner: 1 scenario, {n_replications} replications ===")
    df = run_replications(
        scenario=None,
        n_replications=n_replications,
        master_seed=master_seed,
        run_stability=True,
        verbose=True,
    )
    summary = summarize_results(df)
    print("=== Smoke Runner Finished Successfully ===")
    return summary


if __name__ == "__main__":
    summary = run_smoke(n_replications=20)
    print(summary.head(20).to_string())
