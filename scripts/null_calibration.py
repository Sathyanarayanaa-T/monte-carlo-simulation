#!/usr/bin/env python3
"""Residual-resampling null interaction test calibration check.

Milestone 6.7 Part E:
On scenario base_n100_none, for M4, M5, and M6:
- Runs tuned fit on observed data
- Runs residual-resampling null with B=50 (tuning-conditional refits)
- Reports:
    * Type I error at alpha=0.05 with MCSE
    * Mean seconds per replication per method
    * Share of replications with p < 0.05, p < 0.10, and p < 0.20
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import time
from typing import Any

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.dgp import build_scenario, generate
from src.methods.m1_hierarchical import prepare_features
from src.methods.m4_random_forest import M4RandomForest
from src.methods.m5_gradient_boosting import M5GradientBoosting
from src.methods.m6_ebm import M6ExplainableBoosting


def run_one_null_rep(
    method_name: str,
    scenario: dict[str, Any],
    rep: int,
    child_seed: np.random.SeedSequence,
    b_null_refits: int = 50,
) -> dict[str, Any]:
    """Execute one replication of tuning + residual-resampling null test."""
    child_seed_int = int(child_seed.generate_state(1)[0])
    df = generate(scenario, seed=child_seed)
    _, X2 = prepare_features(df)
    y = df["score_C"].values

    t0 = time.perf_counter()
    if method_name == "M4":
        m4 = M4RandomForest(random_state=child_seed_int).fit(X2, y, tune_oob=True)
        res = m4.run_interaction_test(X2, y, b_null_refits=b_null_refits, alpha=0.05)
    elif method_name == "M5":
        m5 = M5GradientBoosting(random_state=child_seed_int).fit(X2, y, tune_cv=True)
        res = m5.run_interaction_test(X2, y, b_null_refits=b_null_refits, alpha=0.05)
    elif method_name == "M6":
        m6 = M6ExplainableBoosting(random_state=child_seed_int).fit(X2, y)
        res = m6.run_interaction_test(X2, y, b_null_refits=b_null_refits, alpha=0.05)
    else:
        raise ValueError(f"Unknown method: {method_name}")

    elapsed = time.perf_counter() - t0
    return {
        "method": method_name,
        "replication": rep,
        "detected": bool(res["detected"]),
        "p_value": float(res["p_value"]),
        "t_observed": float(res["t_observed"]),
        "runtime_seconds": float(elapsed),
    }


def run_calibration(
    reps: int = 100,
    reps_m6: int = 50,
    n_jobs: int = 6,
    b_null: int = 50,
    out_dir: str | Path = "calibration/",
    master_seed: int = 20261013,
) -> pd.DataFrame:
    out_path = Path(out_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    scenario = build_scenario({"name": "base_n100_none", "n": 100, "non_linear_form": "none", "psi": 0.0})
    master_seq = np.random.SeedSequence(master_seed)
    method_seqs = master_seq.spawn(3)

    plan = [
        ("M4", reps, method_seqs[0]),
        ("M5", reps, method_seqs[1]),
        ("M6", reps_m6, method_seqs[2]),
    ]

    all_results: list[dict[str, Any]] = []

    print("=" * 95)
    print("RESIDUAL-RESAMPLING NULL CALIBRATION CHECK (SCENARIO: base_n100_none, B=50)")
    print(f"Workers (n_jobs): {n_jobs} | M4/M5 reps: {reps} | M6 reps: {reps_m6}")
    print("=" * 95)

    for m_name, n_r, m_seq in plan:
        print(f"\nRunning {m_name} ({n_r} replications)...")
        child_seeds = m_seq.spawn(n_r)
        t_start_m = time.time()

        if n_jobs == 1:
            m_res = [
                run_one_null_rep(m_name, scenario, r, child_seeds[r], b_null_refits=b_null)
                for r in range(n_r)
            ]
        else:
            m_res = Parallel(n_jobs=n_jobs)(
                delayed(run_one_null_rep)(m_name, scenario, r, child_seeds[r], b_null_refits=b_null)
                for r in range(n_r)
            )

        dur_m = time.time() - t_start_m
        print(f"  -> {m_name} finished in {dur_m:.1f}s ({dur_m / n_r:.2f}s/rep wall clock)")
        all_results.extend(m_res)

    df_all = pd.DataFrame(all_results)
    df_all.to_parquet(out_path / "null_calibration_results.parquet")

    # Summary table
    print("\n" + "=" * 95)
    print("NULL CALIBRATION SUMMARY TABLE (Target nominal alpha = 0.05)")
    print("-" * 95)
    print(f"{'Method':<8s} | {'Reps':<6s} | {'Mean Time':<12s} | {'Type I Rate (MCSE)':<22s} | {'p < 0.10':<12s} | {'p < 0.20':<12s} | {'Status'}")
    print("-" * 95)

    summary_records = []
    for m_name, n_r, _ in plan:
        sub = df_all[df_all["method"] == m_name]
        mean_time = float(sub["runtime_seconds"].mean())
        p_05 = float(np.mean(sub["p_value"] < 0.05))
        mcse_05 = float(np.sqrt(max(0.0, p_05 * (1.0 - p_05)) / len(sub)))
        p_10 = float(np.mean(sub["p_value"] < 0.10))
        p_20 = float(np.mean(sub["p_value"] < 0.20))

        # Status: calibrated if Type I rate is within ~2 MCSEs of 0.05, warning if > 0.10
        if p_05 <= 0.10:
            status = "CALIBRATED (<= 0.10)"
        else:
            status = "MISCALIBRATED (> 0.10 - RE-TUNE INSIDE NULL)"

        summary_records.append({
            "method": m_name,
            "reps": len(sub),
            "mean_runtime_s": mean_time,
            "type1_rate_05": p_05,
            "mcse_05": mcse_05,
            "share_p_lt_10": p_10,
            "share_p_lt_20": p_20,
            "status": status,
        })

        t1_str = f"{p_05:.3f} ({mcse_05:.3f})"
        print(f"{m_name:<8s} | {len(sub):<6d} | {mean_time:<12.2f}s | {t1_str:<22s} | {p_10:<12.3f} | {p_20:<12.3f} | {status}")

    print("=" * 95)
    with open(out_path / "null_calibration_summary.json", "w") as f:
        json.dump(summary_records, f, indent=2)

    return df_all


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Residual-resampling null interaction test calibration runner.")
    parser.add_argument("--reps", type=int, default=100, help="Replications for M4 and M5 (default: 100)")
    parser.add_argument("--reps-m6", type=int, default=50, help="Replications for M6 (default: 50)")
    parser.add_argument("--n-jobs", type=int, default=6, help="Parallel workers (default: 6)")
    parser.add_argument("--b-null", type=int, default=50, help="Residual resampling null refits (default: 50)")
    parser.add_argument("--out", type=str, default="calibration/", help="Output directory (default: calibration/)")
    parser.add_argument("--seed", type=int, default=20261013, help="Master seed (default: 20261013)")
    args = parser.parse_args()

    run_calibration(
        reps=args.reps,
        reps_m6=args.reps_m6,
        n_jobs=args.n_jobs,
        b_null=args.b_null,
        out_dir=args.out,
        master_seed=args.seed,
    )
