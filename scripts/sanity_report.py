#!/usr/bin/env python3
"""Sanity reporting script for completed Monte Carlo simulation shards.

Milestone 6.5 Part G / Milestone 6.6 Part F:
Reads results/*.parquet and prints, per scenario:
- Replications completed and failed count
- Type I error calibration for null scenarios (with MCSE)
- Base scenarios empirical power by n
- Structural-validity gate section:
    * HTMT > 0.85 share (with MCSE) for cross-loading (A-i/B-i) and contamination (C/G) at n in {100, 150}
    * Base none scenarios: false-flag share for the three pairs
    * Parallel analysis factor-count distributions
    * Harman first-factor share > 0.50 share (with MCSE)
- Method-bias section:
    * Mean marker correlation r_M
    * Share of replications where m1_adj conclusions differ from m1 for acquiescence scenarios
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
from typing import Any

import numpy as np
import pandas as pd


def compute_rate_and_mcse(series: pd.Series) -> tuple[float, float]:
    """Compute empirical event rate and its Monte Carlo Standard Error."""
    clean = series.dropna()
    r = len(clean)
    if r == 0:
        return np.nan, np.nan
    p = float(clean.mean())
    mcse = float(np.sqrt(max(0.0, p * (1.0 - p)) / r))
    return p, mcse


def generate_sanity_report(results_dir: Path | str = "results") -> None:
    results_path = Path(results_dir)
    if not results_path.exists():
        print(f"Results directory '{results_path}' does not exist.")
        return

    shard_files = sorted(list(results_path.glob("*.parquet")))
    if not shard_files:
        print(f"No parquet shards found in '{results_path}'.")
        return

    print("=" * 105)
    print(f"MONTE CARLO SIMULATION SANITY REPORT: {results_path}")
    print(f"Total Shards Found: {len(shard_files)}")
    print("=" * 105)

    summary_rows = []
    base_rows = []
    null_rows = []
    shards_data: dict[str, pd.DataFrame] = {}

    for shard_file in shard_files:
        sc_name = shard_file.stem
        try:
            df = pd.read_parquet(shard_file)
            shards_data[sc_name] = df
        except Exception as e:
            print(f"Error reading {shard_file}: {e}")
            continue

        n_reps = len(df)
        n_success = int((df.get("status") == "SUCCESS").sum()) if "status" in df else n_reps
        n_failed = int((df.get("status") == "FAILED").sum()) if "status" in df else 0

        runtime_sec = float(df["runtime_seconds"].mean()) if "runtime_seconds" in df else np.nan

        row_info: dict[str, Any] = {
            "scenario": sc_name,
            "reps_completed": n_success,
            "reps_failed": n_failed,
            "mean_runtime_s": runtime_sec,
        }

        # Check scenario categories
        is_h1_null = "h1_null" in sc_name
        is_h2_null = "h2_null" in sc_name
        is_additive_none = "_none" in sc_name and not (is_h1_null or is_h2_null)
        is_base = sc_name.startswith("base_n")

        # Type I detections
        if "h1_f_hit" in df:
            row_info["h1_f_rate"], row_info["h1_f_mcse"] = compute_rate_and_mcse(df["h1_f_hit"])
        if "h1_t_hit" in df:
            row_info["h1_t_rate"], row_info["h1_t_mcse"] = compute_rate_and_mcse(df["h1_t_hit"])
        if "h2_hit" in df:
            row_info["h2_rate"], row_info["h2_mcse"] = compute_rate_and_mcse(df["h2_hit"])
        if "m2a_detected" in df:
            row_info["m2a_rate"], row_info["m2a_mcse"] = compute_rate_and_mcse(df["m2a_detected"])
        if "m2b_detected" in df:
            row_info["m2b_rate"], row_info["m2b_mcse"] = compute_rate_and_mcse(df["m2b_detected"])
        if "m3_detected" in df:
            row_info["m3_rate"], row_info["m3_mcse"] = compute_rate_and_mcse(df["m3_detected"])

        # ML detections
        for m in ["m4", "m5", "m6"]:
            col = f"{m}_detected"
            if col in df:
                row_info[f"{m}_rate"], row_info[f"{m}_mcse"] = compute_rate_and_mcse(df[col])

        summary_rows.append(row_info)
        if is_h1_null or is_h2_null or is_additive_none:
            null_rows.append(row_info)
        if is_base:
            base_rows.append(row_info)

    # 1. Overall Completion Table
    print("\n[1] Shard Completion Summary:")
    print("-" * 80)
    print(f"{'Scenario':<38s} | {'Completed':<10s} | {'Failed':<8s} | {'Runtime (s)':<12s}")
    print("-" * 80)
    for r in summary_rows:
        rt_str = f"{r['mean_runtime_s']:.2f}" if not np.isnan(r['mean_runtime_s']) else "N/A"
        print(f"{r['scenario']:<38s} | {r['reps_completed']:<10d} | {r['reps_failed']:<8d} | {rt_str:<12s}")

    # 2. Type I Error Rates for Null Scenarios
    if null_rows:
        print("\n\n[2] Type I Error Calibration for Null Scenarios (with MCSE):")
        print("-" * 105)
        print(f"{'Scenario':<30s} | {'H1 (F-test)':<14s} | {'H2 (Contrast)':<14s} | {'M2a':<12s} | {'M2b':<12s} | {'M3':<12s}")
        print("-" * 105)
        for r in null_rows:
            h1_s = f"{r.get('h1_f_rate', np.nan):.3f} ({r.get('h1_f_mcse', np.nan):.3f})" if 'h1_f_rate' in r else "N/A"
            h2_s = f"{r.get('h2_rate', np.nan):.3f} ({r.get('h2_mcse', np.nan):.3f})" if 'h2_rate' in r else "N/A"
            m2a_s = f"{r.get('m2a_rate', np.nan):.3f} ({r.get('m2a_mcse', np.nan):.3f})" if 'm2a_rate' in r else "N/A"
            m2b_s = f"{r.get('m2b_rate', np.nan):.3f} ({r.get('m2b_mcse', np.nan):.3f})" if 'm2b_rate' in r else "N/A"
            m3_s = f"{r.get('m3_rate', np.nan):.3f} ({r.get('m3_mcse', np.nan):.3f})" if 'm3_rate' in r else "N/A"
            print(f"{r['scenario']:<30s} | {h1_s:<14s} | {h2_s:<14s} | {m2a_s:<12s} | {m2b_s:<12s} | {m3_s:<12s}")

        has_ml = any('m4_rate' in r for r in null_rows)
        if has_ml:
            print("\n  -> ML Interaction Detection on Additive/Null Data:")
            print("  " + "-" * 75)
            print(f"  {'Scenario':<30s} | {'M4 (RF)':<12s} | {'M5 (GB)':<12s} | {'M6 (EBM)':<12s}")
            print("  " + "-" * 75)
            for r in null_rows:
                if any(k in r for k in ['m4_rate', 'm5_rate', 'm6_rate']):
                    m4_s = f"{r.get('m4_rate', np.nan):.3f} ({r.get('m4_mcse', np.nan):.3f})" if 'm4_rate' in r else "N/A"
                    m5_s = f"{r.get('m5_rate', np.nan):.3f} ({r.get('m5_mcse', np.nan):.3f})" if 'm5_rate' in r else "N/A"
                    m6_s = f"{r.get('m6_rate', np.nan):.3f} ({r.get('m6_mcse', np.nan):.3f})" if 'm6_rate' in r else "N/A"
                    print(f"  {r['scenario']:<30s} | {m4_s:<12s} | {m5_s:<12s} | {m6_s:<12s}")

    # 3. Base Scenarios: Power by n
    if base_rows:
        print("\n\n[3] Base Scenarios: Empirical Power by Sample Size (n):")
        print("-" * 80)
        print(f"{'Scenario':<38s} | {'H1 Power (F-test)':<18s} | {'H2 Power (Contrast)':<18s}")
        print("-" * 80)
        for r in sorted(base_rows, key=lambda x: x['scenario']):
            h1_s = f"{r.get('h1_f_rate', np.nan):.3f} ({r.get('h1_f_mcse', np.nan):.3f})" if 'h1_f_rate' in r else "N/A"
            h2_s = f"{r.get('h2_rate', np.nan):.3f} ({r.get('h2_mcse', np.nan):.3f})" if 'h2_rate' in r else "N/A"
            print(f"{r['scenario']:<38s} | {h1_s:<18s} | {h2_s:<18s}")

    # 4. Structural-Validity Gate Section (Milestone 6.6 Part F)
    print("\n\n[4] Structural-Validity Gate & Psychometric Diagnostics:")
    print("-" * 105)

    # 4a. Base Scenarios False-Flag Rates (HTMT > 0.85)
    print("  4a. Base Scenarios (none): HTMT False-Flag Rate (> 0.85) [Target: <= 0.05]")
    print("  " + "-" * 95)
    print(f"  {'Scenario':<25s} | {'A-i / B-i':<20s} | {'A-ii / B-ii':<20s} | {'C / G':<20s}")
    print("  " + "-" * 95)
    for sc_name, df in sorted(shards_data.items()):
        if sc_name.startswith("base_n") and "_none" in sc_name:
            rates = []
            for col in ["htmt_A_tech_B_i", "htmt_A_mkt_B_ii", "htmt_C_G"]:
                if col in df:
                    p, se = compute_rate_and_mcse(df[col] > 0.85)
                    rates.append(f"{p:.3f} ({se:.3f})")
                else:
                    rates.append("N/A")
            print(f"  {sc_name:<25s} | {rates[0]:<20s} | {rates[1]:<20s} | {rates[2]:<20s}")

    # 4b. Cross-loading & Contamination Flag Rates at n in {100, 150}
    print("\n  4b. Sensitivity Scenarios: Structural-Validity Gate Trigger Rate (HTMT > 0.85)")
    print("  " + "-" * 95)
    print(f"  {'Scenario':<38s} | {'Target Pair':<14s} | {'Flag Rate (>0.85)':<20s} | {'Status':<14s}")
    print("  " + "-" * 95)
    for sc_name, df in sorted(shards_data.items()):
        if any(sc_name.startswith(f"sens_cross_loading_n{n}") for n in [100, 150]):
            col = "htmt_A_tech_B_i"
            if col in df:
                p, se = compute_rate_and_mcse(df[col] > 0.85)
                stat = "GATE PASSED" if p >= 0.70 else "GATE WEAK"
                print(f"  {sc_name:<38s} | {'A-i / B-i':<14s} | {f'{p:.3f} ({se:.3f})':<20s} | {stat:<14s}")
        elif any(sc_name.startswith(f"sens_contaminated_n{n}") for n in [100, 150]):
            col = "htmt_C_G"
            if col in df:
                p, se = compute_rate_and_mcse(df[col] > 0.85)
                stat = "GATE PASSED" if p >= 0.70 else "GATE WEAK"
                print(f"  {sc_name:<38s} | {'C / G':<14s} | {f'{p:.3f} ({se:.3f})':<20s} | {stat:<14s}")

    # 4c. Parallel-Analysis Factor Count Distribution & Harman Share > 0.50
    print("\n  4c. Parallel Analysis Factor Count Distribution & Harman Single-Factor Share > 0.50:")
    print("  " + "-" * 95)
    print(f"  {'Scenario':<32s} | {'Harman > 0.50':<18s} | {'PA A-i/B-i (mean factors)':<24s} | {'PA C/G (mean)':<14s}")
    print("  " + "-" * 95)
    for sc_name, df in sorted(shards_data.items()):
        if sc_name.startswith("base_n") and "_none" in sc_name:
            h_str = "N/A"
            if "harman_first_factor_share" in df:
                hp, hse = compute_rate_and_mcse(df["harman_first_factor_share"] > 0.50)
                h_str = f"{hp:.3f} ({hse:.3f})"
            pa_ab = f"{df['pa_factors_A_tech_B_i'].mean():.2f}" if "pa_factors_A_tech_B_i" in df else "N/A"
            pa_cg = f"{df['pa_factors_C_G'].mean():.2f}" if "pa_factors_C_G" in df else "N/A"
            print(f"  {sc_name:<32s} | {h_str:<18s} | {pa_ab:<24s} | {pa_cg:<14s}")

    # 5. Method-Bias Section (Milestone 6.6 Part F)
    print("\n\n[5] Method-Bias Section (Acquiescence & Marker Adjustment):")
    print("-" * 105)
    print(f"{'Scenario':<35s} | {'Mean r_M (SE)':<18s} | {'H1 Conclusion Shift':<22s} | {'H2 Conclusion Shift':<22s}")
    print("-" * 105)
    acq_found = False
    for sc_name, df in sorted(shards_data.items()):
        if "acquiescence" in sc_name:
            acq_found = True
            rm_str = f"{df['marker_r_M'].mean():.3f} ({df['marker_r_M'].std() / np.sqrt(len(df)):.3f})" if "marker_r_M" in df else "N/A"
            h1_ch = "N/A"
            h2_ch = "N/A"
            if "h1_conclusion_changed" in df:
                p1, se1 = compute_rate_and_mcse(df["h1_conclusion_changed"])
                h1_ch = f"{p1:.3f} ({se1:.3f})"
            if "h2_conclusion_changed" in df:
                p2, se2 = compute_rate_and_mcse(df["h2_conclusion_changed"])
                h2_ch = f"{p2:.3f} ({se2:.3f})"
            print(f"{sc_name:<35s} | {rm_str:<18s} | {h1_ch:<22s} | {h2_ch:<22s}")

    if not acq_found:
        print("  (No acquiescence scenario shards found in results directory yet)")

    print("\n" + "=" * 105)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation Sanity Reporter.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing shard parquet files (default: results)")
    args = parser.parse_args()
    generate_sanity_report(results_dir=args.results_dir)
