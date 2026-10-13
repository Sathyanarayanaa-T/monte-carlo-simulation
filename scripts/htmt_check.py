#!/usr/bin/env bash
# scripts/htmt_check.py
"""Compute structural-validity numbers (HTMT and Parallel Analysis) on DGP data.

Milestone 6.7 Part B:
For base none, cross_loading 0.3, and contaminated kappa 0.4 at n=100 and n=150
(500 replications each):
- Mean HTMT for A-i/B-i, A-ii/B-ii, and C/G
- Share above 0.75, 0.80, 0.85, and 0.90 with MCSE
- Parallel-analysis factor-count distribution for each pair
- Print Milestone 3 claim (>85% flag rate at 0.3 cross-loading) next to measured number.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import numpy as np
import pandas as pd

from src.dgp import build_scenario, generate
from src.measurement import htmt, parallel_analysis


def compute_rate_and_mcse(vals: np.ndarray, threshold: float) -> tuple[float, float]:
    hits = (vals > threshold).astype(float)
    p = float(np.mean(hits))
    mcse = float(np.sqrt(max(0.0, p * (1.0 - p)) / len(vals)))
    return p, mcse


def run_checks(n_reps: int = 500) -> None:
    conditions = [
        ("Base (none)", {"cross_loading": 0.0, "kappa": 0.0}, [100, 150]),
        ("Cross-Loading (0.3)", {"cross_loading": 0.3, "kappa": 0.0}, [100, 150]),
        ("Contaminated (kappa 0.4)", {"cross_loading": 0.0, "kappa": 0.4}, [100, 150]),
    ]

    thresholds = [0.75, 0.80, 0.85, 0.90]

    print("=" * 115)
    print(f"STRUCTURAL-VALIDITY DGP AUDIT (R = {n_reps} Replications per Condition)")
    print("=" * 115)

    results_table = []
    pa_distributions = []

    for cond_name, overrides, sample_sizes in conditions:
        for n in sample_sizes:
            sc_cfg = {
                "name": f"audit_{cond_name}_n{n}",
                "n": n,
                "non_linear_form": "none",
                "psi": 0.0,
                **overrides,
            }
            sc = build_scenario(sc_cfg)

            htmt_ab = []
            htmt_am_bm = []
            htmt_cg = []

            pa_ab = []
            pa_am_bm = []
            pa_cg = []

            cols_ai = [f"A_tech_{i}" for i in range(1, 6)]
            cols_bi = [f"B_i_{i}" for i in range(1, 6)]
            cols_aii = [f"A_mkt_{i}" for i in range(1, 5)]
            cols_bii = [f"B_ii_{i}" for i in range(1, 5)]
            cols_c = [f"C_{i}" for i in range(1, 5)]
            cols_g = [f"G_{i}" for i in range(1, 5)]

            for rep in range(n_reps):
                df = generate(sc, seed=rep + 50000)

                v_ab = float(htmt(df[cols_ai + cols_bi], {"A_tech": cols_ai, "B_i": cols_bi}))
                v_am_bm = float(htmt(df[cols_aii + cols_bii], {"A_mkt": cols_aii, "B_ii": cols_bii}))
                v_cg = float(htmt(df[cols_c + cols_g], {"C": cols_c, "G": cols_g}))

                htmt_ab.append(v_ab)
                htmt_am_bm.append(v_am_bm)
                htmt_cg.append(v_cg)

                p_ab = parallel_analysis(df[cols_ai + cols_bi].to_numpy(dtype=np.float64), n_iter=100)
                p_am_bm = parallel_analysis(df[cols_aii + cols_bii].to_numpy(dtype=np.float64), n_iter=100)
                p_cg = parallel_analysis(df[cols_c + cols_g].to_numpy(dtype=np.float64), n_iter=100)

                pa_ab.append(p_ab)
                pa_am_bm.append(p_am_bm)
                pa_cg.append(p_cg)

            arr_ab = np.array(htmt_ab)
            arr_am_bm = np.array(htmt_am_bm)
            arr_cg = np.array(htmt_cg)

            for pair_name, arr, pa_arr in [
                ("A-i / B-i", arr_ab, pa_ab),
                ("A-ii / B-ii", arr_am_bm, pa_am_bm),
                ("C / G", arr_cg, pa_cg),
            ]:
                rates = [compute_rate_and_mcse(arr, t) for t in thresholds]
                results_table.append({
                    "condition": cond_name,
                    "n": n,
                    "pair": pair_name,
                    "mean_htmt": float(np.mean(arr)),
                    "rates": rates,
                })

                # PA counts
                counts = pd.Series(pa_arr).value_counts(normalize=True).to_dict()
                pa_distributions.append({
                    "condition": cond_name,
                    "n": n,
                    "pair": pair_name,
                    "counts": counts,
                })

    # Print Table 1: HTMT Distribution & Threshold Shares
    print(f"\n[1] HTMT Empirical Distribution & Threshold Shares:")
    print("-" * 115)
    print(f"{'Condition':<25s} | {'n':<4s} | {'Pair':<12s} | {'Mean HTMT':<10s} | {'> 0.75 (MCSE)':<15s} | {'> 0.80 (MCSE)':<15s} | {'> 0.85 (MCSE)':<15s} | {'> 0.90 (MCSE)':<15s}")
    print("-" * 115)

    for r in results_table:
        r_strs = [f"{p:.3f} ({se:.3f})" for p, se in r["rates"]]
        print(f"{r['condition']:<25s} | {r['n']:<4d} | {r['pair']:<12s} | {r['mean_htmt']:<10.4f} | {r_strs[0]:<15s} | {r_strs[1]:<15s} | {r_strs[2]:<15s} | {r_strs[3]:<15s}")

    # Print Table 2: Parallel Analysis Factor Counts
    print(f"\n[2] Parallel Analysis Factor Count Distribution (Horn's Rule):")
    print("-" * 80)
    print(f"{'Condition':<25s} | {'n':<4s} | {'Pair':<12s} | {'Factor Count Distribution (%)'}")
    print("-" * 80)
    for p in pa_distributions:
        cnt_strs = [f"{k} factors: {v*100:.1f}%" for k, v in sorted(p["counts"].items())]
        print(f"{p['condition']:<25s} | {p['n']:<4d} | {p['pair']:<12s} | {', '.join(cnt_strs)}")

    # Print Milestone 3 Comparison
    print("\n" + "=" * 115)
    print("[3] Milestone 3 Claim Comparison:")
    print("-" * 115)
    print("Milestone 3 Claim: '> 85% flag rate (HTMT > 0.85) at 0.3 cross-loading'")

    for r in results_table:
        if r["condition"] == "Cross-Loading (0.3)" and r["pair"] == "A-i / B-i":
            p_85, se_85 = r["rates"][2]
            supported = "SUPPORTED" if p_85 >= 0.85 else "NOT SUPPORTED (fell below 85%)"
            print(f"  -> Measured at n={r['n']}: {p_85*100:.1f}% (MCSE: {se_85*100:.2f}%) | Result: {supported}")

    print("=" * 115)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-reps", type=int, default=500)
    args = parser.parse_args()
    run_checks(n_reps=args.n_reps)
