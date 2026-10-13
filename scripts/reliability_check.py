#!/usr/bin/env python3
"""Reliability check script (Milestone 6.8 Part A).

Computes Cronbach's alpha and McDonald's omega for constructs:
A_tech, A_mkt, B_i, B_ii, B_iii, C, G.

1. Population level: n=200,000 (seed 42), both scored and raw modes.
   Reports alpha, omega to 5 decimals, item count, and item loadings.
2. Finite samples: n=100 and n=150, 500 replications (mean and SD of alpha/omega).
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np
from sklearn.decomposition import FactorAnalysis

# Ensure repo root is on sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import src.dgp as dgp


def calc_cronbach_alpha(items: np.ndarray) -> float:
    k = items.shape[1]
    if k <= 1:
        return 1.0
    item_vars = np.var(items, axis=0, ddof=1)
    tot_var = np.var(np.sum(items, axis=1), ddof=1)
    if tot_var <= 0:
        return 0.0
    return float((k / (k - 1)) * (1.0 - np.sum(item_vars) / tot_var))


def calc_mcdonald_omega(items: np.ndarray) -> float:
    k = items.shape[1]
    if k <= 1:
        return 1.0
    fa = FactorAnalysis(n_components=1, random_state=42)
    fa.fit(items)
    loadings = fa.components_[0]
    uniquenesses = fa.noise_variance_
    sum_loadings = np.sum(np.abs(loadings))
    numerator = sum_loadings**2
    denominator = numerator + np.sum(uniquenesses)
    if denominator <= 0:
        return 0.0
    return float(numerator / denominator)


def evaluate_population(mode: str, seed: int = 42, n_pop: int = 200_000):
    sc = dgp.build_scenario({"n": n_pop, "ceiling_shift_direction": mode})
    df = dgp.generate(sc, seed=seed)
    
    # Get loadings and item counts from scenario
    loadings_dict = dgp.draw_scenario_loadings(sc)
    constructs = ["A_tech", "A_mkt", "B_i", "B_ii", "B_iii", "C", "G"]
    
    results = {}
    for c in constructs:
        cols = [col for col in df.columns if col.startswith(f"{c}_") and col[len(c) + 1:].isdigit()]
        items = df[cols].values
        alpha = calc_cronbach_alpha(items)
        omega = calc_mcdonald_omega(items)
        lam = loadings_dict[c].tolist()
        results[c] = {
            "k": len(cols),
            "loadings": lam,
            "alpha": alpha,
            "omega": omega,
        }
    return results


def evaluate_finite_samples(n: int, n_reps: int = 500, mode: str = "scored", master_seed: int = 20261013):
    sc = dgp.build_scenario({"n": n, "ceiling_shift_direction": mode})
    master_seq = np.random.SeedSequence(master_seed)
    child_seeds = master_seq.spawn(n_reps)
    
    constructs = ["A_tech", "A_mkt", "B_i", "B_ii", "B_iii", "C", "G"]
    alphas = {c: [] for c in constructs}
    omegas = {c: [] for c in constructs}
    
    for i in range(n_reps):
        seed_i = child_seeds[i]
        df = dgp.generate(sc, seed=seed_i)
        for c in constructs:
            cols = [col for col in df.columns if col.startswith(f"{c}_") and col[len(c) + 1:].isdigit()]
            items = df[cols].values
            alphas[c].append(calc_cronbach_alpha(items))
            omegas[c].append(calc_mcdonald_omega(items))
            
    summary = {}
    for c in constructs:
        summary[c] = {
            "alpha_mean": float(np.mean(alphas[c])),
            "alpha_sd": float(np.std(alphas[c], ddof=1)),
            "omega_mean": float(np.mean(omegas[c])),
            "omega_sd": float(np.std(omegas[c], ddof=1)),
        }
    return summary


def main():
    parser = argparse.ArgumentParser(description="Reliability evaluation script")
    parser.add_argument("--out", type=str, default="reports/reliability_check.txt", help="Output report path")
    args = parser.parse_args()
    
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    
    lines = []
    lines.append("================================================================================")
    lines.append("RELIABILITY AUDIT REPORT (Cronbach's alpha and McDonald's omega)")
    lines.append("================================================================================\n")
    
    print("Evaluating population N=200,000 (scored mode)...")
    pop_scored = evaluate_population("scored", seed=42)
    print("Evaluating population N=200,000 (raw mode)...")
    pop_raw = evaluate_population("raw", seed=42)
    
    lines.append("--- POPULATION LEVEL (N=200,000, seed 42) ---")
    header = f"{'Construct':<10} | {'Items':<5} | {'Loadings':<38} | {'Scored Alpha':<12} | {'Scored Omega':<12} | {'Raw Alpha':<12} | {'Raw Omega':<12}"
    lines.append(header)
    lines.append("-" * len(header))
    
    for c in ["A_tech", "A_mkt", "B_i", "B_ii", "B_iii", "C", "G"]:
        k = pop_scored[c]["k"]
        lams = "[" + ", ".join(f"{x:.3f}" for x in pop_scored[c]["loadings"]) + "]"
        a_sc = f"{pop_scored[c]['alpha']:.5f}"
        w_sc = f"{pop_scored[c]['omega']:.5f}"
        a_raw = f"{pop_raw[c]['alpha']:.5f}"
        w_raw = f"{pop_raw[c]['omega']:.5f}"
        lines.append(f"{c:<10} | {k:<5} | {lams:<38} | {a_sc:<12} | {w_sc:<12} | {a_raw:<12} | {w_raw:<12}")
    lines.append("")
    
    print("Evaluating finite sample n=100 (500 replications, scored)...")
    fin_100 = evaluate_finite_samples(n=100, n_reps=500, mode="scored")
    print("Evaluating finite sample n=150 (500 replications, scored)...")
    fin_150 = evaluate_finite_samples(n=150, n_reps=500, mode="scored")
    
    lines.append("--- FINITE SAMPLE: n=100 (500 replications, scored mode) ---")
    h100 = f"{'Construct':<10} | {'Alpha Mean':<12} | {'Alpha SD':<12} | {'Omega Mean':<12} | {'Omega SD':<12}"
    lines.append(h100)
    lines.append("-" * len(h100))
    for c in ["A_tech", "A_mkt", "B_i", "B_ii", "B_iii", "C", "G"]:
        lines.append(f"{c:<10} | {fin_100[c]['alpha_mean']:.5f}      | {fin_100[c]['alpha_sd']:.5f}      | {fin_100[c]['omega_mean']:.5f}      | {fin_100[c]['omega_sd']:.5f}")
    lines.append("")
    
    lines.append("--- FINITE SAMPLE: n=150 (500 replications, scored mode) ---")
    h150 = f"{'Construct':<10} | {'Alpha Mean':<12} | {'Alpha SD':<12} | {'Omega Mean':<12} | {'Omega SD':<12}"
    lines.append(h150)
    lines.append("-" * len(h150))
    for c in ["A_tech", "A_mkt", "B_i", "B_ii", "B_iii", "C", "G"]:
        lines.append(f"{c:<10} | {fin_150[c]['alpha_mean']:.5f}      | {fin_150[c]['alpha_sd']:.5f}      | {fin_150[c]['omega_mean']:.5f}      | {fin_150[c]['omega_sd']:.5f}")
    lines.append("")
    
    report_text = "\n".join(lines) + "\n"
    out_path.write_text(report_text, encoding="utf-8")
    print(f"\nWrote report to {out_path}")
    print(report_text)


if __name__ == "__main__":
    main()
