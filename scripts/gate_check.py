#!/usr/bin/env python3
"""Gate design check script (Milestone 6.8 Part E).

Evaluates candidate structural-validity gate rules on DGP synthetic data
(500 replications, no model fitting):
- Rule 1: HTMT > 0.85
- Rule 2: Parallel Analysis < 2 factors
- Rule 3: Either rule (HTMT > 0.85 OR PA < 2 factors)

Scenarios evaluated:
- base_none (n=100, 150)
- cross_loading 0.3 (n=100, 150)
- cross_loading 0.5 (n=100, 150)
- contaminated kappa 0.4 (n=100, 150)
- high_latent_corr (n=100, 150)

Evaluates pairs:
- A-i / B-i (A_tech and B_i)
- A-ii / B-ii (A_mkt and B_ii)
- C / G (C and G)

Reports to reports/gate_check.txt.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
import numpy as np

# Ensure repo root on path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

import src.dgp as dgp
from src.measurement import htmt, parallel_analysis


def compute_rate_and_mcse(hits: np.ndarray) -> tuple[float, float]:
    p = float(np.mean(hits))
    mcse = float(np.sqrt(max(0.0, p * (1.0 - p)) / len(hits)))
    return p, mcse


def format_rate_mcse(p: float, mcse: float) -> str:
    return f"{p * 100:5.1f}% (±{mcse * 100:4.1f}%)"


def run_gate_checks(n_reps: int = 500, out_path: Path | str = "reports/gate_check.txt") -> None:
    catalog = dgp.get_scenario_catalog()
    
    conditions = [
        ("base_none", "base_n{n}_none", "clean"),
        ("cross_loading 0.3", "sens_cross_loading_n{n}_none", "sens_mild"),
        ("cross_loading 0.5", "sens_cross_loading_0p5_n{n}_none", "violation"),
        ("contaminated kappa 0.4", "sens_contaminated_n{n}_none", "violation"),
        ("high_latent_corr", "sens_high_latent_corr_n{n}_none", "clean"),
    ]
    
    pairs = [
        ("A-i / B-i", "A_tech", "B_i"),
        ("A-ii / B-ii", "A_mkt", "B_ii"),
        ("C / G", "C", "G"),
    ]
    
    lines = []
    lines.append("=" * 125)
    lines.append("STRUCTURAL-VALIDITY GATE DESIGN AUDIT (DGP ONLY, R = 500 Replications)")
    lines.append("Rules evaluated:")
    lines.append("  - Rule 1: HTMT > 0.85")
    lines.append("  - Rule 2: Parallel Analysis < 2 factors")
    lines.append("  - Rule 3: Either rule (HTMT > 0.85 OR PA < 2 factors)")
    lines.append("Markings:")
    lines.append("  - [!] False-flag > 5% on clean constructs (base_none, high_latent_corr)")
    lines.append("  - [?] Detection < 70% on violation constructs (cross_loading 0.5 on A/B pairs, kappa 0.4 on C/G)")
    lines.append("=" * 125)
    lines.append("")
    
    header = (
        f"{'Scenario':<25} | {'n':<4} | {'Pair':<12} | "
        f"{'Mean HTMT':<9} | {'Rule 1 (HTMT>0.85)':<18} | {'Rule 2 (PA<2)':<18} | {'Rule 3 (Either)':<18} | Notes"
    )
    lines.append(header)
    lines.append("-" * len(header))
    
    for cond_label, sc_template, cond_type in conditions:
        for n in [100, 150]:
            sc_name = sc_template.format(n=n)
            sc = catalog[sc_name]
            
            # Replications
            master_seq = np.random.SeedSequence(int(sc.get("seed", 20261013)) + n)
            child_seeds = master_seq.spawn(n_reps)
            
            htmt_data = {p_name: [] for p_name, _, _ in pairs}
            pa_data = {p_name: [] for p_name, _, _ in pairs}
            
            for rep in range(n_reps):
                df = dgp.generate(sc, seed=child_seeds[rep])
                
                for p_name, c1, c2 in pairs:
                    cols1 = [c for c in df.columns if c.startswith(f"{c1}_") and c[len(c1)+1:].isdigit()]
                    cols2 = [c for c in df.columns if c.startswith(f"{c2}_") and c[len(c2)+1:].isdigit()]
                    all_cols = cols1 + cols2
                    
                    # HTMT
                    h_res = htmt(df[all_cols], {c1: cols1, c2: cols2})
                    v_htmt = float(h_res.get((c1, c2), 0.0))
                    htmt_data[p_name].append(v_htmt)
                    
                    # Parallel Analysis (using cached null thresholds)
                    mat = df[all_cols].to_numpy(dtype=np.float64)
                    n_fac = parallel_analysis(mat, n_iter=500, use_cache=True)
                    pa_data[p_name].append(n_fac)
                    
            for p_name, c1, c2 in pairs:
                h_arr = np.array(htmt_data[p_name])
                pa_arr = np.array(pa_data[p_name])
                
                r1_hits = (h_arr > 0.85).astype(float)
                r2_hits = (pa_arr < 2).astype(float)
                r3_hits = ((h_arr > 0.85) | (pa_arr < 2)).astype(float)
                
                mean_h = float(np.mean(h_arr))
                r1_p, r1_se = compute_rate_and_mcse(r1_hits)
                r2_p, r2_se = compute_rate_and_mcse(r2_hits)
                r3_p, r3_se = compute_rate_and_mcse(r3_hits)
                
                # Check markings
                notes = []
                if cond_type == "clean":
                    # Mark if false-flag exceeds 5%
                    if r1_p > 0.05:
                        notes.append("R1>5% false-flag")
                    if r2_p > 0.05:
                        notes.append("R2>5% false-flag")
                    if r3_p > 0.05:
                        notes.append("R3>5% false-flag")
                elif cond_type == "violation":
                    # For cross_loading 0.5, violation targets A-i/B-i and A-ii/B-ii
                    # For contaminated kappa 0.4, violation targets C/G
                    is_target_pair = (
                        ("cross_loading" in cond_label and p_name in ("A-i / B-i", "A-ii / B-ii")) or
                        ("contaminated" in cond_label and p_name == "C / G")
                    )
                    if is_target_pair:
                        if r1_p < 0.70:
                            notes.append("R1<70% detect")
                        if r2_p < 0.70:
                            notes.append("R2<70% detect")
                        if r3_p < 0.70:
                            notes.append("R3<70% detect")
                
                note_str = ", ".join(notes) if notes else "-"
                row = (
                    f"{cond_label:<25} | {n:<4} | {p_name:<12} | "
                    f"{mean_h:<9.4f} | {format_rate_mcse(r1_p, r1_se):<18} | "
                    f"{format_rate_mcse(r2_p, r2_se):<18} | {format_rate_mcse(r3_p, r3_se):<18} | {note_str}"
                )
                lines.append(row)
        lines.append("")
        
    report_text = "\n".join(lines) + "\n"
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report_text, encoding="utf-8")
    print(report_text)
    print(f"\nSaved gate design check report to: {out}")


def main():
    parser = argparse.ArgumentParser(description="Gate Design Audit Script")
    parser.add_argument("--reps", type=int, default=500, help="Replications per scenario (default: 500)")
    parser.add_argument("--out", type=str, default="reports/gate_check.txt", help="Output report path")
    args = parser.parse_args()
    run_gate_checks(n_reps=args.reps, out_path=args.out)


if __name__ == "__main__":
    main()
