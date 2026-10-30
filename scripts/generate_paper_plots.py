#!/usr/bin/env python3
"""Post-simulation publication plot and table generator.

SPEC.md Section 10 & 12 (Milestone 7):
Generates the 8 standard paper figures and summary appendix tables from the
Monte Carlo simulation Parquet shards in results/:
  Figure 1: Power curves by n for H1 and H2 (Base DGP, MCSE bands, 80% target)
  Figure 2: Interaction detection (TPR & FPR) across methods and n (Panel A: Interaction, Panel B: Threshold)
  Figure 3: ROC & Empirical AUC comparison at n in {100, 150}
  Figure 4: Feature importance ranking accuracy (Kendall tau vs Oracle) and bootstrap stability
  Figure 5: Out-of-sample R^2 and delta R^2 vs Elastic Net (M3) across n
  Figure 6: Leaky-design demonstration (Delta R^2 inflation and H1 FPR under H1-null)
  Figure 7: Sensitivity heatmap (shifts in H1/H2 power and bias across sensitivity conditions)
  Tables: Appendix summary tables (LaTeX & CSV) with Monte Carlo standard errors

SPEC.md line 18: Every figure and table carries the mandatory label '[Simulated Data]'.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

import matplotlib
matplotlib.use("Agg")  # Headless backend
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import roc_curve, auc

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from src.leaky import run_leaky_demonstration

# Color palette: Academic publication grade
PALETTE = {
    "M0": "#7f7f7f",       # Grey
    "M1": "#1f77b4",       # Deep Blue
    "M1-adj": "#17becf",   # Cyan
    "M2a": "#2ca02c",      # Forest Green
    "M2b": "#8c564b",      # Brown
    "M3": "#ff7f0e",       # Orange
    "M4": "#9467bd",       # Purple
    "M5": "#d62728",       # Crimson Red
    "M6": "#e377c2",       # Pink/Magenta
    "H1_F": "#1f77b4",
    "H1_t": "#2ca02c",
    "H2": "#ff7f0e",
    "correct": "#2ca02c",
    "leaky": "#d62728",
}

METHOD_LABELS = {
    "M0": "M0: Intercept Null",
    "M1": "M1: Hierarchical OLS",
    "M1-adj": "M1-adj: Marker OLS",
    "M2a": "M2a: Pre-specified OLS",
    "M2b": "M2b: Exploratory OLS (Holm)",
    "M3": "M3: Elastic Net",
    "M4": "M4: Random Forest",
    "M5": "M5: Gradient Boosting",
    "M6": "M6: EBM",
}


def apply_plot_style(ax: plt.Axes, title: str, xlabel: str, ylabel: str) -> None:
    """Apply consistent styling and the mandatory SPEC line 18 label."""
    ax.set_title(f"{title}\n[Simulated Data]", fontsize=11, fontweight="bold", pad=8)
    ax.set_xlabel(xlabel, fontsize=10, fontweight="medium")
    ax.set_ylabel(ylabel, fontsize=10, fontweight="medium")
    ax.grid(True, linestyle="--", alpha=0.5, color="#cccccc")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def load_all_shards(results_dir: Path) -> dict[str, pd.DataFrame]:
    """Load all completed parquet shards from the results directory."""
    shards: dict[str, pd.DataFrame] = {}
    if not results_dir.exists():
        return shards

    for p in sorted(results_dir.glob("*.parquet")):
        # Skip leaky demo cache from regular scenario catalog if needed
        sc_name = p.stem
        try:
            df = pd.read_parquet(p)
            shards[sc_name] = df
        except Exception as e:
            print(f"Warning: Failed to load shard {p}: {e}")
    return shards


# =============================================================================
# FIGURE 1: Power Curves by n for H1 and H2
# =============================================================================
def generate_figure1_power_curves(
    shards: dict[str, pd.DataFrame],
    out_dir: Path,
    dpi: int = 300,
) -> Path | None:
    """Figure 1: Power curves by n for H1 and H2 under Base DGP (SPEC.md Section 10.1)."""
    n_vals = [50, 100, 150, 300]
    records = []

    for n in n_vals:
        sc_name = f"base_n{n}_none"
        if sc_name not in shards:
            continue
        df = shards[sc_name]
        r = len(df)
        if r == 0:
            continue

        for col, label, color in [
            ("h1_f_hit", "H1 (Block Gain F-test)", PALETTE["H1_F"]),
            ("h1_t_hit", "H1 (A_tech t-test)", PALETTE["H1_t"]),
            ("h2_hit", "H2 (B-i > B-ii Contrast)", PALETTE["H2"]),
        ]:
            if col in df.columns:
                p = float(df[col].mean())
                se = float(np.sqrt(p * (1.0 - p) / r))
                records.append({
                    "n": n,
                    "metric": label,
                    "power": p,
                    "mcse": se,
                    "color": color,
                })

    if not records:
        print("[-] Figure 1 skipped: No base scenario shards found.")
        return None

    df_p = pd.DataFrame(records)

    fig, ax = plt.subplots(figsize=(7.5, 5.0))

    for label, group in df_p.groupby("metric"):
        grp = group.sort_values("n")
        color = grp["color"].iloc[0]
        ax.plot(grp["n"], grp["power"], marker="o", linewidth=2.2, label=label, color=color)
        ax.fill_between(
            grp["n"],
            np.clip(grp["power"] - 1.96 * grp["mcse"], 0.0, 1.0),
            np.clip(grp["power"] + 1.96 * grp["mcse"], 0.0, 1.0),
            color=color,
            alpha=0.15,
        )

    # 80% power benchmark line
    ax.axhline(0.80, color="#d62728", linestyle=":", linewidth=1.5, label="80% Power Threshold")
    ax.axhline(0.05, color="#7f7f7f", linestyle="--", linewidth=1.0, alpha=0.7, label="Nominal alpha = 0.05")

    ax.set_ylim(-0.02, 1.05)
    ax.set_xticks(n_vals)
    apply_plot_style(ax, "Figure 1: Statistical Power for H1 and H2 by Sample Size (Base DGP)", "Sample Size (n)", "Empirical Power (Hit Rate)")
    ax.legend(frameon=True, facecolor="white", framealpha=0.9, loc="lower right", fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "figure1_power_curves.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[+] Saved Figure 1 to {out_path}")
    return out_path


# =============================================================================
# FIGURE 2: Interaction Detection (TPR & FPR) across Methods and n
# =============================================================================
def generate_figure2_interaction_detection(
    shards: dict[str, pd.DataFrame],
    out_dir: Path,
    dpi: int = 300,
) -> Path | None:
    """Figure 2: Interaction detection (TPR and FPR) by method and n (SPEC.md Section 10.2)."""
    n_vals = [50, 100, 150, 300]
    methods = [
        ("m2a_detected", "M2a (Oracle OLS)", PALETTE["M2a"], "-"),
        ("m2b_detected", "M2b (Holm OLS)", PALETTE["M2b"], "-"),
        ("m3_detected", "M3 (Elastic Net)", PALETTE["M3"], "--"),
        ("m4_detected", "M4 (Random Forest)", PALETTE["M4"], "-."),
        ("m5_detected", "M5 (Gradient Boosting)", PALETTE["M5"], "-"),
        ("m6_detected", "M6 (EBM)", PALETTE["M6"], ":"),
    ]

    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.2), sharey=True)

    # Panel A: Bilinear Interaction DGP (psi = 0.08)
    # Panel B: Threshold DGP (psi = 0.08)
    panels = [
        (axes[0], "inter_med", "Panel A: Bilinear Interaction (psi = 0.08)"),
        (axes[1], "thresh_med", "Panel B: Localized Threshold (psi = 0.08)"),
    ]

    has_data = False
    for ax, cell_type, title in panels:
        for col, label, color, lstyle in methods:
            tpr_points = []
            for n in n_vals:
                sc_name = f"base_n{n}_{cell_type}"
                if sc_name in shards and col in shards[sc_name].columns:
                    val = float(shards[sc_name][col].mean())
                    tpr_points.append((n, val))

            if tpr_points:
                has_data = True
                xs, ys = zip(*tpr_points)
                ax.plot(xs, ys, marker="s", linestyle=lstyle, linewidth=2.0, color=color, label=label)

        # Plot Null FPR (psi = 0) as baseline reference
        for col, label, color, _ in methods:
            fpr_points = []
            for n in n_vals:
                sc_null = f"base_n{n}_none"
                if sc_null in shards and col in shards[sc_null].columns:
                    val = float(shards[sc_null][col].mean())
                    fpr_points.append((n, val))
            if fpr_points:
                xs, ys = zip(*fpr_points)
                ax.plot(xs, ys, marker="x", linestyle=":", linewidth=1.0, color=color, alpha=0.4)

        ax.axhline(0.05, color="black", linestyle="--", linewidth=1.0, alpha=0.6, label="Nominal alpha = 0.05")
        ax.set_ylim(-0.02, 1.05)
        ax.set_xticks(n_vals)
        apply_plot_style(ax, title, "Sample Size (n)", "Detection Rate (TPR / FPR)")

    if not has_data:
        plt.close(fig)
        print("[-] Figure 2 skipped: No interaction scenario shards found.")
        return None

    axes[1].legend(frameon=True, facecolor="white", framealpha=0.9, loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=9)
    fig.tight_layout()
    out_path = out_dir / "figure2_interaction_detection.png"
    fig.savefig(out_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    print(f"[+] Saved Figure 2 to {out_path}")
    return out_path


# =============================================================================
# FIGURE 3: ROC and AUC Comparison at n in {100, 150}
# =============================================================================
def generate_figure3_roc_auc(
    shards: dict[str, pd.DataFrame],
    out_dir: Path,
    dpi: int = 300,
) -> Path | None:
    """Figure 3: ROC and empirical AUC comparison at n in {100, 150} (SPEC.md Section 10.3)."""
    fig, axes = plt.subplots(1, 2, figsize=(12.0, 5.0))

    methods = [
        ("M2a", "m2a_pvalue", True, PALETTE["M2a"]),
        ("M2b", "m2b_min_corrected_p", True, PALETTE["M2b"]),
        ("M3", "m3_max_freq", False, PALETTE["M3"]),
        ("M4", "m4_pvalue", True, PALETTE["M4"]),
        ("M5", "m5_pvalue", True, PALETTE["M5"]),
        ("M6", "m6_pvalue", True, PALETTE["M6"]),
    ]

    has_data = False
    for idx, n in enumerate([100, 150]):
        ax = axes[idx]
        sc_null = f"base_n{n}_none"
        sc_signal = f"base_n{n}_inter_med"

        if sc_null not in shards or sc_signal not in shards:
            ax.text(0.5, 0.5, f"Data not available for n = {n}", ha="center", va="center", color="gray")
            apply_plot_style(ax, f"ROC Comparison (n = {n})", "False Positive Rate", "True Positive Rate")
            continue

        df_null = shards[sc_null]
        df_sig = shards[sc_signal]

        for m_name, score_col, invert, color in methods:
            if score_col in df_null.columns and score_col in df_sig.columns:
                null_scores = df_null[score_col].dropna().values
                sig_scores = df_sig[score_col].dropna().values
                if len(null_scores) == 0 or len(sig_scores) == 0:
                    continue

                y_true = np.concatenate([np.zeros(len(null_scores)), np.ones(len(sig_scores))])
                y_scores = np.concatenate([null_scores, sig_scores])
                if invert:
                    y_scores = 1.0 - y_scores

                fpr, tpr, _ = roc_curve(y_true, y_scores)
                roc_auc = auc(fpr, tpr)
                ax.plot(fpr, tpr, label=f"{m_name} (AUC = {roc_auc:.2f})", color=color, linewidth=2.0)
                has_data = True

        ax.plot([0, 1], [0, 1], "k--", alpha=0.5, label="Chance (AUC = 0.50)")
        ax.set_xlim(-0.02, 1.02)
        ax.set_ylim(-0.02, 1.02)
        apply_plot_style(ax, f"ROC Curves at Sample Size n = {n}", "False Positive Rate", "True Positive Rate")
        ax.legend(loc="lower right", fontsize=8.5, frameon=True)

    if not has_data:
        plt.close(fig)
        print("[-] Figure 3 skipped: Insufficient null and signal shards for ROC curves.")
        return None

    fig.tight_layout()
    out_path = out_dir / "figure3_roc_auc_comparison.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[+] Saved Figure 3 to {out_path}")
    return out_path


# =============================================================================
# FIGURE 4: Importance & Ranking Stability by Method
# =============================================================================
def generate_figure4_ranking_stability(
    shards: dict[str, pd.DataFrame],
    out_dir: Path,
    dpi: int = 300,
) -> Path | None:
    """Figure 4: Stability and ranking accuracy by method (SPEC.md Section 10.4)."""
    n_vals = [50, 100, 150, 300]
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.8))

    methods = [
        ("M1", "m1", PALETTE["M1"]),
        ("M4", "m4", PALETTE["M4"]),
        ("M5", "m5", PALETTE["M5"]),
        ("M6", "m6", PALETTE["M6"]),
    ]

    has_data = False
    # Panel A: Kendall's Tau vs Oracle Ranking
    ax1 = axes[0]
    for m_label, m_key, color in methods:
        pts = []
        for n in n_vals:
            sc_name = f"base_n{n}_none"
            col = f"{m_key}_kendall_tau"
            if sc_name in shards and col in shards[sc_name].columns:
                mean_tau = float(shards[sc_name][col].dropna().mean())
                pts.append((n, mean_tau))
        if pts:
            has_data = True
            xs, ys = zip(*pts)
            ax1.plot(xs, ys, marker="o", linewidth=2.0, color=color, label=f"{m_label}")

    if not has_data:
        plt.close(fig)
        print("[-] Figure 4 skipped: No importance or stability metrics found.")
        return None

    apply_plot_style(ax1, "Panel A: Ranking Accuracy vs. Oracle (Kendall's tau)", "Sample Size (n)", "Kendall's tau_b")
    ax1.set_xticks(n_vals)
    ax1.set_ylim(-0.2, 1.05)
    ax1.legend(loc="lower right", fontsize=9)

    # Panel B: Bootstrap Stability (50 resamples)
    ax2 = axes[1]
    has_stab = False
    for m_label, m_key, color in methods:
        pts = []
        for n in n_vals:
            sc_name = f"base_n{n}_none"
            col = f"{m_key}_stability"
            if sc_name in shards and col in shards[sc_name].columns:
                mean_stab = float(shards[sc_name][col].dropna().mean())
                pts.append((n, mean_stab))
        if pts:
            has_stab = True
            xs, ys = zip(*pts)
            ax2.plot(xs, ys, marker="s", linewidth=2.0, color=color, label=f"{m_label}")

    apply_plot_style(ax2, "Panel B: Bootstrap Ranking Stability (50 Resamples)", "Sample Size (n)", "Mean Bootstrap Kendall's tau")
    ax2.set_xticks(n_vals)
    ax2.set_ylim(-0.2, 1.05)
    if has_stab:
        ax2.legend(loc="lower right", fontsize=9)

    fig.tight_layout()
    out_path = out_dir / "figure4_ranking_stability.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[+] Saved Figure 4 to {out_path}")
    return out_path


# =============================================================================
# FIGURE 5: Out-of-Sample R^2 by Method and n
# =============================================================================
def generate_figure5_oos_r2(
    shards: dict[str, pd.DataFrame],
    out_dir: Path,
    dpi: int = 300,
) -> Path | None:
    """Figure 5: Out-of-sample R^2 and delta R^2 vs M3 (SPEC.md Section 10.5)."""
    n_vals = [50, 100, 150, 300]
    fig, axes = plt.subplots(1, 2, figsize=(13.0, 5.0))

    methods = [
        ("M0 (Baseline)", "m0_r2_oos", PALETTE["M0"]),
        ("M1 (Hierarchical OLS)", "m1_r2_oos", PALETTE["M1"]),
        ("M2a (Oracle OLS)", "m2a_r2_oos", PALETTE["M2a"]),
        ("M2b (Holm OLS)", "m2b_r2_oos", PALETTE["M2b"]),
        ("M3 (Elastic Net)", "m3_r2_oos", PALETTE["M3"]),
        ("M4 (Random Forest)", "m4_r2_oos", PALETTE["M4"]),
        ("M5 (Gradient Boosting)", "m5_r2_oos", PALETTE["M5"]),
        ("M6 (EBM)", "m6_r2_oos", PALETTE["M6"]),
    ]

    has_data = False
    # Panel A: Mean Out-of-sample R^2
    ax1 = axes[0]
    for label, col, color in methods:
        pts = []
        for n in n_vals:
            sc_name = f"base_n{n}_none"
            if sc_name in shards and col in shards[sc_name].columns:
                val = float(shards[sc_name][col].dropna().mean())
                pts.append((n, val))
        if pts:
            has_data = True
            xs, ys = zip(*pts)
            ax1.plot(xs, ys, marker="o", linewidth=1.8, color=color, label=label)

    if not has_data:
        plt.close(fig)
        print("[-] Figure 5 skipped: No out-of-sample prediction columns found.")
        return None

    apply_plot_style(ax1, "Panel A: Mean Out-of-Sample R^2 Across Methods", "Sample Size (n)", "Out-of-Sample R^2")
    ax1.set_xticks(n_vals)
    ax1.legend(loc="lower right", fontsize=8.5)

    # Panel B: Delta R^2 vs Elastic Net (M3)
    ax2 = axes[1]
    ml_methods = [
        ("M4 (Random Forest)", "m4_delta_r2_vs_m3", PALETTE["M4"]),
        ("M5 (Gradient Boosting)", "m5_delta_r2_vs_m3", PALETTE["M5"]),
        ("M6 (EBM)", "m6_delta_r2_vs_m3", PALETTE["M6"]),
    ]
    has_delta = False
    for label, col, color in ml_methods:
        pts = []
        for n in n_vals:
            sc_name = f"base_n{n}_none"
            if sc_name in shards and col in shards[sc_name].columns:
                val = float(shards[sc_name][col].dropna().mean())
                pts.append((n, val))
        if pts:
            has_delta = True
            xs, ys = zip(*pts)
            ax2.plot(xs, ys, marker="s", linewidth=2.0, color=color, label=label)

    # SPEC line 193: Decision rule threshold delta = 0.02
    ax2.axhline(0.02, color="#d62728", linestyle="--", linewidth=1.2, label="Pre-registered delta = 0.02 rule")
    ax2.axhline(0.00, color="gray", linestyle=":", linewidth=1.0)
    apply_plot_style(ax2, "Panel B: ML Added Value vs. Elastic Net (Delta R^2)", "Sample Size (n)", "Delta R^2 (Model - M3)")
    ax2.set_xticks(n_vals)
    if has_delta:
        ax2.legend(loc="upper right", fontsize=8.5)

    fig.tight_layout()
    out_path = out_dir / "figure5_oos_r2_prediction.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[+] Saved Figure 5 to {out_path}")
    return out_path


# =============================================================================
# FIGURE 6: Leaky-Design Demonstration
# =============================================================================
def generate_figure6_leaky_design(
    shards: dict[str, pd.DataFrame],
    out_dir: Path,
    results_dir: Path | None = None,
    dpi: int = 300,
) -> Path | None:
    """Figure 6: Leaky-design bias and FPR inflation (SPEC.md Section 7 & 10.6)."""
    # Check if pre-run cached shard exists in results
    target_res = results_dir if results_dir is not None else (repo_root / "results")
    cache_path = target_res / "leaky_demonstration.parquet"
    if cache_path.exists():
        df_leaky = pd.read_parquet(cache_path)
    else:
        # Run 500-rep demonstration fast
        print("[i] Generating Leaky-Design empirical demonstration (R = 500, n = 150)...")
        df_leaky, _ = run_leaky_demonstration(n=150, n_replications=500, n_jobs=2, verbose=False)
        try:
            target_res.mkdir(parents=True, exist_ok=True)
            df_leaky.to_parquet(cache_path)
        except Exception:
            pass

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.8))

    # Panel A: Delta R^2 Distribution
    ax1 = axes[0]
    d_corr = df_leaky["delta_r2_correct"].values
    d_leak = df_leaky["delta_r2_leaky"].values

    box_data = [d_corr, d_leak]
    bp = ax1.boxplot(box_data, tick_labels=["Correct Design\n(Outcome Excluded)", "Leaky Design\n(Outcome Included)"], patch_artist=True)
    bp["boxes"][0].set_facecolor("#2ca02c")
    bp["boxes"][1].set_facecolor("#d62728")
    for box in bp["boxes"]:
        box.set_alpha(0.7)

    mean_corr = np.mean(d_corr)
    mean_leak = np.mean(d_leak)
    inflation = mean_leak / mean_corr if mean_corr > 1e-6 else float("nan")

    ax1.text(0.5, 0.85, f"Mean Delta R^2 Inflation: {inflation:.1f}x\n({mean_corr:.3f} -> {mean_leak:.3f})",
             transform=ax1.transAxes, ha="center", bbox=dict(boxstyle="round", facecolor="white", alpha=0.9))
    apply_plot_style(ax1, "Panel A: Delta R^2 Inflation under H1-Null", "Study Survey Design", "Delta R^2 (Block Gain)")

    # Panel B: False Positive Rate (Nominal alpha = 0.05)
    ax2 = axes[1]
    fpr_corr = float(df_leaky["significant_correct"].mean())
    fpr_leak = float(df_leaky["significant_leaky"].mean())

    bars = ax2.bar(["Correct Design\n(Outcome Excluded)", "Leaky Design\n(Outcome Included)"],
                   [fpr_corr, fpr_leak], color=["#2ca02c", "#d62728"], alpha=0.8, width=0.55)
    ax2.axhline(0.05, color="black", linestyle="--", linewidth=1.2, label="Nominal alpha = 0.05")

    for bar in bars:
        h = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width() / 2.0, h + 0.02, f"{h:.1%}", ha="center", fontweight="bold")

    ax2.set_ylim(0.0, 1.15)
    apply_plot_style(ax2, "Panel B: H1 False Positive Rate under H1-Null", "Study Survey Design", "False Positive Rate (Hit Rate)")
    ax2.legend(loc="upper left")

    fig.tight_layout()
    out_path = out_dir / "figure6_leaky_design_bias.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[+] Saved Figure 6 to {out_path}")
    return out_path


# =============================================================================
# FIGURE 7: Sensitivity Heatmap
# =============================================================================
def generate_figure7_sensitivity_heatmap(
    shards: dict[str, pd.DataFrame],
    out_dir: Path,
    dpi: int = 300,
) -> Path | None:
    """Figure 7: Sensitivity heatmap showing power shifts vs base (SPEC.md Section 10.7)."""
    # Sensitivity conditions at n = 150
    sens_factors = [
        ("sens_cross_loading_0p3_n150_none", "Cross-Loading 0.3"),
        ("sens_cross_loading_0p5_n150_none", "Cross-Loading 0.5"),
        ("sens_high_latent_corr_n150_none", "High Latent Corr (0.60)"),
        ("sens_cmv_gamma_0p3_n150_none", "Acquiescence (gamma=0.3)"),
        ("sens_low_reliability_n150_none", "Low Reliability"),
        ("sens_low_r2_n150_none", "Low Target R^2 (0.20)"),
        ("sens_ceiling_shift_0p6_n150_none", "Ceiling Shift (+0.6)"),
        ("sens_contaminated_outcome_n150_none", "Contaminated Outcome (kappa=0.4)"),
    ]

    base_shard = shards.get("base_n150_none")
    if base_shard is None:
        print("[-] Figure 7 skipped: Base scenario 'base_n150_none' shard missing.")
        return None

    base_h1 = float(base_shard["h1_f_hit"].mean())
    base_h2 = float(base_shard["h2_hit"].mean())
    base_bias_a = float(base_shard["bias_score_A_tech"].mean()) if "bias_score_A_tech" in base_shard.columns else 0.0

    matrix_rows = []
    labels = []

    for sc_key, lbl in sens_factors:
        if sc_key in shards:
            df = shards[sc_key]
            delta_h1 = float(df["h1_f_hit"].mean()) - base_h1
            delta_h2 = float(df["h2_hit"].mean()) - base_h2
            bias_a = float(df["bias_score_A_tech"].mean()) if "bias_score_A_tech" in df.columns else 0.0
            delta_bias = bias_a - base_bias_a
            matrix_rows.append([delta_h1, delta_h2, delta_bias])
            labels.append(lbl)

    if not matrix_rows:
        print("[-] Figure 7 skipped: No sensitivity shards found.")
        return None

    mat = np.array(matrix_rows)

    fig, ax = plt.subplots(figsize=(8.5, max(4.0, len(labels) * 0.6)))
    cax = ax.matshow(mat, cmap="coolwarm", vmin=-0.5, vmax=0.5)

    fig.colorbar(cax, ax=ax, label="Shift Relative to Base Benchmark", fraction=0.046, pad=0.04)

    ax.set_xticks([0, 1, 2])
    ax.set_xticklabels(["Delta H1 Power", "Delta H2 Power", "Delta A_tech Bias"], fontsize=9.5, fontweight="bold")
    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)

    for i in range(len(labels)):
        for j in range(3):
            val = mat[i, j]
            text_color = "white" if abs(val) > 0.25 else "black"
            ax.text(j, i, f"{val:+.3f}", ha="center", va="center", color=text_color, fontweight="bold")

    ax.set_title("Figure 7: Sensitivity Analysis Heatmap (n = 150)\n[Simulated Data]", fontsize=11, fontweight="bold", pad=15)
    fig.tight_layout()
    out_path = out_dir / "figure7_sensitivity_heatmap.png"
    fig.savefig(out_path, dpi=dpi)
    plt.close(fig)
    print(f"[+] Saved Figure 7 to {out_path}")
    return out_path


def format_dataframe_to_latex(df: pd.DataFrame, caption: str) -> str:
    """Format DataFrame as a standard LaTeX tabular block without requiring jinja2."""
    cols = list(df.columns)
    col_align = "l" + "r" * (len(cols) - 1)
    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\begin{{tabular}}{{{col_align}}}",
        "\\toprule",
        " & ".join(str(c).replace("_", "\\_") for c in cols) + " \\\\",
        "\\midrule",
    ]
    for _, row in df.iterrows():
        row_str = " & ".join(str(val).replace("_", "\\_") for val in row.values) + " \\\\"
        lines.append(row_str)
    lines.extend([
        "\\bottomrule",
        "\\end{tabular}",
        "\\end{table}",
        "",
    ])
    return "\n".join(lines)


# =============================================================================
# APPENDIX TABLES: Full Quantitative Summaries
# =============================================================================
def generate_appendix_tables(shards: dict[str, pd.DataFrame], tables_dir: Path) -> list[Path]:
    """Generate LaTeX and CSV appendix tables with Monte Carlo SEs (SPEC.md Section 10.8)."""
    tables_dir.mkdir(parents=True, exist_ok=True)
    generated_files: list[Path] = []

    # Table 1: Inferential Performance
    t1_rows = []
    for sc_name, df in sorted(shards.items()):
        if not ("base_n" in sc_name or "h1_null" in sc_name or "h2_null" in sc_name):
            continue
        r = len(df)
        if r == 0:
            continue

        h1_p = float(df["h1_f_hit"].mean()) if "h1_f_hit" in df.columns else np.nan
        h1_se = np.sqrt(h1_p * (1.0 - h1_p) / r) if not np.isnan(h1_p) else np.nan

        h2_p = float(df["h2_hit"].mean()) if "h2_hit" in df.columns else np.nan
        h2_se = np.sqrt(h2_p * (1.0 - h2_p) / r) if not np.isnan(h2_p) else np.nan

        d_r2 = float(df["delta_r2"].mean()) if "delta_r2" in df.columns else np.nan

        t1_rows.append({
            "Scenario": sc_name,
            "Replications": r,
            "H1_Power": f"{h1_p:.3f} (±{h1_se:.3f})",
            "H2_Power": f"{h2_p:.3f} (±{h2_se:.3f})",
            "Mean_Delta_R2": f"{d_r2:.4f}",
        })

    if t1_rows:
        df_t1 = pd.DataFrame(t1_rows)
        csv_p = tables_dir / "table1_inferential_performance.csv"
        tex_p = tables_dir / "table1_inferential_performance.tex"
        df_t1.to_csv(csv_p, index=False)
        tex_content = format_dataframe_to_latex(df_t1, caption="Inferential Performance (H1, H2) with MCSE [Simulated Data]")
        with open(tex_p, "w") as f:
            f.write(tex_content)
        generated_files.extend([csv_p, tex_p])
        print(f"[+] Generated Table 1 ({csv_p.name}, {tex_p.name})")

    # Table 2: Psychometric Screening Gates
    t2_rows = []
    for sc_name, df in sorted(shards.items()):
        if not ("htmt_A_tech_B_i" in df.columns or "harman_first_factor_share" in df.columns):
            continue
        r = len(df)
        htmt_val = float(df["htmt_A_tech_B_i"].mean()) if "htmt_A_tech_B_i" in df.columns else np.nan
        flag_rate = float((df["htmt_A_tech_B_i"] > 0.85).mean()) if "htmt_A_tech_B_i" in df.columns else np.nan
        harman = float(df["harman_first_factor_share"].mean()) if "harman_first_factor_share" in df.columns else np.nan

        t2_rows.append({
            "Scenario": sc_name,
            "Replications": r,
            "Mean_HTMT_Ai_Bi": f"{htmt_val:.3f}",
            "HTMT_Flag_Rate_gt_0.85": f"{flag_rate:.3f}",
            "Harman_First_Factor_Share": f"{harman:.3f}",
        })

    if t2_rows:
        df_t2 = pd.DataFrame(t2_rows)
        csv_p = tables_dir / "table2_psychometrics_and_gates.csv"
        tex_p = tables_dir / "table2_psychometrics_and_gates.tex"
        df_t2.to_csv(csv_p, index=False)
        tex_content = format_dataframe_to_latex(df_t2, caption="Measurement Diagnostics & Screening Gates [Simulated Data]")
        with open(tex_p, "w") as f:
            f.write(tex_content)
        generated_files.extend([csv_p, tex_p])
        print(f"[+] Generated Table 2 ({csv_p.name}, {tex_p.name})")

    return generated_files


# =============================================================================
# CLI Entrypoint
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description="Generate publication paper plots and tables from simulation results.")
    parser.add_argument("--results-dir", type=str, default="results", help="Directory containing Parquet shards (default: results)")
    parser.add_argument("--out-dir", type=str, default="report", help="Output directory for plots and tables (default: report)")
    parser.add_argument("--dpi", type=int, default=300, help="Resolution for generated figures in DPI (default: 300)")
    args = parser.parse_args()

    results_dir = repo_root / args.results_dir
    out_dir = repo_root / args.out_dir
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"

    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 80)
    print("MONTE CARLO SIMULATION: PUBLICATION PLOT & TABLE GENERATOR")
    print(f"Results Directory: {results_dir}")
    print(f"Output Figures:   {fig_dir}")
    print(f"Output Tables:    {tab_dir}")
    print(f"DPI Resolution:   {args.dpi}")
    print("=" * 80)

    shards = load_all_shards(results_dir)
    print(f"Loaded {len(shards)} Parquet scenario shards from {results_dir}.\n")

    # Generate Figures 1 through 7
    generate_figure1_power_curves(shards, fig_dir, dpi=args.dpi)
    generate_figure2_interaction_detection(shards, fig_dir, dpi=args.dpi)
    generate_figure3_roc_auc(shards, fig_dir, dpi=args.dpi)
    generate_figure4_ranking_stability(shards, fig_dir, dpi=args.dpi)
    generate_figure5_oos_r2(shards, fig_dir, dpi=args.dpi)
    generate_figure6_leaky_design(shards, fig_dir, results_dir=results_dir, dpi=args.dpi)
    generate_figure7_sensitivity_heatmap(shards, fig_dir, dpi=args.dpi)

    # Generate Appendix Tables
    generate_appendix_tables(shards, tab_dir)

    print("\n" + "=" * 80)
    print("Plot and Table Generation Finished.")
    print("=" * 80)


if __name__ == "__main__":
    main()
