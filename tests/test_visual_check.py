"""Validation check 7: Visual diagnostics (SPEC.md Section 9 & Section 12).

Produces:
1. Histograms of item responses (demonstrating discrete Likert 1..5 distribution and ceiling).
2. Correlation matrix of sub-construct scores.
3. True function surfaces for interaction and threshold terms.
Saved to report/diagnostic_plots.png.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import src.dgp as dgp


def test_generate_diagnostic_plots():
    """Generate diagnostic plots satisfying SPEC.md Section 9 check 7."""
    sc = dgp.build_scenario({"n": 5_000})
    df = dgp.generate(sc, seed=42)

    fig = plt.figure(figsize=(16, 12))

    # 1. Histograms of selected item responses
    ax1 = fig.add_subplot(2, 2, 1)
    items_to_plot = ["A_tech_1", "A_mkt_1", "B_i_1", "C_1"]
    x_positions = np.arange(1, 6)
    width = 0.2
    for idx, col in enumerate(items_to_plot):
        counts = [np.mean(df[col] == k) for k in range(1, 6)]
        ax1.bar(x_positions + (idx - 1.5) * width, counts, width=width, label=col, alpha=0.85)
    ax1.set_xlabel("Likert Category")
    ax1.set_ylabel("Proportion")
    ax1.set_title("Item Response Distributions (Check 4: Likert & Mild Ceiling)")
    ax1.set_xticks(x_positions)
    ax1.legend()
    ax1.grid(axis="y", linestyle="--", alpha=0.5)

    # 2. Correlation matrix of sub-construct scores
    ax2 = fig.add_subplot(2, 2, 2)
    score_cols = [
        "score_A_tech", "score_A_mkt", "score_B_i", "score_B_ii",
        "score_B_iii", "score_C", "score_G", "score_M"
    ]
    labels = ["A-i", "A-ii", "B-i", "B-ii", "B-iii", "C", "G", "M"]
    corr_mat = df[score_cols].corr().values
    im = ax2.imshow(corr_mat, cmap="coolwarm", vmin=-0.2, vmax=1.0)
    fig.colorbar(im, ax=ax2, shrink=0.8)
    ax2.set_xticks(np.arange(len(labels)))
    ax2.set_yticks(np.arange(len(labels)))
    ax2.set_xticklabels(labels)
    ax2.set_yticklabels(labels)
    ax2.set_title("Correlation Matrix of Scored Sub-constructs")
    for i in range(len(labels)):
        for j in range(len(labels)):
            ax2.text(j, i, f"{corr_mat[i, j]:.2f}", ha="center", va="center", color="black", fontsize=8)

    # 3. True function surface: Interaction term (A_tech * B_i)
    ax3 = fig.add_subplot(2, 2, 3, projection="3d")
    a_grid = np.linspace(-2.5, 2.5, 40)
    b_grid = np.linspace(-2.5, 2.5, 40)
    A_mesh, B_mesh = np.meshgrid(a_grid, b_grid)
    # Interaction: A_tech * B_i
    Z_int = A_mesh * B_mesh
    surf1 = ax3.plot_surface(A_mesh, B_mesh, Z_int, cmap="viridis", alpha=0.85)
    ax3.set_xlabel("A_tech")
    ax3.set_ylabel("B_i")
    ax3.set_zlabel("Interaction Signal")
    ax3.set_title("True Function Surface: Interaction (A_tech × B_i)")

    # 4. True function surface: Threshold term (B_iii * 1[A_tech < -0.5])
    ax3_thr = fig.add_subplot(2, 2, 4, projection="3d")
    a_grid_t = np.linspace(-2.5, 2.5, 40)
    b_grid_t = np.linspace(-2.5, 2.5, 40)
    A_mesh_t, B_mesh_t = np.meshgrid(a_grid_t, b_grid_t)
    # Threshold: B_iii * 1[A_tech < -0.5]
    Z_thr = B_mesh_t * (A_mesh_t < -0.5).astype(float)
    surf2 = ax3_thr.plot_surface(A_mesh_t, B_mesh_t, Z_thr, cmap="plasma", alpha=0.85)
    ax3_thr.set_xlabel("A_tech")
    ax3_thr.set_ylabel("B_iii")
    ax3_thr.set_zlabel("Threshold Signal")
    ax3_thr.set_title("True Function Surface: Threshold (B_iii × 1[A_tech < -0.5])")

    plt.tight_layout()
    out_dir = Path("report")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "diagnostic_plots.png"
    fig.savefig(out_path, dpi=150)
    plt.close(fig)

    assert out_path.exists()
    assert out_path.stat().st_size > 0
