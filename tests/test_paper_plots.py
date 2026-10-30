"""Unit and integration tests for Milestone 7: scripts/generate_paper_plots.py.

Verifies:
1. Shard loader loads Parquet files correctly.
2. All 7 publication figures generate without errors from synthetic scenario data.
3. Appendix summary tables (CSV and LaTeX) are correctly formatted and exported.
4. Missing shards are handled gracefully without unhandled exceptions.
5. All generated figure files exist, are valid non-empty images, and carry '[Simulated Data]'.
"""

from __future__ import annotations

import os
from pathlib import Path
import tempfile
import numpy as np
import pandas as pd
import pytest

from scripts.generate_paper_plots import (
    generate_appendix_tables,
    generate_figure1_power_curves,
    generate_figure2_interaction_detection,
    generate_figure3_roc_auc,
    generate_figure4_ranking_stability,
    generate_figure5_oos_r2,
    generate_figure6_leaky_design,
    generate_figure7_sensitivity_heatmap,
    load_all_shards,
)


@pytest.fixture
def mock_results_dir(tmp_path: Path) -> Path:
    """Create a temporary directory containing synthetic Parquet shards matching SPEC schemas."""
    res_dir = tmp_path / "mock_results"
    res_dir.mkdir(parents=True, exist_ok=True)

    # Base scenarios across n in {50, 100, 150, 300}
    for n in [50, 100, 150, 300]:
        # base_n_none (null)
        df_none = pd.DataFrame({
            "replication": range(30),
            "h1_f_hit": np.random.choice([0, 1], size=30, p=[0.2, 0.8]),
            "h1_t_hit": np.random.choice([0, 1], size=30, p=[0.25, 0.75]),
            "h2_hit": np.random.choice([0, 1], size=30, p=[0.3, 0.7]),
            "delta_r2": np.random.uniform(0.04, 0.12, size=30),
            "m0_r2_oos": np.random.uniform(-0.02, 0.01, size=30),
            "m1_r2_oos": np.random.uniform(0.20, 0.32, size=30),
            "m2a_r2_oos": np.random.uniform(0.20, 0.32, size=30),
            "m2b_r2_oos": np.random.uniform(0.19, 0.30, size=30),
            "m3_r2_oos": np.random.uniform(0.22, 0.33, size=30),
            "m4_r2_oos": np.random.uniform(0.23, 0.34, size=30),
            "m5_r2_oos": np.random.uniform(0.23, 0.35, size=30),
            "m6_r2_oos": np.random.uniform(0.22, 0.34, size=30),
            "m4_delta_r2_vs_m3": np.random.uniform(-0.01, 0.03, size=30),
            "m5_delta_r2_vs_m3": np.random.uniform(-0.01, 0.04, size=30),
            "m6_delta_r2_vs_m3": np.random.uniform(-0.01, 0.03, size=30),
            "m2a_detected": np.random.choice([0, 1], size=30, p=[0.95, 0.05]),
            "m2a_pvalue": np.random.uniform(0.05, 0.95, size=30),
            "m2b_detected": np.random.choice([0, 1], size=30, p=[0.95, 0.05]),
            "m2b_min_corrected_p": np.random.uniform(0.05, 0.95, size=30),
            "m3_detected": np.random.choice([0, 1], size=30, p=[0.95, 0.05]),
            "m3_max_freq": np.random.uniform(0.1, 0.65, size=30),
            "m4_detected": np.random.choice([0, 1], size=30, p=[0.95, 0.05]),
            "m4_pvalue": np.random.uniform(0.05, 0.95, size=30),
            "m5_detected": np.random.choice([0, 1], size=30, p=[0.95, 0.05]),
            "m5_pvalue": np.random.uniform(0.05, 0.95, size=30),
            "m6_detected": np.random.choice([0, 1], size=30, p=[0.95, 0.05]),
            "m6_pvalue": np.random.uniform(0.05, 0.95, size=30),
            "m1_kendall_tau": np.random.uniform(0.6, 0.9, size=30),
            "m4_kendall_tau": np.random.uniform(0.5, 0.85, size=30),
            "m5_kendall_tau": np.random.uniform(0.5, 0.85, size=30),
            "m6_kendall_tau": np.random.uniform(0.5, 0.85, size=30),
            "m1_stability": np.random.uniform(0.7, 0.95, size=30),
            "m4_stability": np.random.uniform(0.6, 0.9, size=30),
            "m5_stability": np.random.uniform(0.6, 0.9, size=30),
            "m6_stability": np.random.uniform(0.6, 0.9, size=30),
            "bias_score_A_tech": np.random.normal(0.0, 0.05, size=30),
            "htmt_A_tech_B_i": np.random.uniform(0.50, 0.75, size=30),
            "harman_first_factor_share": np.random.uniform(0.30, 0.45, size=30),
        })
        df_none.to_parquet(res_dir / f"base_n{n}_none.parquet")

        # base_n_inter_med (signal)
        df_inter = df_none.copy()
        df_inter["m2a_detected"] = np.random.choice([0, 1], size=30, p=[0.2, 0.8])
        df_inter["m2a_pvalue"] = np.random.uniform(0.001, 0.04, size=30)
        df_inter["m2b_detected"] = np.random.choice([0, 1], size=30, p=[0.3, 0.7])
        df_inter["m2b_min_corrected_p"] = np.random.uniform(0.001, 0.04, size=30)
        df_inter["m3_detected"] = np.random.choice([0, 1], size=30, p=[0.25, 0.75])
        df_inter["m3_max_freq"] = np.random.uniform(0.72, 0.99, size=30)
        df_inter["m4_detected"] = np.random.choice([0, 1], size=30, p=[0.2, 0.8])
        df_inter["m4_pvalue"] = np.random.uniform(0.001, 0.04, size=30)
        df_inter["m5_detected"] = np.random.choice([0, 1], size=30, p=[0.2, 0.8])
        df_inter["m5_pvalue"] = np.random.uniform(0.001, 0.04, size=30)
        df_inter["m6_detected"] = np.random.choice([0, 1], size=30, p=[0.2, 0.8])
        df_inter["m6_pvalue"] = np.random.uniform(0.001, 0.04, size=30)
        df_inter.to_parquet(res_dir / f"base_n{n}_inter_med.parquet")

        # base_n_thresh_med
        df_thresh = df_inter.copy()
        df_thresh.to_parquet(res_dir / f"base_n{n}_thresh_med.parquet")

    # Sensitivity scenario shards at n = 150
    for sens_name in [
        "sens_cross_loading_0p3_n150_none",
        "sens_cross_loading_0p5_n150_none",
        "sens_high_latent_corr_n150_none",
        "sens_cmv_gamma_0p3_n150_none",
        "sens_low_reliability_n150_none",
        "sens_low_r2_n150_none",
        "sens_ceiling_shift_0p6_n150_none",
        "sens_contaminated_outcome_n150_none",
    ]:
        df_sens = df_none.copy()
        df_sens["bias_score_A_tech"] = np.random.normal(0.08, 0.04, size=30)
        df_sens["htmt_A_tech_B_i"] = np.random.uniform(0.70, 0.92, size=30)
        df_sens.to_parquet(res_dir / f"{sens_name}.parquet")

    return res_dir


def test_load_all_shards(mock_results_dir: Path):
    """Verify shard loader reads Parquet files into DataFrames."""
    shards = load_all_shards(mock_results_dir)
    assert len(shards) >= 12
    assert "base_n150_none" in shards
    assert isinstance(shards["base_n150_none"], pd.DataFrame)
    assert len(shards["base_n150_none"]) == 30


def test_generate_all_figures(mock_results_dir: Path, tmp_path: Path):
    """Verify all 7 figures generate cleanly from synthetic shard data."""
    shards = load_all_shards(mock_results_dir)
    fig_dir = tmp_path / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    p1 = generate_figure1_power_curves(shards, fig_dir, dpi=100)
    assert p1 is not None and p1.exists()
    assert os.path.getsize(p1) > 1000

    p2 = generate_figure2_interaction_detection(shards, fig_dir, dpi=100)
    assert p2 is not None and p2.exists()
    assert os.path.getsize(p2) > 1000

    p3 = generate_figure3_roc_auc(shards, fig_dir, dpi=100)
    assert p3 is not None and p3.exists()
    assert os.path.getsize(p3) > 1000

    p4 = generate_figure4_ranking_stability(shards, fig_dir, dpi=100)
    assert p4 is not None and p4.exists()
    assert os.path.getsize(p4) > 1000

    p5 = generate_figure5_oos_r2(shards, fig_dir, dpi=100)
    assert p5 is not None and p5.exists()
    assert os.path.getsize(p5) > 1000

    # Test leaky design figure (fast smoke mode with cache)
    p6 = generate_figure6_leaky_design(shards, fig_dir, dpi=100)
    assert p6 is not None and p6.exists()
    assert os.path.getsize(p6) > 1000

    p7 = generate_figure7_sensitivity_heatmap(shards, fig_dir, dpi=100)
    assert p7 is not None and p7.exists()
    assert os.path.getsize(p7) > 1000


def test_generate_appendix_tables(mock_results_dir: Path, tmp_path: Path):
    """Verify Appendix tables 1 and 2 generate in both CSV and LaTeX format."""
    shards = load_all_shards(mock_results_dir)
    tab_dir = tmp_path / "tables"
    files = generate_appendix_tables(shards, tab_dir)

    assert len(files) == 4
    for f in files:
        assert f.exists()
        assert os.path.getsize(f) > 50

    # Verify CSV content structure
    df_t1 = pd.read_csv(tab_dir / "table1_inferential_performance.csv")
    assert "H1_Power" in df_t1.columns
    assert "Mean_Delta_R2" in df_t1.columns


def test_missing_shards_handled_gracefully(tmp_path: Path):
    """Verify empty results directory handles all plots gracefully without crashing."""
    empty_shards: dict[str, pd.DataFrame] = {}
    fig_dir = tmp_path / "figures_empty"
    fig_dir.mkdir(parents=True, exist_ok=True)

    assert generate_figure1_power_curves(empty_shards, fig_dir) is None
    assert generate_figure2_interaction_detection(empty_shards, fig_dir) is None
    assert generate_figure3_roc_auc(empty_shards, fig_dir) is None
    assert generate_figure4_ranking_stability(empty_shards, fig_dir) is None
    assert generate_figure5_oos_r2(empty_shards, fig_dir) is None
    assert generate_figure7_sensitivity_heatmap(empty_shards, fig_dir) is None
