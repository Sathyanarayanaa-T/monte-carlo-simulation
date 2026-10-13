"""Tests for Milestone 6: src/run.py and src/leaky.py.

Validation checks:
1. Re-running resumes without recomputing finished shards.
2. The same seed gives identical results at n_jobs = 1 and n_jobs = 4.
3. Embedded parquet metadata and sidecar JSON log package versions and git commit hash.
4. Failure logging records full traceback and status FAILED for raising replications.
5. Dry-run outputs task counts and timing estimates without executing tasks.
6. Leaky-design demonstration exhibits Delta R^2 inflation and high FPR under H1-null.
"""

from __future__ import annotations

import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pytest

from src.dgp import build_scenario
from src.leaky import run_leaky_demonstration
from src.run import (
    check_shard_completed,
    get_environment_metadata,
    get_scenarios_for_tier,
    run_scenario,
    run_simulation,
    run_single_replication,
)


@pytest.fixture
def fast_scenario():
    """Fast scenario for runner testing."""
    return build_scenario({
        "name": "test_fast_base",
        "scenario_idx": 999,
        "n": 60,
        "non_linear_form": "none",
        "psi": 0.0,
    })


def test_environment_metadata():
    """Verify git hash and package versions are logged."""
    meta = get_environment_metadata()
    assert "git_hash" in meta
    assert isinstance(meta["git_hash"], str)
    assert len(meta["git_hash"]) > 0
    assert "packages" in meta
    for pkg in ["numpy", "pandas", "scipy", "scikit-learn", "pyarrow", "joblib"]:
        assert pkg in meta["packages"]
        assert meta["packages"][pkg] != "unknown"


def test_run_resumes_without_recomputing(tmp_path: Path, fast_scenario: dict):
    """Verify re-running resumes without recomputing finished shards."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    methods = ["M0", "M1"]
    n_reps = 2
    master_seed = 20261013

    # First run: should compute and save shard
    df1, was_computed1 = run_scenario(
        scenario=fast_scenario,
        n_replications=n_reps,
        methods=methods,
        master_seed=master_seed,
        results_dir=results_dir,
        n_jobs=1,
        run_stability_selection=False,
        run_ml_interaction_tests=False,
        force=False,
        verbose=False,
    )
    assert was_computed1 is True
    assert len(df1) == n_reps

    shard_path = results_dir / f"{fast_scenario['name']}.parquet"
    meta_path = results_dir / f"{fast_scenario['name']}_meta.json"
    assert shard_path.exists()
    assert meta_path.exists()
    assert check_shard_completed(shard_path, n_reps) is True

    mtime_before = shard_path.stat().st_mtime

    # Sleep briefly to ensure stat mtime would change if rewritten
    time.sleep(0.05)

    # Second run without force: should skip computation and return existing shard
    df2, was_computed2 = run_scenario(
        scenario=fast_scenario,
        n_replications=n_reps,
        methods=methods,
        master_seed=master_seed,
        results_dir=results_dir,
        n_jobs=1,
        run_stability_selection=False,
        run_ml_interaction_tests=False,
        force=False,
        verbose=False,
    )
    assert was_computed2 is False
    assert shard_path.stat().st_mtime == mtime_before
    pd.testing.assert_frame_equal(df1, df2)

    # Third run WITH force: should recompute and overwrite
    df3, was_computed3 = run_scenario(
        scenario=fast_scenario,
        n_replications=n_reps,
        methods=methods,
        master_seed=master_seed,
        results_dir=results_dir,
        n_jobs=1,
        run_stability_selection=False,
        run_ml_interaction_tests=False,
        force=True,
        verbose=False,
    )
    assert was_computed3 is True
    timing_cols = [c for c in df1.columns if "time" in c or c in ("t_total", "runtime_seconds")]
    data_cols = [c for c in df1.columns if c not in timing_cols]
    pd.testing.assert_frame_equal(df1[data_cols], df3[data_cols])
    assert (df3["t_total"] > 0).all()


def test_seed_invariance_n_jobs_1_vs_4(tmp_path: Path, fast_scenario: dict):
    """Verify the same master seed produces identical results at n_jobs = 1 and n_jobs = 4."""
    results_dir1 = tmp_path / "results_jobs1"
    results_dir4 = tmp_path / "results_jobs4"
    results_dir1.mkdir(parents=True, exist_ok=True)
    results_dir4.mkdir(parents=True, exist_ok=True)

    methods = ["M0", "M1", "M2a"]
    n_reps = 3
    master_seed = 20261013

    # Run with n_jobs = 1
    df_jobs1, _ = run_scenario(
        scenario=fast_scenario,
        n_replications=n_reps,
        methods=methods,
        master_seed=master_seed,
        results_dir=results_dir1,
        n_jobs=1,
        run_stability_selection=False,
        run_ml_interaction_tests=False,
        force=True,
        verbose=False,
    )

    # Run with n_jobs = 4
    df_jobs4, _ = run_scenario(
        scenario=fast_scenario,
        n_replications=n_reps,
        methods=methods,
        master_seed=master_seed,
        results_dir=results_dir4,
        n_jobs=4,
        run_stability_selection=False,
        run_ml_interaction_tests=False,
        force=True,
        verbose=False,
    )

    # DataFrames must be identical for statistical outputs
    timing_cols = [c for c in df_jobs1.columns if "time" in c or c in ("t_total", "runtime_seconds")]
    data_cols = [c for c in df_jobs1.columns if c not in timing_cols]
    pd.testing.assert_frame_equal(df_jobs1[data_cols], df_jobs4[data_cols])
    assert (df_jobs4["t_total"] > 0).all()


def test_parquet_embedded_and_sidecar_metadata(tmp_path: Path, fast_scenario: dict):
    """Verify parquet schema metadata and JSON sidecar metadata."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    n_reps = 2

    run_scenario(
        scenario=fast_scenario,
        n_replications=n_reps,
        methods=["M0", "M1"],
        master_seed=20261013,
        results_dir=results_dir,
        n_jobs=1,
        run_stability_selection=False,
        run_ml_interaction_tests=False,
        force=True,
        verbose=False,
    )

    shard_path = results_dir / f"{fast_scenario['name']}.parquet"
    meta_path = results_dir / f"{fast_scenario['name']}_meta.json"

    # Verify Parquet schema metadata
    table_meta = pq.read_metadata(shard_path)
    custom_meta = table_meta.metadata
    assert b"git_hash" in custom_meta
    assert b"packages" in custom_meta
    assert b"scenario_name" in custom_meta
    assert custom_meta[b"scenario_name"].decode("utf-8") == fast_scenario["name"]

    # Verify JSON sidecar
    with open(meta_path, "r") as f:
        sidecar = json.load(f)
    assert sidecar["num_replications"] == n_reps
    assert sidecar["environment"]["git_hash"] == custom_meta[b"git_hash"].decode("utf-8")
    assert "numpy" in sidecar["environment"]["packages"]


def test_failure_logging(tmp_path: Path, fast_scenario: dict):
    """Verify failure log records errors and returns FAILED status row."""
    failure_log = tmp_path / "failures.log"
    child_seq = np.random.SeedSequence(12345)

    # Corrupt scenario with invalid weights to provoke exception
    bad_scenario = dict(fast_scenario)
    bad_scenario["linear_weights"] = "not_a_valid_dict_structure"

    res = run_single_replication(
        scenario=bad_scenario,
        rep=0,
        child_seed=child_seq,
        methods=["M0", "M1"],
        failure_log_path=failure_log,
        git_hash="testhash123",
    )

    assert res["status"] == "FAILED"
    assert "error_type" in res
    assert failure_log.exists()

    content = failure_log.read_text()
    assert "FAILURE in Scenario" in content
    assert "TypeError" in content or "KeyError" in content or "AttributeError" in content


def test_dry_run_flag(capsys):
    """Verify --dry-run prints task count and timing estimates without creating files."""
    run_simulation(
        tier="tier1",
        dry_run=True,
        verbose=True,
    )
    captured = capsys.readouterr().out
    assert "MONTE CARLO SIMULATION DRY-RUN SUMMARY" in captured
    assert "Pending Tasks to Run:      64,000 across 64 scenarios" in captured
    assert "Estimated Wall-Clock Time:" in captured


def test_tier_resolution():
    """Verify tier scenario resolutions match specification."""
    tier1_scenarios = get_scenarios_for_tier("tier1")
    assert len(tier1_scenarios) == 64

    tier2_scenarios = get_scenarios_for_tier("tier2")
    assert len(tier2_scenarios) == 14  # 14 key core scenarios at n in {100, 150}


def test_leaky_design_demonstration():
    """Verify leaky design produces massive Delta R^2 inflation and high FPR."""
    df_results, summary = run_leaky_demonstration(
        n=150,
        n_replications=30,
        master_seed=20261013,
        n_jobs=2,
        verbose=False,
    )

    assert len(df_results) == 30
    assert "delta_r2_correct" in df_results.columns
    assert "delta_r2_leaky" in df_results.columns

    # Mean Delta R2 in leaky design must be substantially larger than in correct design
    assert summary["mean_delta_r2_leaky"] > 0.10
    assert summary["mean_delta_r2_correct"] < 0.03
    assert summary["delta_r2_inflation_ratio"] > 5.0

    # False positive rate of H1 test in leaky design must be massive (near 100%)
    assert summary["fpr_leaky"] >= 0.90
    assert summary["fpr_correct"] <= 0.20


def test_shard_measurement_columns(tmp_path: Path, fast_scenario: dict):
    """Verify all 8 measurement checks are recorded per replication in shard output (Part B)."""
    results_dir = tmp_path / "results"
    results_dir.mkdir(parents=True, exist_ok=True)
    n_reps = 2

    df_out, _ = run_scenario(
        scenario=fast_scenario,
        n_replications=n_reps,
        methods=["M0", "M1"],
        master_seed=20261013,
        results_dir=results_dir,
        n_jobs=1,
        run_stability_selection=False,
        run_ml_interaction_tests=False,
        force=True,
        verbose=False,
    )

    required_meas_cols = [
        "htmt_A_tech_B_i",
        "htmt_A_mkt_B_ii",
        "htmt_C_G",
        "pa_factors_A_tech_B_i",
        "pa_factors_A_mkt_B_ii",
        "pa_factors_C_G",
        "harman_first_factor_share",
        "marker_r_M",
    ]

    for col in required_meas_cols:
        assert col in df_out.columns, f"Missing measurement column: {col}"
        assert not df_out[col].isna().any(), f"Measurement column {col} contains NaN values"


def test_seedsequence_spawn_prefix_invariance():
    """Verify spawn(200) equals the first 200 children of spawn(300) (Milestone 6.7 Part D)."""
    master_seq = np.random.SeedSequence(20261013)
    children_200 = master_seq.spawn(200)

    master_seq_again = np.random.SeedSequence(20261013)
    children_300 = master_seq_again.spawn(300)

    for i in range(200):
        assert children_200[i].state == children_300[i].state, f"Mismatch at spawn child {i}"


def test_chunked_run_resume_identical_to_uninterrupted(tmp_path: Path, fast_scenario: dict):
    """Verify stopping after first chunk and resuming gives identical results to an uninterrupted run."""
    dir_full = tmp_path / "results_full"
    dir_full.mkdir(parents=True, exist_ok=True)

    dir_staged = tmp_path / "results_staged"
    dir_staged.mkdir(parents=True, exist_ok=True)

    # Uninterrupted run: 50 reps with chunk_size 25
    df_full, _ = run_scenario(
        scenario=fast_scenario,
        n_replications=50,
        methods=["M0", "M1"],
        master_seed=424242,
        results_dir=dir_full,
        n_jobs=1,
        run_stability_selection=False,
        run_ml_interaction_tests=False,
        force=True,
        verbose=False,
        chunk_size=25,
    )

    # Staged run: first chunk only (25 reps)
    df_chunk1, _ = run_scenario(
        scenario=fast_scenario,
        n_replications=25,
        methods=["M0", "M1"],
        master_seed=424242,
        results_dir=dir_staged,
        n_jobs=1,
        run_stability_selection=False,
        run_ml_interaction_tests=False,
        force=True,
        verbose=False,
        chunk_size=25,
    )
    assert len(df_chunk1) == 25

    # Resume staged run: extend to 50 reps
    df_resumed, was_computed = run_scenario(
        scenario=fast_scenario,
        n_replications=50,
        methods=["M0", "M1"],
        master_seed=424242,
        results_dir=dir_staged,
        n_jobs=1,
        run_stability_selection=False,
        run_ml_interaction_tests=False,
        force=False,
        verbose=False,
        chunk_size=25,
    )
    assert was_computed is True
    assert len(df_resumed) == 50

    # Assert metric results match exactly across all 50 replications
    cols_to_check = ["m1_f_hit", "m1_t_hit", "m1_h2_hit", "r2_block1", "r2_block2"]
    for col in cols_to_check:
        if col in df_full.columns:
            np.testing.assert_allclose(
                df_full[col].values,
                df_resumed[col].values,
                err_msg=f"Mismatch in column {col} between uninterrupted and resumed runs",
            )


def test_extending_replications_preserves_earlier_rows(tmp_path: Path, fast_scenario: dict):
    """Verify extending from 25 to 50 replications gives identical rows for 0 to 24."""
    results_dir = tmp_path / "results_extend"
    results_dir.mkdir(parents=True, exist_ok=True)

    # Initial run: 25 reps
    df_25, _ = run_scenario(
        scenario=fast_scenario,
        n_replications=25,
        methods=["M0", "M1"],
        master_seed=99999,
        results_dir=results_dir,
        n_jobs=1,
        run_stability_selection=False,
        run_ml_interaction_tests=False,
        force=True,
        verbose=False,
        chunk_size=25,
    )

    # Extend to 50 reps
    df_50, _ = run_scenario(
        scenario=fast_scenario,
        n_replications=50,
        methods=["M0", "M1"],
        master_seed=99999,
        results_dir=results_dir,
        n_jobs=1,
        run_stability_selection=False,
        run_ml_interaction_tests=False,
        force=False,
        verbose=False,
        chunk_size=25,
    )

    cols_to_check = ["m1_f_hit", "m1_t_hit", "m1_h2_hit", "r2_block1", "r2_block2"]
    for col in cols_to_check:
        if col in df_25.columns:
            np.testing.assert_allclose(
                df_25[col].values,
                df_50.iloc[:25][col].values,
                err_msg=f"Extending replications corrupted earlier row data in {col}",
            )

    # Verify sidecar metadata JSON contains replication_ids
    meta_path = results_dir / f"{fast_scenario['name']}_meta.json"
    assert meta_path.exists()
    with open(meta_path) as f:
        meta = json.load(f)
    assert "replication_ids" in meta
    assert meta["replication_ids"] == list(range(50))


def test_shard_timing_columns(tmp_path: Path, fast_scenario: dict):
    """Verify M3-M6 timing columns and t_total are recorded in shard output (Part C)."""
    results_dir = tmp_path / "results_timing"
    results_dir.mkdir(parents=True, exist_ok=True)

    df_out, _ = run_scenario(
        scenario=fast_scenario,
        n_replications=2,
        methods=["M0", "M1", "M3"],
        master_seed=123,
        results_dir=results_dir,
        n_jobs=1,
        run_stability_selection=True,
        run_ml_interaction_tests=False,
        force=True,
        verbose=False,
        chunk_size=25,
    )

    assert "t_total" in df_out.columns
    assert "m3_time_nested_cv" in df_out.columns
    assert "m3_time_full_fit" in df_out.columns
    assert "m3_time_null_test" in df_out.columns
    assert "m3_time_bootstrap" in df_out.columns
    assert "m3_t_total" in df_out.columns
    assert (df_out["t_total"] > 0).all()
