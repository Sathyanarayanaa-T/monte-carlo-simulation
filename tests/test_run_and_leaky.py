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
    pd.testing.assert_frame_equal(df1, df3)


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

    # DataFrames must be identical
    pd.testing.assert_frame_equal(df_jobs1, df_jobs4)


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
    assert "Pending Tasks to Run:      60,000 across 60 scenarios" in captured
    assert "Estimated Wall-Clock Time:" in captured


def test_tier_resolution():
    """Verify tier scenario resolutions match specification."""
    tier1_scenarios = get_scenarios_for_tier("tier1")
    assert len(tier1_scenarios) == 60

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
