"""Monte Carlo Simulation Experiment Runner (SPEC.md Section 8 & Milestone 6).

Key features:
- Loop over scenario x replication parallelized with joblib (n_jobs from config).
- Independent child seeds from SeedSequence.spawn (Rule 8 & SPEC line 163).
- One parquet shard per scenario (results/{scenario}.parquet).
- Resume from existing shards without recomputation.
- Logging of package versions and git commit hash with every shard.
- Failure log (results/failures.log) for any replication that raises an exception.
- Tiers in config (tier1: OLS family on all scenarios; tier2: reduced ML grid).
- --dry-run flag that prints total task count and estimated time from timing-pilot output.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
from pathlib import Path
import platform
import subprocess
import sys
import time
import traceback
from typing import Any

# Ensure repo root is on sys.path
repo_root = Path(__file__).resolve().parent.parent
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from joblib import Parallel, delayed
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq

from src.dgp import build_scenario, generate, get_scenario_catalog, load_config
from src.harness import run_nested_cv
from src.metrics import compute_replication_metrics


PACKAGE_LIST = [
    "numpy",
    "scipy",
    "pandas",
    "statsmodels",
    "scikit-learn",
    "shap",
    "interpret-core",
    "joblib",
    "pyarrow",
    "matplotlib",
]


def get_environment_metadata() -> dict[str, Any]:
    """Inspect environment, git commit hash, and pinned package versions."""
    # Git hash and dirty state
    try:
        git_hash = subprocess.check_output(
            ["git", "rev-parse", "HEAD"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
    except Exception:
        git_hash = "unknown"

    try:
        status_out = subprocess.check_output(
            ["git", "status", "--porcelain"], stderr=subprocess.DEVNULL
        ).decode("utf-8").strip()
        git_dirty = bool(status_out)
    except Exception:
        git_dirty = False

    # Package versions
    pkg_versions: dict[str, str] = {}
    for pkg in PACKAGE_LIST:
        try:
            pkg_versions[pkg] = importlib.metadata.version(pkg)
        except Exception:
            try:
                mod = __import__(pkg)
                pkg_versions[pkg] = getattr(mod, "__version__", "unknown")
            except Exception:
                pkg_versions[pkg] = "unknown"

    return {
        "git_hash": git_hash,
        "git_dirty": git_dirty,
        "python_version": sys.version,
        "packages": pkg_versions,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform.platform(),
    }


def load_methods_config(config_path: str | Path = "config/methods.yaml") -> dict[str, Any]:
    """Load methods configuration file."""
    p = Path(config_path)
    if not p.exists():
        p = repo_root / config_path
    return load_config(p)


def get_scenarios_for_tier(
    tier_name: str,
    methods_cfg: dict[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Resolve scenario list for a given tier."""
    if methods_cfg is None:
        methods_cfg = load_methods_config()

    tier_cfg = methods_cfg.get("tiers", {}).get(tier_name, {})
    scenarios_rule = tier_cfg.get("scenarios", "all")
    catalog = get_scenario_catalog()

    # Filter out helper aliases
    valid_scenarios = [
        sc for k, sc in catalog.items()
        if not k.startswith("base_scenario") and k != "base"
    ]

    if scenarios_rule == "all":
        return valid_scenarios
    elif scenarios_rule == "core":
        return [
            sc for sc in valid_scenarios
            if sc["name"].startswith("base_n") or sc["name"].startswith("h1_null") or sc["name"].startswith("h2_null")
        ]
    elif scenarios_rule == "reduced_ml":
        # Exactly 14 scenarios:
        # {none, inter_small, inter_med, thresh_small, thresh_med} at n=100 and n=150 (10 scenarios),
        # plus {none, inter_med} at n=50 and n=300 (4 scenarios).
        target_names = set()
        for n in [100, 150]:
            for cell in ["none", "inter_small", "inter_med", "thresh_small", "thresh_med"]:
                target_names.add(f"base_n{n}_{cell}")
        for n in [50, 300]:
            for cell in ["none", "inter_med"]:
                target_names.add(f"base_n{n}_{cell}")
        return [sc for sc in valid_scenarios if sc["name"] in target_names]
    elif isinstance(scenarios_rule, list):
        return [catalog[k] if k in catalog else build_scenario(k) for k in scenarios_rule]
    else:
        return valid_scenarios


def check_shard_completed(shard_path: Path, expected_reps: int) -> bool:
    """Check if an existing parquet shard has completed all expected replications."""
    if not shard_path.exists():
        return False
    try:
        meta = pq.read_metadata(shard_path)
        return meta.num_rows >= expected_reps
    except Exception:
        return False


def run_single_replication(
    scenario: dict[str, Any],
    rep: int,
    child_seed: np.random.SeedSequence,
    methods: list[str],
    run_stability_selection: bool = True,
    run_ml_interaction_tests: bool = True,
    b_null_refits: int = 50,
    failure_log_path: Path | None = None,
    git_hash: str = "unknown",
) -> dict[str, Any]:
    """Execute one replication with exception catching and failure logging."""
    sc_name = scenario.get("name", "unnamed")
    try:
        # Generate synthetic data
        df = generate(scenario, seed=child_seed)
        child_seed_int = int(child_seed.generate_state(1)[0])

        # Run nested CV harness
        cv_out = run_nested_cv(
            df,
            scenario_dict=scenario,
            methods=methods,
            seed=child_seed_int,
            run_stability_selection=run_stability_selection,
            run_ml_interaction_tests=run_ml_interaction_tests,
            b_null_refits=b_null_refits,
        )

        # Compute metrics
        row = compute_replication_metrics(cv_out, scenario, data=df)
        row["replication"] = rep
        row["scenario"] = sc_name
        row["status"] = "SUCCESS"
        row["git_hash"] = git_hash
        return row

    except Exception as exc:
        # Format traceback and record to failure log
        tb_str = traceback.format_exc()
        ts = datetime.now(timezone.utc).isoformat()
        log_entry = (
            f"[{ts}] FAILURE in Scenario '{sc_name}', Replication {rep}\n"
            f"Error: {type(exc).__name__}: {str(exc)}\n"
            f"{tb_str}\n"
            f"{'-' * 80}\n"
        )
        if failure_log_path is not None:
            try:
                failure_log_path.parent.mkdir(parents=True, exist_ok=True)
                with open(failure_log_path, "a") as f:
                    f.write(log_entry)
            except Exception:
                pass

        return {
            "replication": rep,
            "scenario": sc_name,
            "status": "FAILED",
            "error_type": type(exc).__name__,
            "error_msg": str(exc),
            "git_hash": git_hash,
        }


def to_serializable(val: Any) -> Any:
    """Recursively convert NumPy types to Python native types for JSON serialization."""
    if isinstance(val, np.ndarray):
        return val.tolist()
    if isinstance(val, (np.floating, float)):
        return float(val)
    if isinstance(val, (np.integer, int)):
        return int(val)
    if isinstance(val, (np.bool_, bool)):
        return bool(val)
    if isinstance(val, dict):
        return {
            str(k): to_serializable(v)
            for k, v in val.items()
            if not callable(v) and k != "loadings"
        }
    if isinstance(val, (list, tuple)):
        return [to_serializable(x) for x in val]
    return val


def save_scenario_shard(
    df: pd.DataFrame,
    scenario: dict[str, Any],
    shard_path: Path,
    env_metadata: dict[str, Any],
) -> None:
    """Save dataframe as Parquet shard with embedded metadata and sidecar JSON."""
    shard_path.parent.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pandas(df)

    clean_scenario = to_serializable(scenario)

    # Embed metadata into Parquet schema key-value table
    schema_meta = {
        b"git_hash": str(env_metadata.get("git_hash", "unknown")).encode("utf-8"),
        b"timestamp_utc": str(env_metadata.get("timestamp_utc", "")).encode("utf-8"),
        b"scenario_name": str(scenario.get("name", "")).encode("utf-8"),
        b"packages": json.dumps(env_metadata.get("packages", {})).encode("utf-8"),
        b"scenario_config": json.dumps(clean_scenario).encode("utf-8"),
    }
    existing_meta = table.schema.metadata or {}
    new_meta = {**existing_meta, **schema_meta}
    table = table.replace_schema_metadata(new_meta)

    # Write parquet shard
    pq.write_table(table, shard_path)

    # Write sidecar JSON metadata
    meta_path = shard_path.with_name(f"{shard_path.stem}_meta.json")
    meta_payload = {
        "scenario": clean_scenario,
        "num_replications": len(df),
        "status_counts": df["status"].value_counts().to_dict() if "status" in df.columns else {},
        "environment": env_metadata,
    }
    with open(meta_path, "w") as f:
        json.dump(meta_payload, f, indent=2)


def run_scenario(
    scenario: dict[str, Any],
    n_replications: int,
    methods: list[str],
    master_seed: int,
    results_dir: Path,
    n_jobs: int = 1,
    run_stability_selection: bool = True,
    run_ml_interaction_tests: bool = True,
    b_null_refits: int = 50,
    failure_log_path: Path | None = None,
    force: bool = False,
    verbose: bool = True,
) -> tuple[pd.DataFrame, bool]:
    """Execute all replications for one scenario and save shard.

    Returns:
    - (results_df, was_computed)
    """
    sc_name = scenario.get("name", "unnamed")
    shard_path = results_dir / f"{sc_name}.parquet"

    # Check resumption
    if not force and check_shard_completed(shard_path, n_replications):
        if verbose:
            print(f"[RESUME] Scenario '{sc_name}' already complete. Skipping.")
        existing_df = pd.read_parquet(shard_path)
        return existing_df, False

    env_metadata = get_environment_metadata()
    git_hash = env_metadata["git_hash"]

    # Child seeds via SeedSequence.spawn (Rule 8)
    sc_idx = int(scenario.get("scenario_idx", 0))
    scenario_seq = np.random.SeedSequence([master_seed, sc_idx])
    child_seeds = scenario_seq.spawn(n_replications)

    t0 = time.time()
    if verbose:
        print(f"[RUN] Starting scenario '{sc_name}': {n_replications} reps (n_jobs = {n_jobs})...")

    if n_jobs == 1:
        results = [
            run_single_replication(
                scenario=scenario,
                rep=rep,
                child_seed=child_seeds[rep],
                methods=methods,
                run_stability_selection=run_stability_selection,
                run_ml_interaction_tests=run_ml_interaction_tests,
                b_null_refits=b_null_refits,
                failure_log_path=failure_log_path,
                git_hash=git_hash,
            )
            for rep in range(n_replications)
        ]
    else:
        results = Parallel(n_jobs=n_jobs)(
            delayed(run_single_replication)(
                scenario=scenario,
                rep=rep,
                child_seed=child_seeds[rep],
                methods=methods,
                run_stability_selection=run_stability_selection,
                run_ml_interaction_tests=run_ml_interaction_tests,
                b_null_refits=b_null_refits,
                failure_log_path=failure_log_path,
                git_hash=git_hash,
            )
            for rep in range(n_replications)
        )

    df_results = pd.DataFrame(results).sort_values("replication").reset_index(drop=True)
    save_scenario_shard(df_results, scenario, shard_path, env_metadata)

    if verbose:
        elapsed = time.time() - t0
        n_fails = (df_results["status"] == "FAILED").sum() if "status" in df_results.columns else 0
        print(f"[DONE] Saved '{shard_path.name}' ({len(df_results)} reps, {n_fails} failed, {elapsed:.1f}s)")

    return df_results, True


def estimate_replication_seconds(
    methods: list[str],
    n: int,
    run_ml_tests: bool = False,
    run_stability: bool = True,
    b_null_refits: int = 50,
) -> float:
    """Estimate seconds per replication based on active methods and sample size n."""
    # Scale factor relative to benchmark n=150
    # OLS methods scale sublinearly to moderately linearly with n
    n_scale = (n / 150.0) ** 0.65

    # Base cost of 10 outer folds (5-fold x 2 repeats) per method at n=150
    method_costs_n150 = {
        "M0": 0.001,
        "M1": 0.060,
        "M2a": 0.050,
        "M2b": 0.700,
        "M3": 1.200 if run_stability else 0.250,
        "M4": 12.000 if run_stability else 4.000,
        "M5": 35.000 if run_stability else 12.000,
        "M6": 10.000 if run_stability else 3.500,
    }

    total_sec = sum(method_costs_n150.get(m, 0.0) for m in methods) * n_scale

    # Measurement checks cost (HTMT, Harman, Horn PA, Lindell-Whitney)
    total_sec += 0.040 * n_scale

    # Residual-resampling null interaction test cost (B=50 or B=100)
    if run_ml_tests:
        null_test_sec_n150 = 6.000 * (b_null_refits / 50.0)
        total_sec += null_test_sec_n150 * n_scale

    return total_sec


def print_dry_run_summary(
    methods_cfg: dict[str, Any],
    n_jobs: int = 6,
    results_dir: Path | str = "results",
    force: bool = False,
) -> None:
    """Print complete dry-run workload and wall-clock estimates for Tier 1 and Tier 2 at n_jobs."""
    results_path = Path(results_dir)
    tiers = methods_cfg.get("tiers", {})

    print("=" * 88)
    print("MONTE CARLO SIMULATION DRY-RUN WORKLOAD ESTIMATOR (SPEC.md Section 8)")
    print(f"Target Resource Allocation: n_jobs = {n_jobs}")
    print("=" * 88)

    grand_total_sec = 0.0

    for tier_name in ["tier1", "tier2"]:
        if tier_name not in tiers:
            continue
        tier_cfg = tiers[tier_name]
        methods = tier_cfg.get("methods", [])
        reps = int(tier_cfg.get("replications", 1000))
        run_ml = bool(tier_cfg.get("run_ml_interaction_tests", False))
        run_stab = bool(tier_cfg.get("run_stability_selection", True))
        b_refits = int(tier_cfg.get("b_null_refits", 50))

        scenarios = get_scenarios_for_tier(tier_name, methods_cfg)
        n_scenarios = len(scenarios)
        total_tasks = n_scenarios * reps

        completed_tasks = 0
        tier_seq_sec = 0.0

        for sc in scenarios:
            s_n = int(sc.get("n", 150))
            sec_rep = estimate_replication_seconds(
                methods=methods,
                n=s_n,
                run_ml_tests=run_ml,
                run_stability=run_stab,
                b_null_refits=b_refits,
            )
            s_path = results_path / f"{sc['name']}.parquet"
            if not force and check_shard_completed(s_path, reps):
                completed_tasks += reps
            else:
                tier_seq_sec += reps * sec_rep

        pending_tasks = total_tasks - completed_tasks
        tier_wall_sec = tier_seq_sec / max(1, n_jobs)
        grand_total_sec += tier_wall_sec

        print(f"\n[{tier_name.upper()}] - {tier_cfg.get('description', '')}")
        print("-" * 88)
        print(f"  Active Methods:          {', '.join(methods)}")
        print(f"  Total Scenarios:         {n_scenarios} scenarios ({reps} reps/scenario)")
        print(f"  Total Workload Tasks:    {total_tasks:,} replications")
        print(f"  Already Completed:       {completed_tasks:,} replications ({n_scenarios - (pending_tasks // reps if reps else 0)} shards)")
        print(f"  Pending Tasks to Run:    {pending_tasks:,} replications")
        print(f"  Est. Sequential Compute: {tier_seq_sec:,.1f}s ({tier_seq_sec / 3600:.2f} core-hours)")
        print(f"  Est. Wall-Clock (n_jobs={n_jobs}): {tier_wall_sec:,.1f}s ({tier_wall_sec / 3600:.2f} hours, or ~{tier_wall_sec / 60:.1f} mins)")

    print("\n" + "=" * 88)
    print(f"Combined Estimated Wall-Clock Time (Tier 1 + Tier 2 at n_jobs={n_jobs}): {grand_total_sec / 3600:.2f} hours")
    print("=" * 88)
def run_simulation(
    tier: str = "tier1",
    scenario_name: str | None = None,
    replications_override: int | None = None,
    n_jobs_override: int | None = None,
    results_dir_override: str | Path | None = None,
    dry_run: bool = False,
    force: bool = False,
    verbose: bool = True,
) -> None:
    """Main simulation driver."""
    methods_cfg = load_methods_config()
    exec_cfg = methods_cfg.get("execution", {})

    n_jobs = n_jobs_override if n_jobs_override is not None else int(exec_cfg.get("n_jobs", 4))
    master_seed = int(exec_cfg.get("master_seed", 20261013))

    if results_dir_override is not None:
        results_dir = Path(results_dir_override)
        failure_log = results_dir / "failures.log"
    else:
        results_dir = Path(exec_cfg.get("results_dir", "results"))
        failure_log = Path(exec_cfg.get("failure_log", results_dir / "failures.log"))
    results_dir.mkdir(parents=True, exist_ok=True)

    # Tier settings
    tier_cfg = methods_cfg.get("tiers", {}).get(tier, {})
    if not tier_cfg:
        raise ValueError(f"Unknown tier: '{tier}'. Available tiers: {list(methods_cfg.get('tiers', {}).keys())}")

    methods = tier_cfg.get("methods", ["M0", "M1", "M2a", "M2b", "M3"])
    reps_per_scenario = replications_override if replications_override is not None else int(tier_cfg.get("replications", 1000))
    run_ml_tests = bool(tier_cfg.get("run_ml_interaction_tests", False))
    run_stability = bool(tier_cfg.get("run_stability_selection", True))
    b_null_refits = int(tier_cfg.get("b_null_refits", 50))

    # Resolve target scenarios
    if scenario_name is not None:
        catalog = get_scenario_catalog()
        if scenario_name in catalog:
            scenarios = [catalog[scenario_name]]
        else:
            scenarios = [build_scenario({"name": scenario_name})]
    else:
        scenarios = get_scenarios_for_tier(tier, methods_cfg)

    # Timing pilot estimates
    timing_pilot_cfg = methods_cfg.get("timing_pilot", {})
    sec_per_rep_dict = timing_pilot_cfg.get("seconds_per_rep", {"tier1": 2.05, "tier2": 65.0})
    sec_per_rep = float(sec_per_rep_dict.get(tier, 2.05))

    # Task accounting
    n_scenarios = len(scenarios)
    total_tasks = n_scenarios * reps_per_scenario

    completed_tasks = 0
    pending_scenarios: list[dict[str, Any]] = []

    for sc in scenarios:
        s_path = results_dir / f"{sc['name']}.parquet"
        if not force and check_shard_completed(s_path, reps_per_scenario):
            completed_tasks += reps_per_scenario
        else:
            pending_scenarios.append(sc)

    pending_tasks = total_tasks - completed_tasks

    # Dry-run handling
    if dry_run:
        est_sequential_sec = sum(
            reps_per_scenario * estimate_replication_seconds(
                methods=methods,
                n=sc.get("n", 150),
                run_ml_tests=run_ml_tests,
                run_stability=run_stability,
                b_null_refits=b_null_refits,
            )
            for sc in pending_scenarios
        )
        est_parallel_sec = est_sequential_sec / max(1, n_jobs)

        print("=" * 88)
        print("MONTE CARLO SIMULATION DRY-RUN SUMMARY (SPEC.md Section 8)")
        print("=" * 88)
        print(f"Tier:                      {tier} ({tier_cfg.get('description', '')})")
        print(f"Active Methods:            {', '.join(methods)}")
        print(f"Master Seed:               {master_seed}")
        print(f"Parallel Workers (n_jobs): {n_jobs}")
        print(f"Results Directory:         {results_dir}")
        print(f"Failure Log:               {failure_log}")
        print(f"Total Scenarios:           {n_scenarios}")
        print(f"Replications / Scenario:   {reps_per_scenario}")
        print(f"Total Tasks:               {total_tasks:,}")
        print(f"Already Completed Shards:  {n_scenarios - len(pending_scenarios)} ({completed_tasks:,} tasks)")
        print(f"Pending Tasks to Run:      {pending_tasks:,} across {len(pending_scenarios)} scenarios")
        print("-" * 88)
        print(f"Estimated Sequential Time: {est_sequential_sec:,.1f}s ({est_sequential_sec / 3600:.2f} hours)")
        print(f"Estimated Wall-Clock Time: {est_parallel_sec:,.1f}s ({est_parallel_sec / 3600:.2f} hours at n_jobs = {n_jobs})")
        print("=" * 88)
        print_dry_run_summary(methods_cfg, n_jobs=6, results_dir=results_dir, force=force)
        return

    # Real execution
    if verbose:
        print(f"=== Starting Run: Tier '{tier}', {len(pending_scenarios)} pending scenarios, {reps_per_scenario} reps/scen, n_jobs = {n_jobs} ===")

    for sc in pending_scenarios:
        run_scenario(
            scenario=sc,
            n_replications=reps_per_scenario,
            methods=methods,
            master_seed=master_seed,
            results_dir=results_dir,
            n_jobs=n_jobs,
            run_stability_selection=run_stability,
            run_ml_interaction_tests=run_ml_tests,
            b_null_refits=b_null_refits,
            failure_log_path=failure_log,
            force=force,
            verbose=verbose,
        )

    if verbose:
        print(f"=== Run Complete for Tier '{tier}' ===")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Monte Carlo Simulation Runner (SPEC.md Section 8).")
    parser.add_argument("--tier", choices=["tier1", "tier2"], default="tier1", help="Execution tier (default: tier1)")
    parser.add_argument("--scenario", type=str, default=None, help="Specific scenario name to run")
    parser.add_argument("--max-reps", "--replications", dest="max_reps", type=int, default=None, help="Override replications per scenario")
    parser.add_argument("--n-jobs", type=int, default=None, help="Override parallel jobs from config")
    parser.add_argument("--out-dir", "--results-dir", dest="out_dir", type=str, default=None, help="Override output directory")
    parser.add_argument("--dry-run", action="store_true", help="Print task count and estimated time without running")
    parser.add_argument("--force", action="store_true", help="Recompute and overwrite existing completed shards")

    args = parser.parse_args()

    run_simulation(
        tier=args.tier,
        scenario_name=args.scenario,
        replications_override=args.max_reps,
        n_jobs_override=args.n_jobs,
        results_dir_override=args.out_dir,
        dry_run=args.dry_run,
        force=args.force,
        verbose=True,
    )
