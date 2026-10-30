#!/usr/bin/env python3
"""Compare measured end-to-end timing against config estimates (Milestone 6.6 Part E).

Reads benchmark timings produced by scripts/time_e2e.sh, compares them
with timing_pilot estimates in config/methods.yaml, and optionally updates
config/methods.yaml if discrepancies exceed 25%.
"""

from __future__ import annotations

import argparse
import difflib
import json
from pathlib import Path
import sys

import yaml


def compare_timing(
    timing_dir: Path | str = "timing_shards",
    config_path: Path | str = "config/methods.yaml",
    auto_confirm: bool = False,
) -> None:
    timing_path = Path(timing_dir)
    cfg_path = Path(config_path)

    benchmark_file = timing_path / "measured_timing.json"
    if not benchmark_file.exists():
        print(f"Error: Benchmark file '{benchmark_file}' not found.")
        print("Please run 'scripts/time_e2e.sh' first to generate empirical timing benchmarks.")
        sys.exit(1)

    with open(benchmark_file, "r") as f:
        measured = json.load(f)

    with open(cfg_path, "r") as f:
        cfg = yaml.safe_load(f)

    timing_pilot = cfg.get("timing_pilot", {})
    est_per_rep = timing_pilot.get("seconds_per_rep", {"tier1": 2.05, "tier2": 65.0})

    measured_t1 = float(measured.get("tier1_sec_per_rep", 0.0))
    measured_t2 = float(measured.get("tier2_sec_per_rep", 0.0))
    est_t1 = float(est_per_rep.get("tier1", 2.05))
    est_t2 = float(est_per_rep.get("tier2", 65.0))

    diff_t1_pct = abs(measured_t1 - est_t1) / est_t1 * 100.0 if est_t1 > 0 else 0.0
    diff_t2_pct = abs(measured_t2 - est_t2) / est_t2 * 100.0 if est_t2 > 0 else 0.0

    print("=" * 80)
    print("TIMING PILOT COMPARISON: MEASURED VS ESTIMATED (n=150)")
    print("=" * 80)
    print(f"{'Tier':<10s} | {'Estimated (s/rep)':<20s} | {'Measured (s/rep)':<18s} | {'Difference (%)':<16s} | {'Status':<10s}")
    print("-" * 80)

    status_t1 = "DIFFER > 25%" if diff_t1_pct > 25.0 else "WITHIN 25%"
    status_t2 = "DIFFER > 25%" if diff_t2_pct > 25.0 else "WITHIN 25%"

    print(f"{'tier1':<10s} | {est_t1:<20.2f} | {measured_t1:<18.2f} | {diff_t1_pct:<16.1f} | {status_t1:<10s}")
    print(f"{'tier2':<10s} | {est_t2:<20.2f} | {measured_t2:<18.2f} | {diff_t2_pct:<16.1f} | {status_t2:<10s}")
    print("=" * 80)

    needs_update = (diff_t1_pct > 25.0) or (diff_t2_pct > 25.0)

    if not needs_update:
        print("\nAll measured timings are within 25% of current estimates. No configuration update required.")
        return

    print("\n[NOTICE] One or more timing measurements differ from config estimates by > 25%.")

    # Prepare updated YAML representation
    updated_cfg = dict(cfg)
    updated_cfg["timing_pilot"]["seconds_per_rep"] = {
        "tier1": round(measured_t1, 2),
        "tier2": round(measured_t2, 2),
    }

    orig_lines = cfg_path.read_text().splitlines(keepends=True)
    new_text = yaml.dump(updated_cfg, sort_keys=False)
    new_lines = new_text.splitlines(keepends=True)

    print("\n--- Proposed Configuration Diff ---")
    diff = difflib.unified_diff(
        orig_lines, new_lines,
        fromfile=str(cfg_path),
        tofile=f"{cfg_path} (updated)",
    )
    sys.stdout.writelines(diff)
    print("-----------------------------------")

    confirmed = auto_confirm
    if not confirmed:
        try:
            resp = input("\nUpdate config/methods.yaml with measured benchmark values? [y/N]: ").strip().lower()
            confirmed = resp in ["y", "yes"]
        except EOFError:
            confirmed = False

    if confirmed:
        # Update specifically the seconds_per_rep block to preserve comments in methods.yaml
        with open(cfg_path, "w") as f:
            yaml.dump(updated_cfg, f, sort_keys=False)
        print(f"Successfully updated '{cfg_path}' with measured timings.")
    else:
        print("Update skipped by user. Configuration remains unchanged.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Compare measured simulation timing against config.")
    parser.add_argument("--timing-dir", type=str, default="timing_shards", help="Directory with benchmark timing output")
    parser.add_argument("--config", type=str, default="config/methods.yaml", help="Path to methods.yaml")
    parser.add_argument("--yes", "-y", action="store_true", help="Automatically confirm update if difference > 25%%")
    args = parser.parse_args()

    compare_timing(
        timing_dir=args.timing_dir,
        config_path=args.config,
        auto_confirm=args.yes,
    )
