#!/usr/bin/env python3
"""Run the full CoherenceBench benchmark suite.

Iterates all configs x providers x seeds.
Runs scripts/run_single.py as a subprocess for each combination.

Usage:
    python scripts/run_benchmark.py
    python scripts/run_benchmark.py --max-seeds 3
    python scripts/run_benchmark.py --resume
    python scripts/run_benchmark.py --configs configs/run_a_baseline.yaml configs/run_b_intervention.yaml
"""

import argparse
import logging
import subprocess
import sys
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_benchmark")

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def is_completed(config_name: str, provider: str, seed: int) -> bool:
    """Check if a particular run already has analyzed results."""
    result_file = RESULTS_DIR / config_name / provider / f"seed_{seed}" / "analyzed_results.json"
    return result_file.exists()


def run_benchmark(
    config_paths: list[Path],
    max_seeds: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
):
    total_runs = 0
    skipped = 0
    failed = 0
    completed = 0

    for config_path in config_paths:
        config = load_config(config_path)
        name = config["name"]
        seeds = config["seeds"]
        providers = config["providers"]

        if max_seeds is not None:
            seeds = seeds[:max_seeds]

        logger.info(
            "Config: %s (%d providers x %d seeds = %d runs)",
            name, len(providers), len(seeds), len(providers) * len(seeds),
        )

        for provider in providers:
            for seed in seeds:
                total_runs += 1

                if resume and is_completed(name, provider, seed):
                    logger.info("  SKIP (completed): %s / %s / seed %d", name, provider, seed)
                    skipped += 1
                    continue

                logger.info("  RUN: %s / %s / seed %d", name, provider, seed)

                cmd = [
                    sys.executable,
                    str(Path(__file__).resolve().parent / "run_single.py"),
                    "--config", str(config_path),
                    "--provider", provider,
                    "--seed", str(seed),
                ]
                if dry_run:
                    cmd.append("--dry-run")

                try:
                    result = subprocess.run(
                        cmd,
                        capture_output=True,
                        text=True,
                        timeout=3600,  # 1 hour max per run
                    )
                    if result.returncode != 0:
                        logger.error(
                            "  FAILED: %s / %s / seed %d\n%s",
                            name, provider, seed, result.stderr[-500:] if result.stderr else "no stderr",
                        )
                        failed += 1
                    else:
                        completed += 1
                except subprocess.TimeoutExpired:
                    logger.error("  TIMEOUT: %s / %s / seed %d", name, provider, seed)
                    failed += 1
                except Exception as e:
                    logger.error("  ERROR: %s / %s / seed %d: %s", name, provider, seed, e)
                    failed += 1

    logger.info(
        "Done. Total=%d Completed=%d Skipped=%d Failed=%d",
        total_runs, completed, skipped, failed,
    )


def main():
    parser = argparse.ArgumentParser(description="Run the full CoherenceBench benchmark suite")
    parser.add_argument(
        "--configs",
        nargs="*",
        help="Specific config files to run (default: all in configs/)",
    )
    parser.add_argument(
        "--max-seeds",
        type=int,
        default=None,
        help="Maximum number of seeds to run per config (default: all)",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Skip runs that already have analyzed_results.json",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Estimate tokens/cost for all runs without API calls",
    )
    args = parser.parse_args()

    if args.configs:
        config_paths = [Path(c) for c in args.configs]
    else:
        config_paths = sorted(DEFAULT_CONFIG_DIR.glob("run_*.yaml"))

    if not config_paths:
        logger.error("No config files found")
        sys.exit(1)

    logger.info("Found %d config files", len(config_paths))
    run_benchmark(
        config_paths,
        max_seeds=args.max_seeds,
        resume=args.resume,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
