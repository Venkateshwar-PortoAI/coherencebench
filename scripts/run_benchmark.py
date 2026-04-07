#!/usr/bin/env python3
from __future__ import annotations

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
import selectors
import subprocess
import sys
import time
from pathlib import Path

import yaml

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_benchmark")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("anthropic._base_client").setLevel(logging.WARNING)

DEFAULT_CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"
RESULTS_DIR = Path(__file__).resolve().parent.parent / "results"
RATE_LIMIT_COOLDOWN_SECONDS = 60
MAX_RATE_LIMIT_RETRIES = 2


def load_config(config_path: Path) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def is_completed(config_name: str, provider: str, seed: int) -> bool:
    """Check if a particular run already has analyzed results."""
    result_file = RESULTS_DIR / config_name / provider / f"seed_{seed}" / "analyzed_results.json"
    return result_file.exists()


def _result_file(config_name: str, provider: str, seed: int) -> Path:
    return RESULTS_DIR / config_name / provider / f"seed_{seed}" / "analyzed_results.json"


def _load_run_summary(config_name: str, provider: str, seed: int) -> dict | None:
    result_file = _result_file(config_name, provider, seed)
    if not result_file.exists():
        return None
    import json

    with open(result_file) as f:
        return json.load(f).get("summary", {})


def _format_run_summary(summary: dict | None) -> str:
    if not summary:
        return "summary unavailable"
    return (
        f"mean FC={summary['mean_fc']:.3f} "
        f"DA={summary['mean_da']:.3f} "
        f"ADR={summary['mean_adr']:.3f} "
        f"FI={summary['mean_fi']:.3f} | "
        f"FC delta={summary['fc_drop_q1_to_q4']:.3f} | "
        f"directional={summary['directional_validation']['verdict']} | "
        f"trunc={summary['context_truncations']}"
    )


def _looks_rate_limited(stderr: str) -> bool:
    haystack = (stderr or "").lower()
    return (
        "429" in haystack
        or "rate limit" in haystack
        or "ratelimit" in haystack
        or "resourceexhausted" in haystack
    )


def _run_with_live_output(cmd: list[str], timeout_seconds: int) -> tuple[int, str]:
    """Stream child output while also retaining it for failure inspection."""
    process = subprocess.Popen(
        cmd,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1,
    )
    captured: list[str] = []
    deadline = time.monotonic() + timeout_seconds
    selector = selectors.DefaultSelector()
    assert process.stdout is not None
    selector.register(process.stdout, selectors.EVENT_READ)

    try:
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                process.kill()
                tail, _ = process.communicate()
                if tail:
                    sys.stdout.write(tail)
                    sys.stdout.flush()
                    captured.append(tail)
                raise subprocess.TimeoutExpired(cmd, timeout_seconds, output="".join(captured))

            events = selector.select(timeout=min(0.5, remaining))
            for key, _ in events:
                line = key.fileobj.readline()
                if not line:
                    continue
                sys.stdout.write(line)
                sys.stdout.flush()
                captured.append(line)

            if process.poll() is not None:
                remainder = process.stdout.read()
                if remainder:
                    sys.stdout.write(remainder)
                    sys.stdout.flush()
                    captured.append(remainder)
                break
    except KeyboardInterrupt:
        process.terminate()
        try:
            process.wait(timeout=5)
        except subprocess.TimeoutExpired:
            process.kill()
            process.wait(timeout=5)
        raise
    finally:
        selector.unregister(process.stdout)
        selector.close()
        process.stdout.close()

    return process.wait(), "".join(captured)


def run_benchmark(
    config_paths: list[Path],
    max_seeds: int | None = None,
    resume: bool = False,
    dry_run: bool = False,
    model_name: str | None = None,
    providers_filter: list[str] | None = None,
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

        if providers_filter:
            providers = [provider for provider in providers if provider in providers_filter]

        if model_name and len(providers) != 1:
            raise ValueError(
                f"--model {model_name!r} is only valid when exactly one provider is selected for "
                f"{name}. Use --providers <provider> to target a single provider."
            )

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
                if model_name:
                    cmd.extend(["--model", model_name])

                attempt = 0
                while attempt < MAX_RATE_LIMIT_RETRIES:
                    attempt += 1
                    started_at = time.monotonic()
                    try:
                        returncode, combined_output = _run_with_live_output(
                            cmd,
                            timeout_seconds=3600,
                        )
                    except subprocess.TimeoutExpired:
                        logger.error("  TIMEOUT: %s / %s / seed %d", name, provider, seed)
                        failed += 1
                        break
                    except KeyboardInterrupt:
                        logger.warning("Interrupted by user. Stopping benchmark.")
                        raise SystemExit(130)
                    except Exception as e:
                        logger.error("  ERROR: %s / %s / seed %d: %s", name, provider, seed, e)
                        failed += 1
                        break

                    duration_seconds = time.monotonic() - started_at

                    if returncode == 0:
                        completed += 1
                        if dry_run:
                            logger.info(
                                "  DRY RUN complete: %s / %s / seed %d | elapsed=%.1fs",
                                name,
                                provider,
                                seed,
                                duration_seconds,
                            )
                        else:
                            summary = _load_run_summary(name, provider, seed)
                            logger.info(
                                "  COMPLETE: %s / %s / seed %d | %s | elapsed=%.1fs",
                                name,
                                provider,
                                seed,
                                _format_run_summary(summary),
                                duration_seconds,
                            )
                        break

                    output_tail = combined_output[-1000:] if combined_output else "no output"
                    if _looks_rate_limited(combined_output) and attempt < MAX_RATE_LIMIT_RETRIES:
                        logger.warning(
                            "  RATE LIMITED: %s / %s / seed %d | cooling down %ds before retry %d/%d",
                            name,
                            provider,
                            seed,
                            RATE_LIMIT_COOLDOWN_SECONDS,
                            attempt + 1,
                            MAX_RATE_LIMIT_RETRIES,
                        )
                        time.sleep(RATE_LIMIT_COOLDOWN_SECONDS)
                        continue

                    logger.error(
                        "  FAILED: %s / %s / seed %d | exit=%d | elapsed=%.1fs\n%s",
                        name,
                        provider,
                        seed,
                        returncode,
                        duration_seconds,
                        output_tail,
                    )
                    failed += 1
                    break

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
    parser.add_argument(
        "--model",
        default=None,
        help="Optional provider model override applied to each run, e.g. gpt-5.4",
    )
    parser.add_argument(
        "--providers",
        nargs="*",
        default=None,
        help="Optional subset of providers to run, e.g. --providers gpt4o",
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
        model_name=args.model,
        providers_filter=args.providers,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user.")
        raise SystemExit(130)
