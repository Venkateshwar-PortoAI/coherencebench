#!/usr/bin/env python3
from __future__ import annotations

"""Run a single benchmark config/provider/seed combination.

Usage:
    python scripts/run_single.py --config configs/run_a_baseline.yaml --provider claude --seed 42
    python scripts/run_single.py --config configs/run_a_baseline.yaml --provider claude --seed 42 --dry-run
"""

import argparse
import json
import logging
import sys
import time
from collections import Counter
from pathlib import Path

import yaml
from dotenv import load_dotenv

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
load_dotenv()

from src.providers import get_provider
from src.runner import BenchmarkRunner
from src.analyzer import ResponseAnalyzer
from src.scenarios import get_scenario

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_single")
logging.getLogger("httpx").setLevel(logging.WARNING)
logging.getLogger("httpcore").setLevel(logging.WARNING)
logging.getLogger("anthropic").setLevel(logging.WARNING)
logging.getLogger("anthropic._base_client").setLevel(logging.WARNING)

# Rough cost estimates per 1K tokens (input + output blended)
COST_PER_1K_TOKENS = {
    "claude": 0.006,    # Sonnet blended
    "gpt4o": 0.005,     # GPT-4o blended
    "gemini": 0.003,    # Gemini 1.5 Pro blended
    "llama": 0.001,     # Together Llama blended
}


def load_config(config_path: str) -> dict:
    with open(config_path) as f:
        return yaml.safe_load(f)


def estimate_cost(config: dict, provider_name: str) -> dict:
    """Estimate token usage and cost for a dry run."""
    num_ticks = config["num_ticks"]

    # Estimate: system prompt ~400 tokens, each tick prompt ~200 tokens,
    # each response ~300 tokens, conversation accumulates.
    sys_tokens = 400
    tick_prompt_tokens = 200
    response_tokens = 300

    total_input = 0
    total_output = 0
    context_len = sys_tokens

    reset_interval = config.get("context_reset_interval")

    for i in range(num_ticks):
        if reset_interval and i > 0 and i % reset_interval == 0:
            context_len = sys_tokens + 200  # state summary

        total_input += context_len + tick_prompt_tokens
        total_output += response_tokens
        context_len += tick_prompt_tokens + response_tokens

    cost_per_1k = COST_PER_1K_TOKENS.get(provider_name, 0.005)
    total_tokens = total_input + total_output
    estimated_cost = (total_tokens / 1000) * cost_per_1k

    return {
        "num_ticks": num_ticks,
        "estimated_input_tokens": total_input,
        "estimated_output_tokens": total_output,
        "estimated_total_tokens": total_tokens,
        "estimated_cost_usd": round(estimated_cost, 2),
        "provider": provider_name,
    }


def _format_elapsed(seconds: float) -> str:
    total_seconds = max(int(seconds), 0)
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _format_factor_list(values: list[str]) -> str:
    return ",".join(values) if values else "none"


def _format_eta(elapsed_seconds: float, completed_ticks: int, total_ticks: int) -> str:
    if completed_ticks <= 0 or total_ticks <= completed_ticks:
        return "--:--"
    avg_seconds = elapsed_seconds / completed_ticks
    remaining_seconds = avg_seconds * (total_ticks - completed_ticks)
    return _format_elapsed(remaining_seconds)


def _verdict_label(analysis, ground_truth: dict) -> str:
    if analysis.decision_accuracy >= 1.0:
        return "hit"
    acceptable = ground_truth.get("acceptable_actions") or []
    acceptable_norm = {action.strip().lower() for action in acceptable}
    if analysis.action.strip().lower() in acceptable_norm:
        return "acceptable"
    return "miss"


def _top_counter_items(counter: Counter, limit: int = 3) -> list[dict]:
    return [
        {"name": name, "count": count}
        for name, count in counter.most_common(limit)
    ]


def _worst_window(analyses: list, window_size: int = 10) -> dict:
    if not analyses:
        return {"start_tick": 0, "end_tick": 0, "mean_da": 0.0}

    actual_window = min(window_size, len(analyses))
    worst = None
    for start in range(0, len(analyses) - actual_window + 1):
        window = analyses[start:start + actual_window]
        mean_da = sum(a.decision_accuracy for a in window) / len(window)
        candidate = {
            "start_tick": window[0].tick_number,
            "end_tick": window[-1].tick_number,
            "mean_da": round(mean_da, 4),
        }
        if worst is None or candidate["mean_da"] < worst["mean_da"]:
            worst = candidate
    return worst or {"start_tick": 0, "end_tick": 0, "mean_da": 0.0}


def _build_failure_cases(raw_results: list[dict], analyses: list) -> list[dict]:
    failures = []
    for result, analysis in zip(raw_results, analyses):
        ground_truth = result["ground_truth"]
        anomalies = ground_truth.get("anomalous_factors", [])
        missed_factors = [
            factor for factor in anomalies
            if factor not in analysis.factors_substantive
        ]
        if (
            analysis.decision_accuracy >= 1.0
            and not missed_factors
            and analysis.factor_coverage >= 1.0
        ):
            continue

        failures.append(
            {
                "tick": result["tick_number"],
                "anomalies": anomalies,
                "relevant_factors": ground_truth.get("relevant_factors", []),
                "expected_action": ground_truth.get("correct_action"),
                "acceptable_actions": ground_truth.get("acceptable_actions", []),
                "chosen_action": analysis.action,
                "verdict": _verdict_label(analysis, ground_truth),
                "missed_factors": missed_factors,
                "factor_coverage": analysis.factor_coverage,
                "decision_accuracy": analysis.decision_accuracy,
                "anomaly_detection_rate": analysis.anomaly_detection_rate,
                "fixation_index": analysis.fixation_index,
                "context_truncated": result["context_truncated"],
            }
        )
    return failures


def _make_progress_callback(
    scenario_name: str,
    total_ticks: int,
    progress_every: int,
):
    analyzer = ResponseAnalyzer(scenario=get_scenario(scenario_name))
    started_at = time.monotonic()
    stats = {
        "count": 0,
        "turn_seconds_total": 0.0,
        "sum_fc": 0.0,
        "sum_da": 0.0,
        "sum_adr": 0.0,
        "sum_fi": 0.0,
        "context_truncations": 0,
        "context_resets": 0,
        "interventions": 0,
        "analyses": [],
        "recent": [],
        "miss_counter": Counter(),
        "wrong_action_counter": Counter(),
    }

    def on_progress(event: dict) -> None:
        kind = event.get("event")

        if kind == "run_started":
            logger.info(
                "Plan | scenario=%s ticks=%d results=%s",
                scenario_name,
                event["num_ticks"],
                event["results_file"],
            )
            return

        if kind == "run_completed":
            elapsed = _format_elapsed(time.monotonic() - started_at)
            logger.info("Run complete | ticks=%d | elapsed=%s", event["num_ticks"], elapsed)
            return

        if kind == "context_reset":
            stats["context_resets"] += 1
            logger.info(
                "Cycle %03d/%03d | context reset | state restored from tick %d",
                event["cycle_index"],
                event["total_ticks"],
                event["restored_from_tick"],
            )
            return

        if kind == "intervention":
            stats["interventions"] += 1
            logger.info(
                "Cycle %03d/%03d | intervention injected",
                event["cycle_index"],
                event["total_ticks"],
            )
            return

        if kind != "tick_completed":
            return

        analysis = analyzer.analyze_tick(
            tick_number=event["tick_number"],
            response=event["response"],
            ground_truth=event["ground_truth"],
        )
        stats["analyses"].append(analysis)
        stats["count"] += 1
        stats["turn_seconds_total"] += event["duration_seconds"]
        stats["sum_fc"] += analysis.factor_coverage
        stats["sum_da"] += analysis.decision_accuracy
        stats["sum_adr"] += analysis.anomaly_detection_rate
        stats["sum_fi"] += analysis.fixation_index
        stats["recent"].append(analysis)
        if len(stats["recent"]) > progress_every:
            stats["recent"].pop(0)
        if event["context_truncated"]:
            stats["context_truncations"] += 1

        tick_num = event["tick_number"]
        elapsed_seconds = time.monotonic() - started_at
        elapsed = _format_elapsed(elapsed_seconds)
        eta = _format_eta(elapsed_seconds, tick_num, total_ticks)
        anomalies = event["ground_truth"].get("anomalous_factors", [])
        missed = [
            factor for factor in anomalies
            if factor not in analysis.factors_substantive
        ]
        for factor in missed:
            stats["miss_counter"][factor] += 1
        if analysis.decision_accuracy < 1.0:
            stats["wrong_action_counter"][analysis.action] += 1
        flags = []
        if event["context_reset"]:
            flags.append("reset")
        if event["intervention"]:
            flags.append("intervention")
        if event["context_truncated"]:
            flags.append("truncated")
        verdict = _verdict_label(analysis, event["ground_truth"])
        logger.info(
            "Tick %03d/%03d | anomaly=%s | chose=%s | expected=%s | %s | turn %.1fs | eta %s%s",
            event["cycle_index"],
            total_ticks,
            _format_factor_list(anomalies),
            analysis.action,
            event["ground_truth"].get("correct_action", "unknown"),
            verdict,
            event["duration_seconds"],
            eta,
            f" | flags={_format_factor_list(flags)}" if flags else "",
        )

        should_log = tick_num == total_ticks or tick_num % progress_every == 0
        if not should_log:
            return

        count = stats["count"]
        recent = stats["recent"]
        logger.info(
            "Checkpoint %03d/%03d | elapsed=%s eta=%s | run DA=%.3f FC=%.3f ADR=%.3f FI=%.3f | "
            "last%d DA=%.3f FC=%.3f ADR=%.3f FI=%.3f | avg turn=%.1fs | top misses=%s | wrong actions=%s | trunc=%d resets=%d interventions=%d",
            tick_num,
            total_ticks,
            elapsed,
            eta,
            stats["sum_da"] / count,
            stats["sum_fc"] / count,
            stats["sum_adr"] / count,
            stats["sum_fi"] / count,
            len(recent),
            sum(a.decision_accuracy for a in recent) / len(recent),
            sum(a.factor_coverage for a in recent) / len(recent),
            sum(a.anomaly_detection_rate for a in recent) / len(recent),
            sum(a.fixation_index for a in recent) / len(recent),
            stats["turn_seconds_total"] / count,
            _format_factor_list([name for name, _ in stats["miss_counter"].most_common(2)]),
            _format_factor_list([name for name, _ in stats["wrong_action_counter"].most_common(2)]),
            stats["context_truncations"],
            stats["context_resets"],
            stats["interventions"],
        )

    return on_progress, stats


def run(
    config_path: str,
    provider_name: str,
    seed: int,
    dry_run: bool = False,
    scenario_name: str | None = None,
    progress_every: int = 10,
    model_name: str | None = None,
):
    config = load_config(config_path)

    if dry_run:
        est = estimate_cost(config, provider_name)
        logger.info("DRY RUN - Token & cost estimates:")
        for k, v in est.items():
            logger.info("  %s: %s", k, f"${v}" if "cost" in k else v)
        print(json.dumps(est, indent=2))
        return

    logger.info(
        "Starting: config=%s provider=%s seed=%d",
        config["name"], provider_name, seed,
    )

    # Scenario: CLI flag > config file > default
    scenario = scenario_name or config.get("scenario", "power_grid")
    progress_interval = max(progress_every, 1)

    provider_kwargs = {}
    if model_name:
        provider_kwargs["model"] = model_name
    provider = get_provider(provider_name, **provider_kwargs)
    output_dir = Path("results") / config["name"] / provider_name / f"seed_{seed}"
    progress_callback, progress_stats = _make_progress_callback(
        scenario_name=scenario,
        total_ticks=config["num_ticks"],
        progress_every=progress_interval,
    )

    runner = BenchmarkRunner(
        provider=provider,
        num_ticks=config["num_ticks"],
        seed=seed,
        output_dir=output_dir,
        context_reset_interval=config.get("context_reset_interval"),
        intervention_ticks=config.get("intervention_ticks") or [],
        force_checklist=config.get("force_checklist", False),
        scenario=scenario,
        progress_callback=progress_callback,
    )

    logger.info("Running %d ticks...", config["num_ticks"])
    raw_results = runner.run()

    # Analyze
    analyses = progress_stats["analyses"]
    if len(analyses) != len(raw_results):
        logger.info("Analyzing responses...")
        analyzer = ResponseAnalyzer(scenario=runner.scenario)
        analyses = analyzer.analyze_run(raw_results)
    else:
        analyzer = ResponseAnalyzer(scenario=runner.scenario)

    # Compute summary metrics
    fc_values = [a.factor_coverage for a in analyses]
    da_values = [a.decision_accuracy for a in analyses]
    adr_values = [a.anomaly_detection_rate for a in analyses]
    fi_values = [a.fixation_index for a in analyses]
    da_at_40 = sum(da_values[:40]) / max(1, len(da_values[:40]))
    da_last = sum(da_values[-40:]) / max(1, len(da_values[-40:]))
    dfg = da_at_40 - da_last

    # Intervention recovery
    ir_values = {}
    for it in config.get("intervention_ticks") or []:
        ir = analyzer.compute_intervention_recovery(analyses, it)
        ir_values[str(it)] = ir

    # Directional validation
    dv = analyzer.directional_validation(analyses)
    failure_cases = _build_failure_cases(raw_results, analyses)
    missed_factor_counter = Counter()
    wrong_action_counter = Counter()
    for case in failure_cases:
        missed_factor_counter.update(case["missed_factors"])
        if case["verdict"] == "miss":
            wrong_action_counter.update([case["chosen_action"]])
    worst_window = _worst_window(analyses, window_size=10)

    analyzed_output = {
        "config": config["name"],
        "provider": provider_name,
        "provider_label": provider.name(),
        "seed": seed,
        "num_ticks": config["num_ticks"],
        "per_tick": [
            {
                "tick": a.tick_number,
                "fc": a.factor_coverage,
                "fi": a.fixation_index,
                "da": a.decision_accuracy,
                "adr": a.anomaly_detection_rate,
                "action": a.action,
                "factors_substantive": a.factors_substantive,
                "word_counts": a.word_counts,
            }
            for a in analyses
        ],
        "summary": {
            "mean_fc": round(sum(fc_values) / len(fc_values), 4),
            "mean_da": round(sum(da_values) / len(da_values), 4),
            "mean_adr": round(sum(adr_values) / len(adr_values), 4),
            "mean_fi": round(sum(fi_values) / len(fi_values), 4),
            "da_at_40": round(da_at_40, 4),
            "da_last": round(da_last, 4),
            "dfg": round(dfg, 4),
            "collapses": dfg > 0.15,
            "fc_first_quarter": round(sum(fc_values[:len(fc_values)//4]) / max(1, len(fc_values)//4), 4),
            "fc_last_quarter": round(sum(fc_values[3*len(fc_values)//4:]) / max(1, len(fc_values) - 3*len(fc_values)//4), 4),
            "fc_drop_q1_to_q4": round(
                (
                    sum(fc_values[:len(fc_values)//4]) / max(1, len(fc_values)//4)
                ) - (
                    sum(fc_values[3*len(fc_values)//4:]) / max(1, len(fc_values) - 3*len(fc_values)//4)
                ),
                4,
            ),
            "context_truncations": sum(1 for r in raw_results if r["context_truncated"]),
            "context_resets": progress_stats["context_resets"],
            "interventions": progress_stats["interventions"],
            "avg_turn_seconds": round(progress_stats["turn_seconds_total"] / max(1, len(raw_results)), 2),
            "intervention_recovery": ir_values,
            "directional_validation": dv,
            "top_missed_factors": _top_counter_items(missed_factor_counter),
            "common_wrong_actions": _top_counter_items(wrong_action_counter),
            "worst_window": worst_window,
            "failure_case_count": len(failure_cases),
        },
    }

    output_file = output_dir / "analyzed_results.json"
    summary_file = output_dir / "summary.json"
    failure_file = output_dir / "failure_cases.jsonl"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(analyzed_output, f, indent=2)
    with open(summary_file, "w") as f:
        json.dump(analyzed_output["summary"], f, indent=2)
    with open(failure_file, "w") as f:
        for case in failure_cases:
            f.write(json.dumps(case) + "\n")

    logger.info("Results saved to %s", output_file)
    logger.info(
        "Run complete | config=%s provider=%s seed=%d scenario=%s",
        config["name"],
        provider_name,
        seed,
        scenario,
    )
    logger.info(
        "Scorecard | DA=%.3f | DA@40=%.3f | DA@last=%.3f | DFG=%.3f | collapses=%s | avg turn=%.1fs",
        analyzed_output["summary"]["mean_da"],
        analyzed_output["summary"]["da_at_40"],
        analyzed_output["summary"]["da_last"],
        analyzed_output["summary"]["dfg"],
        "yes" if analyzed_output["summary"]["collapses"] else "no",
        analyzed_output["summary"]["avg_turn_seconds"],
    )
    logger.info(
        "Diagnostics | FC=%.3f ADR=%.3f FI=%.3f | directional=%s | trunc=%d resets=%d interventions=%d",
        analyzed_output["summary"]["mean_fc"],
        analyzed_output["summary"]["mean_adr"],
        analyzed_output["summary"]["mean_fi"],
        analyzed_output["summary"]["directional_validation"]["verdict"],
        analyzed_output["summary"]["context_truncations"],
        analyzed_output["summary"]["context_resets"],
        analyzed_output["summary"]["interventions"],
    )
    logger.info(
        "Failure patterns | top missed=%s | wrong actions=%s | worst window=%d-%d (DA=%.3f) | cases=%d",
        _format_factor_list([item["name"] for item in analyzed_output["summary"]["top_missed_factors"]]),
        _format_factor_list([item["name"] for item in analyzed_output["summary"]["common_wrong_actions"]]),
        analyzed_output["summary"]["worst_window"]["start_tick"],
        analyzed_output["summary"]["worst_window"]["end_tick"],
        analyzed_output["summary"]["worst_window"]["mean_da"],
        analyzed_output["summary"]["failure_case_count"],
    )
    if ir_values:
        logger.info("Intervention recovery | %s", ir_values)
    logger.info("Artifacts | summary=%s failures=%s", summary_file, failure_file)


def main():
    parser = argparse.ArgumentParser(description="Run a single CoherenceBench benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--provider", required=True, help="Provider name (claude, gpt4o, gemini, llama)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Estimate tokens and cost without API calls")
    parser.add_argument("--scenario", default=None, help="Scenario name (power_grid, hospital, network). Overrides config.")
    parser.add_argument("--model", default=None, help="Optional provider model override, e.g. gpt-5.4")
    parser.add_argument(
        "--progress-every",
        type=int,
        default=10,
        help="Emit rollup summary every N ticks while still logging each cycle (default: 10)",
    )
    args = parser.parse_args()
    run(
        args.config,
        args.provider,
        args.seed,
        args.dry_run,
        args.scenario,
        args.progress_every,
        args.model,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        logger.warning("Interrupted by user. Partial results remain on disk if the run had already started.")
        raise SystemExit(130)
