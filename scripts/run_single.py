#!/usr/bin/env python3
"""Run a single benchmark config/provider/seed combination.

Usage:
    python scripts/run_single.py --config configs/run_a_baseline.yaml --provider claude --seed 42
    python scripts/run_single.py --config configs/run_a_baseline.yaml --provider claude --seed 42 --dry-run
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import yaml

# Add project root to path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.providers import get_provider
from src.runner import BenchmarkRunner
from src.analyzer import ResponseAnalyzer
from src.scenarios import get_scenario

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("run_single")

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


def run(config_path: str, provider_name: str, seed: int, dry_run: bool = False, scenario_name: str | None = None):
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

    provider = get_provider(provider_name)
    output_dir = Path("results") / config["name"] / provider_name / f"seed_{seed}"

    runner = BenchmarkRunner(
        provider=provider,
        num_ticks=config["num_ticks"],
        seed=seed,
        output_dir=output_dir,
        context_reset_interval=config.get("context_reset_interval"),
        intervention_ticks=config.get("intervention_ticks") or [],
        force_checklist=config.get("force_checklist", False),
        scenario=scenario,
    )

    logger.info("Running %d ticks...", config["num_ticks"])
    raw_results = runner.run()

    # Analyze
    logger.info("Analyzing responses...")
    analyzer = ResponseAnalyzer()
    analyses = analyzer.analyze_run(raw_results)

    # Compute summary metrics
    fc_values = [a.factor_coverage for a in analyses]
    da_values = [a.decision_accuracy for a in analyses]
    adr_values = [a.anomaly_detection_rate for a in analyses]
    fi_values = [a.fixation_index for a in analyses]

    # Intervention recovery
    ir_values = {}
    for it in config.get("intervention_ticks") or []:
        ir = analyzer.compute_intervention_recovery(analyses, it)
        ir_values[str(it)] = ir

    # Directional validation
    dv = analyzer.directional_validation(analyses)

    analyzed_output = {
        "config": config["name"],
        "provider": provider_name,
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
                "token_counts": a.token_counts,
            }
            for a in analyses
        ],
        "summary": {
            "mean_fc": round(sum(fc_values) / len(fc_values), 4),
            "mean_da": round(sum(da_values) / len(da_values), 4),
            "mean_adr": round(sum(adr_values) / len(adr_values), 4),
            "mean_fi": round(sum(fi_values) / len(fi_values), 4),
            "fc_first_quarter": round(sum(fc_values[:len(fc_values)//4]) / max(1, len(fc_values)//4), 4),
            "fc_last_quarter": round(sum(fc_values[3*len(fc_values)//4:]) / max(1, len(fc_values) - 3*len(fc_values)//4), 4),
            "intervention_recovery": ir_values,
            "directional_validation": dv,
        },
    }

    output_file = output_dir / "analyzed_results.json"
    output_dir.mkdir(parents=True, exist_ok=True)
    with open(output_file, "w") as f:
        json.dump(analyzed_output, f, indent=2)

    logger.info("Results saved to %s", output_file)
    logger.info(
        "Summary: FC=%.3f DA=%.3f ADR=%.3f FI=%.3f",
        analyzed_output["summary"]["mean_fc"],
        analyzed_output["summary"]["mean_da"],
        analyzed_output["summary"]["mean_adr"],
        analyzed_output["summary"]["mean_fi"],
    )


def main():
    parser = argparse.ArgumentParser(description="Run a single CoherenceBench benchmark")
    parser.add_argument("--config", required=True, help="Path to YAML config file")
    parser.add_argument("--provider", required=True, help="Provider name (claude, gpt4o, gemini, llama)")
    parser.add_argument("--seed", type=int, required=True, help="Random seed")
    parser.add_argument("--dry-run", action="store_true", help="Estimate tokens and cost without API calls")
    parser.add_argument("--scenario", default=None, help="Scenario name (power_grid, hospital, network). Overrides config.")
    args = parser.parse_args()
    run(args.config, args.provider, args.seed, args.dry_run, args.scenario)


if __name__ == "__main__":
    main()
