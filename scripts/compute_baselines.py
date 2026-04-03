#!/usr/bin/env python3
"""Compute random, majority, and most-common-action baselines for CoherenceBench.

Answers the question: is X% DA meaningful, or trivially achievable by guessing?

Usage:
    python scripts/compute_baselines.py
    python scripts/compute_baselines.py --scenario hospital
    python scripts/compute_baselines.py --trials 500
"""

import argparse
import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generator import TickGenerator
from src.scenarios import get_scenario, SCENARIOS

SEEDS = [42, 123, 456, 789, 1001]
NUM_TICKS = 200
DEFAULT_TRIALS = 1000


def compute_da(predicted: str, acceptable: list[str]) -> float:
    return 1.0 if predicted in acceptable else 0.0


def find_most_common_acceptable_action(ticks: list[dict]) -> str:
    """Find the action that appears in the most acceptable_actions lists."""
    counter: Counter = Counter()
    for tick in ticks:
        gt = tick["ground_truth"]
        for a in gt["acceptable_actions"]:
            counter[a] += 1
    if counter:
        return counter.most_common(1)[0][0]
    return "hold_steady"


def compute_baselines_for_seed(
    seed: int,
    scenario_name: str,
    num_trials: int,
) -> dict:
    """Compute all three baselines for a single seed."""
    scenario = get_scenario(scenario_name)
    gen = TickGenerator(seed=seed, num_ticks=NUM_TICKS, scenario=scenario)
    ticks = gen.generate()

    actions = scenario.actions
    most_common_action = find_most_common_acceptable_action(ticks)

    # Determine "hold_steady" equivalent: the no-op action
    # For power_grid it's "hold_steady", for hospital "no_action_needed", for network "no_action_needed"
    noop_actions = ["hold_steady", "no_action_needed"]
    majority_action = next((a for a in noop_actions if a in actions), actions[0])

    # --- Random baseline: 1000 trials ---
    random_da_trials = []
    for trial in range(num_trials):
        rng = random.Random(seed * 10000 + trial)
        trial_da = []
        for tick in ticks:
            gt = tick["ground_truth"]
            predicted = rng.choice(actions)
            trial_da.append(compute_da(predicted, gt["acceptable_actions"]))
        random_da_trials.append(trial_da)

    # Aggregate random trials
    random_mean_da = sum(sum(t) / len(t) for t in random_da_trials) / len(random_da_trials)
    random_da_at_40 = sum(
        sum(t[:40]) / 40 for t in random_da_trials
    ) / len(random_da_trials)
    random_da_last = sum(
        sum(t[160:]) / 40 for t in random_da_trials
    ) / len(random_da_trials)

    # --- Majority baseline: always pick noop ---
    majority_da = []
    for tick in ticks:
        gt = tick["ground_truth"]
        majority_da.append(compute_da(majority_action, gt["acceptable_actions"]))

    majority_mean = sum(majority_da) / len(majority_da)
    majority_at_40 = sum(majority_da[:40]) / 40
    majority_last = sum(majority_da[160:]) / 40

    # --- Most-common-action baseline ---
    common_da = []
    for tick in ticks:
        gt = tick["ground_truth"]
        common_da.append(compute_da(most_common_action, gt["acceptable_actions"]))

    common_mean = sum(common_da) / len(common_da)
    common_at_40 = sum(common_da[:40]) / 40
    common_last = sum(common_da[160:]) / 40

    # DFG approximation: difference between DA@40 and DA@last
    random_dfg = random_da_at_40 - random_da_last
    majority_dfg = majority_at_40 - majority_last
    common_dfg = common_at_40 - common_last

    return {
        "seed": seed,
        "majority_action": majority_action,
        "most_common_action": most_common_action,
        "random": {
            "mean_da": random_mean_da,
            "da_at_40": random_da_at_40,
            "da_last": random_da_last,
            "dfg": random_dfg,
        },
        "majority": {
            "mean_da": majority_mean,
            "da_at_40": majority_at_40,
            "da_last": majority_last,
            "dfg": majority_dfg,
        },
        "most_common": {
            "mean_da": common_mean,
            "da_at_40": common_at_40,
            "da_last": common_last,
            "dfg": common_dfg,
        },
    }


def main():
    parser = argparse.ArgumentParser(description="Compute baselines for CoherenceBench")
    parser.add_argument("--scenario", default="power_grid", choices=list(SCENARIOS.keys()),
                        help="Scenario to compute baselines for")
    parser.add_argument("--trials", type=int, default=DEFAULT_TRIALS,
                        help="Number of random trials per seed for statistical significance")
    args = parser.parse_args()

    print(f"Computing baselines for scenario: {args.scenario}")
    print(f"Seeds: {SEEDS}, Ticks: {NUM_TICKS}, Random trials: {args.trials}")
    print("=" * 80)

    all_results = []
    for seed in SEEDS:
        result = compute_baselines_for_seed(seed, args.scenario, args.trials)
        all_results.append(result)
        print(f"\nSeed {seed}:")
        print(f"  Majority action: {result['majority_action']}")
        print(f"  Most common acceptable action: {result['most_common_action']}")
        for baseline_name in ["random", "majority", "most_common"]:
            b = result[baseline_name]
            print(f"  {baseline_name:15s}: DA={b['mean_da']:.1%}  DA@40={b['da_at_40']:.1%}  "
                  f"DA@last={b['da_last']:.1%}  DFG={b['dfg']:+.1%}")

    # Aggregate across seeds
    print("\n" + "=" * 80)
    print("AGGREGATE (mean across all seeds)")
    print("=" * 80)

    header = f"{'Baseline':20s} {'DA':>8s} {'DA@40':>8s} {'DA@last':>8s} {'DFG':>8s}"
    print(header)
    print("-" * len(header))

    for baseline_name in ["random", "majority", "most_common"]:
        mean_da = sum(r[baseline_name]["mean_da"] for r in all_results) / len(all_results)
        mean_at_40 = sum(r[baseline_name]["da_at_40"] for r in all_results) / len(all_results)
        mean_last = sum(r[baseline_name]["da_last"] for r in all_results) / len(all_results)
        mean_dfg = sum(r[baseline_name]["dfg"] for r in all_results) / len(all_results)
        print(f"{baseline_name:20s} {mean_da:>7.1%} {mean_at_40:>7.1%} "
              f"{mean_last:>7.1%} {mean_dfg:>+7.1%}")

    # Comparison with model results
    print("\n" + "=" * 80)
    print("COMPARISON TABLE")
    print("=" * 80)
    print(f"{'Agent':25s} {'DA':>8s} {'DA@40':>8s} {'DA@last':>8s} {'DFG':>8s} {'Collapses?':>12s}")
    print("-" * 85)

    for baseline_name, label in [
        ("random", "Random (uniform)"),
        ("majority", "Majority (always noop)"),
        ("most_common", "Most-common action"),
    ]:
        mean_da = sum(r[baseline_name]["mean_da"] for r in all_results) / len(all_results)
        mean_at_40 = sum(r[baseline_name]["da_at_40"] for r in all_results) / len(all_results)
        mean_last = sum(r[baseline_name]["da_last"] for r in all_results) / len(all_results)
        mean_dfg = sum(r[baseline_name]["dfg"] for r in all_results) / len(all_results)
        collapse = "NO" if abs(mean_at_40 - mean_last) < 0.15 else "YES"
        print(f"{label:25s} {mean_da:>7.1%} {mean_at_40:>7.1%} "
              f"{mean_last:>7.1%} {mean_dfg:>+7.1%} {collapse:>12s}")

    # Known model results (power_grid only)
    if args.scenario == "power_grid":
        print(f"{'Claude Haiku 4.5':25s} {'33%':>8s} {'58%':>8s} {'22%':>8s} {'+3%':>8s} {'YES':>12s}")
        print(f"{'GPT-5.4 (Codex)':25s} {'28%':>8s} {'30%':>8s} {'30%':>8s} {'+1%':>8s} {'NO':>12s}")


if __name__ == "__main__":
    main()
