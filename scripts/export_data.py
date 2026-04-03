#!/usr/bin/env python3
"""Export pre-generated tick data as JSON files for reproducibility.

Generates deterministic tick data for all seeds and scenarios so that
researchers can inspect the benchmark inputs without running the generator.

Usage:
    python scripts/export_data.py
    python scripts/export_data.py --scenario hospital
    python scripts/export_data.py --scenario all
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.generator import TickGenerator
from src.scenarios import get_scenario, SCENARIOS

SEEDS = [42, 123, 456, 789, 1001]
NUM_TICKS = 200
DATA_DIR = Path(__file__).resolve().parent.parent / "data"


def export_scenario(scenario_name: str):
    """Export tick data for a single scenario across all seeds."""
    scenario = get_scenario(scenario_name)
    # network is the held-out evaluation set, stored under data/eval/
    if scenario_name == "network":
        scenario_dir = DATA_DIR / "eval" / scenario_name
    else:
        scenario_dir = DATA_DIR / scenario_name

    for seed in SEEDS:
        gen = TickGenerator(seed=seed, num_ticks=NUM_TICKS, scenario=scenario)
        ticks = gen.generate()

        # Build the export structure
        factor_descriptions = {
            f.name: {"display_name": f.display_name, "description": f.description}
            for f in scenario.factors
        }

        export = {
            "scenario": scenario_name,
            "seed": seed,
            "num_ticks": NUM_TICKS,
            "factors": factor_descriptions,
            "actions": scenario.actions,
            "system_prompt": scenario.system_prompt(),
            "ticks": [],
        }

        for tick in ticks:
            tick_export = {
                "tick_number": tick["tick_number"],
                "prompt": scenario.format_tick(tick["tick_number"], tick["data"]),
                "ground_truth": {
                    "anomalous_factors": tick["ground_truth"]["anomalous_factors"],
                    "correct_action": tick["ground_truth"]["correct_action"],
                    "acceptable_actions": tick["ground_truth"]["acceptable_actions"],
                    "is_multi_factor": tick["ground_truth"]["is_multi_factor"],
                },
            }
            export["ticks"].append(tick_export)

        # Write the file
        scenario_dir.mkdir(parents=True, exist_ok=True)
        output_file = scenario_dir / f"seed_{seed}.json"
        with open(output_file, "w") as f:
            json.dump(export, f, indent=2)

        print(f"  Exported: {output_file} ({len(ticks)} ticks)")


def write_data_readme():
    """Write a README explaining the data format."""
    readme = DATA_DIR / "README.md"
    content = """# CoherenceBench Pre-Generated Data

This directory contains pre-generated tick data for all CoherenceBench scenarios.
Files are deterministic -- regenerating with the same seed produces identical output.

## Train/Eval Split

CoherenceBench enforces a strict train/eval split:

- **`power_grid/` and `hospital/`** are the public **development** set.
  Use these for development, debugging, and model tuning.
- **`eval/network/`** is the held-out **evaluation** set.
  Do not use ground truth from these files for training or prompt engineering.

## Structure

```
data/
  power_grid/           # DEVELOPMENT SET
    seed_42.json
    seed_123.json
    ...
  hospital/             # DEVELOPMENT SET
    seed_42.json
    ...
  eval/
    README.md           # "Do not use for development or fine-tuning"
    network/            # EVALUATION SET (held-out)
      seed_42.json
      ...
```

## File Format

Each JSON file contains:

```json
{
  "scenario": "power_grid",
  "seed": 42,
  "num_ticks": 200,
  "factors": {
    "load": {"display_name": "Load (Consumer Demand)", "description": "..."},
    ...
  },
  "actions": ["shed_load", "start_gas_turbine", ...],
  "system_prompt": "You are an autonomous power grid...",
  "ticks": [
    {
      "tick_number": 1,
      "prompt": "TICK 1:\\n\\nF1 (Load): ...",
      "ground_truth": {
        "anomalous_factors": ["load"],
        "correct_action": "shed_load",
        "acceptable_actions": ["shed_load", "start_gas_turbine"],
        "is_multi_factor": false
      }
    },
    ...
  ]
}
```

## Usage

You can use these files to:
1. Inspect the exact prompts given to models
2. Run your own evaluation without using the generator
3. Verify reproducibility of the benchmark
4. Build alternative scoring scripts

## Seeds

| Seed | Purpose |
|------|---------|
| 42 | Primary benchmark seed |
| 123 | Validation seed |
| 456 | Validation seed |
| 789 | Validation seed |
| 1001 | Validation seed |
"""
    with open(readme, "w") as f:
        f.write(content)
    print(f"  Wrote: {readme}")


def main():
    parser = argparse.ArgumentParser(description="Export CoherenceBench tick data as JSON")
    parser.add_argument("--scenario", default="all",
                        help="Scenario to export (power_grid, hospital, network, or 'all')")
    args = parser.parse_args()

    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if args.scenario == "all":
        scenarios = list(SCENARIOS.keys())
    else:
        if args.scenario not in SCENARIOS:
            print(f"Unknown scenario: {args.scenario}. Available: {list(SCENARIOS.keys())}")
            sys.exit(1)
        scenarios = [args.scenario]

    for scenario_name in scenarios:
        print(f"\nExporting scenario: {scenario_name}")
        export_scenario(scenario_name)

    write_data_readme()
    print("\nDone.")


if __name__ == "__main__":
    main()
