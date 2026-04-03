# CoherenceBench Pre-Generated Data

This directory contains pre-generated tick data for all CoherenceBench scenarios.
Files are deterministic -- regenerating with the same seed produces identical output.

## Structure

```
data/
  power_grid/
    seed_42.json
    seed_123.json
    ...
  hospital/
    seed_42.json
    ...
  network/
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
      "prompt": "TICK 1:\n\nF1 (Load): ...",
      "ground_truth": {
        "anomalous_factors": ["load"],
        "correct_action": "shed_load",
        "acceptable_actions": ["shed_load", "start_gas_turbine", "ramp_plant"],
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
