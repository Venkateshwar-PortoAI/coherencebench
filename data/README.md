# CoherenceBench Pre-Generated Data

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
      "prompt": "TICK 1:\n\nF1 (Load): ...",
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
