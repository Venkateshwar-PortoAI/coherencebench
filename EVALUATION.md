# How to Evaluate a Model on CoherenceBench

This document describes the standard evaluation protocol for CoherenceBench.
Follow these steps exactly so that results are comparable across labs.

## 1. Install

```bash
git clone https://github.com/Venkateshwar-PortoAI/coherencebench.git
cd coherencebench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

cp .env.example .env
# Add your API keys (Anthropic, OpenAI, Google, Together -- whichever you need).
# The benchmark scripts load `.env` automatically.
```

## 2. Standard Run Configuration

All published results must use these settings unless explicitly stated otherwise:

| Parameter | Value |
|-----------|-------|
| Ticks per run | 200 |
| Temperature | 0 |
| Seeds | 5 minimum (42, 123, 456, 789, 1001) |
| Scenario | `power_grid` (primary), plus `hospital` and `network` if reporting multi-scenario |
| Config | `configs/run_a_baseline.yaml` (baseline condition) |

## 3. Run the Benchmark

```bash
# Dry run first (estimates tokens and cost, no API calls)
python scripts/run_single.py --config configs/run_a_baseline.yaml \
    --provider <your-provider> --seed 42 --dry-run

# Execute a single seed
python scripts/run_single.py --config configs/run_a_baseline.yaml \
    --provider <your-provider> --seed 42

# Execute all 5 seeds
for seed in 42 123 456 789 1001; do
    python scripts/run_single.py --config configs/run_a_baseline.yaml \
        --provider <your-provider> --seed $seed
done
```

Results are written to `results/run_a_baseline/<provider>/seed_<N>/`.

## 4. Metrics to Report

Report the following metrics, averaged across all seeds:

| Metric | Description | How to compute |
|--------|-------------|---------------|
| **DA** | Decision Accuracy | Fraction of ticks where the agent chose an acceptable action |
| **DA@40** | DA in first 40 ticks | DA restricted to ticks 1-40 |
| **DA@last** | DA in final 40 ticks | DA restricted to ticks 161-200 |
| **DFG** | DA Drift (First-to-last Gap) | DA@40 minus DA@last (positive = accuracy degraded) |
| **Collapses?** | Whether DA degrades by >15pp | YES if DA@40 - DA@last > 0.15 |

Optional but encouraged:

| Metric | Description |
|--------|-------------|
| **FC** | Factor Coverage (fraction of 6 factors substantively analyzed) |
| **FI** | Fixation Index (token share of the most-discussed factor) |
| **ADR** | Anomaly Mention Rate (fraction of anomalous factors substantively discussed; proxy, not validated detection) |

## 5. Interpreting Results

- **DA is the primary metric.** It measures whether the agent makes correct decisions,
  not whether it writes well-formatted responses.
- **Collapses? is the headline finding.** A model that collapses (DA drops >15pp from
  start to end) is exhibiting attention collapse -- the core phenomenon CoherenceBench
  measures.
- **DFG direction matters.** Positive DFG means accuracy was higher early than late
  (typical degradation pattern). Negative DFG means the model improved over time (rare).
- **FC can mask DA.** A model can maintain FC = 1.0 (mentions all 6 factors every tick)
  while its DA collapses. This is the "invisible collapse" that CoherenceBench detects.

## 6. Submitting Results to the Leaderboard

Submit a PR to the repository with:

1. Raw results in `results/run_a_baseline/<provider>/seed_<N>/raw_results.jsonl`
   for all 5 seeds.
2. An update to the leaderboard table in `README.md` with your aggregated numbers.
3. Include in the PR description:
   - Model name and version (exact API model string)
   - Date of evaluation
   - Temperature setting (must be 0)
   - Number of seeds
   - Total token count and cost (if available)

### PR Format

```
## Leaderboard Submission: <Model Name>

- Model: <exact API model string, e.g. claude-sonnet-4-20250514>
- Date: <YYYY-MM-DD>
- Temperature: 0
- Seeds: 5 (42, 123, 456, 789, 1001)
- Config: run_a_baseline

| Metric | Value |
|--------|-------|
| DA     | XX.X% |
| DA@40  | XX.X% |
| DA@last| XX.X% |
| DFG    | +X.X% |
| Collapses? | YES/NO |
```

## 7. Running Additional Conditions

Beyond the baseline (Run A), CoherenceBench supports 4 mitigation conditions:

| Config | Condition | What it tests |
|--------|-----------|---------------|
| `run_b_intervention.yaml` | Periodic reminders | Does prompting "analyze all factors" help? |
| `run_c_context_reset.yaml` | Context clearing | Does resetting context every 40 ticks help? |
| `run_d_checklist.yaml` | Mandatory checklist | Does forcing structured output help? |
| `run_e_cross_model.yaml` | Cross-model | Compare multiple models under identical conditions |

These are optional but encouraged for research papers.
