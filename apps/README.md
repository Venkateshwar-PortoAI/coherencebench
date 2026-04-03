# CoherenceBench Human Benchmark

A browser-based interface for humans to take the CoherenceBench benchmark. No server, no dependencies — just open the HTML file.

## Quick Start

```bash
open apps/human_benchmark.html
# or just double-click the file in Finder
```

## How It Works

1. **Open** `human_benchmark.html` in any modern browser
2. **Choose** the embedded Power Grid scenario (seed 42) or load any CoherenceBench JSON file
3. **Read** the 6-factor status update for each tick — the same data the AI models see
4. **Select** an action from the 10 buttons (or press keys 1-9, 0)
5. **Review** immediate feedback: correct/incorrect, acceptable actions, anomalous factors
6. **Repeat** for 50 ticks (configurable)
7. **View** your summary: overall DA, per-factor DA, DA over time, decision speed

## Keyboard Shortcuts

| Key | Action |
|-----|--------|
| `1`-`9`, `0` | Select action 1-10 |
| `Enter` / `Space` | Advance to next tick |

## Scoring

- **Decision Accuracy (DA)**: Your action is correct if it appears in `acceptable_actions` for that tick
- **Per-Factor DA**: How well you perform when a specific factor is anomalous
- **DA@first10 vs DA@last10**: Detects whether you experience "coherence collapse" — the same degradation the benchmark measures in AI models
- **Multi-Factor DA**: Accuracy on ticks where multiple factors are simultaneously anomalous

## Loading Custom Data

Click "Load JSON file" and select any CoherenceBench scenario JSON (`data/power_grid/seed_42.json`, `data/hospital/seed_42.json`, `data/network/seed_42.json`, etc.). The viewer auto-detects the scenario type and adapts factor labels and action buttons.

## Output

Click "Copy Results as JSON" on the summary screen to get a structured result object:

```json
{
  "benchmark": "CoherenceBench",
  "participant": "human",
  "scenario": "power_grid",
  "seed": 42,
  "num_ticks": 50,
  "overall_da": 82.0,
  "da_first_10": 90.0,
  "da_last_10": 70.0,
  "avg_decision_time_ms": 8500,
  "per_tick": [...]
}
```

## Requirements

- Any modern browser (Chrome, Firefox, Safari, Edge)
- Works fully offline — no network requests
- Single self-contained HTML file (~65KB)
