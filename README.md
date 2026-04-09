<p align="center">
  <img src="assets/banner.png" alt="CoherenceBench — Measuring Attention Collapse in Long-Running Autonomous Agents" width="900"/>
</p>

<p align="center">
  <a href="https://github.com/Venkateshwar-PortoAI/coherencebench/actions/workflows/ci.yml"><img src="https://github.com/Venkateshwar-PortoAI/coherencebench/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
</p>

CoherenceBench is a benchmark for sustained multi-factor decision coherence in long-running agent loops. A model monitors 6 subsystems across 200 sequential decisions in simulated control-room scenarios, while the benchmark measures whether decision quality degrades as anomalies shift across factors over time.

> **[Live Demo](https://venkateshwar-portoai.github.io/coherencebench/)** — Watch GPT-5.4 lose coherence over 200 ticks. No install needed.

> **Status:** v0.1.0 released. 4 scenarios, 11 providers, replay viewer. We welcome model submissions. See [EVALUATION.md](EVALUATION.md).

## How It Works

```mermaid
graph LR
    subgraph Scenario["Scenario (e.g. Power Grid, ATC)"]
        F1[6 Subsystems]
        A1[10 Possible Actions]
        PW[Phase-Shifted<br/>Anomalies]
    end

    subgraph Loop["200-Tick Benchmark Loop"]
        TG[Tick Generator] -->|tick prompt| LLM[LLM Agent]
        LLM -->|structured response| PA[Response Analyzer]
        PA -->|DA, FC, FI, ADR| M[Metrics]
    end

    F1 --> TG
    PW --> TG
    A1 --> PA

    style LLM fill:#f9f,stroke:#333
    style M fill:#9f9,stroke:#333
```

Each tick, the agent receives sensor readings from 6 subsystems and must pick one of 10 actions. Anomalies shift across subsystems over 5 phases, creating an attention trap: models that stay fluent but stop tracking the right factors can still look coherent while choosing worse actions.

## Results

These are early reference results, not a final leaderboard. All runs are reproducible from the raw data in `results/`. Some runs are incomplete due to API rate limits or cost constraints — we document this honestly rather than exclude partial data.

### Reference Results (Power Grid)

| Agent | Ticks | Seeds | DA | DA@40 | DA@last | DFG | Collapses? |
|-------|-------|-------|-----|-------|---------|-----|------------|
| Most-common action (baseline) | 200 | — | 54.8% | 45.5% | 70.0% | -24.5% | NO |
| Majority / always hold_steady | 200 | — | 24.9% | 26.0% | 21.5% | +4.5% | NO |
| Random uniform | 200 | — | 24.1% | 22.7% | 25.2% | -2.5% | NO |
| Llama 3.3 70B (Groq) | 14* | 1 | 71.4% | 71.4% | — | — | — |
| GPT-5.4 (Codex) | 200 | 3 complete | 26.8% | 30.8% | 15.8% | +15.0% | YES |
| Nemotron 3 Super 120B | 50* | 1 | 2.0% | 2.5% | 0.0% | +2.5% | — |
| Claude Haiku 4.5 | 12* | 4 partial | — | — | — | — | — |

> **DA** = Decision Accuracy (% correct actions). **DA@40** = first 40 ticks. **DA@last** = final 40 ticks.
> **DFG** = DA@40 minus DA@last (positive = accuracy degraded). **Collapses?** = DFG > 15pp.
>
> \* **Incomplete runs.** Llama 3.3 70B stopped at tick 14 due to Groq free-tier daily token limit (100k TPD). Nemotron stopped at tick 50 due to OpenRouter free-tier rate limit (50 req/day). Claude Haiku runs are partial (10-12 ticks each) due to API key expiration during early development. We include these as honest data points rather than omitting them. Full 200-tick runs with 5 seeds are needed for definitive conclusions. See `results/` for raw data.

### Key findings

- **GPT-5.4 shows coherence collapse.** Across 3 complete seeds, DA degrades from ~31% (first 40 ticks) to ~16% (last 40 ticks) while FC stays at 100%. The model keeps writing perfect analysis of all 6 factors but its decisions get worse. DFG ranges from +10pp to +20pp across seeds. **[See it happen in the replay viewer.](https://venkateshwar-portoai.github.io/coherencebench/)**
- **Nemotron 120B: format-behavior dissociation.** FC=0.94 (near-perfect format) but DA=2%. The model writes thorough analysis then picks `adjust_voltage` or `charge_battery` almost every tick regardless of the actual anomaly. This is the clearest example of the dissociation the benchmark is designed to detect.
- **Llama 3.3 70B: promising early signal.** DA=71.4% in 14 ticks (far above baselines). Full 200-tick run needed to see if it sustains or collapses.

### Decision Accuracy Over Time

![Decision Accuracy Over Time](assets/da_over_time.png)

Per-run outputs are written to `results/*/`, including `summary.json`, `failure_cases.jsonl`, `raw_results.jsonl`, and `analyzed_results.json`.

### Contributing runs

We are actively seeking complete 200-tick runs across more models. If you have API access, even a single seed helps. See [EVALUATION.md](EVALUATION.md) for the standard protocol.

**[Add your model](EVALUATION.md)** -- submit a PR with your results.

## Replay Viewer

The replay viewer lets you scrub through 200 ticks and watch coherence collapse happen. See the model's analysis, its chosen action, the correct action, and the anomaly heatmap tick by tick.

**[Try the live demo](https://venkateshwar-portoai.github.io/coherencebench/)** (no install needed)

Or generate a viewer from any run:

```bash
python -m src.cli view results/run_a_baseline/codex/seed_123/
```

This opens a self-contained HTML file in your browser with 5 panes: collapse curve, subsystem heatmap, tick scrubber, transcript panel, and score dashboard.

## Quick Start

```bash
git clone https://github.com/Venkateshwar-PortoAI/coherencebench.git
cd coherencebench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Set up API keys
cp .env.example .env
# Edit .env with your API keys

# Run a benchmark (Groq is free)
python -m src.cli run --provider groq --scenario power_grid --seed 42

# View results
python -m src.cli view results/run_a_baseline/groq/seed_42/

# Or use the scripts directly
python scripts/run_single.py --config configs/run_a_baseline.yaml --provider groq --seed 42
```

The main flow:

1. Run one seed.
2. Open the replay viewer to see the collapse curve.
3. Inspect `failure_cases.jsonl` for concrete mistakes.
4. Run more seeds or more providers if you want stable comparisons.

### Typical Usage

Run one seed for a quick inspection:

```bash
python scripts/run_single.py \
  --config configs/run_a_baseline.yaml \
  --provider codex \
  --seed 42
```

Run 5 seeds for one provider:

```bash
for seed in 42 123 456 789 1001; do
  python scripts/run_single.py \
    --config configs/run_a_baseline.yaml \
    --provider codex \
    --seed "$seed"
done
```

Run a batch benchmark across selected providers:

```bash
python scripts/run_benchmark.py \
  --configs configs/run_a_baseline.yaml \
  --providers codex claude
```

Override a specific model string for one provider:

```bash
python scripts/run_single.py \
  --config configs/run_a_baseline.yaml \
  --provider gpt4o \
  --model gpt-5 \
  --seed 42
```

## Outputs

Each run writes results to:

```text
results/<config>/<provider>/seed_<N>/
```

Key artifacts:

| File | Purpose |
|------|---------|
| `summary.json` | Compact scorecard for the run |
| `failure_cases.jsonl` | Ticks where the model missed, partially covered, or otherwise failed |
| `raw_results.jsonl` | Raw per-tick model responses and ground truth references |
| `analyzed_results.json` | Full per-tick analysis plus aggregate metrics |

The main public-facing report is `summary.json`. The main debugging artifact is `failure_cases.jsonl`.

## What This Benchmark Measures

CoherenceBench is not a general reasoning benchmark. It is a controlled benchmark for one specific failure mode:

- can a model sustain correct multi-factor decisions over a long sequential run?

The benchmark separates:

- **Outcome quality** via `DA`, `DA@40`, `DA@last`, and `DFG`
- **Behavior diagnostics** via `FC`, `FI`, and `ADR`

That distinction matters because a model can preserve fluent, structured analysis while still choosing the wrong action.

## Scenarios

4 scenarios across different domains. Each has 6 subsystems, 10 actions, and phase-shifted anomalies.

| Scenario | Domain | Subsystems |
|----------|--------|------------|
| `power_grid` | Electricity grid | Load, Generation, Frequency, Voltage, Weather, Reserve |
| `hospital` | Hospital triage | Vitals, Labs, Imaging, Medications, History, Capacity |
| `air_traffic_control` | ATC tower | Radar, Weather, Runway, Comms, Traffic Flow, Systems |
| `network` | Network security SOC | Traffic, Auth, Endpoints, Firewall, Logs, Threats |

The first three are the main public scenarios. `network` is packaged separately under `data/eval/network/` with held-out ground truth and should be treated as an evaluation-only scenario unless you are extending the scoring flow yourself.

```bash
# Run a different scenario
python scripts/run_single.py --config configs/run_a_baseline.yaml --provider claude --seed 42 --scenario hospital
```

## Metrics

| Metric | What It Measures |
|--------|-----------------|
| **DA** (Decision Accuracy) | Did the agent choose a correct action? **(primary)** |
| **FC** (Factor Coverage) | How many of 6 subsystems were substantively analyzed? |
| **FI** (Fixation Index) | How much attention goes to a single subsystem? |
| **ADR** (Anomaly Mention Rate) | Did the agent discuss the anomalous subsystems? |

High FC + low DA = invisible collapse. The agent writes about all subsystems but picks the wrong action.

For public reporting, the headline metrics are:

| Metric | Meaning |
|--------|---------|
| **DA** | Overall decision accuracy |
| **DA@40** | Accuracy in the first 40 ticks |
| **DA@last** | Accuracy in the final 40 ticks |
| **DFG** | `DA@40 - DA@last` |
| **Collapses?** | Whether `DFG > 0.15` |

## Experimental Conditions

| Run | Condition | What It Tests |
|-----|-----------|---------------|
| A | Baseline | Natural degradation over 200 ticks |
| B | Intervention | Do "analyze all factors" reminders help? |
| C | Context Reset | Does clearing context every 40 ticks help? |
| D | Checklist | Does a mandatory checklist prevent collapse? |
| E | Cross-Model | Same test across all providers |

## Supported Models

| Provider Flag | Default Model | Via | Cost |
|---------------|---------------|-----|------|
| `groq` | `llama-3.3-70b-versatile` | Groq API | Free (rate-limited) |
| `ollama` | `deepseek-r1:14b` | Local Ollama | Free (local GPU) |
| `openrouter` | `nvidia/nemotron-3-super-120b-a12b:free` | OpenRouter | Free (50 req/day) |
| `mlvoca` | `deepseek-r1:1.5b` | MLvoca API | Free (no key needed) |
| `claude` | `claude-haiku-4-5-20251001` | Anthropic API | Paid |
| `gpt4o` | `gpt-4o` | OpenAI API | Paid |
| `gemini` | `gemini-2.0-flash` | Google GenAI API | Paid |
| `llama` | `meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo` | Together API | Paid |
| `codex` | `gpt-5.4` | Codex CLI | Paid |
| `claude-cli` | `sonnet` | Claude Code CLI | Paid |

### Adding Your Own Model

Implement `LLMProvider` in `src/providers/base.py`, register in `src/providers/__init__.py`, run with `--provider your-model`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Project Structure

```
coherencebench/
  configs/           # YAML run configurations (A-E)
  data/              # Pre-generated tick data (deterministic, JSON)
  docs/              # GitHub Pages demo (replay viewer HTML)
  results/           # Run outputs and scorecards
  scripts/           # run_single.py, run_benchmark.py, compute_baselines.py
  src/
    cli.py           # CLI entry point (run, view, analyze)
    analyzer.py      # Response parsing + metrics
    generator.py     # Tick data with planted anomalies
    metrics.py       # DA, FC, FI, ADR, DFG
    runner.py        # Benchmark loop with context management
    visualizer.py    # Matplotlib figures
    viewer/          # Replay viewer (self-contained HTML generator)
    providers/       # LLM API adapters (11 providers)
    scenarios/       # Scenario definitions (4 domains)
  tests/             # Test suite (101 tests)
```

## Limitations

- **Controlled session loop.** The benchmark primarily measures foundation-model behavior inside this harness, not arbitrary external agent stacks.
- **Single-turn decisions.** No multi-step planning or tool-using sub-policies within a tick.
- **Synthetic environments.** Simplified simulations, not real-world monitoring.
- **Binary scoring.** No partial credit for reasonable but non-matching actions.
- **Limited model coverage.** Early reference runs only. Most runs are partial due to API cost/rate constraints. Community submissions welcome.

## Related Work

- **SWE-bench**: Code repair (one-shot). CoherenceBench: continuous monitoring (200 turns).
- **AgentBench**: Task completion. CoherenceBench: degradation measurement.
- **Beyond pass@1** (2026): Reliability surfaces for long-horizon agents. CoherenceBench catches subtle drift, not obvious meltdowns.

## Citation

```bibtex
@software{coherencebench2026,
  author       = {Venkateshwar Reddy Jambula},
  title        = {{CoherenceBench}: Measuring Attention Collapse in
                  Long-Running Autonomous Agents},
  year         = {2026},
  publisher    = {GitHub},
  url          = {https://github.com/Venkateshwar-PortoAI/coherencebench},
  note         = {Open-source benchmark, MIT License}
}
```

## License

[MIT](LICENSE)

---

Built by [PranaAlpha Labs](https://pranaalpha.com)
