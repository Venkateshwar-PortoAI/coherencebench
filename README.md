<p align="center">
  <img src="assets/banner.png" alt="CoherenceBench — Measuring Attention Collapse in Long-Running Autonomous Agents" width="900"/>
</p>

<p align="center">
  <a href="https://github.com/Venkateshwar-PortoAI/coherencebench/actions/workflows/ci.yml"><img src="https://github.com/Venkateshwar-PortoAI/coherencebench/actions/workflows/ci.yml/badge.svg" alt="CI"></a>
  <a href="https://opensource.org/licenses/MIT"><img src="https://img.shields.io/badge/License-MIT-yellow.svg" alt="License: MIT"></a>
  <a href="https://www.python.org/downloads/"><img src="https://img.shields.io/badge/python-3.11+-blue.svg" alt="Python 3.11+"></a>
</p>

CoherenceBench measures how LLM agents degrade over extended interactions. Agents monitor 6 subsystems across 200 decisions in simulated control-room scenarios. The key finding: agents maintain perfect format compliance while their decision accuracy quietly collapses below random.

> **Status:** Early research release (2 models evaluated). We welcome model submissions. See [EVALUATION.md](EVALUATION.md).

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

Each tick, the agent receives sensor readings from 6 subsystems and must pick one of 10 actions. Anomalies shift across subsystems over 5 phases, creating an attention trap: agents that fixate on where problems *were* miss where problems *are now*.

## Leaderboard (Power Grid Scenario)

| Agent | DA | DA@40 | DA@last | DFG | Collapses? |
|-------|-----|-------|---------|-----|------------|
| Most-common action (baseline) | 54.8% | 45.5% | 70.0% | -24.5% | NO |
| Claude Haiku 4.5 | 33% | 58% | 22% | +3% | **YES** (-36pp) |
| GPT-5.4 (Codex) | 28% | 30% | 30% | +1% | NO |
| Random uniform (baseline) | 24.1% | 22.7% | 25.2% | -2.5% | NO |
| Majority / always hold_steady (baseline) | 24.9% | 26.0% | 21.5% | +4.5% | NO |

**DA** = Decision Accuracy. **DA@40** = DA in first 40 ticks. **DA@last** = DA in final 40 ticks. **DFG** = DA@40 minus DA@last (positive = degraded).

Claude Haiku starts at 58% but collapses to 22% (below random) by tick 200. GPT-5.4 stays flat at ~30%. Both maintain perfect format compliance the entire time. The collapse is invisible without behavioral metrics.

![Decision Accuracy Over Time](assets/da_over_time.png)

**[Add your model](EVALUATION.md)** -- submit a PR with your results.

## Quick Start

```bash
git clone https://github.com/Venkateshwar-PortoAI/coherencebench.git
cd coherencebench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Set up API keys
cp .env.example .env
# Edit .env with your API keys

# Estimate cost before running
python scripts/run_single.py --config configs/run_a_baseline.yaml --provider claude --seed 42 --dry-run

# Run the benchmark
python scripts/run_single.py --config configs/run_a_baseline.yaml --provider claude --seed 42
```

## Scenarios

4 scenarios across different domains. Each has 6 subsystems, 10 actions, and phase-shifted anomalies.

| Scenario | Domain | Split | Subsystems |
|----------|--------|-------|------------|
| `power_grid` | Electricity grid | Development | Load, Generation, Frequency, Voltage, Weather, Reserve |
| `hospital` | Hospital triage | Development | Vitals, Labs, Imaging, Medications, History, Capacity |
| `air_traffic_control` | ATC tower | Development | Radar, Weather, Runway, Comms, Traffic Flow, Systems |
| `network` | Network security SOC | **Evaluation** | Traffic, Auth, Endpoints, Firewall, Logs, Threats |

```bash
# Run a different scenario
python scripts/run_single.py --config configs/run_a_baseline.yaml --provider claude --seed 42 --scenario hospital
```

## Metrics

| Metric | What It Measures |
|--------|-----------------|
| **DA** (Decision Accuracy) | Did the agent choose a correct action? (primary metric) |
| **FC** (Factor Coverage) | How many of 6 subsystems were substantively analyzed? |
| **FI** (Fixation Index) | How much attention goes to a single subsystem? |
| **ADR** (Anomaly Mention Rate) | Did the agent discuss the anomalous subsystems? |

**DA is the primary metric.** High FC + low DA = invisible collapse. The agent writes about all subsystems but picks the wrong action.

## Experimental Conditions

| Run | Condition | What It Tests |
|-----|-----------|---------------|
| A | Baseline | Natural degradation over 200 ticks |
| B | Intervention | Do "analyze all factors" reminders help? |
| C | Context Reset | Does clearing context every 40 ticks help? |
| D | Checklist | Does a mandatory checklist prevent collapse? |
| E | Cross-Model | Same test across all providers |

## Train/Eval Split

- **`power_grid`**, **`hospital`**, **`air_traffic_control`** are the public development set. Use freely.
- **`network`** is the held-out evaluation set. Ground truth is stripped. Submit results for server-side scoring via [EVALUATION.md](EVALUATION.md).

## Supported Models

| Provider | Model | Via |
|----------|-------|----|
| **Claude** | Haiku 4.5 | Anthropic API |
| **GPT-4o** | GPT-4o | OpenAI API |
| **Gemini** | 1.5 Pro | Google AI API |
| **Llama** | 3.1 405B | Together API |
| **Claude CLI** | Sonnet 4 | Claude Code CLI |

### Adding Your Own Model

Implement `LLMProvider` in `src/providers/base.py`, register in `src/providers/__init__.py`, run with `--provider your-model`. See [CONTRIBUTING.md](CONTRIBUTING.md).

## Project Structure

```
coherencebench/
  configs/           # YAML run configurations (A-E)
  data/              # Pre-generated tick data (JSON, deterministic)
  results/           # Benchmark output (gitignored)
  scripts/           # CLI entry points
  src/
    analyzer.py      # Response parsing + metrics
    generator.py     # Tick data with planted anomalies
    metrics.py       # DA, FC, FI, ADR, IR
    runner.py        # Benchmark loop with context management
    visualizer.py    # Plotting
    providers/       # LLM API adapters
    scenarios/       # Scenario definitions (base + 4 domains)
  tests/             # 94 tests
```

## Limitations

- **Single-turn decisions.** No multi-step planning or stateful reasoning across ticks.
- **Synthetic environments.** Simplified simulations, not real-world monitoring.
- **Binary scoring.** No partial credit for reasonable but non-matching actions.
- **Limited model coverage.** 2 models so far. Community submissions welcome.

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
