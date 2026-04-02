# CoherenceBench

**Measuring Attention Collapse in Long-Running Autonomous Agents**

CoherenceBench is a benchmark that measures how LLM-based autonomous agents degrade their multi-factor analysis quality over extended interactions. In a controlled power grid control room scenario, the benchmark tracks whether agents continue to analyze all 6 required subsystems across 200 sequential decision ticks, or whether they progressively narrow their attention to a subset of factors.

## Quick Start

```bash
# Clone and install
git clone https://github.com/pranaalpha/coherencebench.git
cd coherencebench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Copy env and add API keys
cp .env.example .env
# Edit .env with your Anthropic/OpenAI/Google/Together API keys

# Dry run (estimate tokens and cost without API calls)
python scripts/run_single.py --config configs/run_a_baseline.yaml --provider claude --seed 42 --dry-run

# Run a single benchmark
python scripts/run_single.py --config configs/run_a_baseline.yaml --provider claude --seed 42

# Run full benchmark suite (all configs x providers x seeds)
python scripts/run_benchmark.py --max-seeds 3

# Resume interrupted runs
python scripts/run_benchmark.py --resume

# Generate paper figures from results
python scripts/generate_paper_figures.py
```

## Benchmark Design

### Scenario

The agent operates as a power grid control room operator receiving updates from 6 subsystems every tick:

| Factor | Description |
|--------|-------------|
| **Load** | Consumer electricity demand across 3 zones |
| **Generation** | Power plant output and fault status |
| **Frequency** | Grid stability (nominal 50.0 Hz) |
| **Voltage** | Transmission line voltage levels |
| **Weather** | Wind speed, solar conditions for renewables |
| **Reserve** | Battery storage, gas turbine, spinning reserve |

The agent must analyze all 6 factors and recommend one of 10 possible actions.

### Anomaly Injection

Anomalies are planted with phase-dependent probabilities that shift over time:
- **Early phase (ticks 1-80):** Anomalies concentrate in Load and Generation
- **Late phase (ticks 121-200):** Anomalies shift to Weather and Reserve

This creates an "attention trap" -- an agent that fixates on where anomalies *were* rather than where they *are* will miss critical late-phase events.

### Experimental Conditions

| Run | Condition | Description |
|-----|-----------|-------------|
| A | Baseline | 200 ticks, continuous session, no mitigation |
| B | Intervention | Explicit "analyze all factors" reminders at ticks 50, 100, 150 |
| C | Context Reset | Context cleared every 40 ticks with state re-injection |
| D | Checklist | Mandatory 6-factor checklist appended to every tick |
| E | Cross-Model | Run A across all 4 providers |

## Metrics

| Metric | Full Name | Definition | Role |
|--------|-----------|------------|------|
| **FC** | Factor Coverage | Fraction of 6 factors substantively analyzed | Format quality |
| **FI** | Fixation Index | Fraction of tokens devoted to top factor | Attention balance |
| **DA** | Decision Accuracy | 1 if correct action chosen, 0 otherwise | Decision quality |
| **ADR** | Anomaly Detection Rate | Fraction of anomalous factors detected | Behavior quality (primary) |
| **IR** | Intervention Recovery | Ticks before FC drops post-intervention | Mitigation durability |

ADR is the **primary metric**: it measures whether the agent behaviorally responds to anomalies, not merely whether it formats its response correctly.

## Supported Models

- **Claude** (Anthropic) -- Sonnet 4
- **GPT-4o** (OpenAI)
- **Gemini 1.5 Pro** (Google)
- **Llama 3.1 405B** (Meta, via Together)

## Project Structure

```
coherencebench/
  configs/           # YAML run configurations (A-E)
  paper/             # LaTeX paper skeleton + figures
  results/           # Benchmark output (gitignored)
  scripts/           # CLI scripts (run_single, run_benchmark, generate_paper_figures)
  src/
    analyzer.py      # Response parsing + metric computation
    generator.py     # Synthetic tick data with planted anomalies
    metrics.py       # FC, FI, DA, ADR, IR implementations
    runner.py        # Benchmark runner with context management
    scenario.py      # Power grid scenario definition
    visualizer.py    # Paper-quality matplotlib/seaborn plots
    providers/       # LLM API adapters (Anthropic, OpenAI, Google, Together)
  tests/             # Pytest test suite
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Citation

```bibtex
@article{jambula2026coherencebench,
  title={CoherenceBench: Measuring Attention Collapse in Long-Running Autonomous Agents},
  author={Jambula, Venkateshwar and Krishnakumar, Anitha},
  journal={arXiv preprint},
  year={2026}
}
```

## License

MIT
