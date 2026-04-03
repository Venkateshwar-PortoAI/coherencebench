# CoherenceBench

**Measuring Attention Collapse in Long-Running Autonomous Agents**

CoherenceBench is an open-source benchmark that measures how LLM-based autonomous agents degrade their decision quality over extended interactions. It places agents in a simulated power grid control room where they must continuously monitor 6 subsystems across 200 sequential decisions.

## Key Finding

Agents maintain near-perfect format compliance throughout a run -- they keep writing structured responses that mention all required subsystems -- while their actual decision accuracy quietly collapses. An agent can look like it is performing well (high format scores) while missing critical anomalies that shift to new subsystems over time. CoherenceBench separates "looks correct" from "is correct" by measuring behavioral metrics independently from format metrics.

## Leaderboard

| Model | Runs | DA | DA@40 | DA@last | DFG | Collapses? |
|-------|------|-----|-------|---------|-----|------------|
| Claude Haiku 4.5 | 3 | 33% | 58% | 22% | +3% | **YES** (-36pp) |
| GPT-5.4 (Codex) | 5 | 28% | 30% | 30% | +1% | NO |

**DA** = Decision Accuracy (% of ticks where the agent chose a correct action).
**DA@40** = DA in the first 40 ticks. **DA@last** = DA in the final window.
**DFG** = Directional Fixation Gap (positive = fixates on early-phase factors).
**Collapses?** = Does DA degrade by >15pp from start to end?

Claude Haiku starts strong at 58% but collapses to 22% by tick 200. GPT-5.4 stays flat at ~30% throughout. Both maintain perfect format compliance (FC = 1.00) the entire time -- the collapse is invisible without behavioral metrics.

**Add your model.** Run the benchmark and submit a PR with your results.

## Quick Start

```bash
# Clone and install
git clone https://github.com/Venkateshwar-PortoAI/coherencebench.git
cd coherencebench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Configure API keys
cp .env.example .env
# Edit .env with your API keys (Anthropic, OpenAI, Google, Together -- use whichever you need)

# Dry run (estimate tokens and cost, no API calls)
python scripts/run_single.py --config configs/run_a_baseline.yaml --provider claude --seed 42 --dry-run

# Run a single benchmark
python scripts/run_single.py --config configs/run_a_baseline.yaml --provider claude --seed 42

# Run the full benchmark suite (all configs x providers x seeds)
python scripts/run_benchmark.py --max-seeds 3

# Resume interrupted runs
python scripts/run_benchmark.py --resume
```

## Supported Models

| Provider | Model | Via |
|----------|-------|----|
| **Claude** | Sonnet 4 | Anthropic API |
| **Claude CLI** | Sonnet 4 | Claude Code CLI |
| **GPT-4o** | GPT-4o | OpenAI API (or Codex CLI) |
| **Gemini** | 1.5 Pro | Google AI API |
| **Llama** | 3.1 405B | Together API |

## How to Add Your Own Model

Implement the `LLMProvider` interface in `src/providers/base.py`:

```python
from src.providers.base import LLMProvider

class MyProvider(LLMProvider):
    def name(self) -> str:
        return "my-model"

    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        # Call your model's API and return the response text
        ...

    def reset(self) -> None:
        # Clear conversation state for a fresh session
        ...
```

Register it in `src/providers/__init__.py`, then run with `--provider my-model`.

See `CONTRIBUTING.md` for full details.

## Benchmark Design

The agent operates as a power grid control room operator receiving updates from 6 subsystems every tick:

| Subsystem | What It Tracks |
|-----------|---------------|
| **Load** | Consumer electricity demand across 3 zones |
| **Generation** | Power plant output and fault status |
| **Frequency** | Grid stability (nominal 50.0 Hz) |
| **Voltage** | Transmission line voltage levels |
| **Weather** | Wind speed, solar conditions for renewables |
| **Reserve** | Battery storage, gas turbine, spinning reserve |

Anomalies are injected on a schedule that shifts over time: early anomalies concentrate in Load and Generation, while late anomalies shift to Weather and Reserve. This creates an attention trap -- agents that fixate on where problems *were* will miss where problems *are now*.

### Experimental Conditions

| Run | Condition | Description |
|-----|-----------|-------------|
| A | Baseline | 200 ticks, continuous session, no mitigation |
| B | Intervention | "Analyze all factors" reminders at ticks 50, 100, 150 |
| C | Context Reset | Context cleared every 40 ticks with state re-injection |
| D | Checklist | Mandatory 6-factor checklist appended to every tick |
| E | Cross-Model | Run A across all providers |

## Metrics

| Metric | What It Measures | Type |
|--------|-----------------|------|
| **FC** (Factor Coverage) | Fraction of 6 subsystems substantively analyzed | Format quality |
| **FI** (Fixation Index) | Fraction of tokens devoted to the top subsystem | Attention balance |
| **DA** (Decision Accuracy) | Whether the correct action was chosen (0 or 1) | Decision quality |
| **ADR** (Anomaly Detection Rate) | Fraction of anomalous subsystems actually detected | Behavioral quality |
| **IR** (Intervention Recovery) | Ticks before coverage drops post-intervention | Mitigation durability |

**DA is the primary metric.** It measures whether the agent makes the correct decision given current conditions, not just whether it writes about all factors. A model with high FC but low DA is experiencing invisible collapse.

## Interpreting Results

Results are saved as JSON in `results/`. Each run produces per-tick metrics. Key things to look for:

- **FC stays high while ADR drops**: The agent is writing about all subsystems but not actually detecting the anomalies in them. This is the core "coherence collapse" signal.
- **FI increases over time**: The agent is devoting more and more of its response to a single subsystem, indicating attention narrowing.
- **DA drops in late ticks**: Decision quality degrades as the anomaly distribution shifts away from where the agent learned to focus.
- **IR is short**: Intervention reminders produce only temporary recovery before the agent reverts to its fixation pattern.

## Project Structure

```
coherencebench/
  configs/           # YAML run configurations (A-E)
  results/           # Benchmark output (gitignored, kept locally)
  scripts/           # CLI entry points
  src/
    analyzer.py      # Response parsing + metric computation
    generator.py     # Synthetic tick data with planted anomalies
    metrics.py       # FC, FI, DA, ADR, IR implementations
    runner.py        # Benchmark runner with context management
    scenario.py      # Power grid scenario definition
    visualizer.py    # Matplotlib/seaborn plotting
    providers/       # LLM API adapters
  tests/             # Pytest test suite
```

## Running Tests

```bash
python -m pytest tests/ -v
```

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for how to add scenarios, providers, and submit changes.

## Citation

```bibtex
@software{coherencebench2026,
  title={CoherenceBench: Measuring Attention Collapse in Long-Running Autonomous Agents},
  author={PranaAlpha Labs},
  year={2026},
  url={https://github.com/Venkateshwar-PortoAI/coherencebench}
}
```

## License

[MIT](LICENSE)

---

Built by [PranaAlpha Labs](https://pranaalpha.com)
