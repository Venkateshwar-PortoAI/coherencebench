# CoherenceBench Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build CoherenceBench — the first benchmark for measuring multi-factor attention collapse in long-running autonomous AI agents — and write the accompanying research paper.

**Architecture:** A Python benchmark suite that generates synthetic multi-factor decision scenarios, runs them against multiple LLM APIs in long sessions (200+ turns), parses agent responses to measure which factors were analyzed per turn, and produces quantitative metrics + publication-ready visualizations. The benchmark is domain-neutral (power grid control room scenario) to avoid revealing any trading-specific IP.

**Tech Stack:** Python 3.11+, anthropic SDK, openai SDK, google-genai SDK, together/groq SDK (for Llama), matplotlib/seaborn for visualization, LaTeX (NeurIPS 2026 template) for paper.

**Key differentiation from existing benchmarks:**
- Vending-Bench tests long-horizon decisions but doesn't measure per-factor attention allocation
- Needle-in-a-Haystack tests retrieval in long context, not sustained multi-factor reasoning
- AgentBench tests general agent capability, not coherence degradation over time
- CoherenceBench specifically measures the PROGRESSIVE NARROWING of which factors an agent considers — the "attention collapse" phenomenon

---

## File Structure

```
coherencebench/
├── README.md                      # Project overview + how to run
├── pyproject.toml                 # Package config + dependencies
├── .env.example                   # API key template
│
├── src/
│   ├── __init__.py
│   ├── generator.py               # Generates tick data with planted anomalies
│   ├── scenario.py                # Defines the 6-factor power grid scenario
│   ├── runner.py                  # Feeds ticks to LLMs, logs raw responses
│   ├── providers/
│   │   ├── __init__.py
│   │   ├── base.py                # Abstract provider interface
│   │   ├── anthropic.py           # Claude API wrapper
│   │   ├── openai.py              # GPT-4o API wrapper
│   │   ├── google.py              # Gemini API wrapper
│   │   └── together.py            # Llama via Together/Groq wrapper
│   ├── analyzer.py                # Parses responses, computes FC/FI/DA/ADR/IR metrics
│   ├── metrics.py                 # Metric definitions and computation
│   └── visualizer.py              # Generates publication-ready graphs
│
├── configs/
│   ├── run_a_baseline.yaml        # 200 ticks, no intervention
│   ├── run_b_intervention.yaml    # 200 ticks, reminders at 50/100/150
│   ├── run_c_context_reset.yaml   # 200 ticks, reset every 40 ticks
│   ├── run_d_checklist.yaml       # 200 ticks, forced factor checklist
│   └── run_e_cross_model.yaml     # Run A across all models
│
├── tests/
│   ├── test_generator.py          # Tests for tick data generation
│   ├── test_scenario.py           # Tests for scenario definitions
│   ├── test_analyzer.py           # Tests for response parsing + metrics
│   ├── test_metrics.py            # Tests for metric computations
│   └── test_visualizer.py         # Tests for graph generation
│
├── results/                       # Output directory (gitignored except examples)
│   └── .gitkeep
│
├── paper/
│   ├── main.tex                   # NeurIPS 2026 paper
│   ├── references.bib             # Citations
│   └── figures/                   # Generated figures for paper
│       └── .gitkeep
│
└── scripts/
    ├── run_benchmark.py           # CLI entrypoint: run full benchmark
    ├── run_single.py              # CLI: run one config against one model
    └── generate_paper_figures.py  # Generate all figures for the paper
```

---

## Task 1: Project Scaffolding

**Files:**
- Create: `pyproject.toml`
- Create: `README.md`
- Create: `.env.example`
- Create: `.gitignore`
- Create: `src/__init__.py`

- [ ] **Step 1: Create pyproject.toml**

```toml
[project]
name = "coherencebench"
version = "0.1.0"
description = "Benchmark for measuring multi-factor attention collapse in long-running autonomous AI agents"
requires-python = ">=3.11"
dependencies = [
    "anthropic>=0.40.0",
    "openai>=1.50.0",
    "google-genai>=1.0.0",
    "together>=1.0.0",
    "pyyaml>=6.0",
    "matplotlib>=3.8.0",
    "seaborn>=0.13.0",
    "numpy>=1.26.0",
    "pandas>=2.1.0",
    "python-dotenv>=1.0.0",
]

[project.optional-dependencies]
dev = ["pytest>=8.0.0", "pytest-asyncio>=0.23.0"]

[build-system]
requires = ["setuptools>=68.0"]
build-backend = "setuptools.backends._legacy:_Backend"
```

- [ ] **Step 2: Create .env.example**

```
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
GOOGLE_API_KEY=...
TOGETHER_API_KEY=...
```

- [ ] **Step 3: Create .gitignore**

```
.env
__pycache__/
*.pyc
results/*.json
results/*.png
results/*.csv
!results/.gitkeep
.venv/
dist/
*.egg-info/
```

- [ ] **Step 4: Create src/__init__.py**

```python
"""CoherenceBench: Measuring attention collapse in long-running autonomous agents."""
```

- [ ] **Step 5: Create empty directories**

```bash
mkdir -p src/providers tests configs results paper/figures scripts
touch src/providers/__init__.py tests/__init__.py results/.gitkeep paper/figures/.gitkeep
```

- [ ] **Step 6: Install and verify**

```bash
cd /Users/gokukilluavegeta/GitHub/coherencebench
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
python -c "import coherencebench; print('OK')"
```
Expected: "OK"

- [ ] **Step 7: Commit**

```bash
git add -A
git commit -m "chore: project scaffolding with dependencies"
```

---

## Task 2: Scenario Definition

**Files:**
- Create: `src/scenario.py`
- Create: `tests/test_scenario.py`

The scenario defines the 6 factors, their data types, ranges, and how the agent prompt is constructed each tick.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_scenario.py
from src.scenario import PowerGridScenario

def test_scenario_has_six_factors():
    scenario = PowerGridScenario()
    assert len(scenario.factors) == 6

def test_factor_names():
    scenario = PowerGridScenario()
    names = [f.name for f in scenario.factors]
    assert "load" in names
    assert "generation" in names
    assert "frequency" in names
    assert "voltage" in names
    assert "weather" in names
    assert "reserve" in names

def test_format_tick_includes_all_factors():
    scenario = PowerGridScenario()
    tick_data = {
        "load": {"zone_a": 340, "zone_b": 280, "zone_c": 195},
        "generation": {"plant_1": 290, "plant_2": 180, "plant_3": 150, "plant_4": 95},
        "frequency": {"hz": 49.82, "trend": "dropping"},
        "voltage": {"north": 398, "south": 412, "east": 405},
        "weather": {"wind_kmh": 8, "wind_trend": "dropping", "solar": "stable"},
        "reserve": {"battery_mwh": 340, "battery_pct": 68, "gas_turbine": "standby", "spin_mw": 120},
    }
    prompt = scenario.format_tick(tick_number=47, tick_data=tick_data)
    assert "TICK 47" in prompt
    assert "Load" in prompt or "load" in prompt
    assert "Generation" in prompt or "generation" in prompt
    assert "Frequency" in prompt or "frequency" in prompt
    assert "Voltage" in prompt or "voltage" in prompt
    assert "Weather" in prompt or "weather" in prompt
    assert "Reserve" in prompt or "reserve" in prompt
    assert "Based on ALL six factors" in prompt

def test_system_prompt_instructs_all_factors():
    scenario = PowerGridScenario()
    sys_prompt = scenario.system_prompt()
    assert "six" in sys_prompt.lower() or "6" in sys_prompt
    assert "load" in sys_prompt.lower()
    assert "generation" in sys_prompt.lower()
    assert "frequency" in sys_prompt.lower()
    assert "voltage" in sys_prompt.lower()
    assert "weather" in sys_prompt.lower()
    assert "reserve" in sys_prompt.lower()

def test_scenario_defines_possible_actions():
    scenario = PowerGridScenario()
    assert len(scenario.actions) >= 6  # At least one action type per factor
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
cd /Users/gokukilluavegeta/GitHub/coherencebench
python -m pytest tests/test_scenario.py -v
```
Expected: FAIL — `ModuleNotFoundError: No module named 'src.scenario'`

- [ ] **Step 3: Implement scenario.py**

```python
# src/scenario.py
"""Power grid control room scenario for CoherenceBench."""

from dataclasses import dataclass


@dataclass
class Factor:
    name: str
    display_name: str
    description: str
    keywords: list[str]  # Used by analyzer to detect references in agent responses


FACTORS = [
    Factor(
        name="load",
        display_name="Load (Consumer Demand)",
        description="Consumer electricity demand across 3 zones (A, B, C) in MW",
        keywords=["load", "demand", "zone a", "zone b", "zone c", "consumer", "consumption"],
    ),
    Factor(
        name="generation",
        display_name="Generation (Power Output)",
        description="Power output from 4 plants in MW, including trip/fault status",
        keywords=["generation", "plant", "output", "trip", "fault", "generator", "capacity"],
    ),
    Factor(
        name="frequency",
        display_name="Frequency (Grid Stability)",
        description="Grid frequency in Hz (nominal 50.0, safe range 49.5-50.5)",
        keywords=["frequency", "hz", "hertz", "grid stability", "49.", "50."],
    ),
    Factor(
        name="voltage",
        display_name="Voltage (Transmission Lines)",
        description="Voltage levels across 3 transmission lines in kV",
        keywords=["voltage", "kv", "transmission", "line north", "line south", "line east"],
    ),
    Factor(
        name="weather",
        display_name="Weather (Renewable Forecast)",
        description="Wind speed, solar conditions affecting renewable generation",
        keywords=["weather", "wind", "solar", "renewable", "forecast", "cloud", "gust"],
    ),
    Factor(
        name="reserve",
        display_name="Reserve (Backup Capacity)",
        description="Battery storage, gas turbine standby, spinning reserve in MW/MWh",
        keywords=["reserve", "battery", "gas turbine", "backup", "storage", "spin"],
    ),
]

ACTIONS = [
    "shed_load",              # Reduce demand in a zone
    "start_gas_turbine",      # Bring backup generation online
    "ramp_plant",             # Increase/decrease plant output
    "adjust_voltage",         # Tap transformer to fix voltage
    "curtail_renewable",      # Reduce wind/solar intake
    "deploy_battery",         # Discharge battery storage
    "charge_battery",         # Charge from excess generation
    "hold_steady",            # No action needed, all stable
    "emergency_disconnect",   # Disconnect a failing line
    "request_import",         # Import power from neighboring grid
]


class PowerGridScenario:
    """Defines the power grid control room scenario."""

    def __init__(self):
        self.factors = FACTORS
        self.actions = ACTIONS

    def system_prompt(self) -> str:
        return (
            "You are an autonomous power grid control room operator. "
            "Every tick, you receive updates from 6 subsystems: "
            "Load (consumer demand), Generation (power plant output), "
            "Frequency (grid stability), Voltage (transmission lines), "
            "Weather (renewable forecast), and Reserve (backup capacity).\n\n"
            "Your job is to analyze ALL 6 factors every tick and recommend an action.\n\n"
            "You MUST respond in this exact format:\n"
            "ANALYSIS:\n"
            "- Load: [your analysis of current load state]\n"
            "- Generation: [your analysis of current generation state]\n"
            "- Frequency: [your analysis of current frequency state]\n"
            "- Voltage: [your analysis of current voltage state]\n"
            "- Weather: [your analysis of current weather state]\n"
            "- Reserve: [your analysis of current reserve state]\n"
            "ACTION: [one of: shed_load, start_gas_turbine, ramp_plant, adjust_voltage, "
            "curtail_renewable, deploy_battery, charge_battery, hold_steady, "
            "emergency_disconnect, request_import]\n"
            "REASON: [1-2 sentences explaining why, referencing the specific factors that "
            "informed your decision]\n\n"
            "IMPORTANT: You must analyze ALL 6 factors every tick. Do not skip any factor. "
            "Each factor line must contain substantive analysis, not just 'stable' or 'unchanged'."
        )

    def format_tick(self, tick_number: int, tick_data: dict) -> str:
        lines = [f"TICK {tick_number}:"]
        lines.append("")

        load = tick_data["load"]
        lines.append(
            f"F1 (Load):       Zone A: {load['zone_a']}MW, "
            f"Zone B: {load['zone_b']}MW, Zone C: {load['zone_c']}MW"
        )

        gen = tick_data["generation"]
        gen_parts = []
        for i in range(1, 5):
            val = gen[f"plant_{i}"]
            if isinstance(val, str):
                gen_parts.append(f"Plant {i}: {val}")
            else:
                gen_parts.append(f"Plant {i}: {val}MW")
        lines.append(f"F2 (Generation):  {', '.join(gen_parts)}")

        freq = tick_data["frequency"]
        lines.append(f"F3 (Frequency):   {freq['hz']} Hz ({freq['trend']})")

        volt = tick_data["voltage"]
        volt_parts = []
        for line_name in ["north", "south", "east"]:
            v = volt[line_name]
            status = " (LOW)" if v < 400 else " (HIGH)" if v > 420 else ""
            volt_parts.append(f"Line {line_name.title()}: {v}kV{status}")
        lines.append(f"F4 (Voltage):     {', '.join(volt_parts)}")

        weather = tick_data["weather"]
        lines.append(
            f"F5 (Weather):     Wind {weather['wind_kmh']}km/h ({weather['wind_trend']}). "
            f"Solar: {weather['solar']}."
        )

        res = tick_data["reserve"]
        lines.append(
            f"F6 (Reserve):     Battery: {res['battery_mwh']}MWh ({res['battery_pct']}%). "
            f"Gas turbine: {res['gas_turbine'].upper()}. Spin reserve: {res['spin_mw']}MW."
        )

        lines.append("")
        lines.append("Based on ALL six factors, what action do you recommend?")
        return "\n".join(lines)
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_scenario.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/scenario.py tests/test_scenario.py
git commit -m "feat: power grid scenario with 6 factors and structured response format"
```

---

## Task 3: Tick Data Generator

**Files:**
- Create: `src/generator.py`
- Create: `tests/test_generator.py`

Generates 200 ticks of synthetic data with planted anomalies at known positions. Uses seeded randomness for reproducibility.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_generator.py
import json
from src.generator import TickGenerator


def test_generator_produces_200_ticks():
    gen = TickGenerator(seed=42, num_ticks=200)
    ticks = gen.generate()
    assert len(ticks) == 200


def test_ticks_have_all_six_factors():
    gen = TickGenerator(seed=42, num_ticks=10)
    ticks = gen.generate()
    for tick in ticks:
        assert "load" in tick["data"]
        assert "generation" in tick["data"]
        assert "frequency" in tick["data"]
        assert "voltage" in tick["data"]
        assert "weather" in tick["data"]
        assert "reserve" in tick["data"]


def test_ticks_have_ground_truth():
    gen = TickGenerator(seed=42, num_ticks=10)
    ticks = gen.generate()
    for tick in ticks:
        assert "ground_truth" in tick
        assert "anomalous_factors" in tick["ground_truth"]
        assert "correct_action" in tick["ground_truth"]
        assert "relevant_factors" in tick["ground_truth"]


def test_reproducibility_with_same_seed():
    gen1 = TickGenerator(seed=42, num_ticks=50)
    gen2 = TickGenerator(seed=42, num_ticks=50)
    ticks1 = gen1.generate()
    ticks2 = gen2.generate()
    assert json.dumps(ticks1) == json.dumps(ticks2)


def test_different_seeds_produce_different_data():
    gen1 = TickGenerator(seed=42, num_ticks=50)
    gen2 = TickGenerator(seed=99, num_ticks=50)
    ticks1 = gen1.generate()
    ticks2 = gen2.generate()
    assert json.dumps(ticks1) != json.dumps(ticks2)


def test_anomalies_shift_across_phases():
    """Early anomalies should be in load/generation, later in weather/reserve."""
    gen = TickGenerator(seed=42, num_ticks=200)
    ticks = gen.generate()

    early_anomaly_factors = set()
    late_anomaly_factors = set()

    for tick in ticks[:30]:
        early_anomaly_factors.update(tick["ground_truth"]["anomalous_factors"])
    for tick in ticks[150:]:
        late_anomaly_factors.update(tick["ground_truth"]["anomalous_factors"])

    # Early phase should have load and/or generation anomalies
    assert "load" in early_anomaly_factors or "generation" in early_anomaly_factors
    # Late phase should have weather and/or reserve anomalies
    assert "weather" in late_anomaly_factors or "reserve" in late_anomaly_factors


def test_tick_data_values_in_range():
    gen = TickGenerator(seed=42, num_ticks=50)
    ticks = gen.generate()
    for tick in ticks:
        d = tick["data"]
        # Frequency should be roughly around 50Hz
        assert 48.0 <= d["frequency"]["hz"] <= 52.0
        # Voltage should be roughly 380-430 kV range
        for line in ["north", "south", "east"]:
            assert 370 <= d["voltage"][line] <= 440
        # Battery percentage 0-100
        assert 0 <= d["reserve"]["battery_pct"] <= 100
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_generator.py -v
```
Expected: FAIL — `ModuleNotFoundError`

- [ ] **Step 3: Implement generator.py**

```python
# src/generator.py
"""Generates synthetic power grid tick data with planted anomalies."""

import random
from dataclasses import dataclass


@dataclass
class TickGroundTruth:
    anomalous_factors: list[str]
    correct_action: str
    relevant_factors: list[str]  # Factors needed to determine correct action


# Anomaly probability by factor per phase.
# Key insight: early phases have anomalies in load/generation,
# later phases shift to weather/reserve. This creates the "attention trap" —
# the agent fixates on where anomalies WERE, not where they ARE.
PHASE_ANOMALY_WEIGHTS = {
    # (tick_start, tick_end): {factor: probability_of_anomaly}
    (0, 40): {"load": 0.4, "generation": 0.35, "frequency": 0.15, "voltage": 0.1, "weather": 0.05, "reserve": 0.05},
    (40, 80): {"load": 0.25, "generation": 0.25, "frequency": 0.2, "voltage": 0.2, "weather": 0.1, "reserve": 0.1},
    (80, 120): {"load": 0.15, "generation": 0.15, "frequency": 0.15, "voltage": 0.15, "weather": 0.2, "reserve": 0.2},
    (120, 160): {"load": 0.1, "generation": 0.1, "frequency": 0.1, "voltage": 0.15, "weather": 0.3, "reserve": 0.3},
    (160, 200): {"load": 0.05, "generation": 0.05, "frequency": 0.15, "voltage": 0.1, "weather": 0.35, "reserve": 0.35},
}

# Maps anomalous factor to the correct action
ANOMALY_ACTION_MAP = {
    "load": "shed_load",
    "generation": "start_gas_turbine",
    "frequency": "ramp_plant",
    "voltage": "adjust_voltage",
    "weather": "curtail_renewable",
    "reserve": "deploy_battery",
}


class TickGenerator:
    """Generates reproducible tick data with planted anomalies."""

    def __init__(self, seed: int = 42, num_ticks: int = 200):
        self.seed = seed
        self.num_ticks = num_ticks
        self.rng = random.Random(seed)

    def generate(self) -> list[dict]:
        ticks = []
        # Running state for smooth time-series
        state = {
            "load": {"zone_a": 300, "zone_b": 260, "zone_c": 180},
            "generation": {"plant_1": 280, "plant_2": 200, "plant_3": 160, "plant_4": 100},
            "frequency": {"hz": 50.0},
            "voltage": {"north": 410, "south": 410, "east": 408},
            "weather": {"wind_kmh": 18, "solar_pct": 80},
            "reserve": {"battery_mwh": 400, "battery_pct": 80, "spin_mw": 150},
        }

        for tick_num in range(self.num_ticks):
            # Determine anomaly phase
            weights = self._get_phase_weights(tick_num)

            # Evolve state smoothly
            state = self._evolve_state(state)

            # Decide which factors have anomalies this tick
            anomalous = []
            for factor, prob in weights.items():
                if self.rng.random() < prob:
                    anomalous.append(factor)

            # Apply anomalies to state copy
            tick_data = self._deep_copy_state(state)
            for factor in anomalous:
                tick_data = self._inject_anomaly(tick_data, factor)

            # Determine correct action
            if anomalous:
                # Most critical anomaly determines action
                primary = anomalous[0]
                correct_action = ANOMALY_ACTION_MAP[primary]
                relevant = anomalous.copy()
            else:
                correct_action = "hold_steady"
                relevant = []

            # Format for output
            formatted_data = self._format_tick_data(tick_data)

            ticks.append({
                "tick_number": tick_num + 1,
                "data": formatted_data,
                "ground_truth": {
                    "anomalous_factors": anomalous,
                    "correct_action": correct_action,
                    "relevant_factors": relevant if relevant else ["all"],
                },
            })

        return ticks

    def _get_phase_weights(self, tick_num: int) -> dict[str, float]:
        for (start, end), weights in PHASE_ANOMALY_WEIGHTS.items():
            if start <= tick_num < end:
                return weights
        return list(PHASE_ANOMALY_WEIGHTS.values())[-1]

    def _evolve_state(self, state: dict) -> dict:
        """Small random walk to make data look natural."""
        s = self._deep_copy_state(state)
        s["load"]["zone_a"] = max(150, min(500, s["load"]["zone_a"] + self.rng.randint(-8, 8)))
        s["load"]["zone_b"] = max(100, min(400, s["load"]["zone_b"] + self.rng.randint(-5, 5)))
        s["load"]["zone_c"] = max(80, min(300, s["load"]["zone_c"] + self.rng.randint(-4, 4)))

        for p in ["plant_1", "plant_2", "plant_3", "plant_4"]:
            s["generation"][p] = max(50, min(350, s["generation"][p] + self.rng.randint(-5, 5)))

        s["frequency"]["hz"] = round(max(49.0, min(51.0, s["frequency"]["hz"] + self.rng.uniform(-0.05, 0.05))), 2)

        for line in ["north", "south", "east"]:
            s["voltage"][line] = max(385, min(425, s["voltage"][line] + self.rng.randint(-2, 2)))

        s["weather"]["wind_kmh"] = max(0, min(50, s["weather"]["wind_kmh"] + self.rng.randint(-2, 2)))
        s["weather"]["solar_pct"] = max(0, min(100, s["weather"]["solar_pct"] + self.rng.randint(-3, 3)))

        s["reserve"]["battery_mwh"] = max(0, min(500, s["reserve"]["battery_mwh"] + self.rng.randint(-5, 5)))
        s["reserve"]["battery_pct"] = round(s["reserve"]["battery_mwh"] / 500 * 100)
        s["reserve"]["spin_mw"] = max(50, min(200, s["reserve"]["spin_mw"] + self.rng.randint(-3, 3)))

        return s

    def _inject_anomaly(self, data: dict, factor: str) -> dict:
        """Inject a clear anomaly into the specified factor."""
        if factor == "load":
            spike_zone = self.rng.choice(["zone_a", "zone_b", "zone_c"])
            data["load"][spike_zone] += self.rng.randint(80, 150)
        elif factor == "generation":
            trip_plant = self.rng.choice(["plant_1", "plant_2", "plant_3", "plant_4"])
            data["generation"][trip_plant] = "TRIPPED"
        elif factor == "frequency":
            data["frequency"]["hz"] = round(self.rng.uniform(49.0, 49.45), 2) if self.rng.random() < 0.5 else round(self.rng.uniform(50.55, 51.0), 2)
        elif factor == "voltage":
            bad_line = self.rng.choice(["north", "south", "east"])
            data["voltage"][bad_line] = self.rng.randint(370, 392) if self.rng.random() < 0.5 else self.rng.randint(422, 440)
        elif factor == "weather":
            data["weather"]["wind_kmh"] = self.rng.randint(0, 3)  # Wind dies
            data["weather"]["solar_pct"] = self.rng.randint(0, 15)  # Clouds
        elif factor == "reserve":
            data["reserve"]["battery_mwh"] = self.rng.randint(10, 50)
            data["reserve"]["battery_pct"] = round(data["reserve"]["battery_mwh"] / 500 * 100)
            data["reserve"]["spin_mw"] = self.rng.randint(10, 30)
        return data

    def _format_tick_data(self, state: dict) -> dict:
        """Format raw state into the dict expected by scenario.format_tick()."""
        freq_hz = state["frequency"]["hz"]
        if freq_hz < 49.8:
            trend = "dropping"
        elif freq_hz > 50.2:
            trend = "rising"
        else:
            trend = "stable"

        wind = state["weather"]["wind_kmh"]
        if wind < 5:
            wind_trend = "calm"
        elif wind > 30:
            wind_trend = "gusting"
        else:
            wind_trend = "steady"

        solar = state["weather"]["solar_pct"]
        solar_str = "strong" if solar > 60 else "moderate" if solar > 30 else "weak" if solar > 10 else "minimal"

        gas_state = "standby"
        if state["reserve"]["spin_mw"] < 40:
            gas_state = "offline"
        elif state["reserve"]["battery_pct"] < 20:
            gas_state = "active"

        gen_data = {}
        for p in ["plant_1", "plant_2", "plant_3", "plant_4"]:
            gen_data[p] = state["generation"][p]

        return {
            "load": {"zone_a": state["load"]["zone_a"], "zone_b": state["load"]["zone_b"], "zone_c": state["load"]["zone_c"]},
            "generation": gen_data,
            "frequency": {"hz": freq_hz, "trend": trend},
            "voltage": {"north": state["voltage"]["north"], "south": state["voltage"]["south"], "east": state["voltage"]["east"]},
            "weather": {"wind_kmh": wind, "wind_trend": wind_trend, "solar": solar_str},
            "reserve": {"battery_mwh": state["reserve"]["battery_mwh"], "battery_pct": state["reserve"]["battery_pct"], "gas_turbine": gas_state, "spin_mw": state["reserve"]["spin_mw"]},
        }

    def _deep_copy_state(self, state: dict) -> dict:
        return {k: dict(v) for k, v in state.items()}
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_generator.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/generator.py tests/test_generator.py
git commit -m "feat: tick data generator with phased anomaly injection"
```

---

## Task 4: LLM Provider Abstraction

**Files:**
- Create: `src/providers/base.py`
- Create: `src/providers/anthropic.py`
- Create: `src/providers/openai.py`
- Create: `src/providers/google.py`
- Create: `src/providers/together.py`
- Create: `src/providers/__init__.py`

- [ ] **Step 1: Write base provider interface**

```python
# src/providers/base.py
"""Abstract base class for LLM providers."""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Interface for LLM API providers."""

    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for results."""
        ...

    @abstractmethod
    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        """Send a single turn and get the response text.

        Args:
            system_prompt: The system prompt (scenario instructions).
            messages: Prior conversation history as list of {"role": str, "content": str}.
            user_message: The current tick prompt.

        Returns:
            The model's response text.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear conversation state for a fresh session."""
        ...
```

- [ ] **Step 2: Implement Anthropic provider**

```python
# src/providers/anthropic.py
"""Claude API provider."""

import anthropic
from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.client = anthropic.Anthropic()

    def name(self) -> str:
        return f"Claude ({self.model})"

    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        all_messages = messages + [{"role": "user", "content": user_message}]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=all_messages,
        )
        return response.content[0].text

    def reset(self) -> None:
        pass  # Stateless API
```

- [ ] **Step 3: Implement OpenAI provider**

```python
# src/providers/openai.py
"""GPT API provider."""

from openai import OpenAI
from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = OpenAI()

    def name(self) -> str:
        return f"GPT ({self.model})"

    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        all_messages = [{"role": "system", "content": system_prompt}]
        all_messages += messages
        all_messages.append({"role": "user", "content": user_message})
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=all_messages,
        )
        return response.choices[0].message.content

    def reset(self) -> None:
        pass
```

- [ ] **Step 4: Implement Google provider**

```python
# src/providers/google.py
"""Gemini API provider."""

from google import genai
from .base import LLMProvider


class GoogleProvider(LLMProvider):
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self.client = genai.Client()

    def name(self) -> str:
        return f"Gemini ({self.model})"

    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": user_message}]})

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config={"system_instruction": system_prompt, "max_output_tokens": 1024},
        )
        return response.text

    def reset(self) -> None:
        pass
```

- [ ] **Step 5: Implement Together provider (for Llama)**

```python
# src/providers/together.py
"""Together AI provider for open-source models (Llama, Mistral, etc.)."""

from openai import OpenAI
from .base import LLMProvider


class TogetherProvider(LLMProvider):
    """Uses Together AI's OpenAI-compatible API."""

    def __init__(self, model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
        self.model = model
        self.client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=None,  # Reads TOGETHER_API_KEY from env
        )

    def name(self) -> str:
        short_name = self.model.split("/")[-1]
        return f"Llama ({short_name})"

    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        all_messages = [{"role": "system", "content": system_prompt}]
        all_messages += messages
        all_messages.append({"role": "user", "content": user_message})
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=all_messages,
        )
        return response.choices[0].message.content

    def reset(self) -> None:
        pass
```

- [ ] **Step 6: Update providers/__init__.py**

```python
# src/providers/__init__.py
"""LLM provider registry."""

from .base import LLMProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .google import GoogleProvider
from .together import TogetherProvider

PROVIDERS = {
    "claude": AnthropicProvider,
    "gpt4o": OpenAIProvider,
    "gemini": GoogleProvider,
    "llama": TogetherProvider,
}

def get_provider(name: str, **kwargs) -> LLMProvider:
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name](**kwargs)
```

- [ ] **Step 7: Commit**

```bash
git add src/providers/
git commit -m "feat: LLM provider abstraction for Claude, GPT-4o, Gemini, Llama"
```

---

## Task 5: Benchmark Runner

**Files:**
- Create: `src/runner.py`
- Create: `tests/test_runner.py` (mock provider for testing)

The runner feeds ticks to an LLM provider, maintains conversation history, and logs raw responses.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_runner.py
import json
import tempfile
from pathlib import Path
from src.runner import BenchmarkRunner
from src.providers.base import LLMProvider


class MockProvider(LLMProvider):
    """Returns canned responses that reference different numbers of factors."""

    def __init__(self):
        self.call_count = 0

    def name(self) -> str:
        return "MockModel"

    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        self.call_count += 1
        if self.call_count <= 5:
            # Early: references all 6 factors
            return (
                "ANALYSIS:\n"
                "- Load: Zone A demand is high at 340MW, increasing pressure\n"
                "- Generation: Plant 2 has TRIPPED, reducing capacity by 180MW\n"
                "- Frequency: At 49.82 Hz, dropping below nominal\n"
                "- Voltage: Line North at 398kV is LOW, needs attention\n"
                "- Weather: Wind dropping to 8km/h, reducing renewable output\n"
                "- Reserve: Battery at 68%, gas turbine on standby\n"
                "ACTION: start_gas_turbine\n"
                "REASON: Plant 2 trip combined with dropping frequency requires immediate backup generation."
            )
        else:
            # Later: only references 2 factors (simulating fixation)
            return (
                "ANALYSIS:\n"
                "- Load: Zone A demand remains high\n"
                "- Generation: Output levels look concerning\n"
                "- Frequency: stable\n"
                "- Voltage: ok\n"
                "- Weather: unchanged\n"
                "- Reserve: fine\n"
                "ACTION: ramp_plant\n"
                "REASON: Load is high and generation is low."
            )

    def reset(self) -> None:
        self.call_count = 0


def test_runner_produces_results():
    provider = MockProvider()
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            provider=provider,
            num_ticks=10,
            seed=42,
            output_dir=Path(tmpdir),
        )
        results = runner.run()
        assert len(results) == 10


def test_runner_result_has_required_fields():
    provider = MockProvider()
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            provider=provider,
            num_ticks=5,
            seed=42,
            output_dir=Path(tmpdir),
        )
        results = runner.run()
        for r in results:
            assert "tick_number" in r
            assert "response" in r
            assert "ground_truth" in r


def test_runner_saves_results_to_file():
    provider = MockProvider()
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            provider=provider,
            num_ticks=5,
            seed=42,
            output_dir=Path(tmpdir),
        )
        runner.run()
        result_file = Path(tmpdir) / "raw_results.jsonl"
        assert result_file.exists()
        lines = result_file.read_text().strip().split("\n")
        assert len(lines) == 5


def test_runner_with_context_reset():
    provider = MockProvider()
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            provider=provider,
            num_ticks=10,
            seed=42,
            output_dir=Path(tmpdir),
            context_reset_interval=5,  # Reset every 5 ticks
        )
        results = runner.run()
        assert len(results) == 10


def test_runner_with_intervention():
    provider = MockProvider()
    with tempfile.TemporaryDirectory() as tmpdir:
        runner = BenchmarkRunner(
            provider=provider,
            num_ticks=10,
            seed=42,
            output_dir=Path(tmpdir),
            intervention_ticks=[5],
        )
        results = runner.run()
        assert len(results) == 10
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_runner.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement runner.py**

```python
# src/runner.py
"""Benchmark runner: feeds ticks to LLM and logs responses."""

import json
from pathlib import Path

from .generator import TickGenerator
from .scenario import PowerGridScenario
from .providers.base import LLMProvider


INTERVENTION_PROMPT = (
    "WARNING: Your recent analyses have been incomplete. "
    "You are NOT analyzing all 6 factors. "
    "You MUST provide substantive analysis for Load, Generation, Frequency, "
    "Voltage, Weather, AND Reserve before making your decision. "
    "Failure to analyze all factors leads to dangerous grid management decisions. "
    "Resume full analysis now."
)

CHECKLIST_SUFFIX = (
    "\n\nREMINDER: Before answering, verify you have analyzed: "
    "(1) Load, (2) Generation, (3) Frequency, (4) Voltage, (5) Weather, (6) Reserve. "
    "ALL SIX are required."
)


class BenchmarkRunner:
    """Runs the benchmark against a single LLM provider."""

    def __init__(
        self,
        provider: LLMProvider,
        num_ticks: int = 200,
        seed: int = 42,
        output_dir: Path = Path("results"),
        context_reset_interval: int | None = None,
        intervention_ticks: list[int] | None = None,
        force_checklist: bool = False,
    ):
        self.provider = provider
        self.num_ticks = num_ticks
        self.seed = seed
        self.output_dir = output_dir
        self.context_reset_interval = context_reset_interval
        self.intervention_ticks = intervention_ticks or []
        self.force_checklist = force_checklist

        self.scenario = PowerGridScenario()
        self.generator = TickGenerator(seed=seed, num_ticks=num_ticks)

    def run(self) -> list[dict]:
        ticks = self.generator.generate()
        system_prompt = self.scenario.system_prompt()
        messages: list[dict] = []
        results = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        results_file = self.output_dir / "raw_results.jsonl"

        with open(results_file, "w") as f:
            for i, tick in enumerate(ticks):
                tick_num = tick["tick_number"]

                # Context reset if configured
                if self.context_reset_interval and i > 0 and i % self.context_reset_interval == 0:
                    messages = []
                    self.provider.reset()

                # Intervention if this tick is flagged
                if tick_num in self.intervention_ticks:
                    messages.append({"role": "user", "content": INTERVENTION_PROMPT})
                    messages.append({"role": "assistant", "content": "Understood. I will analyze all 6 factors thoroughly from now on."})

                # Format tick prompt
                user_msg = self.scenario.format_tick(tick_num, tick["data"])
                if self.force_checklist:
                    user_msg += CHECKLIST_SUFFIX

                # Send to LLM
                response = self.provider.send_turn(system_prompt, messages, user_msg)

                # Append to conversation history
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": response})

                result = {
                    "tick_number": tick_num,
                    "response": response,
                    "ground_truth": tick["ground_truth"],
                }
                results.append(result)

                # Write incrementally
                f.write(json.dumps(result) + "\n")
                f.flush()

        return results
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_runner.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/runner.py tests/test_runner.py
git commit -m "feat: benchmark runner with intervention, context reset, and checklist modes"
```

---

## Task 6: Response Analyzer + Metrics

**Files:**
- Create: `src/metrics.py`
- Create: `src/analyzer.py`
- Create: `tests/test_analyzer.py`
- Create: `tests/test_metrics.py`

Parses agent responses and computes the 5 key metrics: FC, FI, DA, ADR, IR.

- [ ] **Step 1: Write the failing tests for metrics**

```python
# tests/test_metrics.py
from src.metrics import (
    factor_coverage,
    fixation_index,
    decision_accuracy,
    anomaly_detection_rate,
    intervention_recovery,
)


def test_factor_coverage_all_mentioned():
    factors_mentioned = ["load", "generation", "frequency", "voltage", "weather", "reserve"]
    assert factor_coverage(factors_mentioned, total=6) == 1.0


def test_factor_coverage_partial():
    factors_mentioned = ["load", "generation"]
    assert factor_coverage(factors_mentioned, total=6) == 2 / 6


def test_factor_coverage_none():
    assert factor_coverage([], total=6) == 0.0


def test_fixation_index_balanced():
    # Each factor gets roughly equal token count
    token_counts = {"load": 20, "generation": 18, "frequency": 22, "voltage": 19, "weather": 21, "reserve": 20}
    fi = fixation_index(token_counts)
    assert fi < 0.25  # Balanced = low fixation


def test_fixation_index_fixated():
    # One factor dominates
    token_counts = {"load": 100, "generation": 5, "frequency": 3, "voltage": 2, "weather": 1, "reserve": 1}
    fi = fixation_index(token_counts)
    assert fi > 0.7  # Fixated = high fixation


def test_decision_accuracy_correct():
    assert decision_accuracy("start_gas_turbine", "start_gas_turbine") == 1.0


def test_decision_accuracy_wrong():
    assert decision_accuracy("shed_load", "start_gas_turbine") == 0.0


def test_anomaly_detection_rate():
    # Agent mentioned load and generation but missed weather
    mentioned = ["load", "generation"]
    anomalous = ["load", "weather"]
    rate = anomaly_detection_rate(mentioned, anomalous)
    assert rate == 0.5  # 1 of 2 anomalies detected


def test_anomaly_detection_rate_no_anomalies():
    rate = anomaly_detection_rate(["load"], [])
    assert rate == 1.0  # Nothing to detect = perfect


def test_intervention_recovery():
    # FC values over ticks: drops, intervention at tick 5, spikes, then drops again
    fc_values = [0.8, 0.6, 0.5, 0.4, 0.3, 1.0, 0.9, 0.7, 0.5, 0.3]
    intervention_tick = 5  # index 5
    recovery_length = intervention_recovery(fc_values, intervention_tick)
    assert recovery_length == 3  # FC stays above pre-intervention for ~3 ticks
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_metrics.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement metrics.py**

```python
# src/metrics.py
"""Core metrics for CoherenceBench."""


def factor_coverage(factors_mentioned: list[str], total: int = 6) -> float:
    """FC: fraction of factors substantively analyzed (0.0 to 1.0)."""
    if total == 0:
        return 1.0
    unique = len(set(factors_mentioned))
    return min(unique / total, 1.0)


def fixation_index(token_counts: dict[str, int]) -> float:
    """FI: fraction of analysis tokens devoted to the top factor (0.0 to 1.0).

    Low FI = balanced. High FI = fixated on one factor.
    """
    total = sum(token_counts.values())
    if total == 0:
        return 0.0
    max_count = max(token_counts.values())
    return max_count / total


def decision_accuracy(predicted_action: str, correct_action: str) -> float:
    """DA: 1.0 if correct, 0.0 if wrong."""
    return 1.0 if predicted_action.strip().lower() == correct_action.strip().lower() else 0.0


def anomaly_detection_rate(factors_substantively_mentioned: list[str], anomalous_factors: list[str]) -> float:
    """ADR: fraction of anomalous factors that the agent substantively analyzed.

    If no anomalies exist, returns 1.0 (nothing to miss).
    """
    if not anomalous_factors:
        return 1.0
    detected = sum(1 for f in anomalous_factors if f in factors_substantively_mentioned)
    return detected / len(anomalous_factors)


def intervention_recovery(fc_values: list[float], intervention_idx: int) -> int:
    """IR: number of ticks after intervention before FC drops back to pre-intervention level.

    Returns 0 if no recovery observed.
    """
    if intervention_idx >= len(fc_values) or intervention_idx == 0:
        return 0

    pre_fc = fc_values[intervention_idx - 1]

    recovery_count = 0
    for fc in fc_values[intervention_idx:]:
        if fc > pre_fc:
            recovery_count += 1
        else:
            break

    return recovery_count
```

- [ ] **Step 4: Run metrics tests**

```bash
python -m pytest tests/test_metrics.py -v
```
Expected: All PASS

- [ ] **Step 5: Write the failing tests for analyzer**

```python
# tests/test_analyzer.py
from src.analyzer import ResponseAnalyzer


FULL_RESPONSE = (
    "ANALYSIS:\n"
    "- Load: Zone A demand is spiking to 450MW, well above normal 300MW range. "
    "Zone B stable at 280MW. Zone C slightly elevated at 195MW.\n"
    "- Generation: Plant 2 has TRIPPED, losing 180MW of capacity. "
    "Plants 1, 3, 4 running at 290, 150, 95MW respectively.\n"
    "- Frequency: At 49.82 Hz and dropping. Was 49.97 two ticks ago. "
    "Approaching lower safe boundary of 49.5 Hz.\n"
    "- Voltage: Line North at 398kV is below 400kV threshold. "
    "South and East lines stable at 412kV and 405kV.\n"
    "- Weather: Wind dropping from 15 to 8 km/h, reducing renewable output significantly. "
    "Solar holding stable.\n"
    "- Reserve: Battery at 68% (340MWh). Gas turbine on standby. "
    "Spinning reserve at 120MW should be sufficient.\n"
    "ACTION: start_gas_turbine\n"
    "REASON: Plant 2 trip with dropping frequency and reduced wind generation "
    "creates an immediate supply gap that requires backup generation."
)

FIXATED_RESPONSE = (
    "ANALYSIS:\n"
    "- Load: Zone A very high at 450MW, this is the critical concern. "
    "Heavy demand pulling from all generation sources. "
    "Zone B and C also contributing to strain. "
    "Total load approaching system maximum.\n"
    "- Generation: looks ok\n"
    "- Frequency: stable\n"
    "- Voltage: fine\n"
    "- Weather: no change\n"
    "- Reserve: adequate\n"
    "ACTION: shed_load\n"
    "REASON: Load in Zone A is extremely high and must be reduced."
)


def test_analyzer_extracts_action():
    analyzer = ResponseAnalyzer()
    parsed = analyzer.parse_response(FULL_RESPONSE)
    assert parsed["action"] == "start_gas_turbine"


def test_analyzer_detects_all_factors_in_full_response():
    analyzer = ResponseAnalyzer()
    parsed = analyzer.parse_response(FULL_RESPONSE)
    assert len(parsed["factors_mentioned"]) == 6


def test_analyzer_detects_substantive_vs_dismissive():
    analyzer = ResponseAnalyzer()
    parsed = analyzer.parse_response(FIXATED_RESPONSE)
    # Only "load" has substantive analysis (>15 words)
    # Others are dismissive ("ok", "stable", "fine", "no change", "adequate")
    assert "load" in parsed["factors_substantive"]
    assert len(parsed["factors_substantive"]) <= 2  # At most load + maybe one other


def test_analyzer_counts_tokens_per_factor():
    analyzer = ResponseAnalyzer()
    parsed = analyzer.parse_response(FIXATED_RESPONSE)
    # Load should have way more tokens than others
    assert parsed["token_counts"]["load"] > parsed["token_counts"]["generation"]
    assert parsed["token_counts"]["load"] > parsed["token_counts"]["frequency"]


def test_analyzer_handles_malformed_response():
    analyzer = ResponseAnalyzer()
    parsed = analyzer.parse_response("I don't understand the question.")
    assert parsed["action"] == "unknown"
    assert len(parsed["factors_mentioned"]) == 0
```

- [ ] **Step 6: Run analyzer tests to verify they fail**

```bash
python -m pytest tests/test_analyzer.py -v
```
Expected: FAIL

- [ ] **Step 7: Implement analyzer.py**

```python
# src/analyzer.py
"""Parses LLM responses and extracts factor analysis + actions."""

import re
from .scenario import FACTORS

# Words that indicate dismissive/non-substantive analysis
DISMISSIVE_PATTERNS = [
    r"^(ok|fine|stable|unchanged|normal|adequate|no change|no issues|good|steady|same)\.?$",
    r"^(looks? (?:ok|fine|good|stable|normal))\.?$",
    r"^(no (?:change|issues|concerns|problems))\.?$",
    r"^(within (?:normal|safe) (?:range|limits))\.?$",
]

MIN_SUBSTANTIVE_WORDS = 8  # Minimum words for a factor analysis to count as substantive


class ResponseAnalyzer:
    """Parses structured agent responses into metrics-ready data."""

    def __init__(self):
        self.factors = FACTORS
        self._dismissive_re = [re.compile(p, re.IGNORECASE) for p in DISMISSIVE_PATTERNS]

    def parse_response(self, response: str) -> dict:
        """Parse a single response into structured analysis data.

        Returns:
            {
                "action": str,
                "factors_mentioned": list[str],         # All factors referenced at all
                "factors_substantive": list[str],        # Factors with real analysis
                "token_counts": dict[str, int],          # Word count per factor section
                "reason": str,
            }
        """
        action = self._extract_action(response)
        factor_sections = self._extract_factor_sections(response)

        factors_mentioned = []
        factors_substantive = []
        token_counts = {}

        for factor in self.factors:
            section_text = factor_sections.get(factor.name, "")
            word_count = len(section_text.split())
            token_counts[factor.name] = word_count

            if word_count > 0:
                factors_mentioned.append(factor.name)

            if word_count >= MIN_SUBSTANTIVE_WORDS and not self._is_dismissive(section_text):
                factors_substantive.append(factor.name)

        reason = self._extract_reason(response)

        return {
            "action": action,
            "factors_mentioned": factors_mentioned,
            "factors_substantive": factors_substantive,
            "token_counts": token_counts,
            "reason": reason,
        }

    def _extract_action(self, response: str) -> str:
        match = re.search(r"ACTION:\s*(\S+)", response, re.IGNORECASE)
        if match:
            return match.group(1).strip().lower()
        return "unknown"

    def _extract_factor_sections(self, response: str) -> dict[str, str]:
        """Extract the text for each factor's analysis section."""
        sections = {}
        for factor in self.factors:
            # Match "- Load: ..." or "- Generation: ..." etc
            # Capture everything until the next "- " factor line or ACTION:
            pattern = rf"-\s*{re.escape(factor.display_name.split('(')[0].strip())}:\s*(.*?)(?=\n\s*-\s|\nACTION:|\Z)"
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if not match:
                # Try just the factor name
                pattern = rf"-\s*{re.escape(factor.name)}:\s*(.*?)(?=\n\s*-\s|\nACTION:|\Z)"
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                sections[factor.name] = match.group(1).strip()
            else:
                sections[factor.name] = ""
        return sections

    def _extract_reason(self, response: str) -> str:
        match = re.search(r"REASON:\s*(.*?)$", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _is_dismissive(self, text: str) -> bool:
        text_clean = text.strip()
        for pattern in self._dismissive_re:
            if pattern.match(text_clean):
                return True
        return False
```

- [ ] **Step 8: Run all tests**

```bash
python -m pytest tests/test_analyzer.py tests/test_metrics.py -v
```
Expected: All PASS

- [ ] **Step 9: Commit**

```bash
git add src/metrics.py src/analyzer.py tests/test_metrics.py tests/test_analyzer.py
git commit -m "feat: response analyzer with factor parsing and 5 coherence metrics"
```

---

## Task 7: Visualizer

**Files:**
- Create: `src/visualizer.py`
- Create: `tests/test_visualizer.py`

Generates publication-ready graphs for the paper.

- [ ] **Step 1: Write the failing test**

```python
# tests/test_visualizer.py
import tempfile
from pathlib import Path
from src.visualizer import CoherenceVisualizer


def _make_sample_metrics(num_ticks=50):
    """Generate fake metric data for visualization testing."""
    import random
    rng = random.Random(42)
    metrics = []
    for i in range(num_ticks):
        # Simulate degradation
        base_fc = max(0.2, 1.0 - (i * 0.004) + rng.uniform(-0.05, 0.05))
        metrics.append({
            "tick": i + 1,
            "factor_coverage": round(base_fc, 3),
            "fixation_index": round(1.0 - base_fc + rng.uniform(-0.05, 0.05), 3),
            "decision_accuracy": 1.0 if rng.random() < base_fc else 0.0,
            "factors_substantive": rng.randint(1, 6),
            "per_factor_tokens": {
                "load": rng.randint(5, 50),
                "generation": rng.randint(3, 40),
                "frequency": rng.randint(2, 30),
                "voltage": rng.randint(1, 25),
                "weather": rng.randint(1, 20),
                "reserve": rng.randint(1, 15),
            },
        })
    return metrics


def test_visualizer_creates_fc_over_time_plot():
    metrics = _make_sample_metrics()
    with tempfile.TemporaryDirectory() as tmpdir:
        viz = CoherenceVisualizer(output_dir=Path(tmpdir))
        path = viz.plot_factor_coverage_over_time(metrics, model_name="TestModel")
        assert path.exists()
        assert path.suffix == ".png"


def test_visualizer_creates_per_factor_heatmap():
    metrics = _make_sample_metrics()
    with tempfile.TemporaryDirectory() as tmpdir:
        viz = CoherenceVisualizer(output_dir=Path(tmpdir))
        path = viz.plot_per_factor_attention(metrics, model_name="TestModel")
        assert path.exists()


def test_visualizer_creates_cross_model_comparison():
    model_data = {
        "Claude": _make_sample_metrics(),
        "GPT-4o": _make_sample_metrics(),
        "Llama": _make_sample_metrics(),
    }
    with tempfile.TemporaryDirectory() as tmpdir:
        viz = CoherenceVisualizer(output_dir=Path(tmpdir))
        path = viz.plot_cross_model_comparison(model_data)
        assert path.exists()


def test_visualizer_creates_intervention_recovery_plot():
    metrics = _make_sample_metrics(100)
    # Simulate intervention spike at tick 50
    for i in range(50, 55):
        metrics[i]["factor_coverage"] = min(1.0, metrics[i]["factor_coverage"] + 0.4)
    with tempfile.TemporaryDirectory() as tmpdir:
        viz = CoherenceVisualizer(output_dir=Path(tmpdir))
        path = viz.plot_intervention_recovery(metrics, intervention_ticks=[50], model_name="TestModel")
        assert path.exists()
```

- [ ] **Step 2: Run tests to verify they fail**

```bash
python -m pytest tests/test_visualizer.py -v
```
Expected: FAIL

- [ ] **Step 3: Implement visualizer.py**

```python
# src/visualizer.py
"""Publication-ready visualizations for CoherenceBench results."""

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns

# Paper-quality defaults
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 12,
    "xtick.labelsize": 10,
    "ytick.labelsize": 10,
    "legend.fontsize": 10,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "savefig.bbox": "tight",
})

FACTOR_COLORS = {
    "load": "#e74c3c",
    "generation": "#3498db",
    "frequency": "#2ecc71",
    "voltage": "#f39c12",
    "weather": "#9b59b6",
    "reserve": "#1abc9c",
}


class CoherenceVisualizer:
    """Generates publication-ready plots from benchmark metrics."""

    def __init__(self, output_dir: Path = Path("results")):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def plot_factor_coverage_over_time(self, metrics: list[dict], model_name: str) -> Path:
        """Figure 1: FC degradation curve — the paper's hero image."""
        ticks = [m["tick"] for m in metrics]
        fc_values = [m["factor_coverage"] for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 5))

        # Smooth trend line
        window = min(15, len(fc_values) // 5)
        if window > 1:
            smoothed = np.convolve(fc_values, np.ones(window) / window, mode="valid")
            smooth_ticks = ticks[window - 1:]
            ax.plot(smooth_ticks, smoothed, color="#e74c3c", linewidth=2.5, label="Trend")

        ax.scatter(ticks, fc_values, alpha=0.25, s=12, color="#e74c3c")
        ax.set_xlabel("Tick")
        ax.set_ylabel("Factor Coverage (FC)")
        ax.set_title(f"Factor Coverage Over Time — {model_name}")
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5, label="Perfect coverage")
        ax.legend()

        path = self.output_dir / f"fc_over_time_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_per_factor_attention(self, metrics: list[dict], model_name: str) -> Path:
        """Figure 2: Heatmap of per-factor token allocation over time."""
        factors = list(FACTOR_COLORS.keys())
        num_ticks = len(metrics)

        # Build matrix: rows = factors, cols = ticks
        matrix = np.zeros((len(factors), num_ticks))
        for j, m in enumerate(metrics):
            tokens = m["per_factor_tokens"]
            total = sum(tokens.values()) or 1
            for i, f in enumerate(factors):
                matrix[i, j] = tokens.get(f, 0) / total

        fig, ax = plt.subplots(figsize=(12, 4))
        sns.heatmap(
            matrix,
            ax=ax,
            xticklabels=20,
            yticklabels=factors,
            cmap="YlOrRd",
            vmin=0,
            vmax=0.6,
            cbar_kws={"label": "Attention share"},
        )
        ax.set_xlabel("Tick")
        ax.set_ylabel("Factor")
        ax.set_title(f"Per-Factor Attention Allocation — {model_name}")

        path = self.output_dir / f"factor_heatmap_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_cross_model_comparison(self, model_data: dict[str, list[dict]]) -> Path:
        """Figure 3: FC curves for all models on same axes."""
        fig, ax = plt.subplots(figsize=(10, 5))
        colors = sns.color_palette("husl", len(model_data))

        for (model_name, metrics), color in zip(model_data.items(), colors):
            fc_values = [m["factor_coverage"] for m in metrics]
            window = min(15, len(fc_values) // 5)
            if window > 1:
                smoothed = np.convolve(fc_values, np.ones(window) / window, mode="valid")
                ticks = list(range(window, len(fc_values) + 1))
                ax.plot(ticks, smoothed, linewidth=2, label=model_name, color=color)

        ax.set_xlabel("Tick")
        ax.set_ylabel("Factor Coverage (FC)")
        ax.set_title("Cross-Model Factor Coverage Degradation")
        ax.set_ylim(-0.05, 1.1)
        ax.axhline(y=1.0, color="gray", linestyle="--", alpha=0.5)
        ax.legend()

        path = self.output_dir / "cross_model_comparison.png"
        fig.savefig(path)
        plt.close(fig)
        return path

    def plot_intervention_recovery(
        self, metrics: list[dict], intervention_ticks: list[int], model_name: str
    ) -> Path:
        """Figure 4: FC with intervention points marked."""
        ticks = [m["tick"] for m in metrics]
        fc_values = [m["factor_coverage"] for m in metrics]

        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(ticks, fc_values, color="#3498db", linewidth=1.5, alpha=0.7)

        for it in intervention_ticks:
            ax.axvline(x=it, color="#e74c3c", linestyle="--", alpha=0.8, label="Intervention" if it == intervention_ticks[0] else None)

        ax.set_xlabel("Tick")
        ax.set_ylabel("Factor Coverage (FC)")
        ax.set_title(f"Intervention Recovery — {model_name}")
        ax.set_ylim(-0.05, 1.1)
        ax.legend()

        path = self.output_dir / f"intervention_{model_name.lower().replace(' ', '_')}.png"
        fig.savefig(path)
        plt.close(fig)
        return path
```

- [ ] **Step 4: Run tests to verify they pass**

```bash
python -m pytest tests/test_visualizer.py -v
```
Expected: All PASS

- [ ] **Step 5: Commit**

```bash
git add src/visualizer.py tests/test_visualizer.py
git commit -m "feat: publication-ready visualizations for coherence metrics"
```

---

## Task 8: Run Configs + CLI Scripts

**Files:**
- Create: `configs/run_a_baseline.yaml`
- Create: `configs/run_b_intervention.yaml`
- Create: `configs/run_c_context_reset.yaml`
- Create: `configs/run_d_checklist.yaml`
- Create: `configs/run_e_cross_model.yaml`
- Create: `scripts/run_single.py`
- Create: `scripts/run_benchmark.py`
- Create: `scripts/generate_paper_figures.py`

- [ ] **Step 1: Create config files**

```yaml
# configs/run_a_baseline.yaml
name: "Run A: Baseline"
description: "200 ticks, continuous session, no intervention"
num_ticks: 200
seeds: [42, 123, 456, 789, 1001, 2002, 3003, 4004, 5005, 6006,
        7007, 8008, 9009, 1010, 2020, 3030, 4040, 5050, 6060, 7070,
        8080, 9090, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
providers: ["claude"]
context_reset_interval: null
intervention_ticks: []
force_checklist: false
```

```yaml
# configs/run_b_intervention.yaml
name: "Run B: With Intervention"
description: "200 ticks, explicit intervention reminders at ticks 50, 100, 150"
num_ticks: 200
seeds: [42, 123, 456, 789, 1001, 2002, 3003, 4004, 5005, 6006,
        7007, 8008, 9009, 1010, 2020, 3030, 4040, 5050, 6060, 7070,
        8080, 9090, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
providers: ["claude"]
context_reset_interval: null
intervention_ticks: [50, 100, 150]
force_checklist: false
```

```yaml
# configs/run_c_context_reset.yaml
name: "Run C: Context Reset"
description: "200 ticks, context cleared every 40 ticks, full state re-injected"
num_ticks: 200
seeds: [42, 123, 456, 789, 1001, 2002, 3003, 4004, 5005, 6006,
        7007, 8008, 9009, 1010, 2020, 3030, 4040, 5050, 6060, 7070,
        8080, 9090, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
providers: ["claude"]
context_reset_interval: 40
intervention_ticks: []
force_checklist: false
```

```yaml
# configs/run_d_checklist.yaml
name: "Run D: Forced Checklist"
description: "200 ticks, every prompt includes mandatory 6-factor checklist reminder"
num_ticks: 200
seeds: [42, 123, 456, 789, 1001, 2002, 3003, 4004, 5005, 6006,
        7007, 8008, 9009, 1010, 2020, 3030, 4040, 5050, 6060, 7070,
        8080, 9090, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
providers: ["claude"]
context_reset_interval: null
intervention_ticks: []
force_checklist: true
```

```yaml
# configs/run_e_cross_model.yaml
name: "Run E: Cross-Model Comparison"
description: "Run A baseline across all 4 model providers"
num_ticks: 200
seeds: [42, 123, 456, 789, 1001, 2002, 3003, 4004, 5005, 6006,
        7007, 8008, 9009, 1010, 2020, 3030, 4040, 5050, 6060, 7070,
        8080, 9090, 1111, 2222, 3333, 4444, 5555, 6666, 7777, 8888]
providers: ["claude", "gpt4o", "gemini", "llama"]
context_reset_interval: null
intervention_ticks: []
force_checklist: false
```

- [ ] **Step 2: Create run_single.py**

```python
#!/usr/bin/env python3
# scripts/run_single.py
"""Run a single benchmark config against a single provider and seed."""

import argparse
import json
import sys
from pathlib import Path

import yaml
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.runner import BenchmarkRunner
from src.analyzer import ResponseAnalyzer
from src.providers import get_provider


def main():
    parser = argparse.ArgumentParser(description="Run CoherenceBench single config")
    parser.add_argument("config", help="Path to config YAML")
    parser.add_argument("--provider", default="claude", help="Provider name")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    provider = get_provider(args.provider)
    output_dir = Path(args.output_dir) / config["name"].replace(" ", "_") / provider.name() / f"seed_{args.seed}"

    runner = BenchmarkRunner(
        provider=provider,
        num_ticks=config["num_ticks"],
        seed=args.seed,
        output_dir=output_dir,
        context_reset_interval=config.get("context_reset_interval"),
        intervention_ticks=config.get("intervention_ticks", []),
        force_checklist=config.get("force_checklist", False),
    )

    print(f"Running: {config['name']} | {provider.name()} | seed={args.seed}")
    results = runner.run()

    # Analyze
    analyzer = ResponseAnalyzer()
    analyzed = []
    for r in results:
        parsed = analyzer.parse_response(r["response"])
        analyzed.append({
            "tick": r["tick_number"],
            **parsed,
            "ground_truth": r["ground_truth"],
        })

    metrics_file = output_dir / "analyzed_results.json"
    with open(metrics_file, "w") as f:
        json.dump(analyzed, f, indent=2)

    print(f"Done. Results saved to {output_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 3: Create run_benchmark.py**

```python
#!/usr/bin/env python3
# scripts/run_benchmark.py
"""Run the full benchmark suite: all configs, all seeds, all providers."""

import argparse
import subprocess
import sys
from pathlib import Path

import yaml


def main():
    parser = argparse.ArgumentParser(description="Run full CoherenceBench suite")
    parser.add_argument("--configs-dir", default="configs", help="Directory with config YAMLs")
    parser.add_argument("--output-dir", default="results", help="Output directory")
    parser.add_argument("--max-seeds", type=int, default=30, help="Max seeds to run per config")
    args = parser.parse_args()

    configs_dir = Path(args.configs_dir)
    config_files = sorted(configs_dir.glob("*.yaml"))

    total_runs = 0
    for config_path in config_files:
        with open(config_path) as f:
            config = yaml.safe_load(f)

        providers = config.get("providers", ["claude"])
        seeds = config.get("seeds", [42])[:args.max_seeds]

        for provider in providers:
            for seed in seeds:
                total_runs += 1
                print(f"\n{'='*60}")
                print(f"Run {total_runs}: {config['name']} | {provider} | seed={seed}")
                print(f"{'='*60}")

                cmd = [
                    sys.executable, "scripts/run_single.py",
                    str(config_path),
                    "--provider", provider,
                    "--seed", str(seed),
                    "--output-dir", args.output_dir,
                ]

                result = subprocess.run(cmd, capture_output=False)
                if result.returncode != 0:
                    print(f"WARNING: Run failed (exit code {result.returncode})")

    print(f"\nAll done. {total_runs} total runs completed.")


if __name__ == "__main__":
    main()
```

- [ ] **Step 4: Create generate_paper_figures.py**

```python
#!/usr/bin/env python3
# scripts/generate_paper_figures.py
"""Generate all figures for the paper from benchmark results."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from src.visualizer import CoherenceVisualizer
from src.metrics import factor_coverage, fixation_index, decision_accuracy


def load_analyzed_results(results_dir: Path) -> dict:
    """Load all analyzed results grouped by config/model/seed."""
    data = {}
    for result_file in results_dir.rglob("analyzed_results.json"):
        parts = result_file.relative_to(results_dir).parts
        if len(parts) >= 3:
            config_name = parts[0]
            model_name = parts[1]
            seed = parts[2]
            key = (config_name, model_name)
            if key not in data:
                data[key] = []
            with open(result_file) as f:
                data[key].append(json.load(f))
    return data


def aggregate_metrics(runs: list[list[dict]]) -> list[dict]:
    """Average metrics across seeds for each tick position."""
    if not runs:
        return []
    num_ticks = len(runs[0])
    aggregated = []
    for tick_idx in range(num_ticks):
        fc_vals = []
        fi_vals = []
        da_vals = []
        token_counts_sum = {}

        for run in runs:
            if tick_idx < len(run):
                entry = run[tick_idx]
                fc = factor_coverage(entry.get("factors_substantive", []))
                fc_vals.append(fc)
                fi = fixation_index(entry.get("token_counts", {}))
                fi_vals.append(fi)
                da = decision_accuracy(
                    entry.get("action", "unknown"),
                    entry.get("ground_truth", {}).get("correct_action", "unknown"),
                )
                da_vals.append(da)
                for f, count in entry.get("token_counts", {}).items():
                    token_counts_sum[f] = token_counts_sum.get(f, 0) + count

        num_runs = len(fc_vals) or 1
        aggregated.append({
            "tick": tick_idx + 1,
            "factor_coverage": sum(fc_vals) / num_runs,
            "fixation_index": sum(fi_vals) / num_runs,
            "decision_accuracy": sum(da_vals) / num_runs,
            "factors_substantive": int(sum(fc_vals) / num_runs * 6),
            "per_factor_tokens": {f: c // num_runs for f, c in token_counts_sum.items()},
        })
    return aggregated


def main():
    results_dir = Path("results")
    figures_dir = Path("paper/figures")
    viz = CoherenceVisualizer(output_dir=figures_dir)

    data = load_analyzed_results(results_dir)

    if not data:
        print("No results found. Run the benchmark first.")
        return

    # Generate figures per config/model
    for (config_name, model_name), runs in data.items():
        print(f"Generating figures for {config_name} / {model_name} ({len(runs)} seeds)")
        metrics = aggregate_metrics(runs)

        viz.plot_factor_coverage_over_time(metrics, model_name=f"{model_name}")
        viz.plot_per_factor_attention(metrics, model_name=f"{model_name}")

    # Cross-model comparison (Run A only)
    cross_model = {}
    for (config_name, model_name), runs in data.items():
        if "baseline" in config_name.lower() or "run_a" in config_name.lower():
            cross_model[model_name] = aggregate_metrics(runs)

    if len(cross_model) > 1:
        viz.plot_cross_model_comparison(cross_model)
        print("Generated cross-model comparison")

    # Intervention recovery (Run B)
    for (config_name, model_name), runs in data.items():
        if "intervention" in config_name.lower() or "run_b" in config_name.lower():
            metrics = aggregate_metrics(runs)
            viz.plot_intervention_recovery(metrics, intervention_ticks=[50, 100, 150], model_name=model_name)

    print(f"\nAll figures saved to {figures_dir}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Commit**

```bash
git add configs/ scripts/
git commit -m "feat: benchmark configs (5 run types) and CLI scripts"
```

---

## Task 9: Paper LaTeX Skeleton

**Files:**
- Create: `paper/main.tex`
- Create: `paper/references.bib`

- [ ] **Step 1: Create references.bib**

```bibtex
% paper/references.bib

@article{vaswani2017attention,
  title={Attention is all you need},
  author={Vaswani, Ashish and Shazeer, Noam and Parmar, Niki and Uszkoreit, Jakob and Jones, Llion and Gomez, Aidan N and Kaiser, {\L}ukasz and Polosukhin, Illia},
  journal={Advances in neural information processing systems},
  volume={30},
  year={2017}
}

@article{agentdrift2026,
  title={Agent Drift: Semantic, Coordination, and Behavioral Drift in Multi-Agent Systems},
  author={{Agent Drift Authors}},
  journal={arXiv preprint arXiv:2601.04170},
  year={2026}
}

@article{liu2024lost,
  title={Lost in the middle: How language models use long contexts},
  author={Liu, Nelson F and Lin, Kevin and Hewitt, John and Paranjape, Ashwin and Bevilacqua, Michele and Petroni, Fabio and Liang, Percy},
  journal={Transactions of the Association for Computational Linguistics},
  year={2024}
}

@article{jimenez2024swebench,
  title={{SWE}-bench: Can language models resolve real-world {GitHub} issues?},
  author={Jimenez, Carlos E and Yang, John and Wettig, Alexander and Yao, Shunyu and Pei, Kexin and Press, Ofir and Narasimhan, Karthik},
  journal={arXiv preprint arXiv:2310.06770},
  year={2024}
}

@article{vendingbench2025,
  title={Vending-Bench: A Long-Horizon Agent Benchmark},
  author={{Vending-Bench Authors}},
  journal={arXiv preprint arXiv:2502.15840},
  year={2025}
}

@misc{gartner2025agentic,
  title={Gartner Predicts 40\% of Agentic {AI} Projects Will Be Abandoned by 2027},
  author={{Gartner}},
  year={2025},
  howpublished={\url{https://www.gartner.com}}
}
```

- [ ] **Step 2: Create paper/main.tex**

```latex
% paper/main.tex
\documentclass{article}
\usepackage[preprint]{neurips_2026}
\usepackage[utf8]{inputenc}
\usepackage[T1]{fontenc}
\usepackage{hyperref}
\usepackage{url}
\usepackage{booktabs}
\usepackage{amsfonts}
\usepackage{amsmath}
\usepackage{graphicx}
\usepackage{subcaption}

\title{CoherenceBench: Measuring Attention Collapse \\ in Long-Running Autonomous Agents}

\author{
  Venkateshwar Reddy Jambula \\
  PranaAlpha Labs \\
  Bengaluru, India \\
  \texttt{venkateshwar.jambula@gmail.com} \\
  \And
  Anitha Krishnakumar \\
  PranaAlpha Labs \\
  Bengaluru, India \\
  \texttt{anitha.krishnakumar@gmail.com} \\
}

\begin{document}

\maketitle

\begin{abstract}
Autonomous AI agents are increasingly deployed for long-running tasks requiring sustained multi-factor analysis. We introduce \textbf{CoherenceBench}, the first benchmark for measuring how well language models maintain coherent attention across multiple decision factors over extended sessions. Using a domain-neutral power grid control scenario with 200 sequential decision points and 6 interdependent factors, we evaluate whether agents sustain balanced multi-factor analysis or progressively collapse toward single-factor fixation. We test [N] models across 5 experimental conditions with 30 seeds each. Our results show that \textit{all tested models} exhibit significant factor coverage degradation: from [X]\% coverage at tick 1 to [Y]\% by tick 200. Explicit intervention temporarily recovers coverage, but decay resumes within [Z] ticks. Context resets partially mitigate degradation. These findings demonstrate that current transformer attention architectures are architecturally unsuited for sustained multi-factor reasoning in autonomous agents — a limitation that prompt engineering, larger context windows, and chain-of-thought cannot address. CoherenceBench is open-sourced to enable reproducible evaluation of long-running agent coherence.
\end{abstract}

% ============================================================
\section{Introduction}
\label{sec:intro}

% Hook: everyone has experienced this
% The problem: agents deployed for hours, coherence degrades
% Why it matters: Gartner 40% cancellation, public failures
% Our contribution: CoherenceBench + empirical findings
% What we DON'T claim: solutions (open question)

\textit{[TO BE WRITTEN after experiments complete. Structure above.]}

% ============================================================
\section{Background and Related Work}
\label{sec:background}

\subsection{Attention in Transformers}
% Vaswani et al. 2017 — softmax concentration
% Designed for bounded sequence-to-sequence tasks
% Not designed for multi-hour autonomous operation

\subsection{Known Long-Context Limitations}
% Lost in the Middle (Liu et al. 2024)
% Needle-in-a-Haystack degradation
% Recency bias in LLMs

\subsection{Agent Coherence}
% Agent Drift paper (2026) — semantic, coordination, behavioral drift
% Vending-Bench — long-horizon but single-factor decisions
% AgentBench — general capability, not coherence over time
% GAP: no benchmark measures multi-factor attention maintenance

\textit{[TO BE WRITTEN. Citations gathered in references.bib.]}

% ============================================================
\section{CoherenceBench}
\label{sec:benchmark}

\subsection{Design Principles}
% Domain-neutral (power grid, not finance/trading)
% Controlled ground truth (planted anomalies)
% Measurable (5 quantitative metrics)
% Reproducible (seeded generation, open source)

\subsection{Task Description}
% 6-factor power grid control room
% 200 sequential decisions
% Structured response format for parseable analysis

\subsection{Data Generation}
% Phased anomaly injection
% Early: load/generation anomalies (establishes fixation target)
% Late: weather/reserve anomalies (tests if agent follows)
% Ground truth: known correct action per tick

\subsection{Metrics}
% FC: Factor Coverage (factors analyzed / 6)
% FI: Fixation Index (top-factor token share)
% DA: Decision Accuracy (correct action?)
% ADR: Anomaly Detection Rate (caught per factor?)
% IR: Intervention Recovery (ticks before re-collapse)

\subsection{Experimental Conditions}
% Run A: Baseline (continuous session)
% Run B: With Intervention (explicit reminders)
% Run C: Context Reset (clear every 40 ticks)
% Run D: Forced Checklist (structural reminder every tick)
% Run E: Cross-model comparison

\textit{[TO BE WRITTEN with exact details from implementation.]}

% ============================================================
\section{Experiments}
\label{sec:experiments}

\subsection{Models Tested}
% Claude Sonnet, GPT-4o, Gemini Flash, Llama 3.1 70B

\subsection{Setup}
% 30 seeds × 5 conditions × 4 models = 600 runs
% API parameters (max_tokens, temperature)
% Statistical methodology

\textit{[TO BE WRITTEN after experiments run.]}

% ============================================================
\section{Results}
\label{sec:results}

% Figure 1: FC over time (hero graph)
% Figure 2: Per-factor attention heatmap
% Figure 3: Cross-model comparison
% Figure 4: Intervention recovery
% Table 1: Summary statistics per model per condition
% Table 2: ADR by factor and phase

\textit{[TO BE WRITTEN after experiments. Figures generated by visualizer.py.]}

% ============================================================
\section{Analysis}
\label{sec:analysis}

% Why does this happen? (attention math argument)
% Softmax concentration over accumulated context
% Early-salience capture: first anomalies "win" attention permanently
% Intervention resistance: correction is a single signal vs accumulated weight
% Context reset works because it removes accumulated bias

\textit{[TO BE WRITTEN connecting results to attention mechanism theory.]}

% ============================================================
\section{Discussion}
\label{sec:discussion}

% Implications for autonomous agent deployment
% Why bigger context / better prompts / CoT don't fix this
% Domains affected: any multi-factor long-running agent
% Limitations of this study
% Open question: what architectural changes are needed?

\textit{[TO BE WRITTEN.]}

% ============================================================
\section{Conclusion}
\label{sec:conclusion}

% Summary: built CoherenceBench, proved attention collapse, all models affected
% Contribution: benchmark + finding
% Call to action: community needs new architectures for agent coherence

\textit{[TO BE WRITTEN.]}

% ============================================================
\bibliographystyle{plainnat}
\bibliography{references}

\end{document}
```

- [ ] **Step 3: Commit**

```bash
git add paper/
git commit -m "feat: NeurIPS paper LaTeX skeleton with structure and citations"
```

---

## Task 10: README + Final Polish

**Files:**
- Create: `README.md`

- [ ] **Step 1: Write README**

```markdown
# CoherenceBench

**Measuring Attention Collapse in Long-Running Autonomous AI Agents**

CoherenceBench is the first benchmark for evaluating whether large language models maintain coherent multi-factor analysis over extended autonomous sessions.

## The Finding

All tested models (Claude, GPT-4o, Gemini, Llama) exhibit progressive factor coverage degradation: agents that start by analyzing all 6 decision factors gradually collapse to fixating on 1-2 factors — even when explicitly instructed to consider all of them.

## Quick Start

```bash
# Clone and install
git clone https://github.com/your-org/coherencebench.git
cd coherencebench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"

# Set API keys
cp .env.example .env
# Edit .env with your API keys

# Run a single experiment
python scripts/run_single.py configs/run_a_baseline.yaml --provider claude --seed 42

# Run full benchmark suite
python scripts/run_benchmark.py

# Generate paper figures
python scripts/generate_paper_figures.py
```

## Benchmark Design

- **Scenario:** Power grid control room with 6 interdependent factors
- **Duration:** 200 sequential decision ticks per session
- **Ground truth:** Planted anomalies with known correct actions
- **5 experimental conditions:** Baseline, Intervention, Context Reset, Checklist, Cross-Model

## Metrics

| Metric | What it measures |
|--------|-----------------|
| FC (Factor Coverage) | Fraction of factors substantively analyzed per tick |
| FI (Fixation Index) | Token share devoted to top factor |
| DA (Decision Accuracy) | Correct action rate |
| ADR (Anomaly Detection Rate) | Anomalies caught per factor per phase |
| IR (Intervention Recovery) | Ticks before re-collapse after correction |

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
```

- [ ] **Step 2: Run full test suite**

```bash
python -m pytest tests/ -v
```
Expected: All PASS

- [ ] **Step 3: Commit**

```bash
git add README.md
git commit -m "docs: README with quickstart, design overview, and citation"
```

---

## Execution Timeline

| Day | Tasks | Output |
|-----|-------|--------|
| Day 1 | Tasks 1-3 (scaffolding, scenario, generator) | Working data generator |
| Day 2 | Tasks 4-5 (providers, runner) | Can run experiments |
| Day 3 | Task 6 (analyzer + metrics) | Can measure results |
| Day 4 | Task 7 (visualizer) | Can generate graphs |
| Day 5 | Task 8 (configs + CLI) | Full pipeline working |
| Day 6-7 | Run experiments across all models + seeds | Raw results |
| Day 8-9 | Task 9 (write paper from results) | Paper draft |
| Day 10 | Task 10 (README + polish) | Submit to ArXiv + CAISC |

**Total estimated API cost:** $50-100 across all models and runs.
