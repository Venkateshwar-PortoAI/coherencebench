# Contributing to CoherenceBench

Thanks for your interest in contributing. This guide covers the main ways to help.

## Setup

```bash
git clone https://github.com/pranaalpha/coherencebench.git
cd coherencebench
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
# Add your API keys to .env
```

## Adding a New Scenario

Scenarios live in `src/scenarios/` and configs live in `configs/`. To add a new scenario:

1. Create a new file in `src/scenarios/` (e.g., `my_scenario.py`) implementing the `BaseScenario` interface from `src/scenarios/base.py`. Define the scenario's factors, actions, anomaly injection rules, and ground-truth action mapping.
2. Register your scenario in `src/scenarios/__init__.py`.
3. Create a YAML config in `configs/` specifying tick count, anomaly schedule, and experimental condition.
4. The analyzer and metrics work automatically via the `BaseScenario` interface — no changes needed.

## Adding a New Provider

Providers live in `src/providers/`. To add support for a new model:

1. Create a new file in `src/providers/` (e.g., `mistral.py`).
2. Implement the `LLMProvider` interface from `src/providers/base.py`:
   - `name()` -- return a human-readable model name.
   - `send_turn(system_prompt, messages, user_message)` -- send a prompt and return the response text.
   - `reset()` -- clear any conversation state.
   - Optionally override `max_context_tokens` (default is 128k).
3. Register your provider in `src/providers/__init__.py`.
4. Add any required API key to `.env.example`.

## Running Tests

```bash
python -m pytest tests/ -v
```

All tests should pass before submitting a PR. The test suite uses mocked API responses so no API keys are needed.

## Code Style

- Standard Python conventions (PEP 8).
- Type hints on function signatures.
- Docstrings on public classes and methods.

## Submitting Changes

1. Fork the repo and create a feature branch.
2. Make your changes with tests.
3. Run the test suite.
4. Open a pull request with a clear description of what you changed and why.
