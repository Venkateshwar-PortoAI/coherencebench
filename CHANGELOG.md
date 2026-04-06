# Changelog

All notable changes to CoherenceBench will be documented in this file.

Format follows [Keep a Changelog](https://keepachangelog.com/).

## [0.1.0] - 2026-04-06

### Added

- Core research framework: tick generator, benchmark runner, response analyzer
- 5 metrics: Factor Coverage (FC), Fixation Index (FI), Decision Accuracy (DA), Anomaly Detection Rate (ADR), Intervention Recovery (IR)
- 3 scenarios: Power Grid (dev), Hospital Triage (dev), Network Security (eval, held-out)
- 5 providers: Claude (API + CLI), GPT-4o/Codex, Gemini 1.5 Pro, Llama 3.1 405B
- Pre-generated data for 5 seeds (42, 123, 456, 789, 1001)
- 3 baselines: random uniform, majority action, most-common action
- Directional validation gate for distinguishing fixation from uniform degradation
- Context budget management with automatic message truncation
- Context reset with state re-injection
- Intervention prompts with recovery measurement
- Browser-based human benchmark viewer (standalone HTML app)
- Visualization suite: FC curves, attention heatmaps, ADR by phase, DA vs FC scatter
- Standard evaluation protocol (EVALUATION.md)
- GitHub Actions CI (Python 3.11 + 3.12)

### Fixed

- ResponseAnalyzer now receives the correct scenario (was defaulting to power_grid for all)
- Intervention and checklist prompts are scenario-aware (were hardcoded to power grid factors)
- Renamed `token_counts` to `word_counts` (was measuring words, labeled as tokens)
- State deep copy handles nested dicts correctly (was shallow, affecting hospital scenario)

### Security

- Eval ground truth stripped from public data (held-out integrity)
- API keys loaded from .env only, never committed
