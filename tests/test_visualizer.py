"""Tests for the CoherenceVisualizer using synthetic metric data."""

import os
import tempfile
from pathlib import Path

import numpy as np

from src.visualizer import CoherenceVisualizer, FACTOR_NAMES


def _make_visualizer(tmp_path: Path) -> CoherenceVisualizer:
    return CoherenceVisualizer(output_dir=tmp_path)


def _synthetic_fc_curve(n: int = 200, seed: int = 42) -> list[float]:
    """Generate a synthetic FC degradation curve: starts ~1.0, decays to ~0.3."""
    rng = np.random.default_rng(seed)
    base = np.linspace(1.0, 0.3, n)
    noise = rng.normal(0, 0.05, n)
    return list(np.clip(base + noise, 0, 1))


def _synthetic_word_counts(n: int = 200, seed: int = 42) -> list[dict[str, int]]:
    """Generate synthetic per-factor word counts that shift over time."""
    rng = np.random.default_rng(seed)
    counts = []
    for t in range(n):
        c = {}
        for f in FACTOR_NAMES:
            if f in ("load", "generation"):
                # High early, low late
                base = max(5, 30 - t * 0.12)
            elif f in ("weather", "reserve"):
                # Low early, high late (if agent adapts)
                base = max(5, 10 + t * 0.05)
            else:
                base = 15
            c[f] = max(0, int(base + rng.integers(-3, 4)))
        counts.append(c)
    return counts


def _synthetic_da_values(fc_values: list[float], seed: int = 42) -> list[float]:
    """DA correlated with FC: lower FC -> higher chance of wrong decision."""
    rng = np.random.default_rng(seed)
    da = []
    for fc in fc_values:
        p_correct = 0.5 + 0.5 * fc  # 100% correct at FC=1, 50% at FC=0
        da.append(1.0 if rng.random() < p_correct else 0.0)
    return da


class TestFactorCoverageOverTime:
    def test_creates_file(self, tmp_path):
        viz = _make_visualizer(tmp_path)
        fc = _synthetic_fc_curve(100)
        path = viz.plot_factor_coverage_over_time(fc, save_as="test_fc.png")
        assert path.exists()
        assert path.stat().st_size > 1000

    def test_with_interventions(self, tmp_path):
        viz = _make_visualizer(tmp_path)
        fc = _synthetic_fc_curve(200)
        path = viz.plot_factor_coverage_over_time(
            fc, intervention_ticks=[50, 100, 150], save_as="test_fc_int.png"
        )
        assert path.exists()

    def test_short_curve(self, tmp_path):
        viz = _make_visualizer(tmp_path)
        fc = [0.8, 0.7, 0.6, 0.5]
        path = viz.plot_factor_coverage_over_time(fc, save_as="test_fc_short.png")
        assert path.exists()


class TestPerFactorAttention:
    def test_creates_heatmap(self, tmp_path):
        viz = _make_visualizer(tmp_path)
        counts = _synthetic_word_counts(100)
        path = viz.plot_per_factor_attention(counts, save_as="test_heatmap.png")
        assert path.exists()
        assert path.stat().st_size > 1000


class TestCrossModelComparison:
    def test_creates_multi_model_plot(self, tmp_path):
        viz = _make_visualizer(tmp_path)
        curves = {
            "claude": _synthetic_fc_curve(200, seed=1),
            "gpt4o": _synthetic_fc_curve(200, seed=2),
            "gemini": _synthetic_fc_curve(200, seed=3),
            "llama": _synthetic_fc_curve(200, seed=4),
        }
        path = viz.plot_cross_model_comparison(curves, save_as="test_cross.png")
        assert path.exists()

    def test_single_model(self, tmp_path):
        viz = _make_visualizer(tmp_path)
        curves = {"claude": _synthetic_fc_curve(100)}
        path = viz.plot_cross_model_comparison(curves, save_as="test_single.png")
        assert path.exists()


class TestInterventionRecovery:
    def test_creates_recovery_plot(self, tmp_path):
        viz = _make_visualizer(tmp_path)
        fc = _synthetic_fc_curve(200)
        # Simulate recovery bumps at intervention points
        for it in [50, 100, 150]:
            for j in range(it - 1, min(it + 10, len(fc))):
                fc[j] = min(1.0, fc[j] + 0.3)
        path = viz.plot_intervention_recovery(
            fc, intervention_ticks=[50, 100, 150], save_as="test_recovery.png"
        )
        assert path.exists()


class TestAnomalyDetectionByPhase:
    def test_creates_adr_phase_plot(self, tmp_path):
        viz = _make_visualizer(tmp_path)
        early = {
            "load": 0.95, "generation": 0.90, "frequency": 0.85,
            "voltage": 0.80, "weather": 0.75, "reserve": 0.70,
        }
        late = {
            "load": 0.88, "generation": 0.85, "frequency": 0.70,
            "voltage": 0.65, "weather": 0.40, "reserve": 0.35,
        }
        path = viz.plot_anomaly_detection_by_phase(early, late, save_as="test_adr.png")
        assert path.exists()
        assert path.stat().st_size > 1000


class TestDecisionAccuracyVsCoverage:
    def test_creates_scatter_plot(self, tmp_path):
        viz = _make_visualizer(tmp_path)
        fc = _synthetic_fc_curve(200)
        da = _synthetic_da_values(fc)
        path = viz.plot_decision_accuracy_vs_coverage(fc, da, save_as="test_da_fc.png")
        assert path.exists()
        assert path.stat().st_size > 1000
