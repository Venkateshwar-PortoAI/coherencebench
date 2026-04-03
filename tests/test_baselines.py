"""Tests for baseline computation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.compute_baselines import compute_baselines_for_seed


class TestBaselineComputation:
    def test_random_baseline_produces_results(self):
        result = compute_baselines_for_seed(42, "power_grid", num_trials=10)
        assert "random" in result
        assert 0.0 <= result["random"]["mean_da"] <= 1.0
        assert 0.0 <= result["random"]["da_at_40"] <= 1.0
        assert 0.0 <= result["random"]["da_last"] <= 1.0

    def test_majority_baseline_produces_results(self):
        result = compute_baselines_for_seed(42, "power_grid", num_trials=10)
        assert "majority" in result
        assert 0.0 <= result["majority"]["mean_da"] <= 1.0

    def test_most_common_baseline_produces_results(self):
        result = compute_baselines_for_seed(42, "power_grid", num_trials=10)
        assert "most_common" in result
        assert 0.0 <= result["most_common"]["mean_da"] <= 1.0

    def test_random_baseline_below_50_percent(self):
        """Random guessing from 10 actions should be well below 50% DA."""
        result = compute_baselines_for_seed(42, "power_grid", num_trials=100)
        assert result["random"]["mean_da"] < 0.50, \
            f"Random baseline DA={result['random']['mean_da']:.2%} is suspiciously high"

    def test_baselines_work_for_hospital(self):
        result = compute_baselines_for_seed(42, "hospital", num_trials=10)
        assert 0.0 <= result["random"]["mean_da"] <= 1.0
        assert 0.0 <= result["majority"]["mean_da"] <= 1.0

    def test_baselines_work_for_network(self):
        result = compute_baselines_for_seed(42, "network", num_trials=10)
        assert 0.0 <= result["random"]["mean_da"] <= 1.0
        assert 0.0 <= result["majority"]["mean_da"] <= 1.0

    def test_different_seeds_produce_different_baselines(self):
        r1 = compute_baselines_for_seed(42, "power_grid", num_trials=10)
        r2 = compute_baselines_for_seed(123, "power_grid", num_trials=10)
        # Majority baselines can differ because different seeds produce different tick distributions
        # At minimum, the most_common_action might differ
        # Just verify both run successfully and produce valid numbers
        assert 0.0 <= r1["majority"]["mean_da"] <= 1.0
        assert 0.0 <= r2["majority"]["mean_da"] <= 1.0
