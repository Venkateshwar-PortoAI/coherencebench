"""Tests for tick data generation, including multi-factor ticks (FIX 2)."""

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
    """Early anomalies in load/generation, later in weather/reserve."""
    gen = TickGenerator(seed=42, num_ticks=200)
    ticks = gen.generate()

    early_anomaly_factors = set()
    late_anomaly_factors = set()

    for tick in ticks[:30]:
        early_anomaly_factors.update(tick["ground_truth"]["anomalous_factors"])
    for tick in ticks[150:]:
        late_anomaly_factors.update(tick["ground_truth"]["anomalous_factors"])

    assert "load" in early_anomaly_factors or "generation" in early_anomaly_factors
    assert "weather" in late_anomaly_factors or "reserve" in late_anomaly_factors


def test_tick_data_values_in_range():
    gen = TickGenerator(seed=42, num_ticks=50)
    ticks = gen.generate()
    for tick in ticks:
        d = tick["data"]
        assert 48.0 <= d["frequency"]["hz"] <= 52.0
        for line in ["north", "south", "east"]:
            assert 370 <= d["voltage"][line] <= 440
        assert 0 <= d["reserve"]["battery_pct"] <= 100


def test_multi_factor_ticks_exist():
    """FIX 2: Some ticks should require integrating 2+ factors."""
    gen = TickGenerator(seed=42, num_ticks=50)
    ticks = gen.generate()
    multi_factor_ticks = [
        t for t in ticks if t["ground_truth"].get("is_multi_factor", False)
    ]
    assert len(multi_factor_ticks) > 0, "Expected at least one multi-factor tick"


def test_multi_factor_ticks_have_multiple_anomalous_factors():
    """FIX 2: Multi-factor ticks should have 2+ anomalous factors."""
    gen = TickGenerator(seed=42, num_ticks=50)
    ticks = gen.generate()
    multi_factor_ticks = [
        t for t in ticks if t["ground_truth"].get("is_multi_factor", False)
    ]
    for tick in multi_factor_ticks:
        assert len(tick["ground_truth"]["anomalous_factors"]) >= 2, (
            f"Multi-factor tick {tick['tick_number']} has fewer than 2 anomalous factors"
        )


def test_multi_factor_ticks_have_multiple_relevant_factors():
    """FIX 2: Multi-factor ticks should require 2+ factors for correct action."""
    gen = TickGenerator(seed=42, num_ticks=50)
    ticks = gen.generate()
    multi_factor_ticks = [
        t for t in ticks if t["ground_truth"].get("is_multi_factor", False)
    ]
    for tick in multi_factor_ticks:
        assert len(tick["ground_truth"]["relevant_factors"]) >= 2


def test_ground_truth_has_is_multi_factor_field():
    gen = TickGenerator(seed=42, num_ticks=10)
    ticks = gen.generate()
    for tick in ticks:
        assert "is_multi_factor" in tick["ground_truth"]
