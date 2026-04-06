"""Tests for all 5 CoherenceBench metrics (FIX 8)."""

from src.metrics import (
    factor_coverage,
    fixation_index,
    decision_accuracy,
    anomaly_detection_rate,
    intervention_recovery,
    compute_all_metrics,
)


# --- FC: Factor Coverage ---

def test_factor_coverage_all_mentioned():
    factors_mentioned = ["load", "generation", "frequency", "voltage", "weather", "reserve"]
    assert factor_coverage(factors_mentioned, total=6) == 1.0


def test_factor_coverage_partial():
    factors_mentioned = ["load", "generation"]
    assert factor_coverage(factors_mentioned, total=6) == 2 / 6


def test_factor_coverage_none():
    assert factor_coverage([], total=6) == 0.0


def test_factor_coverage_zero_total():
    assert factor_coverage([], total=0) == 1.0


# --- FI: Fixation Index ---

def test_fixation_index_balanced():
    token_counts = {
        "load": 20, "generation": 18, "frequency": 22,
        "voltage": 19, "weather": 21, "reserve": 20,
    }
    fi = fixation_index(token_counts)
    assert fi < 0.25


def test_fixation_index_fixated():
    token_counts = {
        "load": 100, "generation": 5, "frequency": 3,
        "voltage": 2, "weather": 1, "reserve": 1,
    }
    fi = fixation_index(token_counts)
    assert fi > 0.7


def test_fixation_index_empty():
    assert fixation_index({}) == 0.0


def test_fixation_index_single_factor():
    assert fixation_index({"load": 50}) == 1.0


# --- DA: Decision Accuracy ---

def test_decision_accuracy_correct():
    assert decision_accuracy("start_gas_turbine", "start_gas_turbine") == 1.0


def test_decision_accuracy_wrong():
    assert decision_accuracy("shed_load", "start_gas_turbine") == 0.0


def test_decision_accuracy_case_insensitive():
    assert decision_accuracy("Start_Gas_Turbine", "start_gas_turbine") == 1.0


def test_decision_accuracy_whitespace():
    assert decision_accuracy("  shed_load  ", "shed_load") == 1.0


# --- ADR: Anomaly Detection Rate (FIX 1 primary metric) ---

def test_anomaly_detection_rate_full():
    mentioned = ["load", "weather"]
    anomalous = ["load", "weather"]
    assert anomaly_detection_rate(mentioned, anomalous) == 1.0


def test_anomaly_detection_rate_partial():
    mentioned = ["load", "generation"]
    anomalous = ["load", "weather"]
    rate = anomaly_detection_rate(mentioned, anomalous)
    assert rate == 0.5


def test_anomaly_detection_rate_none_detected():
    mentioned = ["frequency", "voltage"]
    anomalous = ["load", "weather"]
    assert anomaly_detection_rate(mentioned, anomalous) == 0.0


def test_anomaly_detection_rate_no_anomalies():
    rate = anomaly_detection_rate(["load"], [])
    assert rate == 1.0


# --- IR: Intervention Recovery ---

def test_intervention_recovery_basic():
    fc_values = [0.8, 0.6, 0.5, 0.4, 0.3, 1.0, 0.9, 0.7, 0.5, 0.3]
    intervention_tick = 5
    recovery_length = intervention_recovery(fc_values, intervention_tick)
    assert recovery_length == 4  # FC stays above 0.3 for 4 ticks (1.0, 0.9, 0.7, 0.5)


def test_intervention_recovery_no_recovery():
    fc_values = [0.8, 0.6, 0.5, 0.3, 0.2]
    assert intervention_recovery(fc_values, 3) == 0  # 0.2 < 0.5


def test_intervention_recovery_at_start():
    fc_values = [0.5, 0.8, 0.7]
    assert intervention_recovery(fc_values, 0) == 0


def test_intervention_recovery_out_of_bounds():
    fc_values = [0.5, 0.4]
    assert intervention_recovery(fc_values, 10) == 0


# --- compute_all_metrics ---

def test_compute_all_metrics():
    result = compute_all_metrics(
        factors_substantive=["load", "generation", "frequency"],
        word_counts={
            "load": 20, "generation": 18, "frequency": 15,
            "voltage": 3, "weather": 2, "reserve": 1,
        },
        predicted_action="start_gas_turbine",
        correct_action="start_gas_turbine",
        anomalous_factors=["load", "generation"],
    )
    assert result["factor_coverage"] == 3 / 6
    assert result["decision_accuracy"] == 1.0
    assert result["anomaly_detection_rate"] == 1.0
    assert 0.0 <= result["fixation_index"] <= 1.0
