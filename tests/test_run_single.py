import logging

from scripts.run_single import _build_failure_cases, _make_progress_callback
from src.analyzer import ResponseAnalyzer


FULL_RESPONSE = (
    "ANALYSIS:\n"
    "- Load: Zone A demand is elevated at 340MW while the other zones are holding steady.\n"
    "- Generation: Plant 2 is offline and reserve generation is covering part of the gap.\n"
    "- Frequency: Frequency is drifting down toward the lower bound and needs correction.\n"
    "- Voltage: Transmission voltage remains acceptable but has started to soften in the north.\n"
    "- Weather: Wind output is weakening, which reduces renewable support over the next interval.\n"
    "- Reserve: Battery charge is still healthy and the gas turbine is available if needed.\n"
    "ACTION: start_gas_turbine\n"
    "REASON: Generation loss plus falling frequency requires fast backup supply."
)


def _event(tick_number: int, *, context_reset: bool = False, intervention: bool = False) -> dict:
    return {
        "event": "tick_completed",
        "tick_number": tick_number,
        "num_ticks": 3,
        "total_ticks": 3,
        "cycle_index": tick_number,
        "duration_seconds": 0.5,
        "response": FULL_RESPONSE,
        "ground_truth": {
            "anomalous_factors": ["generation", "frequency"],
            "correct_action": "start_gas_turbine",
            "acceptable_actions": ["start_gas_turbine"],
        },
        "context_truncated": False,
        "context_reset": context_reset,
        "intervention": intervention,
    }


def test_progress_callback_logs_progress_and_events(caplog):
    callback, _stats = _make_progress_callback("power_grid", total_ticks=3, progress_every=2)

    with caplog.at_level(logging.INFO, logger="run_single"):
        callback(_event(1))
        callback(_event(2, context_reset=True))
        callback(_event(3, intervention=True))

    assert "Tick 002/003" in caplog.text
    assert "Tick 003/003" in caplog.text
    assert "Checkpoint 002/003" in caplog.text
    assert "anomaly=generation,frequency" in caplog.text
    assert "expected=start_gas_turbine" in caplog.text
    assert "flags=reset" in caplog.text
    assert "flags=intervention" in caplog.text
    assert "hit" in caplog.text


def test_build_failure_cases_only_keeps_non_clean_ticks():
    analyzer = ResponseAnalyzer()
    raw_results = [
        {
            "tick_number": 1,
            "response": FULL_RESPONSE,
            "ground_truth": {
                "anomalous_factors": ["generation", "frequency"],
                "correct_action": "start_gas_turbine",
                "acceptable_actions": ["start_gas_turbine"],
                "relevant_factors": ["generation", "frequency"],
            },
            "context_truncated": False,
        },
        {
            "tick_number": 2,
            "response": FULL_RESPONSE.replace("ACTION: start_gas_turbine", "ACTION: hold_steady"),
            "ground_truth": {
                "anomalous_factors": ["generation"],
                "correct_action": "start_gas_turbine",
                "acceptable_actions": ["start_gas_turbine"],
                "relevant_factors": ["generation"],
            },
            "context_truncated": True,
        },
    ]
    analyses = analyzer.analyze_run(raw_results)

    failures = _build_failure_cases(raw_results, analyses)

    assert len(failures) == 1
    assert failures[0]["tick"] == 2
    assert failures[0]["chosen_action"] == "hold_steady"
    assert failures[0]["expected_action"] == "start_gas_turbine"
    assert failures[0]["verdict"] == "miss"
    assert failures[0]["context_truncated"] is True
