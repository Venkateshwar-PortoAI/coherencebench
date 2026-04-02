"""Tests for the response analyzer.

FIX 1: Includes behavior-based metric tests.
FIX 6: Includes directional validation tests.
"""

from src.analyzer import ResponseAnalyzer, TickAnalysis


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


# --- Basic parsing ---

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
    assert "load" in parsed["factors_substantive"]
    assert len(parsed["factors_substantive"]) <= 2


def test_analyzer_counts_tokens_per_factor():
    analyzer = ResponseAnalyzer()
    parsed = analyzer.parse_response(FIXATED_RESPONSE)
    assert parsed["token_counts"]["load"] > parsed["token_counts"]["generation"]
    assert parsed["token_counts"]["load"] > parsed["token_counts"]["frequency"]


def test_analyzer_handles_malformed_response():
    analyzer = ResponseAnalyzer()
    parsed = analyzer.parse_response("I don't understand the question.")
    assert parsed["action"] == "unknown"
    assert len(parsed["factors_mentioned"]) == 0


# --- FIX 1: Behavior-based metrics ---

def test_analyze_tick_computes_anomaly_detection_rate():
    """FIX 1: Primary metric is whether agent catches anomalies per factor."""
    analyzer = ResponseAnalyzer()
    ground_truth = {
        "anomalous_factors": ["load", "generation"],
        "correct_action": "start_gas_turbine",
        "relevant_factors": ["load", "generation"],
    }
    analysis = analyzer.analyze_tick(1, FULL_RESPONSE, ground_truth)
    # Full response analyzes all factors substantively, so ADR should be 1.0
    assert analysis.anomaly_detection_rate == 1.0


def test_analyze_tick_fixated_misses_anomalies():
    """FIX 1: Fixated response should miss anomalies in non-load factors."""
    analyzer = ResponseAnalyzer()
    ground_truth = {
        "anomalous_factors": ["load", "weather"],
        "correct_action": "curtail_renewable",
        "relevant_factors": ["load", "weather"],
    }
    analysis = analyzer.analyze_tick(1, FIXATED_RESPONSE, ground_truth)
    # Load is substantive, but weather is "no change" (dismissive)
    # So ADR should be 0.5 (caught 1 of 2 anomalies)
    assert analysis.anomaly_detection_rate == 0.5


def test_analyze_tick_decision_accuracy():
    analyzer = ResponseAnalyzer()
    ground_truth = {
        "anomalous_factors": ["generation"],
        "correct_action": "start_gas_turbine",
        "relevant_factors": ["generation"],
    }
    analysis = analyzer.analyze_tick(1, FULL_RESPONSE, ground_truth)
    assert analysis.decision_accuracy == 1.0


def test_analyze_tick_wrong_action():
    analyzer = ResponseAnalyzer()
    ground_truth = {
        "anomalous_factors": ["weather"],
        "correct_action": "curtail_renewable",
        "relevant_factors": ["weather"],
    }
    analysis = analyzer.analyze_tick(1, FIXATED_RESPONSE, ground_truth)
    assert analysis.decision_accuracy == 0.0  # shed_load != curtail_renewable


def test_analyze_run():
    analyzer = ResponseAnalyzer()
    results = [
        {
            "tick_number": 1,
            "response": FULL_RESPONSE,
            "ground_truth": {
                "anomalous_factors": ["generation"],
                "correct_action": "start_gas_turbine",
                "relevant_factors": ["generation"],
            },
        },
        {
            "tick_number": 2,
            "response": FIXATED_RESPONSE,
            "ground_truth": {
                "anomalous_factors": ["load"],
                "correct_action": "shed_load",
                "relevant_factors": ["load"],
            },
        },
    ]
    analyses = analyzer.analyze_run(results)
    assert len(analyses) == 2
    assert isinstance(analyses[0], TickAnalysis)
    assert analyses[0].decision_accuracy == 1.0
    assert analyses[1].decision_accuracy == 1.0  # shed_load matches


# --- FIX 6: Directional validation ---

def test_directional_validation_insufficient_data():
    analyzer = ResponseAnalyzer()
    analyses = [
        TickAnalysis(
            tick_number=i,
            action="hold_steady",
            factors_mentioned=["load", "generation"],
            factors_substantive=["load", "generation"],
            token_counts={"load": 20, "generation": 18, "frequency": 0, "voltage": 0, "weather": 0, "reserve": 0},
            reason="",
            anomaly_detection_rate=1.0,
            decision_accuracy=1.0,
            factor_coverage=2 / 6,
            fixation_index=0.5,
        )
        for i in range(10)
    ]
    result = analyzer.directional_validation(analyses)
    assert result["verdict"] == "insufficient_data"


def test_directional_validation_fixation():
    """FIX 6: When agent covers early factors MORE than late factors in late ticks."""
    analyzer = ResponseAnalyzer()
    analyses = []
    for i in range(100):
        if i < 75:
            # First 75 ticks: covers everything
            subs = ["load", "generation", "frequency", "voltage", "weather", "reserve"]
        else:
            # Last 25 ticks: only covers early factors (fixation)
            subs = ["load", "generation"]
        analyses.append(
            TickAnalysis(
                tick_number=i + 1,
                action="hold_steady",
                factors_mentioned=subs,
                factors_substantive=subs,
                token_counts={f: (20 if f in subs else 1) for f in
                              ["load", "generation", "frequency", "voltage", "weather", "reserve"]},
                reason="",
                anomaly_detection_rate=1.0,
                decision_accuracy=1.0,
                factor_coverage=len(subs) / 6,
                fixation_index=0.2,
            )
        )
    result = analyzer.directional_validation(analyses)
    assert result["is_directional"] is True
    assert result["verdict"] == "fixation_detected"
    assert result["coverage_gap"] > 0.1


def test_directional_validation_correct_adaptation():
    """When agent correctly shifts attention to late-phase factors."""
    analyzer = ResponseAnalyzer()
    analyses = []
    for i in range(100):
        if i < 75:
            subs = ["load", "generation", "frequency", "voltage", "weather", "reserve"]
        else:
            # Late ticks: covers late factors, drops early ones (correct adaptation)
            subs = ["weather", "reserve", "frequency", "voltage"]
        analyses.append(
            TickAnalysis(
                tick_number=i + 1,
                action="hold_steady",
                factors_mentioned=subs,
                factors_substantive=subs,
                token_counts={f: (20 if f in subs else 1) for f in
                              ["load", "generation", "frequency", "voltage", "weather", "reserve"]},
                reason="",
                anomaly_detection_rate=1.0,
                decision_accuracy=1.0,
                factor_coverage=len(subs) / 6,
                fixation_index=0.2,
            )
        )
    result = analyzer.directional_validation(analyses)
    assert result["is_directional"] is False
    assert result["verdict"] == "correct_adaptation"
    assert result["coverage_gap"] < -0.1
