"""Tests for the power grid scenario definition."""

from src.scenarios.power_grid import PowerGridScenario


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
        "reserve": {
            "battery_mwh": 340, "battery_pct": 68,
            "gas_turbine": "standby", "spin_mw": 120,
        },
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
    assert len(scenario.actions) >= 6


def test_format_state_summary():
    """FIX 7: State summary for context reset re-injection."""
    scenario = PowerGridScenario()
    tick_data = {
        "load": {"zone_a": 340, "zone_b": 280, "zone_c": 195},
        "generation": {"plant_1": 290, "plant_2": "TRIPPED", "plant_3": 150, "plant_4": 95},
        "frequency": {"hz": 49.82, "trend": "dropping"},
        "voltage": {"north": 398, "south": 412, "east": 405},
        "weather": {"wind_kmh": 8, "wind_trend": "dropping", "solar": "stable"},
        "reserve": {
            "battery_mwh": 340, "battery_pct": 68,
            "gas_turbine": "standby", "spin_mw": 120,
        },
    }
    summary = scenario.format_state_summary(tick_data, tick_number=50)
    assert "STATE SUMMARY" in summary
    assert "tick 50" in summary
    assert "TRIPPED" in summary
    assert "Continue monitoring" in summary
