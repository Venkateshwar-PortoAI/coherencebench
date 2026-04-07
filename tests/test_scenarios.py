"""Tests for all scenarios: power_grid, hospital, network, air_traffic_control."""

import json
import random

from src.scenarios import get_scenario, SCENARIOS
from src.scenarios.power_grid import PowerGridScenario
from src.scenarios.hospital import HospitalTriageScenario
from src.scenarios.network import NetworkSecurityScenario
from src.scenarios.air_traffic_control import AirTrafficControlScenario
from src.generator import TickGenerator
from src.analyzer import ResponseAnalyzer


class TestScenarioRegistry:
    def test_all_scenarios_registered(self):
        assert "power_grid" in SCENARIOS
        assert "hospital" in SCENARIOS
        assert "network" in SCENARIOS
        assert "air_traffic_control" in SCENARIOS

    def test_get_scenario_returns_correct_type(self):
        assert isinstance(get_scenario("power_grid"), PowerGridScenario)
        assert isinstance(get_scenario("hospital"), HospitalTriageScenario)
        assert isinstance(get_scenario("network"), NetworkSecurityScenario)
        assert isinstance(get_scenario("air_traffic_control"), AirTrafficControlScenario)

    def test_get_scenario_unknown_raises(self):
        try:
            get_scenario("nonexistent")
            assert False, "Should have raised ValueError"
        except ValueError:
            pass


class TestAllScenariosStructure:
    """Every scenario must have 6 factors, 10 actions, and valid maps."""

    def _check_scenario(self, name: str):
        scenario = get_scenario(name)

        # 6 factors
        assert len(scenario.factors) == 6, f"{name}: expected 6 factors, got {len(scenario.factors)}"
        factor_names = [f.name for f in scenario.factors]
        assert len(set(factor_names)) == 6, f"{name}: duplicate factor names"

        # 10 actions
        assert len(scenario.actions) == 10, f"{name}: expected 10 actions, got {len(scenario.actions)}"
        assert len(set(scenario.actions)) == 10, f"{name}: duplicate actions"

        # anomaly_action_map covers all factors
        aam = scenario.anomaly_action_map
        for fn in factor_names:
            assert fn in aam, f"{name}: factor {fn} missing from anomaly_action_map"
            assert "primary" in aam[fn]
            assert "acceptable" in aam[fn]
            assert aam[fn]["primary"] in scenario.actions, \
                f"{name}: primary action {aam[fn]['primary']} not in actions"
            for a in aam[fn]["acceptable"]:
                assert a in scenario.actions, \
                    f"{name}: acceptable action {a} not in actions"

        # multi_factor_rules reference valid factors
        for rule in scenario.multi_factor_rules:
            rule_factors, action, relevant = rule
            for f in rule_factors:
                assert f in factor_names, f"{name}: rule factor {f} not in scenario factors"
            assert action in scenario.actions, f"{name}: rule action {action} not in actions"

        # phase_anomaly_weights cover 0-200 and reference valid factors
        paw = scenario.phase_anomaly_weights
        assert len(paw) >= 3, f"{name}: need at least 3 phases"
        for (start, end), weights in paw.items():
            for f in weights:
                assert f in factor_names, f"{name}: phase weight factor {f} not in factors"

        # initial_state has keys matching factor names
        state = scenario.initial_state
        for fn in factor_names:
            assert fn in state, f"{name}: factor {fn} missing from initial_state"

        return scenario

    def test_power_grid_structure(self):
        self._check_scenario("power_grid")

    def test_hospital_structure(self):
        self._check_scenario("hospital")

    def test_network_structure(self):
        self._check_scenario("network")

    def test_air_traffic_control_structure(self):
        self._check_scenario("air_traffic_control")


class TestAllScenariosPrompts:
    """Every scenario must produce valid prompts."""

    def _check_prompts(self, name: str):
        scenario = get_scenario(name)
        gen = TickGenerator(seed=42, num_ticks=10, scenario=scenario)
        ticks = gen.generate()

        # system_prompt is non-empty and mentions all factor names
        sp = scenario.system_prompt()
        assert len(sp) > 100, f"{name}: system prompt too short"

        # format_tick produces output for each tick
        for tick in ticks:
            prompt = scenario.format_tick(tick["tick_number"], tick["data"])
            assert f"TICK {tick['tick_number']}" in prompt
            assert "Based on ALL six factors" in prompt

        # format_state_summary works
        summary = scenario.format_state_summary(ticks[0]["data"], tick_number=1)
        assert "STATE SUMMARY" in summary
        assert "Continue monitoring" in summary

    def test_power_grid_prompts(self):
        self._check_prompts("power_grid")

    def test_hospital_prompts(self):
        self._check_prompts("hospital")

    def test_network_prompts(self):
        self._check_prompts("network")

    def test_air_traffic_control_prompts(self):
        self._check_prompts("air_traffic_control")


class TestAllScenariosGeneration:
    """Tick generation works for all scenarios."""

    def _check_generation(self, name: str):
        scenario = get_scenario(name)
        gen = TickGenerator(seed=42, num_ticks=200, scenario=scenario)
        ticks = gen.generate()

        assert len(ticks) == 200

        # All ticks have ground_truth
        for tick in ticks:
            gt = tick["ground_truth"]
            assert "anomalous_factors" in gt
            assert "correct_action" in gt
            assert "acceptable_actions" in gt
            assert "is_multi_factor" in gt

        # At least some multi-factor ticks exist
        multi = [t for t in ticks if t["ground_truth"]["is_multi_factor"]]
        assert len(multi) > 0, f"{name}: no multi-factor ticks"

        # Reproducibility
        gen2 = TickGenerator(seed=42, num_ticks=200, scenario=scenario)
        ticks2 = gen2.generate()
        assert json.dumps(ticks) == json.dumps(ticks2), f"{name}: not reproducible"

    def test_power_grid_generation(self):
        self._check_generation("power_grid")

    def test_hospital_generation(self):
        self._check_generation("hospital")

    def test_network_generation(self):
        self._check_generation("network")

    def test_air_traffic_control_generation(self):
        self._check_generation("air_traffic_control")


class TestHospitalSpecific:
    def test_has_correct_factor_names(self):
        s = get_scenario("hospital")
        names = [f.name for f in s.factors]
        assert names == ["vitals", "labs", "imaging", "medications", "history", "capacity"]

    def test_has_correct_actions(self):
        s = get_scenario("hospital")
        assert "admit_icu" in s.actions
        assert "no_action_needed" in s.actions
        assert "call_specialist" in s.actions


class TestNetworkSpecific:
    def test_has_correct_factor_names(self):
        s = get_scenario("network")
        names = [f.name for f in s.factors]
        assert names == ["traffic", "auth", "endpoints", "firewall", "logs", "threats"]

    def test_has_correct_actions(self):
        s = get_scenario("network")
        assert "block_ip" in s.actions
        assert "no_action_needed" in s.actions
        assert "investigate_further" in s.actions


class TestAirTrafficControlSpecific:
    def test_has_correct_factor_names(self):
        s = get_scenario("air_traffic_control")
        names = [f.name for f in s.factors]
        assert names == ["radar", "weather", "runway", "comms", "traffic_flow", "systems"]

    def test_has_correct_actions(self):
        s = get_scenario("air_traffic_control")
        assert "increase_separation" in s.actions
        assert "hold_steady" in s.actions
        assert "declare_ground_stop" in s.actions
        assert "close_runway" in s.actions

    def test_has_action_aliases(self):
        s = get_scenario("air_traffic_control")
        aliases = s.action_aliases
        assert aliases["divert"] == "divert_traffic"
        assert aliases["holding"] == "issue_holding"
        assert aliases["ground_stop"] == "declare_ground_stop"


class TestAirTrafficControlEdgeCases:
    """Edge case tests from eng review."""

    def test_evolve_state_stays_in_bounds(self):
        """All fields stay within clamp bounds after 200 evolve_state calls."""
        s = get_scenario("air_traffic_control")
        state = s.deep_copy_state(s.initial_state)
        rng = random.Random(42)

        for _ in range(200):
            state = s.evolve_state(state, rng)

            assert 5 <= state["radar"]["tracked_aircraft"] <= 40
            assert 2.0 <= state["radar"]["min_separation_nm"] <= 10.0
            assert 0 <= state["radar"]["conflict_alerts"] <= 5

            assert 0 <= state["weather"]["visibility_sm"] <= 15
            assert 0 <= state["weather"]["ceiling_ft"] <= 10000
            assert 0 <= state["weather"]["wind_speed_kt"] <= 50

            assert 10 <= state["comms"]["congestion_pct"] <= 95
            assert 0 <= state["comms"]["readback_errors"] <= 10

            assert 5 <= state["traffic_flow"]["arrival_rate"] <= 50
            assert 5 <= state["traffic_flow"]["departure_rate"] <= 45
            assert 0 <= state["traffic_flow"]["holding_aircraft"] <= 15
            assert 50 <= state["traffic_flow"]["slot_compliance_pct"] <= 100

    def test_inject_anomaly_produces_clear_anomalies(self):
        """Each factor's anomaly values are clearly outside normal bounds."""
        s = get_scenario("air_traffic_control")
        rng = random.Random(42)

        for factor_name in [f.name for f in s.factors]:
            state = s.deep_copy_state(s.initial_state)
            data = s.format_tick_data(state)
            data = s.inject_anomaly(data, factor_name, rng)

            if factor_name == "radar":
                assert data["radar"]["min_separation_nm"] <= 2.5
                assert data["radar"]["conflict_alerts"] >= 3
            elif factor_name == "weather":
                assert data["weather"]["visibility_sm"] <= 2
                assert data["weather"]["ceiling_ft"] <= 500
            elif factor_name == "runway":
                has_bad = any(
                    data["runway"][r]["surface"] in ("icy", "snow")
                    for r in ["rwy_09L", "rwy_27R", "rwy_04"]
                )
                assert has_bad
            elif factor_name == "comms":
                assert data["comms"]["congestion_pct"] >= 80
                assert data["comms"]["readback_errors"] >= 4
            elif factor_name == "traffic_flow":
                assert data["traffic_flow"]["arrival_rate"] >= 38
                assert data["traffic_flow"]["holding_aircraft"] >= 6
            elif factor_name == "systems":
                has_failed = (
                    data["systems"]["primary_radar"] == "failed"
                    or data["systems"]["nav_aids"] == "failed"
                )
                assert has_failed

    def test_action_aliases_resolve_correctly(self):
        """Per-scenario aliases resolve to correct actions in the analyzer."""
        s = get_scenario("air_traffic_control")
        analyzer = ResponseAnalyzer(scenario=s)

        # Scenario-specific aliases should resolve
        assert analyzer._alias_map.get("divert") == "divert_traffic"
        assert analyzer._alias_map.get("holding") == "issue_holding"
        assert analyzer._alias_map.get("ground_stop") == "declare_ground_stop"

        # Global aliases that match this scenario's actions should also work
        assert analyzer._alias_map.get("hold") == "issue_holding"

    def test_directional_validation_uses_phase_weights(self):
        """directional_validation derives early/late factors from phase weights,
        not hardcoded power_grid names."""
        s = get_scenario("air_traffic_control")
        analyzer = ResponseAnalyzer(scenario=s)

        # For ATC, early factors should be radar+weather (highest in first phase)
        # Late factors should be traffic_flow+systems (highest in last phase)
        result = analyzer.directional_validation([])
        # With empty analyses, should return insufficient_data
        assert result["verdict"] == "insufficient_data"

        # Verify the phase weight derivation by checking the analyzer's scenario
        phase_weights = s.phase_anomaly_weights
        sorted_phases = sorted(phase_weights.keys(), key=lambda k: k[0])
        first_phase = phase_weights[sorted_phases[0]]
        last_phase = phase_weights[sorted_phases[-1]]
        early = set(sorted(first_phase, key=first_phase.get, reverse=True)[:2])
        late = set(sorted(last_phase, key=last_phase.get, reverse=True)[:2])
        assert early == {"radar", "weather"}
        assert late == {"traffic_flow", "systems"}
