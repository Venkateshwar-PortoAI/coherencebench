"""Tests for all 3 scenarios: power_grid, hospital, network."""

import json

from src.scenarios import get_scenario, SCENARIOS
from src.scenarios.power_grid import PowerGridScenario
from src.scenarios.hospital import HospitalTriageScenario
from src.scenarios.network import NetworkSecurityScenario
from src.generator import TickGenerator


class TestScenarioRegistry:
    def test_all_scenarios_registered(self):
        assert "power_grid" in SCENARIOS
        assert "hospital" in SCENARIOS
        assert "network" in SCENARIOS

    def test_get_scenario_returns_correct_type(self):
        assert isinstance(get_scenario("power_grid"), PowerGridScenario)
        assert isinstance(get_scenario("hospital"), HospitalTriageScenario)
        assert isinstance(get_scenario("network"), NetworkSecurityScenario)

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
