"""Generates synthetic tick data with planted anomalies.

Scenario-agnostic: uses the scenario's anomaly maps, state evolution,
and anomaly injection. Includes multi-factor ticks (FIX 2) where the
correct action requires integrating 2+ factors simultaneously.
"""

import random
from dataclasses import dataclass

from .scenarios.base import BaseScenario


@dataclass
class TickGroundTruth:
    anomalous_factors: list[str]
    correct_action: str
    acceptable_actions: list[str]  # Any of these count as "correct"
    relevant_factors: list[str]
    is_multi_factor: bool = False


# Every Nth tick in certain ranges is forced to be a multi-factor tick.
MULTI_FACTOR_INTERVAL = 7


class TickGenerator:
    """Generates reproducible tick data with planted anomalies."""

    def __init__(self, seed: int = 42, num_ticks: int = 200, scenario: BaseScenario | None = None):
        self.seed = seed
        self.num_ticks = num_ticks
        self.rng = random.Random(seed)

        # Default to power grid for backward compatibility
        if scenario is None:
            from .scenarios.power_grid import PowerGridScenario
            self.scenario = PowerGridScenario()
        else:
            self.scenario = scenario

    def generate(self) -> list[dict]:
        ticks = []
        state = self.scenario.deep_copy_state(self.scenario.initial_state)

        anomaly_action_map = self.scenario.anomaly_action_map
        multi_factor_rules = self.scenario.multi_factor_rules
        phase_anomaly_weights = self.scenario.phase_anomaly_weights

        for tick_num in range(self.num_ticks):
            weights = self._get_phase_weights(tick_num, phase_anomaly_weights)
            state = self.scenario.evolve_state(state, self.rng)

            # Decide if this is a multi-factor tick (FIX 2)
            is_multi_factor = (tick_num % MULTI_FACTOR_INTERVAL == 0) and tick_num > 0

            if is_multi_factor:
                tick_data, ground_truth = self._generate_multi_factor_tick(
                    state, tick_num, weights, anomaly_action_map, multi_factor_rules
                )
            else:
                tick_data, ground_truth = self._generate_single_factor_tick(
                    state, weights, anomaly_action_map
                )

            formatted_data = self.scenario.format_tick_data(tick_data)

            ticks.append({
                "tick_number": tick_num + 1,
                "data": formatted_data,
                "ground_truth": {
                    "anomalous_factors": ground_truth.anomalous_factors,
                    "correct_action": ground_truth.correct_action,
                    "acceptable_actions": ground_truth.acceptable_actions,
                    "relevant_factors": ground_truth.relevant_factors,
                    "is_multi_factor": ground_truth.is_multi_factor,
                },
            })

        return ticks

    def _generate_single_factor_tick(
        self, state: dict, weights: dict[str, float], anomaly_action_map: dict
    ) -> tuple[dict, TickGroundTruth]:
        """Standard tick: independent anomaly injection per factor."""
        tick_data = self.scenario.deep_copy_state(state)
        anomalous = []

        for factor, prob in weights.items():
            if self.rng.random() < prob:
                anomalous.append(factor)

        for factor in anomalous:
            tick_data = self.scenario.inject_anomaly(tick_data, factor, self.rng)

        if anomalous:
            primary = anomalous[0]
            action_info = anomaly_action_map[primary]
            correct_action = action_info["primary"]
            # Merge acceptable actions from all anomalous factors
            all_acceptable = set()
            for f in anomalous:
                all_acceptable.update(anomaly_action_map[f]["acceptable"])
            relevant = anomalous.copy()
        else:
            correct_action = "hold_steady"
            all_acceptable = {"hold_steady"}
            relevant = []

        return tick_data, TickGroundTruth(
            anomalous_factors=anomalous,
            correct_action=correct_action,
            acceptable_actions=sorted(all_acceptable),
            relevant_factors=relevant if relevant else ["all"],
            is_multi_factor=False,
        )

    def _generate_multi_factor_tick(
        self, state: dict, tick_num: int, weights: dict[str, float],
        anomaly_action_map: dict, multi_factor_rules: list[tuple],
    ) -> tuple[dict, TickGroundTruth]:
        """FIX 2: Multi-factor tick where correct action requires integrating 2+ factors."""
        tick_data = self.scenario.deep_copy_state(state)

        # Pick a multi-factor rule whose factors have combined phase weight > 0.3
        phase_eligible = [
            (i, rule)
            for i, rule in enumerate(multi_factor_rules)
            if sum(weights.get(f, 0) for f in rule[0]) > 0.3
        ]

        if phase_eligible:
            rule_idx = tick_num % len(phase_eligible)
            _, (rule_factors, correct_action, relevant) = phase_eligible[rule_idx]
        else:
            # Fall back to single-factor tick when no rule fits the phase
            return self._generate_single_factor_tick(state, weights, anomaly_action_map)

        anomalous = list(rule_factors)
        for factor in anomalous:
            tick_data = self.scenario.inject_anomaly(tick_data, factor, self.rng)

        # Possibly add a third anomaly to make it harder
        extra_candidates = [f for f in weights if f not in rule_factors]
        if extra_candidates and self.rng.random() < 0.3:
            extra = self.rng.choice(extra_candidates)
            tick_data = self.scenario.inject_anomaly(tick_data, extra, self.rng)
            anomalous.append(extra)

        # Merge acceptable actions from all anomalous factors
        all_acceptable = {correct_action}
        for f in anomalous:
            all_acceptable.update(anomaly_action_map[f]["acceptable"])

        return tick_data, TickGroundTruth(
            anomalous_factors=anomalous,
            correct_action=correct_action,
            acceptable_actions=sorted(all_acceptable),
            relevant_factors=relevant,
            is_multi_factor=True,
        )

    def _get_phase_weights(self, tick_num: int, phase_anomaly_weights: dict) -> dict[str, float]:
        for (start, end), weights in phase_anomaly_weights.items():
            if start <= tick_num < end:
                return weights
        return list(phase_anomaly_weights.values())[-1]
