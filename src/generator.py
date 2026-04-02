"""Generates synthetic power grid tick data with planted anomalies.

Includes multi-factor ticks (FIX 2) where the correct action requires
integrating 2+ factors simultaneously.
"""

import random
from dataclasses import dataclass


@dataclass
class TickGroundTruth:
    anomalous_factors: list[str]
    correct_action: str
    acceptable_actions: list[str]  # Any of these count as "correct"
    relevant_factors: list[str]
    is_multi_factor: bool = False


# Anomaly probability by factor per phase.
# Early phases: anomalies in load/generation.
# Late phases: anomalies shift to weather/reserve.
# This creates the "attention trap" -- the agent fixates on where anomalies
# WERE, not where they ARE.
PHASE_ANOMALY_WEIGHTS = {
    (0, 40): {
        "load": 0.4, "generation": 0.35, "frequency": 0.15,
        "voltage": 0.1, "weather": 0.05, "reserve": 0.05,
    },
    (40, 80): {
        "load": 0.25, "generation": 0.25, "frequency": 0.2,
        "voltage": 0.2, "weather": 0.1, "reserve": 0.1,
    },
    (80, 120): {
        "load": 0.15, "generation": 0.15, "frequency": 0.15,
        "voltage": 0.15, "weather": 0.2, "reserve": 0.2,
    },
    (120, 160): {
        "load": 0.1, "generation": 0.1, "frequency": 0.1,
        "voltage": 0.15, "weather": 0.3, "reserve": 0.3,
    },
    (160, 200): {
        "load": 0.05, "generation": 0.05, "frequency": 0.15,
        "voltage": 0.1, "weather": 0.35, "reserve": 0.35,
    },
}

# Single-factor anomaly -> primary action + acceptable alternatives
ANOMALY_ACTION_MAP = {
    "load": {"primary": "shed_load", "acceptable": ["shed_load", "start_gas_turbine", "ramp_plant"]},
    "generation": {"primary": "start_gas_turbine", "acceptable": ["start_gas_turbine", "ramp_plant", "deploy_battery"]},
    "frequency": {"primary": "ramp_plant", "acceptable": ["ramp_plant", "start_gas_turbine"]},
    "voltage": {"primary": "adjust_voltage", "acceptable": ["adjust_voltage", "ramp_plant"]},
    "weather": {"primary": "curtail_renewable", "acceptable": ["curtail_renewable", "deploy_battery", "start_gas_turbine"]},
    "reserve": {"primary": "deploy_battery", "acceptable": ["deploy_battery", "start_gas_turbine", "charge_battery"]},
}

# FIX 2: Multi-factor combinations and their correct actions.
# Each entry: (frozenset of required factors, correct_action, relevant_factors)
MULTI_FACTOR_RULES = [
    (
        frozenset({"load", "generation"}),
        "start_gas_turbine",
        ["load", "generation"],
    ),
    (
        frozenset({"weather", "reserve"}),
        "deploy_battery",
        ["weather", "reserve"],
    ),
    (
        frozenset({"frequency", "voltage"}),
        "ramp_plant",
        ["frequency", "voltage"],
    ),
    (
        frozenset({"load", "reserve"}),
        "deploy_battery",
        ["load", "reserve"],
    ),
    (
        frozenset({"generation", "frequency"}),
        "start_gas_turbine",
        ["generation", "frequency"],
    ),
    (
        frozenset({"weather", "generation"}),
        "curtail_renewable",
        ["weather", "generation"],
    ),
]

# Every Nth tick in certain ranges is forced to be a multi-factor tick.
MULTI_FACTOR_INTERVAL = 7


class TickGenerator:
    """Generates reproducible tick data with planted anomalies."""

    def __init__(self, seed: int = 42, num_ticks: int = 200):
        self.seed = seed
        self.num_ticks = num_ticks
        self.rng = random.Random(seed)

    def generate(self) -> list[dict]:
        ticks = []
        state = {
            "load": {"zone_a": 300, "zone_b": 260, "zone_c": 180},
            "generation": {"plant_1": 280, "plant_2": 200, "plant_3": 160, "plant_4": 100},
            "frequency": {"hz": 50.0},
            "voltage": {"north": 410, "south": 410, "east": 408},
            "weather": {"wind_kmh": 18, "solar_pct": 80},
            "reserve": {"battery_mwh": 400, "battery_pct": 80, "spin_mw": 150},
        }

        for tick_num in range(self.num_ticks):
            weights = self._get_phase_weights(tick_num)
            state = self._evolve_state(state)

            # Decide if this is a multi-factor tick (FIX 2)
            is_multi_factor = (tick_num % MULTI_FACTOR_INTERVAL == 0) and tick_num > 0

            if is_multi_factor:
                tick_data, ground_truth = self._generate_multi_factor_tick(
                    state, tick_num, weights
                )
            else:
                tick_data, ground_truth = self._generate_single_factor_tick(
                    state, weights
                )

            formatted_data = self._format_tick_data(tick_data)

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
        self, state: dict, weights: dict[str, float]
    ) -> tuple[dict, TickGroundTruth]:
        """Standard tick: independent anomaly injection per factor."""
        tick_data = self._deep_copy_state(state)
        anomalous = []

        for factor, prob in weights.items():
            if self.rng.random() < prob:
                anomalous.append(factor)

        for factor in anomalous:
            tick_data = self._inject_anomaly(tick_data, factor)

        if anomalous:
            primary = anomalous[0]
            action_info = ANOMALY_ACTION_MAP[primary]
            correct_action = action_info["primary"]
            # Merge acceptable actions from all anomalous factors
            all_acceptable = set()
            for f in anomalous:
                all_acceptable.update(ANOMALY_ACTION_MAP[f]["acceptable"])
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
        self, state: dict, tick_num: int, weights: dict[str, float]
    ) -> tuple[dict, TickGroundTruth]:
        """FIX 2: Multi-factor tick where correct action requires integrating 2+ factors."""
        tick_data = self._deep_copy_state(state)

        # Pick a multi-factor rule, biased by current phase weights
        rule_idx = tick_num % len(MULTI_FACTOR_RULES)
        rule_factors, correct_action, relevant = MULTI_FACTOR_RULES[rule_idx]

        anomalous = list(rule_factors)
        for factor in anomalous:
            tick_data = self._inject_anomaly(tick_data, factor)

        # Possibly add a third anomaly to make it harder
        extra_candidates = [f for f in weights if f not in rule_factors]
        if extra_candidates and self.rng.random() < 0.3:
            extra = self.rng.choice(extra_candidates)
            tick_data = self._inject_anomaly(tick_data, extra)
            anomalous.append(extra)

        # Merge acceptable actions from all anomalous factors
        all_acceptable = {correct_action}
        for f in anomalous:
            all_acceptable.update(ANOMALY_ACTION_MAP[f]["acceptable"])

        return tick_data, TickGroundTruth(
            anomalous_factors=anomalous,
            correct_action=correct_action,
            acceptable_actions=sorted(all_acceptable),
            relevant_factors=relevant,
            is_multi_factor=True,
        )

    def _get_phase_weights(self, tick_num: int) -> dict[str, float]:
        for (start, end), weights in PHASE_ANOMALY_WEIGHTS.items():
            if start <= tick_num < end:
                return weights
        return list(PHASE_ANOMALY_WEIGHTS.values())[-1]

    def _evolve_state(self, state: dict) -> dict:
        """Small random walk to make data look natural."""
        s = self._deep_copy_state(state)
        s["load"]["zone_a"] = max(150, min(500, s["load"]["zone_a"] + self.rng.randint(-8, 8)))
        s["load"]["zone_b"] = max(100, min(400, s["load"]["zone_b"] + self.rng.randint(-5, 5)))
        s["load"]["zone_c"] = max(80, min(300, s["load"]["zone_c"] + self.rng.randint(-4, 4)))

        for p in ["plant_1", "plant_2", "plant_3", "plant_4"]:
            s["generation"][p] = max(50, min(350, s["generation"][p] + self.rng.randint(-5, 5)))

        s["frequency"]["hz"] = round(
            max(49.0, min(51.0, s["frequency"]["hz"] + self.rng.uniform(-0.05, 0.05))), 2
        )

        for line in ["north", "south", "east"]:
            s["voltage"][line] = max(385, min(425, s["voltage"][line] + self.rng.randint(-2, 2)))

        s["weather"]["wind_kmh"] = max(0, min(50, s["weather"]["wind_kmh"] + self.rng.randint(-2, 2)))
        s["weather"]["solar_pct"] = max(0, min(100, s["weather"]["solar_pct"] + self.rng.randint(-3, 3)))

        s["reserve"]["battery_mwh"] = max(0, min(500, s["reserve"]["battery_mwh"] + self.rng.randint(-5, 5)))
        s["reserve"]["battery_pct"] = round(s["reserve"]["battery_mwh"] / 500 * 100)
        s["reserve"]["spin_mw"] = max(50, min(200, s["reserve"]["spin_mw"] + self.rng.randint(-3, 3)))

        return s

    def _inject_anomaly(self, data: dict, factor: str) -> dict:
        """Inject a clear anomaly into the specified factor."""
        if factor == "load":
            spike_zone = self.rng.choice(["zone_a", "zone_b", "zone_c"])
            data["load"][spike_zone] += self.rng.randint(80, 150)
        elif factor == "generation":
            trip_plant = self.rng.choice(["plant_1", "plant_2", "plant_3", "plant_4"])
            data["generation"][trip_plant] = "TRIPPED"
        elif factor == "frequency":
            if self.rng.random() < 0.5:
                data["frequency"]["hz"] = round(self.rng.uniform(49.0, 49.45), 2)
            else:
                data["frequency"]["hz"] = round(self.rng.uniform(50.55, 51.0), 2)
        elif factor == "voltage":
            bad_line = self.rng.choice(["north", "south", "east"])
            if self.rng.random() < 0.5:
                data["voltage"][bad_line] = self.rng.randint(370, 392)
            else:
                data["voltage"][bad_line] = self.rng.randint(422, 440)
        elif factor == "weather":
            data["weather"]["wind_kmh"] = self.rng.randint(0, 3)
            data["weather"]["solar_pct"] = self.rng.randint(0, 15)
        elif factor == "reserve":
            data["reserve"]["battery_mwh"] = self.rng.randint(10, 50)
            data["reserve"]["battery_pct"] = round(data["reserve"]["battery_mwh"] / 500 * 100)
            data["reserve"]["spin_mw"] = self.rng.randint(10, 30)
        return data

    def _format_tick_data(self, state: dict) -> dict:
        """Format raw state into the dict expected by scenario.format_tick()."""
        freq_hz = state["frequency"]["hz"]
        if freq_hz < 49.8:
            trend = "dropping"
        elif freq_hz > 50.2:
            trend = "rising"
        else:
            trend = "stable"

        wind = state["weather"]["wind_kmh"]
        if wind < 5:
            wind_trend = "calm"
        elif wind > 30:
            wind_trend = "gusting"
        else:
            wind_trend = "steady"

        solar = state["weather"]["solar_pct"]
        solar_str = "strong" if solar > 60 else "moderate" if solar > 30 else "weak" if solar > 10 else "minimal"

        gas_state = "standby"
        if state["reserve"]["spin_mw"] < 40:
            gas_state = "offline"
        elif state["reserve"]["battery_pct"] < 20:
            gas_state = "active"

        gen_data = {}
        for p in ["plant_1", "plant_2", "plant_3", "plant_4"]:
            gen_data[p] = state["generation"][p]

        return {
            "load": {
                "zone_a": state["load"]["zone_a"],
                "zone_b": state["load"]["zone_b"],
                "zone_c": state["load"]["zone_c"],
            },
            "generation": gen_data,
            "frequency": {"hz": freq_hz, "trend": trend},
            "voltage": {
                "north": state["voltage"]["north"],
                "south": state["voltage"]["south"],
                "east": state["voltage"]["east"],
            },
            "weather": {
                "wind_kmh": wind,
                "wind_trend": wind_trend,
                "solar": solar_str,
            },
            "reserve": {
                "battery_mwh": state["reserve"]["battery_mwh"],
                "battery_pct": state["reserve"]["battery_pct"],
                "gas_turbine": gas_state,
                "spin_mw": state["reserve"]["spin_mw"],
            },
        }

    def _deep_copy_state(self, state: dict) -> dict:
        return {k: dict(v) for k, v in state.items()}
