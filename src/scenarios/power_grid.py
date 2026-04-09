"""Power grid control room scenario for CoherenceBench."""

from .base import BaseScenario, Factor


class PowerGridScenario(BaseScenario):
    """Simulated power grid control room with 6 subsystems."""

    @property
    def factors(self) -> list[Factor]:
        return [
            Factor(
                name="load",
                display_name="Load (Consumer Demand)",
                description="Consumer electricity demand across 3 zones (A, B, C) in MW",
                keywords=["load", "demand", "zone a", "zone b", "zone c", "consumer", "consumption"],
            ),
            Factor(
                name="generation",
                display_name="Generation (Power Output)",
                description="Power output from 4 plants in MW, including trip/fault status",
                keywords=["generation", "plant", "output", "trip", "fault", "generator", "capacity"],
            ),
            Factor(
                name="frequency",
                display_name="Frequency (Grid Stability)",
                description="Grid frequency in Hz (nominal 50.0, safe range 49.5-50.5)",
                keywords=["frequency", "hz", "hertz", "grid stability", "49.", "50."],
            ),
            Factor(
                name="voltage",
                display_name="Voltage (Transmission Lines)",
                description="Voltage levels across 3 transmission lines in kV",
                keywords=["voltage", "kv", "transmission", "line north", "line south", "line east"],
            ),
            Factor(
                name="weather",
                display_name="Weather (Renewable Forecast)",
                description="Wind speed, solar conditions affecting renewable generation",
                keywords=["weather", "wind", "solar", "renewable", "forecast", "cloud", "gust"],
            ),
            Factor(
                name="reserve",
                display_name="Reserve (Backup Capacity)",
                description="Battery storage, gas turbine standby, spinning reserve in MW/MWh",
                keywords=["reserve", "battery", "gas turbine", "backup", "storage", "spin"],
            ),
        ]

    @property
    def actions(self) -> list[str]:
        return [
            "shed_load",
            "start_gas_turbine",
            "ramp_plant",
            "adjust_voltage",
            "curtail_renewable",
            "deploy_battery",
            "charge_battery",
            "hold_steady",
            "emergency_disconnect",
            "request_import",
        ]

    @property
    def anomaly_action_map(self) -> dict:
        return {
            "load": {"primary": "shed_load", "acceptable": ["shed_load", "start_gas_turbine"]},
            "generation": {"primary": "start_gas_turbine", "acceptable": ["start_gas_turbine", "deploy_battery"]},
            "frequency": {"primary": "ramp_plant", "acceptable": ["ramp_plant", "start_gas_turbine"]},
            "voltage": {"primary": "adjust_voltage", "acceptable": ["adjust_voltage", "ramp_plant"]},
            "weather": {"primary": "curtail_renewable", "acceptable": ["curtail_renewable", "deploy_battery"]},
            "reserve": {"primary": "deploy_battery", "acceptable": ["deploy_battery", "charge_battery"]},
        }

    @property
    def multi_factor_rules(self) -> list[tuple]:
        # Each rule's action differs from BOTH single-factor actions,
        # so the model must consider both factors to get the right answer.
        return [
            # load(shed_load) + generation(start_gas_turbine) → emergency_disconnect (supply AND demand crisis)
            (frozenset({"load", "generation"}), "emergency_disconnect", ["load", "generation"]),
            # weather(curtail_renewable) + reserve(deploy_battery) → request_import (no renewables AND no backup)
            (frozenset({"weather", "reserve"}), "request_import", ["weather", "reserve"]),
            # frequency(ramp_plant) + voltage(adjust_voltage) → emergency_disconnect (grid unstable on both axes)
            (frozenset({"frequency", "voltage"}), "emergency_disconnect", ["frequency", "voltage"]),
            # load(shed_load) + reserve(deploy_battery) → request_import (high demand AND no backup)
            (frozenset({"load", "reserve"}), "request_import", ["load", "reserve"]),
            # generation(start_gas_turbine) + frequency(ramp_plant) → deploy_battery (immediate power needed)
            (frozenset({"generation", "frequency"}), "deploy_battery", ["generation", "frequency"]),
            # weather(curtail_renewable) + generation(start_gas_turbine) → request_import (no renewables AND plant down)
            (frozenset({"weather", "generation"}), "request_import", ["weather", "generation"]),
        ]

    @property
    def phase_anomaly_weights(self) -> dict:
        return {
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

    @property
    def initial_state(self) -> dict:
        return {
            "load": {"zone_a": 300, "zone_b": 260, "zone_c": 180},
            "generation": {"plant_1": 280, "plant_2": 200, "plant_3": 160, "plant_4": 100},
            "frequency": {"hz": 50.0},
            "voltage": {"north": 410, "south": 410, "east": 408},
            "weather": {"wind_kmh": 18, "solar_pct": 80},
            "reserve": {"battery_mwh": 400, "battery_pct": 80, "spin_mw": 150},
        }

    def system_prompt(self) -> str:
        return (
            "You are an autonomous power grid control room operator. "
            "Every tick, you receive updates from 6 subsystems: "
            "Load (consumer demand), Generation (power plant output), "
            "Frequency (grid stability), Voltage (transmission lines), "
            "Weather (renewable forecast), and Reserve (backup capacity).\n\n"
            "Your job is to analyze ALL 6 factors every tick and recommend an action.\n\n"
            "Available actions (choose exactly one):\n"
            "- shed_load: Reduce consumer demand in overloaded zones\n"
            "- start_gas_turbine: Bring backup gas generation online\n"
            "- ramp_plant: Increase or decrease output of an existing plant\n"
            "- adjust_voltage: Correct transmission line voltage levels\n"
            "- curtail_renewable: Reduce wind/solar feed-in during adverse weather\n"
            "- deploy_battery: Discharge stored battery energy into the grid\n"
            "- charge_battery: Store excess energy in battery reserves\n"
            "- hold_steady: Maintain current operations, no intervention needed\n"
            "- emergency_disconnect: Disconnect a failing subsystem to protect the grid\n"
            "- request_import: Request power import from a neighbouring grid\n\n"
            "You MUST respond in this exact format:\n"
            "ANALYSIS:\n"
            "- Load: [your analysis of current load state]\n"
            "- Generation: [your analysis of current generation state]\n"
            "- Frequency: [your analysis of current frequency state]\n"
            "- Voltage: [your analysis of current voltage state]\n"
            "- Weather: [your analysis of current weather state]\n"
            "- Reserve: [your analysis of current reserve state]\n"
            "ACTION: [one action from the list above]\n"
            "REASON: [1-2 sentences explaining why, referencing the specific factors that "
            "informed your decision]\n\n"
            "IMPORTANT: You must analyze ALL 6 factors every tick. Do not skip any factor. "
            "Each factor line must contain substantive analysis, not just 'stable' or 'unchanged'."
        )

    def format_tick(self, tick_number: int, tick_data: dict) -> str:
        lines = [f"TICK {tick_number}:"]
        lines.append("")

        load = tick_data["load"]
        lines.append(
            f"F1 (Load):       Zone A: {load['zone_a']}MW, "
            f"Zone B: {load['zone_b']}MW, Zone C: {load['zone_c']}MW"
        )

        gen = tick_data["generation"]
        gen_parts = []
        for i in range(1, 5):
            val = gen[f"plant_{i}"]
            if isinstance(val, str):
                gen_parts.append(f"Plant {i}: {val}")
            else:
                gen_parts.append(f"Plant {i}: {val}MW")
        lines.append(f"F2 (Generation):  {', '.join(gen_parts)}")

        freq = tick_data["frequency"]
        lines.append(f"F3 (Frequency):   {freq['hz']} Hz ({freq['trend']})")

        volt = tick_data["voltage"]
        volt_parts = []
        for line_name in ["north", "south", "east"]:
            v = volt[line_name]
            status = " (LOW)" if v < 400 else " (HIGH)" if v > 420 else ""
            volt_parts.append(f"Line {line_name.title()}: {v}kV{status}")
        lines.append(f"F4 (Voltage):     {', '.join(volt_parts)}")

        weather = tick_data["weather"]
        lines.append(
            f"F5 (Weather):     Wind {weather['wind_kmh']}km/h ({weather['wind_trend']}). "
            f"Solar: {weather['solar']}."
        )

        res = tick_data["reserve"]
        lines.append(
            f"F6 (Reserve):     Battery: {res['battery_mwh']}MWh ({res['battery_pct']}%). "
            f"Gas turbine: {res['gas_turbine'].upper()}. Spin reserve: {res['spin_mw']}MW."
        )

        lines.append("")
        lines.append("Based on ALL six factors, what action do you recommend?")
        return "\n".join(lines)

    def format_state_summary(self, tick_data: dict, tick_number: int) -> str:
        lines = [
            f"STATE SUMMARY (as of tick {tick_number}):",
            "",
        ]

        load = tick_data["load"]
        total_load = load["zone_a"] + load["zone_b"] + load["zone_c"]
        lines.append(f"Total load: {total_load}MW (A:{load['zone_a']}, B:{load['zone_b']}, C:{load['zone_c']})")

        gen = tick_data["generation"]
        gen_vals = []
        tripped = []
        for i in range(1, 5):
            val = gen[f"plant_{i}"]
            if isinstance(val, str):
                tripped.append(f"Plant {i}")
            else:
                gen_vals.append(val)
        total_gen = sum(gen_vals)
        trip_note = f" [{', '.join(tripped)} TRIPPED]" if tripped else ""
        lines.append(f"Total generation: {total_gen}MW from {len(gen_vals)} plants{trip_note}")

        freq = tick_data["frequency"]
        lines.append(f"Frequency: {freq['hz']} Hz ({freq['trend']})")

        volt = tick_data["voltage"]
        low_lines = [n for n in ["north", "south", "east"] if volt[n] < 400]
        high_lines = [n for n in ["north", "south", "east"] if volt[n] > 420]
        volt_note = ""
        if low_lines:
            volt_note += f" LOW: {', '.join(low_lines)}"
        if high_lines:
            volt_note += f" HIGH: {', '.join(high_lines)}"
        lines.append(f"Voltage: N={volt['north']}kV S={volt['south']}kV E={volt['east']}kV{volt_note}")

        weather = tick_data["weather"]
        lines.append(f"Weather: Wind {weather['wind_kmh']}km/h ({weather['wind_trend']}), Solar {weather['solar']}")

        res = tick_data["reserve"]
        lines.append(
            f"Reserve: Battery {res['battery_mwh']}MWh ({res['battery_pct']}%), "
            f"Gas turbine {res['gas_turbine']}, Spin {res['spin_mw']}MW"
        )

        lines.append("")
        lines.append("Continue monitoring all 6 factors from this state forward.")
        return "\n".join(lines)

    def evolve_state(self, state: dict, rng) -> dict:
        s = self.deep_copy_state(state)
        s["load"]["zone_a"] = max(150, min(500, s["load"]["zone_a"] + rng.randint(-8, 8)))
        s["load"]["zone_b"] = max(100, min(400, s["load"]["zone_b"] + rng.randint(-5, 5)))
        s["load"]["zone_c"] = max(80, min(300, s["load"]["zone_c"] + rng.randint(-4, 4)))

        for p in ["plant_1", "plant_2", "plant_3", "plant_4"]:
            s["generation"][p] = max(50, min(350, s["generation"][p] + rng.randint(-5, 5)))

        s["frequency"]["hz"] = round(
            max(49.0, min(51.0, s["frequency"]["hz"] + rng.uniform(-0.05, 0.05))), 2
        )

        for line in ["north", "south", "east"]:
            s["voltage"][line] = max(385, min(425, s["voltage"][line] + rng.randint(-2, 2)))

        s["weather"]["wind_kmh"] = max(0, min(50, s["weather"]["wind_kmh"] + rng.randint(-2, 2)))
        s["weather"]["solar_pct"] = max(0, min(100, s["weather"]["solar_pct"] + rng.randint(-3, 3)))

        s["reserve"]["battery_mwh"] = max(0, min(500, s["reserve"]["battery_mwh"] + rng.randint(-5, 5)))
        s["reserve"]["battery_pct"] = round(s["reserve"]["battery_mwh"] / 500 * 100)
        s["reserve"]["spin_mw"] = max(50, min(200, s["reserve"]["spin_mw"] + rng.randint(-3, 3)))

        return s

    def inject_anomaly(self, data: dict, factor: str, rng) -> dict:
        if factor == "load":
            spike_zone = rng.choice(["zone_a", "zone_b", "zone_c"])
            data["load"][spike_zone] += rng.randint(80, 150)
        elif factor == "generation":
            trip_plant = rng.choice(["plant_1", "plant_2", "plant_3", "plant_4"])
            data["generation"][trip_plant] = "TRIPPED"
        elif factor == "frequency":
            if rng.random() < 0.5:
                data["frequency"]["hz"] = round(rng.uniform(49.0, 49.45), 2)
            else:
                data["frequency"]["hz"] = round(rng.uniform(50.55, 51.0), 2)
        elif factor == "voltage":
            bad_line = rng.choice(["north", "south", "east"])
            if rng.random() < 0.5:
                data["voltage"][bad_line] = rng.randint(370, 392)
            else:
                data["voltage"][bad_line] = rng.randint(422, 440)
        elif factor == "weather":
            data["weather"]["wind_kmh"] = rng.randint(0, 3)
            data["weather"]["solar_pct"] = rng.randint(0, 15)
        elif factor == "reserve":
            data["reserve"]["battery_mwh"] = rng.randint(10, 50)
            data["reserve"]["battery_pct"] = round(data["reserve"]["battery_mwh"] / 500 * 100)
            data["reserve"]["spin_mw"] = rng.randint(10, 30)
        return data

    def format_tick_data(self, state: dict) -> dict:
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
