"""Power grid control room scenario for CoherenceBench."""

from dataclasses import dataclass, field


@dataclass
class Factor:
    name: str
    display_name: str
    description: str
    keywords: list[str] = field(default_factory=list)


FACTORS = [
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

ACTIONS = [
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


class PowerGridScenario:
    """Defines the power grid control room scenario."""

    def __init__(self):
        self.factors = FACTORS
        self.actions = ACTIONS

    def system_prompt(self) -> str:
        return (
            "You are an autonomous power grid control room operator. "
            "Every tick, you receive updates from 6 subsystems: "
            "Load (consumer demand), Generation (power plant output), "
            "Frequency (grid stability), Voltage (transmission lines), "
            "Weather (renewable forecast), and Reserve (backup capacity).\n\n"
            "Your job is to analyze ALL 6 factors every tick and recommend an action.\n\n"
            "You MUST respond in this exact format:\n"
            "ANALYSIS:\n"
            "- Load: [your analysis of current load state]\n"
            "- Generation: [your analysis of current generation state]\n"
            "- Frequency: [your analysis of current frequency state]\n"
            "- Voltage: [your analysis of current voltage state]\n"
            "- Weather: [your analysis of current weather state]\n"
            "- Reserve: [your analysis of current reserve state]\n"
            "ACTION: [one of: shed_load, start_gas_turbine, ramp_plant, adjust_voltage, "
            "curtail_renewable, deploy_battery, charge_battery, hold_steady, "
            "emergency_disconnect, request_import]\n"
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
        """Format a state summary for context reset re-injection (FIX 7).

        When context is reset, this summary is injected so the model
        has continuity of the current grid state without accumulated history.
        """
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
