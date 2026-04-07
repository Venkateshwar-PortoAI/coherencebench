"""Air traffic control scenario for CoherenceBench."""

from .base import BaseScenario, Factor


class AirTrafficControlScenario(BaseScenario):
    """Simulated air traffic control tower with 6 subsystems."""

    @property
    def factors(self) -> list[Factor]:
        return [
            Factor(
                name="radar",
                display_name="Radar (Aircraft Separation)",
                description="Aircraft positions, separation distances in NM, conflict alerts",
                keywords=[
                    "radar", "separation", "conflict", "aircraft", "position",
                    "nm", "nautical", "track", "blip", "proximity",
                ],
            ),
            Factor(
                name="weather",
                display_name="Weather (Conditions & Forecast)",
                description="Visibility in SM, ceiling in ft, wind speed/direction, turbulence, wind shear",
                keywords=[
                    "weather", "visibility", "ceiling", "wind", "turbulence",
                    "shear", "cloud", "fog", "storm", "gust",
                ],
            ),
            Factor(
                name="runway",
                display_name="Runway (Operations & Surface)",
                description="Runway occupancy status, surface conditions (dry/wet/icy), braking action",
                keywords=[
                    "runway", "surface", "braking", "occupancy", "closed",
                    "wet", "icy", "snow", "friction", "threshold",
                ],
            ),
            Factor(
                name="comms",
                display_name="Comms (Frequency & Coordination)",
                description="Frequency congestion level, pilot readback errors, ATIS currency in minutes",
                keywords=[
                    "comms", "frequency", "readback", "atis", "congestion",
                    "pilot", "radio", "transmission", "coordination", "handoff",
                ],
            ),
            Factor(
                name="traffic_flow",
                display_name="Traffic Flow (Arrivals & Departures)",
                description="Arrival rate, departure rate, holding aircraft count, slot compliance %",
                keywords=[
                    "traffic", "flow", "arrival", "departure", "holding",
                    "slot", "sequence", "delay", "queue", "rate",
                ],
            ),
            Factor(
                name="systems",
                display_name="Systems (Equipment Health)",
                description="Primary radar status, nav aid status, backup power, ATIS broadcast status",
                keywords=[
                    "systems", "equipment", "radar health", "nav aid", "backup",
                    "power", "ils", "vor", "transponder", "atis broadcast",
                ],
            ),
        ]

    @property
    def actions(self) -> list[str]:
        return [
            "increase_separation",
            "issue_holding",
            "divert_traffic",
            "close_runway",
            "reduce_arrival_rate",
            "switch_runway_config",
            "issue_weather_advisory",
            "activate_backup_systems",
            "hold_steady",
            "declare_ground_stop",
        ]

    @property
    def action_aliases(self) -> dict[str, str]:
        return {
            "separate": "increase_separation",
            "holding": "issue_holding",
            "divert": "divert_traffic",
            "close": "close_runway",
            "reduce": "reduce_arrival_rate",
            "switch": "switch_runway_config",
            "advisory": "issue_weather_advisory",
            "backup": "activate_backup_systems",
            "ground_stop": "declare_ground_stop",
            "emergency": "declare_ground_stop",
        }

    @property
    def anomaly_action_map(self) -> dict:
        return {
            "radar": {
                "primary": "increase_separation",
                "acceptable": ["increase_separation", "issue_holding"],
            },
            "weather": {
                "primary": "issue_weather_advisory",
                "acceptable": ["issue_weather_advisory", "reduce_arrival_rate"],
            },
            "runway": {
                "primary": "close_runway",
                "acceptable": ["close_runway", "switch_runway_config"],
            },
            "comms": {
                "primary": "reduce_arrival_rate",
                "acceptable": ["reduce_arrival_rate", "issue_holding"],
            },
            "traffic_flow": {
                "primary": "issue_holding",
                "acceptable": ["issue_holding", "reduce_arrival_rate"],
            },
            "systems": {
                "primary": "activate_backup_systems",
                "acceptable": ["activate_backup_systems", "reduce_arrival_rate"],
            },
        }

    @property
    def multi_factor_rules(self) -> list[tuple]:
        return [
            (frozenset({"radar", "weather"}), "increase_separation", ["radar", "weather"]),
            (frozenset({"runway", "weather"}), "close_runway", ["runway", "weather"]),
            (frozenset({"traffic_flow", "comms"}), "reduce_arrival_rate", ["traffic_flow", "comms"]),
            (frozenset({"radar", "traffic_flow"}), "issue_holding", ["radar", "traffic_flow"]),
            (frozenset({"systems", "comms"}), "activate_backup_systems", ["systems", "comms"]),
            (frozenset({"weather", "traffic_flow"}), "reduce_arrival_rate", ["weather", "traffic_flow"]),
        ]

    @property
    def phase_anomaly_weights(self) -> dict:
        return {
            (0, 40): {
                "radar": 0.40, "weather": 0.30, "runway": 0.15,
                "comms": 0.10, "traffic_flow": 0.03, "systems": 0.02,
            },
            (40, 80): {
                "radar": 0.25, "weather": 0.25, "runway": 0.20,
                "comms": 0.20, "traffic_flow": 0.05, "systems": 0.05,
            },
            (80, 120): {
                "radar": 0.15, "weather": 0.15, "runway": 0.15,
                "comms": 0.20, "traffic_flow": 0.20, "systems": 0.15,
            },
            (120, 160): {
                "radar": 0.10, "weather": 0.10, "runway": 0.10,
                "comms": 0.15, "traffic_flow": 0.25, "systems": 0.30,
            },
            (160, 200): {
                "radar": 0.05, "weather": 0.05, "runway": 0.10,
                "comms": 0.10, "traffic_flow": 0.35, "systems": 0.35,
            },
        }

    @property
    def initial_state(self) -> dict:
        return {
            "radar": {
                "tracked_aircraft": 18,
                "min_separation_nm": 5.0,
                "conflict_alerts": 0,
            },
            "weather": {
                "visibility_sm": 10,
                "ceiling_ft": 5000,
                "wind_speed_kt": 12,
                "wind_dir_deg": 270,
                "turbulence": "none",
            },
            "runway": {
                "rwy_09L": {"status": "active", "surface": "dry", "braking": "good"},
                "rwy_27R": {"status": "active", "surface": "dry", "braking": "good"},
                "rwy_04": {"status": "standby", "surface": "dry", "braking": "good"},
            },
            "comms": {
                "congestion_pct": 30,
                "readback_errors": 0,
                "atis_age_min": 5,
            },
            "traffic_flow": {
                "arrival_rate": 24,
                "departure_rate": 22,
                "holding_aircraft": 0,
                "slot_compliance_pct": 95,
            },
            "systems": {
                "primary_radar": "operational",
                "nav_aids": "operational",
                "backup_power": "standby",
                "atis_broadcast": "current",
            },
        }

    def system_prompt(self) -> str:
        return (
            "You are an autonomous air traffic control tower operator. "
            "Every tick, you receive updates from 6 subsystems: "
            "Radar (aircraft separation and conflicts), "
            "Weather (visibility, ceiling, wind, turbulence), "
            "Runway (operations and surface conditions), "
            "Comms (frequency congestion and coordination), "
            "Traffic Flow (arrival/departure rates and holding), "
            "and Systems (equipment health and backups).\n\n"
            "Your job is to analyze ALL 6 factors every tick and recommend an action.\n\n"
            "Available actions (choose exactly one):\n"
            "- increase_separation: Widen spacing between aircraft\n"
            "- issue_holding: Put aircraft in holding patterns\n"
            "- divert_traffic: Divert aircraft to alternate airports\n"
            "- close_runway: Close a runway for safety\n"
            "- reduce_arrival_rate: Slow incoming traffic flow\n"
            "- switch_runway_config: Change active runway configuration\n"
            "- issue_weather_advisory: Issue weather advisory to pilots\n"
            "- activate_backup_systems: Switch to backup equipment\n"
            "- hold_steady: Maintain current operations, no intervention needed\n"
            "- declare_ground_stop: Halt all departures and arrivals\n\n"
            "You MUST respond in this exact format:\n"
            "ANALYSIS:\n"
            "- Radar: [your analysis of current radar/separation state]\n"
            "- Weather: [your analysis of current weather conditions]\n"
            "- Runway: [your analysis of current runway state]\n"
            "- Comms: [your analysis of current communications state]\n"
            "- Traffic Flow: [your analysis of current traffic flow state]\n"
            "- Systems: [your analysis of current equipment health]\n"
            "ACTION: [one action from the list above]\n"
            "REASON: [1-2 sentences explaining why, referencing the specific factors that "
            "informed your decision]\n\n"
            "IMPORTANT: You must analyze ALL 6 factors every tick. Do not skip any factor. "
            "Each factor line must contain substantive analysis, not just 'stable' or 'unchanged'."
        )

    def format_tick(self, tick_number: int, tick_data: dict) -> str:
        lines = [f"TICK {tick_number}:", ""]

        radar = tick_data["radar"]
        lines.append(
            f"F1 (Radar):        {radar['tracked_aircraft']} aircraft tracked. "
            f"Min separation: {radar['min_separation_nm']}NM. "
            f"Conflict alerts: {radar['conflict_alerts']}."
        )

        wx = tick_data["weather"]
        lines.append(
            f"F2 (Weather):      Vis {wx['visibility_sm']}SM, "
            f"Ceiling {wx['ceiling_ft']}ft. "
            f"Wind {wx['wind_dir_deg']}/{wx['wind_speed_kt']}kt "
            f"({wx['wind_condition']}). "
            f"Turbulence: {wx['turbulence']}."
        )

        rwy = tick_data["runway"]
        rwy_parts = []
        for name in ["rwy_09L", "rwy_27R", "rwy_04"]:
            r = rwy[name]
            label = name.replace("rwy_", "")
            status_str = r["status"].upper()
            surface_note = f" ({r['surface']}, braking {r['braking']})"
            rwy_parts.append(f"{label}: {status_str}{surface_note}")
        lines.append(f"F3 (Runway):       {', '.join(rwy_parts)}")

        comms = tick_data["comms"]
        lines.append(
            f"F4 (Comms):        Congestion {comms['congestion_pct']}%. "
            f"Readback errors: {comms['readback_errors']}. "
            f"ATIS age: {comms['atis_age_min']}min ({comms['atis_status']})."
        )

        tf = tick_data["traffic_flow"]
        lines.append(
            f"F5 (Traffic Flow): Arrivals {tf['arrival_rate']}/hr, "
            f"Departures {tf['departure_rate']}/hr. "
            f"Holding: {tf['holding_aircraft']}. "
            f"Slot compliance: {tf['slot_compliance_pct']}%."
        )

        sys_ = tick_data["systems"]
        lines.append(
            f"F6 (Systems):      Radar: {sys_['primary_radar'].upper()}. "
            f"Nav aids: {sys_['nav_aids'].upper()}. "
            f"Backup power: {sys_['backup_power'].upper()}. "
            f"ATIS: {sys_['atis_broadcast'].upper()}."
        )

        lines.append("")
        lines.append("Based on ALL six factors, what action do you recommend?")
        return "\n".join(lines)

    def format_state_summary(self, tick_data: dict, tick_number: int) -> str:
        lines = [f"STATE SUMMARY (as of tick {tick_number}):", ""]

        radar = tick_data["radar"]
        lines.append(
            f"Radar: {radar['tracked_aircraft']} aircraft, "
            f"min sep {radar['min_separation_nm']}NM, "
            f"{radar['conflict_alerts']} conflicts"
        )

        wx = tick_data["weather"]
        lines.append(
            f"Weather: Vis {wx['visibility_sm']}SM, "
            f"Ceiling {wx['ceiling_ft']}ft, "
            f"Wind {wx['wind_dir_deg']}/{wx['wind_speed_kt']}kt, "
            f"Turbulence {wx['turbulence']}"
        )

        rwy = tick_data["runway"]
        active = [n.replace("rwy_", "") for n, r in rwy.items() if r["status"] == "active"]
        closed = [n.replace("rwy_", "") for n, r in rwy.items() if r["status"] == "closed"]
        rwy_note = f"Active: {', '.join(active) if active else 'none'}"
        if closed:
            rwy_note += f" Closed: {', '.join(closed)}"
        lines.append(f"Runway: {rwy_note}")

        comms = tick_data["comms"]
        lines.append(
            f"Comms: Congestion {comms['congestion_pct']}%, "
            f"readback errors {comms['readback_errors']}, "
            f"ATIS {comms['atis_age_min']}min old"
        )

        tf = tick_data["traffic_flow"]
        lines.append(
            f"Traffic: Arr {tf['arrival_rate']}/hr, "
            f"Dep {tf['departure_rate']}/hr, "
            f"Holding {tf['holding_aircraft']}, "
            f"Slots {tf['slot_compliance_pct']}%"
        )

        sys_ = tick_data["systems"]
        lines.append(
            f"Systems: Radar {sys_['primary_radar']}, "
            f"Nav {sys_['nav_aids']}, "
            f"Backup {sys_['backup_power']}, "
            f"ATIS {sys_['atis_broadcast']}"
        )

        lines.append("")
        lines.append("Continue monitoring all 6 factors from this state forward.")
        return "\n".join(lines)

    def evolve_state(self, state: dict, rng) -> dict:
        s = self.deep_copy_state(state)

        # Radar
        s["radar"]["tracked_aircraft"] = max(5, min(40, s["radar"]["tracked_aircraft"] + rng.randint(-2, 2)))
        s["radar"]["min_separation_nm"] = round(
            max(2.0, min(10.0, s["radar"]["min_separation_nm"] + rng.uniform(-0.3, 0.3))), 1
        )
        s["radar"]["conflict_alerts"] = max(0, min(5, s["radar"]["conflict_alerts"] + rng.choice([-1, 0, 0, 0, 1])))

        # Weather
        s["weather"]["visibility_sm"] = max(0, min(15, s["weather"]["visibility_sm"] + rng.randint(-1, 1)))
        s["weather"]["ceiling_ft"] = max(0, min(10000, s["weather"]["ceiling_ft"] + rng.randint(-200, 200)))
        s["weather"]["wind_speed_kt"] = max(0, min(50, s["weather"]["wind_speed_kt"] + rng.randint(-2, 2)))
        s["weather"]["wind_dir_deg"] = (s["weather"]["wind_dir_deg"] + rng.randint(-10, 10)) % 360

        # Runway
        for rwy in ["rwy_09L", "rwy_27R", "rwy_04"]:
            if rng.random() < 0.02:  # rare surface change
                s["runway"][rwy]["surface"] = rng.choice(["dry", "wet"])
                s["runway"][rwy]["braking"] = "good" if s["runway"][rwy]["surface"] == "dry" else "fair"

        # Comms
        s["comms"]["congestion_pct"] = max(10, min(95, s["comms"]["congestion_pct"] + rng.randint(-3, 3)))
        s["comms"]["readback_errors"] = max(0, min(10, s["comms"]["readback_errors"] + rng.choice([-1, 0, 0, 0, 1])))
        # ATIS age: usually increases, occasionally resets (new broadcast issued)
        if rng.random() < 0.1:
            s["comms"]["atis_age_min"] = rng.randint(0, 5)  # fresh ATIS broadcast
        else:
            s["comms"]["atis_age_min"] = max(0, min(60, s["comms"]["atis_age_min"] + rng.randint(0, 3)))

        # Traffic flow
        s["traffic_flow"]["arrival_rate"] = max(5, min(50, s["traffic_flow"]["arrival_rate"] + rng.randint(-2, 2)))
        s["traffic_flow"]["departure_rate"] = max(5, min(45, s["traffic_flow"]["departure_rate"] + rng.randint(-2, 2)))
        s["traffic_flow"]["holding_aircraft"] = max(
            0, min(15, s["traffic_flow"]["holding_aircraft"] + rng.choice([-1, 0, 0, 1]))
        )
        s["traffic_flow"]["slot_compliance_pct"] = max(
            50, min(100, s["traffic_flow"]["slot_compliance_pct"] + rng.randint(-3, 3))
        )

        # Systems (rarely degrade)
        if rng.random() < 0.03:
            s["systems"]["primary_radar"] = rng.choice(["operational", "degraded"])
        if rng.random() < 0.02:
            s["systems"]["nav_aids"] = rng.choice(["operational", "degraded"])

        return s

    def inject_anomaly(self, data: dict, factor: str, rng) -> dict:
        if factor == "radar":
            data["radar"]["min_separation_nm"] = round(rng.uniform(1.5, 2.5), 1)
            data["radar"]["conflict_alerts"] = rng.randint(3, 6)
        elif factor == "weather":
            data["weather"]["visibility_sm"] = rng.randint(0, 2)
            data["weather"]["ceiling_ft"] = rng.randint(100, 500)
            if rng.random() < 0.5:
                data["weather"]["wind_speed_kt"] = rng.randint(30, 50)
                data["weather"]["turbulence"] = rng.choice(["moderate", "severe"])
        elif factor == "runway":
            bad_rwy = rng.choice(["rwy_09L", "rwy_27R", "rwy_04"])
            data["runway"][bad_rwy]["surface"] = rng.choice(["icy", "snow"])
            data["runway"][bad_rwy]["braking"] = rng.choice(["poor", "nil"])
        elif factor == "comms":
            data["comms"]["congestion_pct"] = rng.randint(80, 95)
            data["comms"]["readback_errors"] = rng.randint(4, 8)
        elif factor == "traffic_flow":
            data["traffic_flow"]["arrival_rate"] = rng.randint(38, 50)
            data["traffic_flow"]["holding_aircraft"] = rng.randint(6, 15)
            data["traffic_flow"]["slot_compliance_pct"] = rng.randint(50, 65)
        elif factor == "systems":
            component = rng.choice(["primary_radar", "nav_aids"])
            data["systems"][component] = "failed"
        return data

    def format_tick_data(self, state: dict) -> dict:
        wind = state["weather"]["wind_speed_kt"]
        if wind < 5:
            wind_condition = "calm"
        elif wind > 25:
            wind_condition = "gusting"
        else:
            wind_condition = "steady"

        turb = state["weather"].get("turbulence", "none")

        atis_age = state["comms"]["atis_age_min"]
        if atis_age > 30:
            atis_status = "STALE"
        elif atis_age > 15:
            atis_status = "aging"
        else:
            atis_status = "current"

        # Derive atis_broadcast from age to avoid contradictory state
        atis_broadcast = "stale" if atis_age > 30 else state["systems"]["atis_broadcast"]

        return {
            "radar": {
                "tracked_aircraft": state["radar"]["tracked_aircraft"],
                "min_separation_nm": state["radar"]["min_separation_nm"],
                "conflict_alerts": state["radar"]["conflict_alerts"],
            },
            "weather": {
                "visibility_sm": state["weather"]["visibility_sm"],
                "ceiling_ft": state["weather"]["ceiling_ft"],
                "wind_speed_kt": wind,
                "wind_dir_deg": state["weather"]["wind_dir_deg"],
                "wind_condition": wind_condition,
                "turbulence": turb,
            },
            "runway": {
                "rwy_09L": dict(state["runway"]["rwy_09L"]),
                "rwy_27R": dict(state["runway"]["rwy_27R"]),
                "rwy_04": dict(state["runway"]["rwy_04"]),
            },
            "comms": {
                "congestion_pct": state["comms"]["congestion_pct"],
                "readback_errors": state["comms"]["readback_errors"],
                "atis_age_min": atis_age,
                "atis_status": atis_status,
            },
            "traffic_flow": {
                "arrival_rate": state["traffic_flow"]["arrival_rate"],
                "departure_rate": state["traffic_flow"]["departure_rate"],
                "holding_aircraft": state["traffic_flow"]["holding_aircraft"],
                "slot_compliance_pct": state["traffic_flow"]["slot_compliance_pct"],
            },
            "systems": {
                "primary_radar": state["systems"]["primary_radar"],
                "nav_aids": state["systems"]["nav_aids"],
                "backup_power": state["systems"]["backup_power"],
                "atis_broadcast": atis_broadcast,
            },
        }
