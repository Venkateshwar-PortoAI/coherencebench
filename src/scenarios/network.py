"""Network security operations scenario for CoherenceBench."""

from .base import BaseScenario, Factor


class NetworkSecurityScenario(BaseScenario):
    """Simulated network security operations center with 6 data feeds."""

    @property
    def factors(self) -> list[Factor]:
        return [
            Factor(
                name="traffic",
                display_name="Traffic (Packet Volume, Bandwidth)",
                description="Packets per second, bandwidth utilization percentage, top talkers",
                keywords=["traffic", "packet", "bandwidth", "volume", "throughput", "pps"],
            ),
            Factor(
                name="auth",
                display_name="Auth (Login Attempts, Failed Auths)",
                description="Login attempts per minute, failed auth count, lockout events",
                keywords=["auth", "login", "failed", "lockout", "authentication", "credential"],
            ),
            Factor(
                name="endpoints",
                display_name="Endpoints (Device Status, Patch Levels)",
                description="Online device count, unpatched devices, compliance percentage",
                keywords=["endpoint", "device", "patch", "compliance", "agent", "unpatched"],
            ),
            Factor(
                name="firewall",
                display_name="Firewall (Rule Triggers, Blocked Connections)",
                description="Rules triggered per minute, blocked connections, top blocked IPs",
                keywords=["firewall", "rule", "blocked", "connection", "trigger", "deny"],
            ),
            Factor(
                name="logs",
                display_name="Logs (System Logs, Error Patterns)",
                description="Error rate per minute, warning count, critical events, log volume",
                keywords=["log", "error", "warning", "critical", "event", "syslog", "pattern"],
            ),
            Factor(
                name="threats",
                display_name="Threats (Known IOCs, Threat Intel)",
                description="IOC matches, threat level, active campaigns, CVE alerts",
                keywords=["threat", "ioc", "intel", "campaign", "cve", "indicator", "compromise"],
            ),
        ]

    @property
    def actions(self) -> list[str]:
        return [
            "block_ip",
            "isolate_endpoint",
            "escalate_alert",
            "update_firewall_rule",
            "force_password_reset",
            "deploy_patch",
            "increase_monitoring",
            "quarantine_file",
            "no_action_needed",
            "investigate_further",
        ]

    @property
    def anomaly_action_map(self) -> dict:
        return {
            "traffic": {"primary": "block_ip", "acceptable": ["block_ip", "increase_monitoring", "escalate_alert"]},
            "auth": {"primary": "force_password_reset", "acceptable": ["force_password_reset", "block_ip", "escalate_alert"]},
            "endpoints": {"primary": "deploy_patch", "acceptable": ["deploy_patch", "isolate_endpoint", "increase_monitoring"]},
            "firewall": {"primary": "update_firewall_rule", "acceptable": ["update_firewall_rule", "block_ip", "escalate_alert"]},
            "logs": {"primary": "investigate_further", "acceptable": ["investigate_further", "escalate_alert", "increase_monitoring"]},
            "threats": {"primary": "escalate_alert", "acceptable": ["escalate_alert", "quarantine_file", "isolate_endpoint"]},
        }

    @property
    def multi_factor_rules(self) -> list[tuple]:
        return [
            (frozenset({"traffic", "auth"}), "block_ip", ["traffic", "auth"]),
            (frozenset({"endpoints", "threats"}), "isolate_endpoint", ["endpoints", "threats"]),
            (frozenset({"firewall", "logs"}), "update_firewall_rule", ["firewall", "logs"]),
            (frozenset({"auth", "threats"}), "force_password_reset", ["auth", "threats"]),
            (frozenset({"traffic", "firewall"}), "block_ip", ["traffic", "firewall"]),
            (frozenset({"logs", "endpoints"}), "deploy_patch", ["logs", "endpoints"]),
        ]

    @property
    def phase_anomaly_weights(self) -> dict:
        return {
            (0, 40): {
                "traffic": 0.4, "auth": 0.35, "endpoints": 0.15,
                "firewall": 0.1, "logs": 0.05, "threats": 0.05,
            },
            (40, 80): {
                "traffic": 0.25, "auth": 0.25, "endpoints": 0.2,
                "firewall": 0.2, "logs": 0.1, "threats": 0.1,
            },
            (80, 120): {
                "traffic": 0.15, "auth": 0.15, "endpoints": 0.15,
                "firewall": 0.15, "logs": 0.2, "threats": 0.2,
            },
            (120, 160): {
                "traffic": 0.1, "auth": 0.1, "endpoints": 0.1,
                "firewall": 0.15, "logs": 0.3, "threats": 0.3,
            },
            (160, 200): {
                "traffic": 0.05, "auth": 0.05, "endpoints": 0.15,
                "firewall": 0.1, "logs": 0.35, "threats": 0.35,
            },
        }

    @property
    def initial_state(self) -> dict:
        return {
            "traffic": {"pps": 50000, "bandwidth_pct": 45, "top_talker_pps": 2000},
            "auth": {"login_attempts_pm": 120, "failed_auths": 5, "lockouts": 0},
            "endpoints": {"online": 450, "unpatched": 12, "compliance_pct": 97},
            "firewall": {"rules_triggered_pm": 30, "blocked_connections": 15, "top_blocked_count": 5},
            "logs": {"errors_pm": 8, "warnings": 25, "critical_events": 0, "volume_gb": 2.1},
            "threats": {"ioc_matches": 0, "threat_level": "low", "active_campaigns": 0, "cve_alerts": 1},
        }

    def system_prompt(self) -> str:
        factor_names = [f.display_name for f in self.factors]
        return (
            "You are an autonomous network security operations center (SOC) analyst. "
            "Every tick, you receive updates from 6 monitoring feeds: "
            f"{', '.join(factor_names)}.\n\n"
            "Your job is to analyze ALL 6 factors every tick and recommend an action.\n\n"
            "You MUST respond in this exact format:\n"
            "ANALYSIS:\n"
            "- Traffic: [your analysis of current network traffic]\n"
            "- Auth: [your analysis of current authentication activity]\n"
            "- Endpoints: [your analysis of current endpoint status]\n"
            "- Firewall: [your analysis of current firewall activity]\n"
            "- Logs: [your analysis of current log patterns]\n"
            "- Threats: [your analysis of current threat intelligence]\n"
            "ACTION: [one of: block_ip, isolate_endpoint, escalate_alert, update_firewall_rule, "
            "force_password_reset, deploy_patch, increase_monitoring, quarantine_file, "
            "no_action_needed, investigate_further]\n"
            "REASON: [1-2 sentences explaining why, referencing the specific factors that "
            "informed your decision]\n\n"
            "IMPORTANT: You must analyze ALL 6 factors every tick. Do not skip any factor. "
            "Each factor line must contain substantive analysis, not just 'stable' or 'unchanged'."
        )

    def format_tick(self, tick_number: int, tick_data: dict) -> str:
        lines = [f"TICK {tick_number}:"]
        lines.append("")

        t = tick_data["traffic"]
        lines.append(
            f"F1 (Traffic):     {t['pps']} pps, Bandwidth: {t['bandwidth_pct']}%, "
            f"Top talker: {t['top_talker_pps']} pps"
        )

        a = tick_data["auth"]
        lines.append(
            f"F2 (Auth):        Login attempts/min: {a['login_attempts_pm']}, "
            f"Failed: {a['failed_auths']}, Lockouts: {a['lockouts']}"
        )

        e = tick_data["endpoints"]
        lines.append(
            f"F3 (Endpoints):   Online: {e['online']}, Unpatched: {e['unpatched']}, "
            f"Compliance: {e['compliance_pct']}%"
        )

        f = tick_data["firewall"]
        lines.append(
            f"F4 (Firewall):    Rules triggered/min: {f['rules_triggered_pm']}, "
            f"Blocked: {f['blocked_connections']}, Top blocked: {f['top_blocked_count']} hits"
        )

        lg = tick_data["logs"]
        lines.append(
            f"F5 (Logs):        Errors/min: {lg['errors_pm']}, Warnings: {lg['warnings']}, "
            f"Critical: {lg['critical_events']}, Volume: {lg['volume_gb']}GB"
        )

        th = tick_data["threats"]
        lines.append(
            f"F6 (Threats):     IOC matches: {th['ioc_matches']}, Level: {th['threat_level'].upper()}, "
            f"Active campaigns: {th['active_campaigns']}, CVE alerts: {th['cve_alerts']}"
        )

        lines.append("")
        lines.append("Based on ALL six factors, what action do you recommend?")
        return "\n".join(lines)

    def format_state_summary(self, tick_data: dict, tick_number: int) -> str:
        lines = [
            f"STATE SUMMARY (as of tick {tick_number}):",
            "",
        ]

        t = tick_data["traffic"]
        lines.append(f"Traffic: {t['pps']} pps, {t['bandwidth_pct']}% bandwidth")

        a = tick_data["auth"]
        lines.append(f"Auth: {a['login_attempts_pm']} logins/min, {a['failed_auths']} failed, {a['lockouts']} lockouts")

        e = tick_data["endpoints"]
        lines.append(f"Endpoints: {e['online']} online, {e['unpatched']} unpatched, {e['compliance_pct']}% compliant")

        f = tick_data["firewall"]
        lines.append(f"Firewall: {f['rules_triggered_pm']} rules/min, {f['blocked_connections']} blocked")

        lg = tick_data["logs"]
        lines.append(f"Logs: {lg['errors_pm']} errors/min, {lg['critical_events']} critical")

        th = tick_data["threats"]
        lines.append(f"Threats: {th['ioc_matches']} IOC matches, level {th['threat_level']}, {th['active_campaigns']} campaigns")

        lines.append("")
        lines.append("Continue monitoring all 6 factors from this state forward.")
        return "\n".join(lines)

    def evolve_state(self, state: dict, rng) -> dict:
        s = self.deep_copy_state(state)

        s["traffic"]["pps"] = max(10000, min(200000, s["traffic"]["pps"] + rng.randint(-2000, 2000)))
        s["traffic"]["bandwidth_pct"] = max(10, min(95, s["traffic"]["bandwidth_pct"] + rng.randint(-3, 3)))
        s["traffic"]["top_talker_pps"] = max(500, min(20000, s["traffic"]["top_talker_pps"] + rng.randint(-200, 200)))

        s["auth"]["login_attempts_pm"] = max(20, min(500, s["auth"]["login_attempts_pm"] + rng.randint(-10, 10)))
        s["auth"]["failed_auths"] = max(0, min(50, s["auth"]["failed_auths"] + rng.randint(-2, 2)))
        s["auth"]["lockouts"] = max(0, min(20, s["auth"]["lockouts"] + rng.randint(-1, 1)))

        s["endpoints"]["online"] = max(300, min(500, s["endpoints"]["online"] + rng.randint(-5, 5)))
        s["endpoints"]["unpatched"] = max(0, min(50, s["endpoints"]["unpatched"] + rng.randint(-2, 2)))
        s["endpoints"]["compliance_pct"] = max(80, min(100, s["endpoints"]["compliance_pct"] + rng.randint(-1, 1)))

        s["firewall"]["rules_triggered_pm"] = max(5, min(200, s["firewall"]["rules_triggered_pm"] + rng.randint(-5, 5)))
        s["firewall"]["blocked_connections"] = max(0, min(100, s["firewall"]["blocked_connections"] + rng.randint(-3, 3)))
        s["firewall"]["top_blocked_count"] = max(0, min(50, s["firewall"]["top_blocked_count"] + rng.randint(-2, 2)))

        s["logs"]["errors_pm"] = max(0, min(50, s["logs"]["errors_pm"] + rng.randint(-2, 2)))
        s["logs"]["warnings"] = max(0, min(100, s["logs"]["warnings"] + rng.randint(-3, 3)))
        s["logs"]["critical_events"] = max(0, min(10, s["logs"]["critical_events"] + rng.randint(-1, 1)))
        s["logs"]["volume_gb"] = round(max(0.5, min(10.0, s["logs"]["volume_gb"] + rng.uniform(-0.2, 0.2))), 1)

        s["threats"]["ioc_matches"] = max(0, min(10, s["threats"]["ioc_matches"] + rng.randint(-1, 1)))
        s["threats"]["cve_alerts"] = max(0, min(10, s["threats"]["cve_alerts"] + rng.randint(-1, 1)))

        return s

    def inject_anomaly(self, data: dict, factor: str, rng) -> dict:
        if factor == "traffic":
            data["traffic"]["pps"] = rng.randint(150000, 500000)
            data["traffic"]["bandwidth_pct"] = rng.randint(85, 99)
            data["traffic"]["top_talker_pps"] = rng.randint(30000, 100000)
        elif factor == "auth":
            data["auth"]["login_attempts_pm"] = rng.randint(500, 2000)
            data["auth"]["failed_auths"] = rng.randint(50, 200)
            data["auth"]["lockouts"] = rng.randint(10, 50)
        elif factor == "endpoints":
            data["endpoints"]["unpatched"] = rng.randint(50, 150)
            data["endpoints"]["compliance_pct"] = rng.randint(50, 75)
            data["endpoints"]["online"] = rng.randint(200, 350)
        elif factor == "firewall":
            data["firewall"]["rules_triggered_pm"] = rng.randint(200, 500)
            data["firewall"]["blocked_connections"] = rng.randint(100, 500)
            data["firewall"]["top_blocked_count"] = rng.randint(50, 200)
        elif factor == "logs":
            data["logs"]["errors_pm"] = rng.randint(50, 200)
            data["logs"]["critical_events"] = rng.randint(5, 20)
            data["logs"]["warnings"] = rng.randint(100, 300)
            data["logs"]["volume_gb"] = round(rng.uniform(8.0, 20.0), 1)
        elif factor == "threats":
            data["threats"]["ioc_matches"] = rng.randint(5, 25)
            data["threats"]["threat_level"] = rng.choice(["high", "critical"])
            data["threats"]["active_campaigns"] = rng.randint(2, 6)
            data["threats"]["cve_alerts"] = rng.randint(5, 15)
        return data

    def format_tick_data(self, state: dict) -> dict:
        # Derive threat_level from current state
        ioc = state["threats"].get("ioc_matches", 0)
        if ioc >= 5:
            threat_level = "critical"
        elif ioc >= 2:
            threat_level = "high"
        elif ioc >= 1:
            threat_level = "medium"
        else:
            threat_level = state["threats"].get("threat_level", "low")

        return {
            "traffic": dict(state["traffic"]),
            "auth": dict(state["auth"]),
            "endpoints": dict(state["endpoints"]),
            "firewall": dict(state["firewall"]),
            "logs": dict(state["logs"]),
            "threats": {
                "ioc_matches": state["threats"]["ioc_matches"],
                "threat_level": threat_level,
                "active_campaigns": state["threats"].get("active_campaigns", 0),
                "cve_alerts": state["threats"]["cve_alerts"],
            },
        }
