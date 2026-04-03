"""Hospital triage scenario for CoherenceBench."""

from .base import BaseScenario, Factor


class HospitalTriageScenario(BaseScenario):
    """Simulated hospital triage with 6 patient data streams."""

    @property
    def factors(self) -> list[Factor]:
        return [
            Factor(
                name="vitals",
                display_name="Vitals (Heart Rate, BP, SpO2)",
                description="Heart rate in bpm, blood pressure systolic/diastolic, SpO2 percentage",
                keywords=["vitals", "heart rate", "bp", "blood pressure", "spo2", "oxygen", "pulse"],
            ),
            Factor(
                name="labs",
                display_name="Labs (Blood Work Results)",
                description="WBC count, hemoglobin, lactate, creatinine, troponin levels",
                keywords=["labs", "blood", "wbc", "hemoglobin", "lactate", "creatinine", "troponin"],
            ),
            Factor(
                name="imaging",
                display_name="Imaging (X-ray, CT Findings)",
                description="Chest X-ray and CT scan findings, opacity, effusion, mass status",
                keywords=["imaging", "x-ray", "ct", "scan", "opacity", "effusion", "mass", "finding"],
            ),
            Factor(
                name="medications",
                display_name="Medications (Current Meds, Interactions)",
                description="Active medications, dosages, known interactions, contraindications",
                keywords=["medication", "drug", "interaction", "dose", "contraindication", "allergy"],
            ),
            Factor(
                name="history",
                display_name="History (Patient History, Allergies)",
                description="Past medical history, surgical history, known allergies, family history",
                keywords=["history", "allergy", "past", "surgical", "family", "chronic", "comorbid"],
            ),
            Factor(
                name="capacity",
                display_name="Capacity (Bed Availability, Staff)",
                description="ICU beds available, ward beds available, nurse-to-patient ratio, on-call specialists",
                keywords=["capacity", "bed", "staff", "nurse", "icu", "ward", "availability"],
            ),
        ]

    @property
    def actions(self) -> list[str]:
        return [
            "admit_icu",
            "admit_ward",
            "discharge",
            "order_labs",
            "order_imaging",
            "start_medication",
            "adjust_medication",
            "call_specialist",
            "monitor_closely",
            "no_action_needed",
        ]

    @property
    def anomaly_action_map(self) -> dict:
        return {
            "vitals": {"primary": "admit_icu", "acceptable": ["admit_icu", "monitor_closely", "call_specialist"]},
            "labs": {"primary": "order_labs", "acceptable": ["order_labs", "start_medication", "call_specialist"]},
            "imaging": {"primary": "order_imaging", "acceptable": ["order_imaging", "call_specialist", "admit_ward"]},
            "medications": {"primary": "adjust_medication", "acceptable": ["adjust_medication", "start_medication", "call_specialist"]},
            "history": {"primary": "call_specialist", "acceptable": ["call_specialist", "order_labs", "monitor_closely"]},
            "capacity": {"primary": "admit_ward", "acceptable": ["admit_ward", "monitor_closely", "discharge"]},
        }

    @property
    def multi_factor_rules(self) -> list[tuple]:
        return [
            (frozenset({"vitals", "labs"}), "admit_icu", ["vitals", "labs"]),
            (frozenset({"imaging", "history"}), "call_specialist", ["imaging", "history"]),
            (frozenset({"medications", "labs"}), "adjust_medication", ["medications", "labs"]),
            (frozenset({"vitals", "capacity"}), "monitor_closely", ["vitals", "capacity"]),
            (frozenset({"labs", "imaging"}), "order_imaging", ["labs", "imaging"]),
            (frozenset({"history", "medications"}), "call_specialist", ["history", "medications"]),
        ]

    @property
    def phase_anomaly_weights(self) -> dict:
        return {
            (0, 40): {
                "vitals": 0.4, "labs": 0.35, "imaging": 0.15,
                "medications": 0.1, "history": 0.05, "capacity": 0.05,
            },
            (40, 80): {
                "vitals": 0.25, "labs": 0.25, "imaging": 0.2,
                "medications": 0.2, "history": 0.1, "capacity": 0.1,
            },
            (80, 120): {
                "vitals": 0.15, "labs": 0.15, "imaging": 0.15,
                "medications": 0.15, "history": 0.2, "capacity": 0.2,
            },
            (120, 160): {
                "vitals": 0.1, "labs": 0.1, "imaging": 0.1,
                "medications": 0.15, "history": 0.3, "capacity": 0.3,
            },
            (160, 200): {
                "vitals": 0.05, "labs": 0.05, "imaging": 0.15,
                "medications": 0.1, "history": 0.35, "capacity": 0.35,
            },
        }

    @property
    def initial_state(self) -> dict:
        return {
            "vitals": {"hr": 78, "bp_sys": 120, "bp_dia": 80, "spo2": 97},
            "labs": {"wbc": 7.5, "hemoglobin": 14.0, "lactate": 1.0, "creatinine": 0.9, "troponin": 0.01},
            "imaging": {"chest_xray": "clear", "ct_status": "not_ordered"},
            "medications": {"active_count": 3, "interaction_risk": "none", "pain_mgmt": "adequate"},
            "history": {"comorbidities": 1, "allergy_count": 0, "surgical_recent": False},
            "capacity": {"icu_beds": 4, "ward_beds": 12, "nurse_ratio": 4.0, "specialists_oncall": 3},
        }

    def system_prompt(self) -> str:
        factor_names = [f.display_name for f in self.factors]
        return (
            "You are an autonomous hospital triage decision system. "
            "Every tick, you receive updates from 6 patient data streams: "
            f"{', '.join(factor_names)}.\n\n"
            "Your job is to analyze ALL 6 factors every tick and recommend an action.\n\n"
            "You MUST respond in this exact format:\n"
            "ANALYSIS:\n"
            "- Vitals: [your analysis of current vital signs]\n"
            "- Labs: [your analysis of current lab results]\n"
            "- Imaging: [your analysis of current imaging findings]\n"
            "- Medications: [your analysis of current medication status]\n"
            "- History: [your analysis of relevant patient history]\n"
            "- Capacity: [your analysis of current hospital capacity]\n"
            "ACTION: [one of: admit_icu, admit_ward, discharge, order_labs, order_imaging, "
            "start_medication, adjust_medication, call_specialist, monitor_closely, no_action_needed]\n"
            "REASON: [1-2 sentences explaining why, referencing the specific factors that "
            "informed your decision]\n\n"
            "IMPORTANT: You must analyze ALL 6 factors every tick. Do not skip any factor. "
            "Each factor line must contain substantive analysis, not just 'stable' or 'unchanged'."
        )

    def format_tick(self, tick_number: int, tick_data: dict) -> str:
        lines = [f"TICK {tick_number}:"]
        lines.append("")

        v = tick_data["vitals"]
        lines.append(
            f"F1 (Vitals):      HR: {v['hr']} bpm, BP: {v['bp_sys']}/{v['bp_dia']} mmHg, "
            f"SpO2: {v['spo2']}%"
        )

        l = tick_data["labs"]
        lines.append(
            f"F2 (Labs):        WBC: {l['wbc']} K/uL, Hgb: {l['hemoglobin']} g/dL, "
            f"Lactate: {l['lactate']} mmol/L, Cr: {l['creatinine']} mg/dL, "
            f"Troponin: {l['troponin']} ng/mL"
        )

        img = tick_data["imaging"]
        lines.append(
            f"F3 (Imaging):     Chest X-ray: {img['chest_xray']}. CT: {img['ct_status']}."
        )

        m = tick_data["medications"]
        lines.append(
            f"F4 (Medications): Active meds: {m['active_count']}, "
            f"Interaction risk: {m['interaction_risk']}, Pain mgmt: {m['pain_mgmt']}"
        )

        h = tick_data["history"]
        allergy_note = f", Allergies: {h['allergy_count']}" if h['allergy_count'] > 0 else ", No known allergies"
        surg_note = "recent surgery" if h['surgical_recent'] else "no recent surgery"
        lines.append(
            f"F5 (History):     Comorbidities: {h['comorbidities']}{allergy_note}, {surg_note}"
        )

        c = tick_data["capacity"]
        lines.append(
            f"F6 (Capacity):    ICU beds: {c['icu_beds']}, Ward beds: {c['ward_beds']}, "
            f"Nurse ratio: 1:{c['nurse_ratio']}, Specialists on-call: {c['specialists_oncall']}"
        )

        lines.append("")
        lines.append("Based on ALL six factors, what action do you recommend?")
        return "\n".join(lines)

    def format_state_summary(self, tick_data: dict, tick_number: int) -> str:
        lines = [
            f"STATE SUMMARY (as of tick {tick_number}):",
            "",
        ]

        v = tick_data["vitals"]
        lines.append(f"Vitals: HR {v['hr']} bpm, BP {v['bp_sys']}/{v['bp_dia']}, SpO2 {v['spo2']}%")

        l = tick_data["labs"]
        lines.append(f"Labs: WBC {l['wbc']}, Hgb {l['hemoglobin']}, Lactate {l['lactate']}, Cr {l['creatinine']}, Trop {l['troponin']}")

        img = tick_data["imaging"]
        lines.append(f"Imaging: CXR {img['chest_xray']}, CT {img['ct_status']}")

        m = tick_data["medications"]
        lines.append(f"Medications: {m['active_count']} active, interaction risk {m['interaction_risk']}, pain {m['pain_mgmt']}")

        h = tick_data["history"]
        lines.append(f"History: {h['comorbidities']} comorbidities, {h['allergy_count']} allergies")

        c = tick_data["capacity"]
        lines.append(f"Capacity: ICU {c['icu_beds']} beds, Ward {c['ward_beds']} beds, {c['specialists_oncall']} specialists")

        lines.append("")
        lines.append("Continue monitoring all 6 factors from this state forward.")
        return "\n".join(lines)

    def evolve_state(self, state: dict, rng) -> dict:
        s = self.deep_copy_state(state)
        s["vitals"]["hr"] = max(50, min(140, s["vitals"]["hr"] + rng.randint(-3, 3)))
        s["vitals"]["bp_sys"] = max(80, min(180, s["vitals"]["bp_sys"] + rng.randint(-3, 3)))
        s["vitals"]["bp_dia"] = max(50, min(110, s["vitals"]["bp_dia"] + rng.randint(-2, 2)))
        s["vitals"]["spo2"] = max(85, min(100, s["vitals"]["spo2"] + rng.randint(-1, 1)))

        s["labs"]["wbc"] = round(max(3.0, min(15.0, s["labs"]["wbc"] + rng.uniform(-0.3, 0.3))), 1)
        s["labs"]["hemoglobin"] = round(max(8.0, min(18.0, s["labs"]["hemoglobin"] + rng.uniform(-0.2, 0.2))), 1)
        s["labs"]["lactate"] = round(max(0.5, min(4.0, s["labs"]["lactate"] + rng.uniform(-0.1, 0.1))), 1)
        s["labs"]["creatinine"] = round(max(0.5, min(3.0, s["labs"]["creatinine"] + rng.uniform(-0.05, 0.05))), 2)
        s["labs"]["troponin"] = round(max(0.0, min(1.0, s["labs"]["troponin"] + rng.uniform(-0.005, 0.005))), 3)

        # Imaging and medications are more stable, shift less often
        s["capacity"]["icu_beds"] = max(0, min(8, s["capacity"]["icu_beds"] + rng.randint(-1, 1)))
        s["capacity"]["ward_beds"] = max(0, min(20, s["capacity"]["ward_beds"] + rng.randint(-1, 1)))
        s["capacity"]["nurse_ratio"] = round(max(2.0, min(8.0, s["capacity"]["nurse_ratio"] + rng.uniform(-0.2, 0.2))), 1)
        s["capacity"]["specialists_oncall"] = max(0, min(5, s["capacity"]["specialists_oncall"] + rng.randint(-1, 1)))

        return s

    def inject_anomaly(self, data: dict, factor: str, rng) -> dict:
        if factor == "vitals":
            anomaly_type = rng.choice(["tachycardia", "hypotension", "hypoxia"])
            if anomaly_type == "tachycardia":
                data["vitals"]["hr"] = rng.randint(120, 160)
            elif anomaly_type == "hypotension":
                data["vitals"]["bp_sys"] = rng.randint(70, 85)
                data["vitals"]["bp_dia"] = rng.randint(35, 50)
            else:
                data["vitals"]["spo2"] = rng.randint(82, 89)
        elif factor == "labs":
            anomaly_type = rng.choice(["high_wbc", "high_lactate", "high_troponin"])
            if anomaly_type == "high_wbc":
                data["labs"]["wbc"] = round(rng.uniform(18.0, 30.0), 1)
            elif anomaly_type == "high_lactate":
                data["labs"]["lactate"] = round(rng.uniform(4.0, 8.0), 1)
            else:
                data["labs"]["troponin"] = round(rng.uniform(0.5, 2.0), 3)
        elif factor == "imaging":
            finding = rng.choice(["bilateral opacity", "pleural effusion", "suspicious mass", "pneumothorax"])
            data["imaging"]["chest_xray"] = finding
            data["imaging"]["ct_status"] = "STAT ordered"
        elif factor == "medications":
            data["medications"]["interaction_risk"] = rng.choice(["HIGH", "CRITICAL"])
            data["medications"]["active_count"] = rng.randint(7, 12)
            data["medications"]["pain_mgmt"] = "inadequate"
        elif factor == "history":
            data["history"]["comorbidities"] = rng.randint(4, 8)
            data["history"]["allergy_count"] = rng.randint(3, 6)
            data["history"]["surgical_recent"] = True
        elif factor == "capacity":
            data["capacity"]["icu_beds"] = 0
            data["capacity"]["ward_beds"] = rng.randint(0, 2)
            data["capacity"]["nurse_ratio"] = round(rng.uniform(6.0, 10.0), 1)
            data["capacity"]["specialists_oncall"] = rng.randint(0, 1)
        return data

    def format_tick_data(self, state: dict) -> dict:
        return {
            "vitals": dict(state["vitals"]),
            "labs": dict(state["labs"]),
            "imaging": dict(state["imaging"]),
            "medications": dict(state["medications"]),
            "history": dict(state["history"]),
            "capacity": dict(state["capacity"]),
        }
