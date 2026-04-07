"""Parses LLM responses and extracts factor analysis + actions.

DA (Decision Accuracy) is the primary metric -- did the agent choose a correct action?
ADR (Anomaly Mention Rate) is a secondary proxy -- did the agent write substantively
about anomalous factors? FC and FI are format diagnostics.
"""

import logging
import re
from dataclasses import dataclass

from .scenario import FACTORS, ACTIONS
from .scenarios.base import BaseScenario
from .metrics import (
    factor_coverage,
    fixation_index,
    decision_accuracy,
    anomaly_detection_rate,
    intervention_recovery,
    compute_all_metrics,
)

logger = logging.getLogger(__name__)

# Common model-generated aliases mapped to canonical action names.
# Scenario-specific aliases are merged at runtime via _build_alias_map().
ACTION_ALIASES = {
    "gas_turbine": "start_gas_turbine",
    "shed": "shed_load",
    "battery": "deploy_battery",
    "hold": "hold_steady",
    "disconnect": "emergency_disconnect",
    "import": "request_import",
    "ramp": "ramp_plant",
    "curtail": "curtail_renewable",
    "voltage": "adjust_voltage",
    "charge": "charge_battery",
    "block": "block_ip",
    "isolate": "isolate_endpoint",
    "escalate": "escalate_alert",
    "patch": "deploy_patch",
    "quarantine": "quarantine_file",
    "investigate": "investigate_further",
    "monitor": "increase_monitoring",
    "password_reset": "force_password_reset",
    "reset_password": "force_password_reset",
    "firewall": "update_firewall_rule",
    "icu": "admit_icu",
    "ward": "admit_ward",
    "labs": "order_labs",
    "imaging": "order_imaging",
    "specialist": "call_specialist",
    "medication": "start_medication",
    "adjust_med": "adjust_medication",
    "adjust_meds": "adjust_medication",
    "closely": "monitor_closely",
    "no_action": "no_action_needed",
    "none": "no_action_needed",
    "noop": "hold_steady",
}

# Words that indicate dismissive/non-substantive analysis
DISMISSIVE_PATTERNS = [
    r"^(ok|fine|stable|unchanged|normal|adequate|no change|no issues|good|steady|same)\.?$",
    r"^(looks? (?:ok|fine|good|stable|normal))\.?$",
    r"^(no (?:change|issues|concerns|problems))\.?$",
    r"^(within (?:normal|safe) (?:range|limits))\.?$",
]

MIN_SUBSTANTIVE_WORDS = 8


@dataclass
class TickAnalysis:
    """Analyzed result for a single tick."""

    tick_number: int
    action: str
    factors_mentioned: list[str]
    factors_substantive: list[str]
    word_counts: dict[str, int]
    reason: str
    # Primary metric
    decision_accuracy: float
    # Secondary metrics
    anomaly_detection_rate: float
    factor_coverage: float
    fixation_index: float


class ResponseAnalyzer:
    """Parses structured agent responses into metrics-ready data."""

    def __init__(self, scenario: BaseScenario | None = None):
        if scenario is not None:
            self.factors = scenario.factors
            self.valid_actions = set(scenario.actions)
        else:
            self.factors = FACTORS
            self.valid_actions = set(ACTIONS)
        self._alias_map = self._build_alias_map()
        self._dismissive_re = [re.compile(p, re.IGNORECASE) for p in DISMISSIVE_PATTERNS]

    def _build_alias_map(self) -> dict[str, str]:
        """Build alias lookup restricted to this scenario's valid actions."""
        mapping: dict[str, str] = {}
        for alias, canonical in ACTION_ALIASES.items():
            if canonical in self.valid_actions:
                mapping[alias] = canonical
        return mapping

    def _is_refusal(self, response: str) -> bool:
        """Detect if the model refused to respond or the session broke."""
        if len(response.strip()) < 50:
            return True
        refusal_patterns = [
            r"(?i)conversation.*(closed|ended|over)",
            r"(?i)cannot.*(continue|respond|assist)",
            r"(?i)session.*(expired|ended|closed)",
            r"(?i)i('m| am) (unable|not able) to",
        ]
        for pattern in refusal_patterns:
            if re.search(pattern, response):
                return True
        return False

    def parse_response(self, response: str) -> dict:
        """Parse a single response into structured analysis data.

        Returns dict with:
            action, factors_mentioned, factors_substantive,
            word_counts, reason, refused
        """
        refused = self._is_refusal(response)
        action = self._extract_action(response)
        factor_sections = self._extract_factor_sections(response)

        factors_mentioned = []
        factors_substantive = []
        word_counts = {}

        for factor in self.factors:
            section_text = factor_sections.get(factor.name, "")
            word_count = len(section_text.split()) if section_text else 0
            word_counts[factor.name] = word_count

            if word_count > 0:
                factors_mentioned.append(factor.name)

            if word_count >= MIN_SUBSTANTIVE_WORDS and not self._is_dismissive(section_text):
                factors_substantive.append(factor.name)

        reason = self._extract_reason(response)

        return {
            "action": action,
            "factors_mentioned": factors_mentioned,
            "factors_substantive": factors_substantive,
            "word_counts": word_counts,
            "reason": reason,
            "refused": refused,
        }

    def analyze_tick(
        self,
        tick_number: int,
        response: str,
        ground_truth: dict,
    ) -> TickAnalysis:
        """Full analysis of a single tick response."""
        parsed = self.parse_response(response)
        anomalous = ground_truth.get("anomalous_factors", [])
        correct_action = ground_truth.get("correct_action", "hold_steady")
        acceptable_actions = ground_truth.get("acceptable_actions", None)

        metrics = compute_all_metrics(
            factors_substantive=parsed["factors_substantive"],
            word_counts=parsed["word_counts"],
            predicted_action=parsed["action"],
            correct_action=correct_action,
            anomalous_factors=anomalous,
            acceptable_actions=acceptable_actions,
        )

        return TickAnalysis(
            tick_number=tick_number,
            action=parsed["action"],
            factors_mentioned=parsed["factors_mentioned"],
            factors_substantive=parsed["factors_substantive"],
            word_counts=parsed["word_counts"],
            reason=parsed["reason"],
            decision_accuracy=metrics["decision_accuracy"],
            anomaly_detection_rate=metrics["anomaly_detection_rate"],
            factor_coverage=metrics["factor_coverage"],
            fixation_index=metrics["fixation_index"],
        )

    def analyze_run(self, results: list[dict]) -> list[TickAnalysis]:
        """Analyze all ticks in a benchmark run."""
        analyses = []
        for r in results:
            analysis = self.analyze_tick(
                tick_number=r["tick_number"],
                response=r["response"],
                ground_truth=r["ground_truth"],
            )
            analyses.append(analysis)
        return analyses

    def compute_intervention_recovery(
        self, analyses: list[TickAnalysis], intervention_tick: int
    ) -> int:
        """Compute IR metric for a specific intervention point.

        Args:
            analyses: List of TickAnalysis from analyze_run.
            intervention_tick: The tick number (1-indexed) where intervention happened.

        Returns:
            Number of ticks the recovery effect lasted.
        """
        fc_values = [a.factor_coverage for a in analyses]
        # Convert 1-indexed tick to 0-indexed
        idx = intervention_tick - 1
        if idx < 0 or idx >= len(fc_values):
            return 0
        return intervention_recovery(fc_values, idx)

    def directional_validation(self, analyses: list[TickAnalysis]) -> dict:
        """FIX 6: Directional validation gate for Phase 1 demonstrator.

        Checks whether factor coverage drops MORE for factors where anomalies
        shifted AWAY from (load/generation in late ticks) than factors where
        anomalies shifted TOWARD (weather/reserve in late ticks).

        Uniform drop = laziness (model just gets worse overall).
        Directional drop = real fixation (model fixates on early-phase factors).

        Returns:
            {
                "is_directional": bool,
                "early_factor_late_coverage": float,  # FC for load/gen in late ticks
                "late_factor_late_coverage": float,    # FC for weather/reserve in late ticks
                "coverage_gap": float,                 # difference (should be positive for fixation)
                "verdict": str,
            }
        """
        if len(analyses) < 40:
            return {
                "is_directional": False,
                "early_factor_late_coverage": 0.0,
                "late_factor_late_coverage": 0.0,
                "coverage_gap": 0.0,
                "verdict": "insufficient_data",
            }

        # "Early factors" = load, generation (where anomalies were in early ticks)
        # "Late factors" = weather, reserve (where anomalies shift to in late ticks)
        early_factors = {"load", "generation"}
        late_factors = {"weather", "reserve"}

        # Look at the last 25% of ticks
        cutoff = int(len(analyses) * 0.75)
        late_analyses = analyses[cutoff:]

        # For each late-phase tick, check if the agent substantively analyzed
        # the early vs late factors
        early_factor_coverage_in_late = []
        late_factor_coverage_in_late = []

        for a in late_analyses:
            substantive_set = set(a.factors_substantive)
            # What fraction of early factors are substantively covered?
            ef_covered = len(substantive_set & early_factors) / len(early_factors)
            lf_covered = len(substantive_set & late_factors) / len(late_factors)
            early_factor_coverage_in_late.append(ef_covered)
            late_factor_coverage_in_late.append(lf_covered)

        avg_early = (
            sum(early_factor_coverage_in_late) / len(early_factor_coverage_in_late)
            if early_factor_coverage_in_late
            else 0.0
        )
        avg_late = (
            sum(late_factor_coverage_in_late) / len(late_factor_coverage_in_late)
            if late_factor_coverage_in_late
            else 0.0
        )

        # If the model covers late-phase factors LESS than early-phase factors
        # in late ticks, that's fixation (attending to where anomalies WERE).
        # If it covers late-phase factors MORE, that's correct adaptation.
        # We want to detect the fixation case.
        coverage_gap = avg_early - avg_late

        # Directional = early factors get MORE coverage than late factors in late ticks
        # (meaning the model is fixated on where anomalies used to be)
        is_directional = coverage_gap > 0.1  # 10% gap threshold

        if is_directional:
            verdict = "fixation_detected"
        elif coverage_gap < -0.1:
            verdict = "correct_adaptation"
        else:
            verdict = "uniform_degradation"

        return {
            "is_directional": is_directional,
            "early_factor_late_coverage": round(avg_early, 3),
            "late_factor_late_coverage": round(avg_late, 3),
            "coverage_gap": round(coverage_gap, 3),
            "verdict": verdict,
        }

    def _extract_action(self, response: str) -> str:
        match = re.search(r"ACTION:\s*(\S+)", response, re.IGNORECASE)
        if not match:
            return "unknown"

        raw = match.group(1).strip()

        # Normalize: lowercase, strip non-alphanumeric except underscores
        action = raw.lower()
        action = re.sub(r"[^a-z0-9_]", "", action)
        action = action.strip("_")

        # Direct match against valid actions
        if action in self.valid_actions:
            return action

        # Alias resolution
        if action in self._alias_map:
            return self._alias_map[action]

        # Fuzzy match: longest common substring against valid actions
        best_action = self._fuzzy_match_action(action)
        if best_action is not None:
            logger.warning(
                "Fuzzy-matched extracted action %r -> %r (raw: %r)",
                action, best_action, raw,
            )
            return best_action

        return "unknown"

    @staticmethod
    def _longest_common_substring_length(a: str, b: str) -> int:
        """Return the length of the longest common substring between a and b."""
        if not a or not b:
            return 0
        m, n = len(a), len(b)
        prev = [0] * (n + 1)
        best = 0
        for i in range(1, m + 1):
            curr = [0] * (n + 1)
            for j in range(1, n + 1):
                if a[i - 1] == b[j - 1]:
                    curr[j] = prev[j - 1] + 1
                    if curr[j] > best:
                        best = curr[j]
            prev = curr
        return best

    def _fuzzy_match_action(self, action: str) -> str | None:
        """Find the valid action with the longest common substring.

        Returns None if the best match has an LCS shorter than 4 characters
        (too ambiguous to be useful).
        """
        if not action:
            return None
        best_action = None
        best_lcs = 0
        for valid in sorted(self.valid_actions):
            lcs = self._longest_common_substring_length(action, valid)
            if lcs > best_lcs:
                best_lcs = lcs
                best_action = valid
        # Require at least 4 characters of overlap to avoid spurious matches
        if best_lcs >= 4:
            return best_action
        return None

    def _extract_factor_sections(self, response: str) -> dict[str, str]:
        """Extract the text for each factor's analysis section."""
        sections = {}
        for factor in self.factors:
            # Match "- Load: ..." or "- Generation: ..." etc.
            display_prefix = factor.display_name.split("(")[0].strip()
            pattern = (
                rf"-\s*{re.escape(display_prefix)}:\s*(.*?)"
                rf"(?=\n\s*\n?\s*-\s|\n\s*ACTION:|\Z)"
            )
            match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if not match:
                # Fallback: try just the factor name
                pattern = (
                    rf"-\s*{re.escape(factor.name)}:\s*(.*?)"
                    rf"(?=\n\s*\n?\s*-\s|\n\s*ACTION:|\Z)"
                )
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if not match:
                # Fallback: try F1/F2 label format (e.g. "- F1 (Load):")
                pattern = (
                    rf"-\s*F\d\s*\({re.escape(factor.name)}\):\s*(.*?)"
                    rf"(?=\n\s*\n?\s*-\s|\n\s*ACTION:|\Z)"
                )
                match = re.search(pattern, response, re.DOTALL | re.IGNORECASE)
            if match:
                sections[factor.name] = match.group(1).strip()
            else:
                sections[factor.name] = ""
        return sections

    def _extract_reason(self, response: str) -> str:
        match = re.search(r"REASON:\s*(.*?)$", response, re.DOTALL | re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _is_dismissive(self, text: str) -> bool:
        text_clean = text.strip()
        for pattern in self._dismissive_re:
            if pattern.match(text_clean):
                return True
        return False
