"""Core metrics for CoherenceBench.

FIX 8: All 5 metrics implemented end-to-end:
  FC  - Factor Coverage
  FI  - Fixation Index
  DA  - Decision Accuracy
  ADR - Anomaly Detection Rate (FIX 1: behavior-based PRIMARY metric)
  IR  - Intervention Recovery
"""

from __future__ import annotations


def factor_coverage(factors_mentioned: list[str], total: int = 6) -> float:
    """FC: fraction of factors substantively analyzed (0.0 to 1.0)."""
    if total == 0:
        return 1.0
    unique = len(set(factors_mentioned))
    return min(unique / total, 1.0)


def fixation_index(token_counts: dict[str, int]) -> float:
    """FI: fraction of analysis tokens devoted to the top factor (0.0 to 1.0).

    Low FI = balanced attention. High FI = fixated on one factor.
    """
    total = sum(token_counts.values())
    if total == 0:
        return 0.0
    max_count = max(token_counts.values())
    return max_count / total


def decision_accuracy(
    predicted_action: str,
    correct_action: str,
    acceptable_actions: list[str] | None = None,
) -> float:
    """DA: 1.0 if the agent chose an acceptable action, 0.0 otherwise.

    If acceptable_actions is provided, any of those count as correct.
    Otherwise falls back to exact match on correct_action.
    """
    pred = predicted_action.strip().lower()
    if acceptable_actions:
        return 1.0 if pred in [a.strip().lower() for a in acceptable_actions] else 0.0
    return 1.0 if pred == correct_action.strip().lower() else 0.0


def anomaly_detection_rate(
    factors_substantively_mentioned: list[str],
    anomalous_factors: list[str],
) -> float:
    """ADR: fraction of anomalous factors that the agent substantively analyzed.

    FIX 1: This is the PRIMARY metric. It measures whether the agent's analysis
    demonstrates awareness of anomalies in each factor, not just format compliance.

    If no anomalies exist, returns 1.0 (nothing to miss).
    """
    if not anomalous_factors:
        return 1.0
    detected = sum(1 for f in anomalous_factors if f in factors_substantively_mentioned)
    return detected / len(anomalous_factors)


def intervention_recovery(fc_values: list[float], intervention_idx: int) -> int:
    """IR: number of ticks after intervention before FC drops back to pre-intervention level.

    Measures how long the "wake-up" effect of an intervention lasts.
    Returns 0 if no recovery observed.
    """
    if intervention_idx >= len(fc_values) or intervention_idx == 0:
        return 0

    pre_fc = fc_values[intervention_idx - 1]

    recovery_count = 0
    for fc in fc_values[intervention_idx:]:
        if fc > pre_fc:
            recovery_count += 1
        else:
            break

    return recovery_count


def compute_all_metrics(
    factors_substantive: list[str],
    token_counts: dict[str, int],
    predicted_action: str,
    correct_action: str,
    anomalous_factors: list[str],
    total_factors: int = 6,
    acceptable_actions: list[str] | None = None,
) -> dict[str, float]:
    """Compute FC, FI, DA, and ADR for a single tick.

    IR is computed across ticks (not per-tick), so it's not included here.
    Use intervention_recovery() separately on a sequence of FC values.
    """
    return {
        "factor_coverage": factor_coverage(factors_substantive, total_factors),
        "fixation_index": fixation_index(token_counts),
        "decision_accuracy": decision_accuracy(
            predicted_action, correct_action, acceptable_actions
        ),
        "anomaly_detection_rate": anomaly_detection_rate(
            factors_substantive, anomalous_factors
        ),
    }
