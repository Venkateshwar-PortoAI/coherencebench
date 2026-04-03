"""Backward-compatible shim for the power grid scenario.

The canonical location is now src/scenarios/power_grid.py.
This module re-exports everything so existing imports keep working.
"""

from .scenarios.base import Factor
from .scenarios.power_grid import PowerGridScenario

# Re-export the factor and action lists for backward compatibility with analyzer.py
_scenario = PowerGridScenario()
FACTORS = _scenario.factors
ACTIONS = _scenario.actions

__all__ = ["Factor", "PowerGridScenario", "FACTORS", "ACTIONS"]
