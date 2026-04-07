"""Scenario registry for CoherenceBench."""

from .power_grid import PowerGridScenario
from .hospital import HospitalTriageScenario
from .network import NetworkSecurityScenario
from .air_traffic_control import AirTrafficControlScenario

SCENARIOS = {
    "power_grid": PowerGridScenario,
    "hospital": HospitalTriageScenario,
    "network": NetworkSecurityScenario,
    "air_traffic_control": AirTrafficControlScenario,
}


def get_scenario(name: str = "power_grid"):
    """Get a scenario instance by name.

    Args:
        name: One of 'power_grid', 'hospital', 'network', 'air_traffic_control'.

    Returns:
        A BaseScenario subclass instance.
    """
    if name not in SCENARIOS:
        raise ValueError(f"Unknown scenario: {name}. Available: {list(SCENARIOS.keys())}")
    return SCENARIOS[name]()
