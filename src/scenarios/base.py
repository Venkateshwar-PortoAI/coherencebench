"""Abstract base class for CoherenceBench scenarios."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field


@dataclass
class Factor:
    name: str
    display_name: str
    description: str
    keywords: list[str] = field(default_factory=list)


class BaseScenario(ABC):
    """Interface that all CoherenceBench scenarios must implement."""

    @property
    @abstractmethod
    def factors(self) -> list[Factor]:
        """The 6 factors monitored in this scenario."""
        ...

    @property
    @abstractmethod
    def actions(self) -> list[str]:
        """The 10 possible actions the agent can take."""
        ...

    @property
    @abstractmethod
    def anomaly_action_map(self) -> dict:
        """Map from factor name to {primary, acceptable} actions."""
        ...

    @property
    @abstractmethod
    def multi_factor_rules(self) -> list[tuple]:
        """Multi-factor combination rules: (frozenset, action, relevant_factors)."""
        ...

    @property
    @abstractmethod
    def phase_anomaly_weights(self) -> dict:
        """Anomaly probability by factor per phase range."""
        ...

    @property
    @abstractmethod
    def initial_state(self) -> dict:
        """Starting state for the tick generator."""
        ...

    @abstractmethod
    def system_prompt(self) -> str:
        """The system prompt given to the LLM."""
        ...

    @abstractmethod
    def format_tick(self, tick_number: int, tick_data: dict) -> str:
        """Format a single tick into the prompt text."""
        ...

    @abstractmethod
    def format_state_summary(self, tick_data: dict, tick_number: int) -> str:
        """Format a state summary for context reset re-injection."""
        ...

    @abstractmethod
    def evolve_state(self, state: dict, rng) -> dict:
        """Random walk the state to make data look natural."""
        ...

    @abstractmethod
    def inject_anomaly(self, data: dict, factor: str, rng) -> dict:
        """Inject a clear anomaly into the specified factor."""
        ...

    @abstractmethod
    def format_tick_data(self, state: dict) -> dict:
        """Format raw state into the dict expected by format_tick()."""
        ...

    def deep_copy_state(self, state: dict) -> dict:
        """Deep copy a state dict (assumes one level of nesting)."""
        return {k: dict(v) for k, v in state.items()}
