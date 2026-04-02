"""Abstract base class for LLM providers.

FIX 3: Includes max_context_tokens for context budget management.
"""

from abc import ABC, abstractmethod


class LLMProvider(ABC):
    """Interface for LLM API providers."""

    @property
    def max_context_tokens(self) -> int:
        """Maximum context window in tokens.

        FIX 3: Used by the runner for context budget management.
        Before each send_turn, the runner estimates total tokens. If over
        budget, it truncates oldest messages (keeping system prompt + last N).

        Override in subclasses to match the model's actual context window.
        Default is 128k tokens (conservative for most modern models).
        """
        return 128_000

    @abstractmethod
    def name(self) -> str:
        """Human-readable model name for results."""
        ...

    @abstractmethod
    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        """Send a single turn and get the response text.

        Args:
            system_prompt: The system prompt (scenario instructions).
            messages: Prior conversation history as list of {"role": str, "content": str}.
            user_message: The current tick prompt.

        Returns:
            The model's response text.
        """
        ...

    @abstractmethod
    def reset(self) -> None:
        """Clear conversation state for a fresh session."""
        ...

    def estimate_tokens(self, text: str) -> int:
        """Rough token estimate. ~4 chars per token for English text."""
        return len(text) // 4
