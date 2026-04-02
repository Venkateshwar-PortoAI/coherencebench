"""LLM provider registry."""

from .base import LLMProvider
from .anthropic import AnthropicProvider
from .openai import OpenAIProvider
from .google import GoogleProvider
from .together import TogetherProvider

PROVIDERS = {
    "claude": AnthropicProvider,
    "gpt4o": OpenAIProvider,
    "gemini": GoogleProvider,
    "llama": TogetherProvider,
}


def get_provider(name: str, **kwargs) -> LLMProvider:
    """Get a provider instance by name.

    Args:
        name: One of 'claude', 'gpt4o', 'gemini', 'llama'.
        **kwargs: Passed to the provider constructor (e.g., model=...).

    Returns:
        An LLMProvider instance.
    """
    if name not in PROVIDERS:
        raise ValueError(f"Unknown provider: {name}. Available: {list(PROVIDERS.keys())}")
    return PROVIDERS[name](**kwargs)
