"""LLM provider registry."""

from importlib import import_module

from .base import LLMProvider

PROVIDERS = {
    "claude": (".anthropic", "AnthropicProvider"),
    "claude-cli": (".claude_cli", "ClaudeCliProvider"),
    "codex": (".codex_cli", "CodexCliProvider"),
    "gpt4o": (".openai", "OpenAIProvider"),
    "gemini": (".google", "GoogleProvider"),
    "llama": (".together", "TogetherProvider"),
    "ollama": (".ollama", "OllamaProvider"),
    "openrouter": (".openrouter", "OpenRouterProvider"),
    "groq": (".groq", "GroqProvider"),
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
    module_name, class_name = PROVIDERS[name]
    module = import_module(module_name, package=__name__)
    provider_class = getattr(module, class_name)
    return provider_class(**kwargs)
