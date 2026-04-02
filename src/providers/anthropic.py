"""Claude API provider with retry logic (FIX 4)."""

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import LLMProvider


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-sonnet-4-6"):
        self.model = model
        self.client = anthropic.Anthropic()
        self._max_context_tokens = 200_000

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def name(self) -> str:
        return f"Claude ({self.model})"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)),
        reraise=True,
    )
    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        all_messages = messages + [{"role": "user", "content": user_message}]
        response = self.client.messages.create(
            model=self.model,
            max_tokens=1024,
            system=system_prompt,
            messages=all_messages,
        )
        return response.content[0].text

    def reset(self) -> None:
        pass
