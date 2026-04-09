"""MLvoca provider — free LLM API with no rate limits or API key required."""

import requests
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import LLMProvider

MODELS = {
    "tinyllama": "tinyllama",
    "deepseek-r1": "deepseek-r1:1.5b",
}

DEFAULT_MODEL = MODELS["deepseek-r1"]
BASE_URL = "https://mlvoca.com"


class MLvocaProvider(LLMProvider):
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = MODELS.get(model, model)
        self._max_context_tokens = 8_000

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def name(self) -> str:
        return f"MLvoca ({self.model})"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
        retry=retry_if_exception_type((requests.ConnectionError, requests.Timeout)),
        reraise=True,
    )
    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        # Build prompt from conversation history since this is a completion API
        prompt_parts = []
        for msg in messages:
            role = msg["role"]
            if role == "user":
                prompt_parts.append(f"User: {msg['content']}")
            elif role == "assistant":
                prompt_parts.append(f"Assistant: {msg['content']}")
        prompt_parts.append(f"User: {user_message}")
        prompt_parts.append("Assistant:")

        response = requests.post(
            f"{BASE_URL}/api/generate",
            json={
                "model": self.model,
                "prompt": "\n".join(prompt_parts),
                "system": system_prompt,
                "stream": False,
            },
            timeout=120,
        )
        response.raise_for_status()
        return response.json()["response"]

    def reset(self) -> None:
        pass
