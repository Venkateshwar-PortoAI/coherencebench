"""Ollama provider for local open-source models (zero cost)."""

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import LLMProvider


class OllamaProvider(LLMProvider):
    """Uses Ollama's OpenAI-compatible local API."""

    def __init__(self, model: str = "deepseek-r1:14b"):
        self.model = model
        self.client = OpenAI(
            base_url="http://localhost:11434/v1",
            api_key="ollama",
        )
        self._max_context_tokens = 128_000

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def name(self) -> str:
        return f"Ollama ({self.model})"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((openai.APIConnectionError, openai.InternalServerError)),
        reraise=True,
    )
    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        all_messages = [{"role": "system", "content": system_prompt}]
        all_messages += messages
        all_messages.append({"role": "user", "content": user_message})
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=1024,
            messages=all_messages,
        )
        return response.choices[0].message.content

    def reset(self) -> None:
        pass
