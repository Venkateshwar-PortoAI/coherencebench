"""GPT API provider with retry logic (FIX 4)."""

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import LLMProvider


class OpenAIProvider(LLMProvider):
    def __init__(self, model: str = "gpt-4o"):
        self.model = model
        self.client = OpenAI()
        self._max_context_tokens = 128_000

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def name(self) -> str:
        return f"GPT ({self.model})"

    @retry(
        stop=stop_after_attempt(10),
        wait=wait_exponential(multiplier=2, min=4, max=120),
        retry=retry_if_exception_type((openai.RateLimitError, openai.APIConnectionError, openai.InternalServerError)),
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
