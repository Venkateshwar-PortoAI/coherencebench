"""Together AI provider for open-source models (Llama, Mistral, etc.).

FIX 5: Explicitly passes api_key from TOGETHER_API_KEY env var.
FIX 4: Includes exponential backoff retry.
"""

import os

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import LLMProvider


class TogetherProvider(LLMProvider):
    """Uses Together AI's OpenAI-compatible API."""

    def __init__(self, model: str = "meta-llama/Meta-Llama-3.1-70B-Instruct-Turbo"):
        self.model = model
        # FIX 5: Explicitly pass api_key from env var
        self.client = OpenAI(
            base_url="https://api.together.xyz/v1",
            api_key=os.environ.get("TOGETHER_API_KEY"),
        )
        self._max_context_tokens = 128_000

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def name(self) -> str:
        short_name = self.model.split("/")[-1]
        return f"Llama ({short_name})"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
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
