"""OpenRouter provider — free-tier open-source models via OpenAI-compatible API."""

import os

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import LLMProvider

# Free models available on OpenRouter (rate-limited, $0).
FREE_MODELS = {
    "nemotron-120b": "nvidia/nemotron-3-super-120b-a12b:free",
    "gemma-31b": "google/gemma-4-31b-it:free",
    "step-flash": "stepfun/step-3.5-flash:free",
    "minimax-m2.5": "minimax/minimax-m2.5:free",
    "nemotron-30b": "nvidia/nemotron-3-nano-30b-a3b:free",
    "gemma-26b": "google/gemma-4-26b-a4b-it:free",
}

DEFAULT_MODEL = FREE_MODELS["nemotron-120b"]


class OpenRouterProvider(LLMProvider):
    def __init__(self, model: str = DEFAULT_MODEL):
        # Allow short aliases like "qwen-72b"
        self.model = FREE_MODELS.get(model, model)
        api_key = os.environ.get("OPENROUTER_API_KEY")
        if not api_key:
            raise ValueError("OPENROUTER_API_KEY not set. Get one free at https://openrouter.ai/keys")
        self.client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=api_key,
        )
        self._max_context_tokens = 128_000

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def name(self) -> str:
        return f"OpenRouter ({self.model})"

    @retry(
        stop=stop_after_attempt(5),
        wait=wait_exponential(multiplier=2, min=4, max=60),
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
