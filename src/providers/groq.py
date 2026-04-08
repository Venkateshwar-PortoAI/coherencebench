"""Groq provider — fast inference for open models via OpenAI-compatible API."""

import os

import openai
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import LLMProvider

MODELS = {
    "llama-70b": "llama-3.3-70b-versatile",
    "llama-8b": "llama-3.1-8b-instant",
    "gemma-27b": "gemma2-9b-it",
    "mixtral": "mixtral-8x7b-32768",
}

DEFAULT_MODEL = MODELS["llama-70b"]


class GroqProvider(LLMProvider):
    def __init__(self, model: str = DEFAULT_MODEL):
        self.model = MODELS.get(model, model)
        api_key = os.environ.get("GROQ_API_KEY")
        if not api_key:
            raise ValueError("GROQ_API_KEY not set. Get one free at https://console.groq.com")
        self.client = OpenAI(
            base_url="https://api.groq.com/openai/v1",
            api_key=api_key,
        )
        self._max_context_tokens = 128_000

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def name(self) -> str:
        return f"Groq ({self.model})"

    @retry(
        stop=stop_after_attempt(6),
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
