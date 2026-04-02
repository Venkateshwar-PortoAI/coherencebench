"""Gemini API provider with retry logic (FIX 4)."""

from google import genai
from google.api_core import exceptions as google_exceptions
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import LLMProvider


class GoogleProvider(LLMProvider):
    def __init__(self, model: str = "gemini-2.0-flash"):
        self.model = model
        self.client = genai.Client()
        self._max_context_tokens = 1_000_000

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def name(self) -> str:
        return f"Gemini ({self.model})"

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((google_exceptions.ResourceExhausted, google_exceptions.ServiceUnavailable)),
        reraise=True,
    )
    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        contents = []
        for msg in messages:
            role = "model" if msg["role"] == "assistant" else "user"
            contents.append({"role": role, "parts": [{"text": msg["content"]}]})
        contents.append({"role": "user", "parts": [{"text": user_message}]})

        response = self.client.models.generate_content(
            model=self.model,
            contents=contents,
            config={"system_instruction": system_prompt, "max_output_tokens": 1024},
        )
        return response.text

    def reset(self) -> None:
        pass
