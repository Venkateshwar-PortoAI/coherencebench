"""Claude API provider with retry logic (FIX 4)."""

import json
import os
import subprocess

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type

from .base import LLMProvider


def _get_claude_code_token() -> str | None:
    """Try to extract the OAuth token from macOS keychain (Claude Code credentials)."""
    try:
        result = subprocess.run(
            ["security", "find-generic-password", "-s", "Claude Code-credentials", "-w"],
            capture_output=True, text=True, timeout=5,
        )
        if result.returncode == 0 and result.stdout.strip():
            creds = json.loads(result.stdout.strip())
            token = creds.get("claudeAiOauth", {}).get("accessToken")
            if token:
                return token
    except (subprocess.TimeoutExpired, json.JSONDecodeError, KeyError):
        pass
    return None


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        self.model = model
        # Priority: explicit key > env var > keychain (Claude Code OAuth)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY") or _get_claude_code_token()
        if key:
            self.client = anthropic.Anthropic(api_key=key)
        else:
            self.client = anthropic.Anthropic()  # Will fail if no key found
        self._max_context_tokens = 200_000

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def name(self) -> str:
        return f"Claude ({self.model})"

    def _refresh_token(self) -> None:
        """Re-fetch OAuth token from keychain if auth fails."""
        token = _get_claude_code_token()
        if token:
            self.client = anthropic.Anthropic(api_key=token)

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=30),
        retry=retry_if_exception_type((anthropic.RateLimitError, anthropic.APIConnectionError, anthropic.InternalServerError)),
        reraise=True,
    )
    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        all_messages = messages + [{"role": "user", "content": user_message}]
        try:
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=all_messages,
            )
            return response.content[0].text
        except anthropic.AuthenticationError:
            self._refresh_token()
            response = self.client.messages.create(
                model=self.model,
                max_tokens=1024,
                system=system_prompt,
                messages=all_messages,
            )
            return response.content[0].text

    def reset(self) -> None:
        pass
