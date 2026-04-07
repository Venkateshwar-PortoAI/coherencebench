from __future__ import annotations

"""Claude API provider with explicit retry and rate-limit handling."""

import logging
import json
import os
import random
import subprocess
import time
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime

import anthropic

from .base import LLMProvider

logger = logging.getLogger(__name__)

MAX_RETRY_ATTEMPTS = 6
MIN_RETRY_DELAY_SECONDS = 5.0
MAX_RETRY_DELAY_SECONDS = 120.0
RATE_LIMIT_DELAY_SECONDS = 15.0


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


def _parse_retry_after_seconds(headers) -> float | None:
    """Best-effort parse of Anthropic retry headers."""
    if not headers:
        return None

    retry_after_ms = headers.get("retry-after-ms")
    if retry_after_ms:
        try:
            return max(float(retry_after_ms) / 1000.0, 0.0)
        except ValueError:
            pass

    retry_after = headers.get("retry-after")
    if not retry_after:
        return None

    try:
        return max(float(retry_after), 0.0)
    except ValueError:
        pass

    try:
        retry_at = parsedate_to_datetime(retry_after)
        if retry_at.tzinfo is None:
            retry_at = retry_at.replace(tzinfo=timezone.utc)
        delay = (retry_at - datetime.now(timezone.utc)).total_seconds()
        return max(delay, 0.0)
    except (TypeError, ValueError, OverflowError):
        return None


def _extract_retry_delay_seconds(exc: Exception) -> float | None:
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    return _parse_retry_after_seconds(headers)


class AnthropicProvider(LLMProvider):
    def __init__(self, model: str = "claude-haiku-4-5-20251001", api_key: str | None = None):
        self.model = model
        # Priority: explicit key > env var > keychain (Claude Code OAuth)
        key = api_key or os.environ.get("ANTHROPIC_API_KEY") or _get_claude_code_token()
        if key:
            self.client = anthropic.Anthropic(api_key=key, max_retries=0)
        else:
            self.client = anthropic.Anthropic(max_retries=0)  # Will fail if no key found
        self._max_context_tokens = 200_000

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def name(self) -> str:
        return f"Claude ({self.model})"

    def _refresh_token(self) -> bool:
        """Re-fetch OAuth token from keychain if auth fails.

        Returns True if a new token was found, False otherwise.
        Waits briefly before fetching to allow token rotation to complete.
        """
        import time
        time.sleep(2)
        token = _get_claude_code_token()
        if token:
            self.client = anthropic.Anthropic(api_key=token, max_retries=0)
            return True
        return False

    def _compute_retry_delay_seconds(self, exc: Exception, attempt: int) -> float:
        retry_after = _extract_retry_delay_seconds(exc)
        if retry_after is not None:
            return min(max(retry_after, 0.0), MAX_RETRY_DELAY_SECONDS)

        if isinstance(exc, anthropic.RateLimitError):
            return min(RATE_LIMIT_DELAY_SECONDS * (2 ** (attempt - 1)), MAX_RETRY_DELAY_SECONDS)

        base_delay = min(MIN_RETRY_DELAY_SECONDS * (2 ** (attempt - 1)), MAX_RETRY_DELAY_SECONDS)
        jitter = random.uniform(0.0, min(5.0, base_delay * 0.2))
        return base_delay + jitter

    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        all_messages = messages + [{"role": "user", "content": user_message}]
        for attempt in range(1, MAX_RETRY_ATTEMPTS + 1):
            try:
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=1024,
                    system=system_prompt,
                    messages=all_messages,
                )
                return response.content[0].text
            except anthropic.AuthenticationError:
                # Token rotated. Try refreshing up to 3 times with increasing waits.
                for refresh_attempt in range(3):
                    if self._refresh_token():
                        logger.warning(
                            "Anthropic authentication failed; refreshed token and retrying (%d/3).",
                            refresh_attempt + 1,
                        )
                        try:
                            response = self.client.messages.create(
                                model=self.model,
                                max_tokens=1024,
                                system=system_prompt,
                                messages=all_messages,
                            )
                            return response.content[0].text
                        except anthropic.AuthenticationError:
                            time.sleep(5 * (refresh_attempt + 1))
                            continue
                raise
            except (
                anthropic.RateLimitError,
                anthropic.APIConnectionError,
                anthropic.InternalServerError,
            ) as exc:
                if attempt >= MAX_RETRY_ATTEMPTS:
                    raise
                delay = self._compute_retry_delay_seconds(exc, attempt)
                if isinstance(exc, anthropic.RateLimitError):
                    logger.warning(
                        "Anthropic rate limit for model=%s. Backing off %.1fs before retry %d/%d (request_id=%s).",
                        self.model,
                        delay,
                        attempt + 1,
                        MAX_RETRY_ATTEMPTS,
                        getattr(exc, "request_id", "-"),
                    )
                else:
                    logger.warning(
                        "Anthropic request failed with %s. Retrying in %.1fs (attempt %d/%d, request_id=%s).",
                        type(exc).__name__,
                        delay,
                        attempt + 1,
                        MAX_RETRY_ATTEMPTS,
                        getattr(exc, "request_id", "-"),
                    )
                time.sleep(delay)

    def reset(self) -> None:
        pass
