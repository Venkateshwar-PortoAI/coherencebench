import httpx
import anthropic

from src.providers.anthropic import (
    AnthropicProvider,
    MAX_RETRY_DELAY_SECONDS,
    RATE_LIMIT_DELAY_SECONDS,
)


def _rate_limit_error(headers=None) -> anthropic.RateLimitError:
    request = httpx.Request("POST", "https://api.anthropic.com/v1/messages")
    response = httpx.Response(
        429,
        headers=headers or {},
        request=request,
    )
    return anthropic.RateLimitError("rate limited", response=response, body={})


def test_retry_delay_respects_retry_after_header():
    provider = AnthropicProvider(api_key="test-key")
    exc = _rate_limit_error({"retry-after": "17"})

    delay = provider._compute_retry_delay_seconds(exc, attempt=1)

    assert delay == 17.0


def test_retry_delay_uses_rate_limit_backoff_without_headers():
    provider = AnthropicProvider(api_key="test-key")
    exc = _rate_limit_error()

    delay = provider._compute_retry_delay_seconds(exc, attempt=3)

    assert delay == min(RATE_LIMIT_DELAY_SECONDS * 4, MAX_RETRY_DELAY_SECONDS)
