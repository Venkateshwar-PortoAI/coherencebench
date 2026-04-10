"""Gemini CLI provider — uses the `gemini` CLI binary for non-interactive prompts."""

import subprocess
import re

from .base import LLMProvider

GEMINI_BIN = "gemini"


class GeminiCliProvider(LLMProvider):
    """Runs the benchmark through Google's Gemini CLI.

    Uses cached Google credentials (from Antigravity/gcloud auth).
    Each tick is a single-shot prompt with conversation history included.
    """

    def __init__(self, model: str = "gemini-2.5-flash"):
        self._model = model
        self._turn_count = 0
        self._max_context_tokens = 1_000_000

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def name(self) -> str:
        return f"Gemini CLI ({self._model})"

    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        self._turn_count += 1

        # Build prompt with conversation history, capped to avoid CLI limits
        MAX_HISTORY_CHARS = 25_000
        parts = [
            f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n---\n",
        ]
        history_chars = 0
        kept_messages = []
        for msg in reversed(messages):
            msg_len = len(msg["content"])
            if history_chars + msg_len > MAX_HISTORY_CHARS:
                break
            kept_messages.append(msg)
            history_chars += msg_len
        kept_messages.reverse()

        for msg in kept_messages:
            role = msg["role"].upper()
            parts.append(f"{role}: {msg['content']}\n")
        parts.append(f"USER: {user_message}")

        prompt = "\n".join(parts)

        result = subprocess.run(
            [GEMINI_BIN, "-p", prompt, "--sandbox", "false", "-m", self._model, "-o", "text"],
            input="",
            capture_output=True,
            text=True,
            timeout=600,
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Gemini CLI failed (exit {result.returncode}): {result.stderr[:500]}"
            )

        return self._clean_output(result.stdout)

    def _clean_output(self, raw: str) -> str:
        """Remove Gemini CLI metadata lines."""
        lines = raw.split("\n")
        content_lines = []
        for line in lines:
            if line.startswith("(node:") or "DeprecationWarning" in line:
                continue
            if "Loaded cached credentials" in line:
                continue
            if "[ERROR]" in line and "IDE companion" in line:
                continue
            if "Hook registry" in line:
                continue
            content_lines.append(line)
        return "\n".join(content_lines).strip()

    def reset(self) -> None:
        self._turn_count = 0
