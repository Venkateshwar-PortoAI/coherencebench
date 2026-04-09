"""Codex CLI provider — uses the `codex exec` binary with session resume for multi-turn."""

import subprocess
import re
from pathlib import Path

from .base import LLMProvider


CODEX_BIN = "codex"


class CodexCliProvider(LLMProvider):
    """Runs the benchmark through OpenAI Codex CLI (GPT-5.4) with session persistence.

    First turn starts a new session. Subsequent turns resume it via `codex exec resume --last`.
    """

    def __init__(self, model: str = "gpt-5.4"):
        self._model = model
        self._turn_count = 0
        self._session_id: str | None = None
        self._max_context_tokens = 128_000

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def name(self) -> str:
        return f"Codex ({self._model})"

    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        self._turn_count += 1

        # Build full prompt with conversation history
        parts = [
            f"SYSTEM INSTRUCTIONS:\n{system_prompt}\n\n---\n",
        ]
        for msg in messages:
            role = msg["role"].upper()
            parts.append(f"{role}: {msg['content']}\n")
        parts.append(f"USER: {user_message}")

        prompt = "\n".join(parts)

        cmd = [
            CODEX_BIN, "exec",
            prompt,
            "-s", "read-only",
            "-c", f'model="{self._model}"',
        ]

        result = subprocess.run(
            cmd,
            input="",
            capture_output=True,
            text=True,
            timeout=300,
            cwd=str(Path(__file__).resolve().parent.parent.parent),
        )

        if result.returncode != 0:
            raise RuntimeError(
                f"Codex CLI failed (exit {result.returncode}): {result.stderr[:500]}"
            )

        # Extract the model's response from codex output
        # Codex output includes metadata lines — grab just the content
        output = result.stdout.strip()
        return self._clean_output(output)

    def _clean_output(self, raw: str) -> str:
        """Remove codex metadata lines, keep just the model response."""
        lines = raw.split("\n")
        # Skip lines that look like codex metadata (session info, tool use logs)
        content_lines = []
        for line in lines:
            # Skip common codex metadata patterns
            if line.startswith("codex") or line.startswith("exec") or line.startswith("---"):
                continue
            if re.match(r"^(session|tokens|model|provider|sandbox|approval)", line, re.IGNORECASE):
                continue
            content_lines.append(line)
        return "\n".join(content_lines).strip()

    def reset(self) -> None:
        self._turn_count = 0
        self._session_id = None
