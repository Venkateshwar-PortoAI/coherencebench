"""Claude Code CLI provider — uses the `claude` CLI binary with session continuity."""

import subprocess
import uuid

from .base import LLMProvider


CLAUDE_BIN = "/Users/gokukilluavegeta/.local/bin/claude"


class ClaudeCliProvider(LLMProvider):
    """Runs the benchmark through Claude Code's CLI with session persistence.

    Each benchmark run gets a unique session ID. The first tick starts a new session,
    subsequent ticks resume it with -c (continue), maintaining full conversation context.
    """

    def __init__(self, model: str = "sonnet"):
        self._model = model
        self._session_id = str(uuid.uuid4())
        self._turn_count = 0
        self._max_context_tokens = 200_000

    @property
    def max_context_tokens(self) -> int:
        return self._max_context_tokens

    def name(self) -> str:
        return f"ClaudeCLI ({self._model})"

    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        """Send a turn via the Claude CLI.

        Note: The CLI manages its own conversation history via session persistence.
        We ignore the `messages` parameter since the CLI already has the full history.
        On the first turn, we prepend the system prompt to the user message.
        """
        self._turn_count += 1

        if self._turn_count == 1:
            # First turn: include system prompt as context, start new session
            prompt = f"SYSTEM INSTRUCTIONS (follow these for the entire session):\n{system_prompt}\n\n---\n\n{user_message}"
            cmd = [
                CLAUDE_BIN, "-p",
                "--model", self._model,
                "--session-id", self._session_id,
                "--permission-mode", "bypassPermissions",
            ]
        else:
            # Subsequent turns: resume the exact session by ID
            prompt = user_message
            cmd = [
                CLAUDE_BIN, "-p",
                "-r", self._session_id,
                "--permission-mode", "bypassPermissions",
            ]

        result = subprocess.run(
            cmd,
            input=prompt,
            capture_output=True,
            text=True,
            timeout=300,  # 5 min — sessions get slow as they grow
        )

        if result.returncode != 0:
            raise RuntimeError(f"Claude CLI failed (exit {result.returncode}): {result.stderr[:500]}")

        return result.stdout.strip()

    def reset(self) -> None:
        """Start a fresh session."""
        self._session_id = str(uuid.uuid4())
        self._turn_count = 0
