from __future__ import annotations

"""Benchmark runner: feeds ticks to LLM and logs responses.

FIX 3: Context budget management -- truncates oldest messages when over budget.
FIX 7: Context reset with state re-injection -- injects state summary on reset.
"""

import json
import logging
import time
from pathlib import Path
from typing import Callable

from .generator import TickGenerator
from .scenarios import get_scenario
from .scenarios.base import BaseScenario
from .providers.base import LLMProvider

logger = logging.getLogger(__name__)


def _intervention_prompt(scenario) -> str:
    factor_names = ", ".join(f.display_name.split("(")[0].strip() for f in scenario.factors)
    n = len(scenario.factors)
    return (
        "WARNING: Your recent analyses have been incomplete. "
        f"You are NOT analyzing all {n} factors. "
        f"You MUST provide substantive analysis for {factor_names} before making your decision. "
        "Failure to analyze all factors leads to dangerous decisions. "
        "Resume full analysis now."
    )


def _checklist_suffix(scenario) -> str:
    items = ", ".join(
        f"({i+1}) {f.display_name.split('(')[0].strip()}"
        for i, f in enumerate(scenario.factors)
    )
    n = len(scenario.factors)
    return f"\n\nREMINDER: Before answering, verify you have analyzed: {items}. ALL {n} are required."


class BenchmarkRunner:
    """Runs the benchmark against a single LLM provider."""

    def __init__(
        self,
        provider: LLMProvider,
        num_ticks: int = 200,
        seed: int = 42,
        output_dir: Path = Path("results"),
        context_reset_interval: int | None = None,
        intervention_ticks: list[int] | None = None,
        force_checklist: bool = False,
        context_budget_ratio: float = 0.8,
        scenario: str | BaseScenario = "power_grid",
        progress_callback: Callable[[dict], None] | None = None,
    ):
        self.provider = provider
        self.num_ticks = num_ticks
        self.seed = seed
        self.output_dir = output_dir
        self.context_reset_interval = context_reset_interval
        self.intervention_ticks = intervention_ticks or []
        self.force_checklist = force_checklist
        self.context_budget_ratio = context_budget_ratio
        self.progress_callback = progress_callback

        if isinstance(scenario, str):
            self.scenario = get_scenario(scenario)
        else:
            self.scenario = scenario
        self.generator = TickGenerator(seed=seed, num_ticks=num_ticks, scenario=self.scenario)

    def _emit_progress(self, event: dict) -> None:
        if self.progress_callback is None:
            return

        try:
            self.progress_callback(event)
        except Exception:
            logger.exception("Progress callback failed for event %s", event.get("event"))

    def _estimate_message_tokens(self, messages: list[dict], system_prompt: str) -> int:
        """FIX 3: Estimate total tokens in the conversation."""
        total = self.provider.estimate_tokens(system_prompt)
        for msg in messages:
            total += self.provider.estimate_tokens(msg["content"])
        return total

    def _truncate_messages(
        self, messages: list[dict], system_prompt: str, new_message: str
    ) -> list[dict]:
        """FIX 3: Truncate oldest messages if over context budget.

        Keeps system prompt + last N message pairs. Always preserves
        at least the last 4 messages (2 turns).
        """
        budget = int(self.provider.max_context_tokens * self.context_budget_ratio)
        new_msg_tokens = self.provider.estimate_tokens(new_message)
        sys_tokens = self.provider.estimate_tokens(system_prompt)

        available = budget - sys_tokens - new_msg_tokens - 1024  # reserve for response
        if available <= 0:
            # Extreme case: just keep last 2 messages
            return messages[-4:] if len(messages) >= 4 else messages

        # Add messages from newest to oldest until we hit budget
        kept = []
        running_tokens = 0
        for msg in reversed(messages):
            msg_tokens = self.provider.estimate_tokens(msg["content"])
            if running_tokens + msg_tokens > available and len(kept) >= 4:
                break
            kept.append(msg)
            running_tokens += msg_tokens

        kept.reverse()

        if len(kept) < len(messages):
            logger.info(
                "Context budget: truncated %d -> %d messages (%.0f%% of %d token budget)",
                len(messages), len(kept),
                (running_tokens / budget) * 100, budget,
            )

        return kept

    def run(self) -> list[dict]:
        ticks = self.generator.generate()
        system_prompt = self.scenario.system_prompt()
        messages: list[dict] = []
        results = []

        self.output_dir.mkdir(parents=True, exist_ok=True)
        results_file = self.output_dir / "raw_results.jsonl"

        last_tick_data = None
        self._emit_progress(
            {
                "event": "run_started",
                "num_ticks": self.num_ticks,
                "seed": self.seed,
                "provider_name": self.provider.name(),
                "scenario_name": type(self.scenario).__name__,
                "results_file": str(results_file),
            }
        )

        with open(results_file, "w") as f:
            for i, tick in enumerate(ticks):
                tick_num = tick["tick_number"]
                cycle_index = i + 1
                context_reset_applied = False

                # FIX 7: Context reset with state re-injection
                if (
                    self.context_reset_interval
                    and i > 0
                    and i % self.context_reset_interval == 0
                ):
                    context_reset_applied = True
                    if last_tick_data is not None:
                        state_summary = self.scenario.format_state_summary(
                            last_tick_data, tick_num - 1
                        )
                        factor_names = ", ".join(
                            f.display_name.split("(")[0].strip()
                            for f in self.scenario.factors
                        )
                        n = len(self.scenario.factors)
                        messages = [
                            {
                                "role": "user",
                                "content": state_summary,
                            },
                            {
                                "role": "assistant",
                                "content": (
                                    f"Understood. I have the current state. "
                                    f"I will continue monitoring all {n} factors: "
                                    f"{factor_names}."
                                ),
                            },
                        ]
                    else:
                        messages = []
                    self.provider.reset()
                    self._emit_progress(
                        {
                            "event": "context_reset",
                            "tick_number": tick_num,
                            "cycle_index": cycle_index,
                            "total_ticks": self.num_ticks,
                            "restored_from_tick": tick_num - 1,
                        }
                    )

                # Intervention if this tick is flagged
                intervention_applied = tick_num in self.intervention_ticks
                if intervention_applied:
                    messages.append({"role": "user", "content": _intervention_prompt(self.scenario)})
                    n = len(self.scenario.factors)
                    messages.append({
                        "role": "assistant",
                        "content": (
                            f"Understood. I will analyze all {n} factors thoroughly from now on."
                        ),
                    })
                    self._emit_progress(
                        {
                            "event": "intervention",
                            "tick_number": tick_num,
                            "cycle_index": cycle_index,
                            "total_ticks": self.num_ticks,
                        }
                    )

                # Format tick prompt
                user_msg = self.scenario.format_tick(tick_num, tick["data"])
                if self.force_checklist:
                    user_msg += _checklist_suffix(self.scenario)

                # FIX 3: Truncate if over context budget
                pre_truncate_len = len(messages)
                messages = self._truncate_messages(messages, system_prompt, user_msg)
                context_truncated = len(messages) < pre_truncate_len

                # Send to LLM
                turn_started_at = time.monotonic()
                response = self.provider.send_turn(system_prompt, messages, user_msg)
                duration_seconds = time.monotonic() - turn_started_at

                # Append to conversation history
                messages.append({"role": "user", "content": user_msg})
                messages.append({"role": "assistant", "content": response})

                last_tick_data = tick["data"]

                result = {
                    "tick_number": tick_num,
                    "response": response,
                    "ground_truth": tick["ground_truth"],
                    "context_truncated": context_truncated,
                }
                results.append(result)

                # Write incrementally
                f.write(json.dumps(result) + "\n")
                f.flush()

                self._emit_progress(
                    {
                        "event": "tick_completed",
                        "tick_number": tick_num,
                        "cycle_index": cycle_index,
                        "total_ticks": self.num_ticks,
                        "response": response,
                        "ground_truth": tick["ground_truth"],
                        "context_truncated": context_truncated,
                        "context_reset": context_reset_applied,
                        "intervention": intervention_applied,
                        "duration_seconds": duration_seconds,
                    }
                )

        self._emit_progress(
            {
                "event": "run_completed",
                "num_ticks": self.num_ticks,
                "seed": self.seed,
                "provider_name": self.provider.name(),
                "scenario_name": type(self.scenario).__name__,
                "results_file": str(results_file),
            }
        )
        return results
