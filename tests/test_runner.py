from pathlib import Path

from src.providers.base import LLMProvider
from src.runner import BenchmarkRunner


class DummyProvider(LLMProvider):
    def name(self) -> str:
        return "Dummy"

    def send_turn(self, system_prompt: str, messages: list[dict], user_message: str) -> str:
        return (
            "ANALYSIS:\n"
            "- Load: Detailed analysis.\n"
            "- Generation: Detailed analysis.\n"
            "- Frequency: Detailed analysis.\n"
            "- Voltage: Detailed analysis.\n"
            "- Weather: Detailed analysis.\n"
            "- Reserve: Detailed analysis.\n"
            "ACTION: hold_steady\n"
            "REASON: Synthetic test response."
        )

    def reset(self) -> None:
        pass


def test_runner_emits_progress_events(tmp_path: Path):
    events = []
    runner = BenchmarkRunner(
        provider=DummyProvider(),
        num_ticks=3,
        seed=42,
        output_dir=tmp_path,
        context_reset_interval=2,
        intervention_ticks=[2],
        scenario="power_grid",
        progress_callback=events.append,
    )

    results = runner.run()

    assert len(results) == 3

    event_names = [event["event"] for event in events]
    assert event_names == [
        "run_started",
        "tick_completed",
        "intervention",
        "tick_completed",
        "context_reset",
        "tick_completed",
        "run_completed",
    ]

    assert events[0]["results_file"] == str(tmp_path / "raw_results.jsonl")
    assert events[2]["tick_number"] == 2
    assert events[4]["tick_number"] == 3
    assert events[4]["restored_from_tick"] == 2

    tick_two = events[3]
    assert tick_two["event"] == "tick_completed"
    assert tick_two["intervention"] is True
    assert tick_two["context_reset"] is False
    assert tick_two["duration_seconds"] >= 0.0

    tick_three = events[5]
    assert tick_three["event"] == "tick_completed"
    assert tick_three["intervention"] is False
    assert tick_three["context_reset"] is True
    assert tick_three["duration_seconds"] >= 0.0
