from pathlib import Path

import pytest

from scripts.run_benchmark import _format_run_summary, _looks_rate_limited
from scripts.run_benchmark import run_benchmark


def test_looks_rate_limited_detects_429_and_text():
    assert _looks_rate_limited("HTTP/1.1 429 Too Many Requests")
    assert _looks_rate_limited("RateLimitError: account is rate limited")
    assert not _looks_rate_limited("ValueError: malformed response")


def test_format_run_summary_includes_key_metrics():
    summary = {
        "mean_fc": 0.75,
        "mean_da": 0.5,
        "mean_adr": 0.6,
        "mean_fi": 0.2,
        "fc_drop_q1_to_q4": 0.18,
        "context_truncations": 4,
        "directional_validation": {"verdict": "fixation_detected"},
    }
    text = _format_run_summary(summary)
    assert "mean FC=0.750" in text
    assert "directional=fixation_detected" in text
    assert "trunc=4" in text


def test_model_override_requires_single_provider_selection():
    with pytest.raises(ValueError, match="exactly one provider"):
        run_benchmark(
            config_paths=[Path("configs/run_a_baseline.yaml")],
            max_seeds=1,
            dry_run=True,
            model_name="gpt-5",
        )
