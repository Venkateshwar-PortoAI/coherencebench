#!/usr/bin/env python3
"""Generate all paper figures from analyzed benchmark results.

Loads all analyzed_results.json files, aggregates metrics across seeds
(mean + std), and generates all visualizations.

Usage:
    python scripts/generate_paper_figures.py
    python scripts/generate_paper_figures.py --results-dir results --output-dir paper/figures
"""

import argparse
import json
import logging
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.visualizer import CoherenceVisualizer, FACTOR_NAMES

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger("generate_figures")


def load_all_results(results_dir: Path) -> dict[str, list[dict]]:
    """Load all analyzed_results.json grouped by config/provider.

    Returns:
        {f"{config_name}/{provider}": [result_dict, ...]}
    """
    groups = defaultdict(list)
    for result_file in results_dir.rglob("analyzed_results.json"):
        with open(result_file) as f:
            data = json.load(f)
        key = f"{data['config']}/{data['provider']}"
        groups[key].append(data)
    return dict(groups)


def aggregate_fc_curves(results: list[dict]) -> tuple[list[float], list[float]]:
    """Aggregate FC curves across seeds: returns (mean, std) per tick."""
    all_fc = []
    for r in results:
        fc = [t["fc"] for t in r["per_tick"]]
        all_fc.append(fc)

    # Align to shortest length
    min_len = min(len(fc) for fc in all_fc)
    trimmed = [fc[:min_len] for fc in all_fc]
    arr = np.array(trimmed)
    return list(arr.mean(axis=0)), list(arr.std(axis=0))


def aggregate_word_counts(results: list[dict]) -> list[dict[str, int]]:
    """Average word counts across seeds."""
    all_counts = []
    for r in results:
        counts = [t.get("word_counts", t.get("token_counts", {})) for t in r["per_tick"]]
        all_counts.append(counts)

    min_len = min(len(c) for c in all_counts)
    avg_counts = []
    for t in range(min_len):
        merged = {}
        for f in FACTOR_NAMES:
            vals = [all_counts[s][t].get(f, 0) for s in range(len(all_counts))]
            merged[f] = int(np.mean(vals))
        avg_counts.append(merged)
    return avg_counts


def compute_adr_by_phase(results: list[dict]) -> tuple[dict, dict]:
    """Compute per-factor ADR in early vs late phase, averaged across seeds."""
    early_adr = defaultdict(list)
    late_adr = defaultdict(list)

    for r in results:
        ticks = r["per_tick"]
        n = len(ticks)
        early_end = n * 2 // 5   # first 40%
        late_start = n * 3 // 5  # last 40%

        for t in ticks[:early_end]:
            for f in t["factors_substantive"]:
                early_adr[f].append(1.0)
            for f in FACTOR_NAMES:
                if f not in t["factors_substantive"]:
                    early_adr[f].append(0.0)

        for t in ticks[late_start:]:
            for f in t["factors_substantive"]:
                late_adr[f].append(1.0)
            for f in FACTOR_NAMES:
                if f not in t["factors_substantive"]:
                    late_adr[f].append(0.0)

    early_avg = {f: np.mean(early_adr[f]) if early_adr[f] else 0.0 for f in FACTOR_NAMES}
    late_avg = {f: np.mean(late_adr[f]) if late_adr[f] else 0.0 for f in FACTOR_NAMES}
    return early_avg, late_avg


def generate_figures(results_dir: Path, output_dir: Path):
    groups = load_all_results(results_dir)

    if not groups:
        logger.warning("No results found in %s. Nothing to generate.", results_dir)
        return

    viz = CoherenceVisualizer(output_dir=output_dir)
    logger.info("Loaded %d config/provider groups", len(groups))

    # --- Figure 1: FC degradation for baseline (hero image) ---
    baseline_groups = {k: v for k, v in groups.items() if "baseline" in k}
    for key, results in baseline_groups.items():
        provider = key.split("/")[-1]
        fc_mean, fc_std = aggregate_fc_curves(results)
        viz.plot_factor_coverage_over_time(
            fc_mean,
            title=f"Factor Coverage Degradation: {provider} (Baseline)",
            label=provider,
            save_as=f"fig1_fc_degradation_{provider}.png",
        )
        logger.info("  Generated FC degradation for %s", provider)

    # --- Figure 2: Per-factor attention heatmap ---
    for key, results in baseline_groups.items():
        provider = key.split("/")[-1]
        word_counts = aggregate_word_counts(results)
        viz.plot_per_factor_attention(
            word_counts,
            title=f"Per-Factor Word Allocation: {provider}",
            save_as=f"fig2_attention_heatmap_{provider}.png",
        )
        logger.info("  Generated attention heatmap for %s", provider)

    # --- Figure 3: Cross-model comparison ---
    cross_model_curves = {}
    for key, results in baseline_groups.items():
        provider = key.split("/")[-1]
        fc_mean, _ = aggregate_fc_curves(results)
        cross_model_curves[provider] = fc_mean

    if cross_model_curves:
        viz.plot_cross_model_comparison(
            cross_model_curves,
            save_as="fig3_cross_model_fc.png",
        )
        logger.info("  Generated cross-model comparison")

    # --- Figure 4: Intervention recovery ---
    intervention_groups = {k: v for k, v in groups.items() if "intervention" in k}
    for key, results in intervention_groups.items():
        provider = key.split("/")[-1]
        fc_mean, _ = aggregate_fc_curves(results)
        # Extract intervention ticks from first result's config
        intervention_ticks = [50, 100, 150]  # from config
        viz.plot_intervention_recovery(
            fc_mean,
            intervention_ticks=intervention_ticks,
            title=f"Intervention Recovery: {provider}",
            save_as=f"fig4_intervention_recovery_{provider}.png",
        )
        logger.info("  Generated intervention recovery for %s", provider)

    # --- Figure 5: ADR by phase (directional fixation) ---
    for key, results in baseline_groups.items():
        provider = key.split("/")[-1]
        early_adr, late_adr = compute_adr_by_phase(results)
        viz.plot_anomaly_detection_by_phase(
            early_adr,
            late_adr,
            title=f"ADR by Phase: {provider} (Directional Fixation)",
            save_as=f"fig5_adr_by_phase_{provider}.png",
        )
        logger.info("  Generated ADR by phase for %s", provider)

    # --- Figure 6: DA vs FC scatter ---
    for key, results in baseline_groups.items():
        provider = key.split("/")[-1]
        all_fc = []
        all_da = []
        for r in results:
            all_fc.extend(t["fc"] for t in r["per_tick"])
            all_da.extend(t["da"] for t in r["per_tick"])
        viz.plot_decision_accuracy_vs_coverage(
            all_fc, all_da,
            title=f"Decision Accuracy vs Coverage: {provider}",
            save_as=f"fig6_da_vs_fc_{provider}.png",
        )
        logger.info("  Generated DA vs FC scatter for %s", provider)

    logger.info("All figures saved to %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Generate paper figures from benchmark results")
    parser.add_argument(
        "--results-dir",
        type=Path,
        default=Path("results"),
        help="Directory containing benchmark results",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("paper/figures"),
        help="Directory to save figures",
    )
    args = parser.parse_args()
    generate_figures(args.results_dir, args.output_dir)


if __name__ == "__main__":
    main()
