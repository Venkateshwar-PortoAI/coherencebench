"""Visualizations for CoherenceBench results.

Paper-quality plots (300 DPI) using matplotlib + seaborn.
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import seaborn as sns

logger = logging.getLogger(__name__)

# Paper-quality defaults
PAPER_RC = {
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "xtick.labelsize": 9,
    "ytick.labelsize": 9,
    "legend.fontsize": 9,
    "font.family": "serif",
    "text.usetex": False,
    "axes.grid": True,
    "grid.alpha": 0.3,
}

FACTOR_NAMES = ["load", "generation", "frequency", "voltage", "weather", "reserve"]

MODEL_COLORS = {
    "claude": "#6B4C9A",
    "gpt4o": "#2E86AB",
    "gemini": "#E8963E",
    "llama": "#44A57C",
}


def _apply_style():
    """Apply paper-quality matplotlib style."""
    sns.set_theme(style="whitegrid", rc=PAPER_RC)


class CoherenceVisualizer:
    """Generates paper-quality plots from CoherenceBench results."""

    def __init__(self, output_dir: Path = Path("paper/figures")):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        _apply_style()

    def plot_factor_coverage_over_time(
        self,
        fc_values: list[float],
        title: str = "Factor Coverage Degradation Over Time",
        label: str = "FC",
        intervention_ticks: Optional[list[int]] = None,
        save_as: str = "fc_degradation.png",
    ) -> Path:
        """Plot FC degradation curve -- the hero image of the paper.

        Args:
            fc_values: Factor coverage per tick (0.0 to 1.0).
            title: Plot title.
            label: Legend label.
            intervention_ticks: Optional tick indices where interventions occurred.
            save_as: Output filename.

        Returns:
            Path to the saved figure.
        """
        fig, ax = plt.subplots(figsize=(10, 4))
        ticks = np.arange(1, len(fc_values) + 1)

        # Smoothed line + raw scatter
        window = min(10, max(1, len(fc_values) // 20))
        if len(fc_values) >= window:
            smoothed = np.convolve(fc_values, np.ones(window) / window, mode="valid")
            smooth_x = ticks[window - 1:]
            ax.plot(smooth_x, smoothed, color="#6B4C9A", linewidth=2, label=f"{label} (smoothed)")
        ax.scatter(ticks, fc_values, alpha=0.15, s=8, color="#6B4C9A", zorder=1)

        # Mark interventions
        if intervention_ticks:
            for it in intervention_ticks:
                if 0 < it <= len(fc_values):
                    ax.axvline(x=it, color="#E74C3C", linestyle="--", alpha=0.7, linewidth=1)
            ax.axvline(x=0, color="#E74C3C", linestyle="--", alpha=0, label="Intervention")

        ax.set_xlabel("Tick")
        ax.set_ylabel("Factor Coverage (FC)")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc="lower left")

        path = self.output_dir / save_as
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

    def plot_per_factor_attention(
        self,
        word_counts_over_time: list[dict[str, int]],
        title: str = "Per-Factor Word Allocation Over Time",
        save_as: str = "factor_attention_heatmap.png",
    ) -> Path:
        """Heatmap showing per-factor word allocation across ticks.

        Args:
            word_counts_over_time: List of {factor_name: word_count} per tick.
            save_as: Output filename.

        Returns:
            Path to saved figure.
        """
        n_ticks = len(word_counts_over_time)
        factors = FACTOR_NAMES

        # Build matrix: rows=factors, cols=ticks
        matrix = np.zeros((len(factors), n_ticks))
        for t, counts in enumerate(word_counts_over_time):
            total = max(sum(counts.values()), 1)
            for f_idx, f_name in enumerate(factors):
                matrix[f_idx, t] = counts.get(f_name, 0) / total

        fig, ax = plt.subplots(figsize=(12, 3.5))
        sns.heatmap(
            matrix,
            ax=ax,
            cmap="YlOrRd",
            vmin=0,
            vmax=0.6,
            xticklabels=False,
            yticklabels=[f.title() for f in factors],
            cbar_kws={"label": "Fraction of tokens"},
        )

        # Add tick markers on x-axis at intervals
        step = max(1, n_ticks // 10)
        ax.set_xticks(range(0, n_ticks, step))
        ax.set_xticklabels(range(1, n_ticks + 1, step), rotation=0)

        ax.set_xlabel("Tick")
        ax.set_title(title)

        path = self.output_dir / save_as
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

    def plot_cross_model_comparison(
        self,
        model_fc_curves: dict[str, list[float]],
        title: str = "Factor Coverage: Cross-Model Comparison",
        save_as: str = "cross_model_fc.png",
    ) -> Path:
        """FC curves for all models on the same axes.

        Args:
            model_fc_curves: {model_name: [fc_per_tick]}.
            save_as: Output filename.

        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(10, 4.5))
        window = 10

        for model_name, fc_values in model_fc_curves.items():
            color = MODEL_COLORS.get(model_name, None)
            ticks = np.arange(1, len(fc_values) + 1)

            if len(fc_values) >= window:
                smoothed = np.convolve(fc_values, np.ones(window) / window, mode="valid")
                smooth_x = ticks[window - 1:]
                ax.plot(smooth_x, smoothed, linewidth=2, label=model_name, color=color)
            else:
                ax.plot(ticks, fc_values, linewidth=2, label=model_name, color=color)

        ax.set_xlabel("Tick")
        ax.set_ylabel("Factor Coverage (FC)")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.1)
        ax.legend(loc="lower left")

        path = self.output_dir / save_as
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

    def plot_intervention_recovery(
        self,
        fc_values: list[float],
        intervention_ticks: list[int],
        title: str = "Factor Coverage with Intervention Recovery",
        save_as: str = "intervention_recovery.png",
    ) -> Path:
        """FC curve with intervention points and recovery windows marked.

        Args:
            fc_values: Factor coverage per tick.
            intervention_ticks: 1-indexed tick numbers where interventions occurred.
            save_as: Output filename.

        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(10, 4.5))
        ticks = np.arange(1, len(fc_values) + 1)

        ax.plot(ticks, fc_values, color="#6B4C9A", linewidth=1.5, alpha=0.7, label="FC")

        # Mark each intervention with a vertical band showing recovery
        colors = ["#E74C3C", "#F39C12", "#27AE60"]
        for i, it in enumerate(intervention_ticks):
            if it < 1 or it > len(fc_values):
                continue
            color = colors[i % len(colors)]
            ax.axvline(x=it, color=color, linestyle="--", linewidth=2, alpha=0.8)

            # Find recovery window: count ticks after intervention where FC > pre-intervention
            idx = it - 1  # 0-indexed
            if idx > 0:
                pre_fc = fc_values[idx - 1]
                recovery_end = idx
                for j in range(idx, len(fc_values)):
                    if fc_values[j] > pre_fc:
                        recovery_end = j + 1
                    else:
                        break
                if recovery_end > idx:
                    ax.axvspan(it, recovery_end + 1, alpha=0.1, color=color)

            ax.annotate(
                f"Intervention {i + 1}",
                xy=(it, 1.02),
                fontsize=8,
                color=color,
                ha="center",
            )

        ax.set_xlabel("Tick")
        ax.set_ylabel("Factor Coverage (FC)")
        ax.set_title(title)
        ax.set_ylim(-0.05, 1.15)
        ax.legend(loc="lower left")

        path = self.output_dir / save_as
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

    def plot_anomaly_detection_by_phase(
        self,
        adr_per_factor_early: dict[str, float],
        adr_per_factor_late: dict[str, float],
        title: str = "Anomaly Detection Rate by Phase",
        save_as: str = "adr_by_phase.png",
    ) -> Path:
        """ADR per factor in early phase vs late phase.

        Proves directional fixation: ADR stays high for factors where anomalies
        were common early, but drops for factors where anomalies shifted to late.

        Args:
            adr_per_factor_early: {factor: ADR} for early phase (ticks 1-80).
            adr_per_factor_late: {factor: ADR} for late phase (ticks 121-200).
            save_as: Output filename.

        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(8, 5))

        factors = FACTOR_NAMES
        x = np.arange(len(factors))
        width = 0.35

        early_vals = [adr_per_factor_early.get(f, 0.0) for f in factors]
        late_vals = [adr_per_factor_late.get(f, 0.0) for f in factors]

        bars_early = ax.bar(x - width / 2, early_vals, width, label="Early Phase (1-80)",
                            color="#3498DB", alpha=0.85)
        bars_late = ax.bar(x + width / 2, late_vals, width, label="Late Phase (121-200)",
                           color="#E74C3C", alpha=0.85)

        ax.set_xlabel("Factor")
        ax.set_ylabel("Anomaly Detection Rate (ADR)")
        ax.set_title(title)
        ax.set_xticks(x)
        ax.set_xticklabels([f.title() for f in factors], rotation=30, ha="right")
        ax.set_ylim(0, 1.15)
        ax.legend()

        # Annotate the drop for late-phase factors
        for i, f in enumerate(factors):
            drop = early_vals[i] - late_vals[i]
            if drop > 0.1:
                ax.annotate(
                    f"{drop:+.0%}",
                    xy=(x[i] + width / 2, late_vals[i] + 0.02),
                    fontsize=8,
                    color="#C0392B",
                    ha="center",
                    fontweight="bold",
                )

        path = self.output_dir / save_as
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path

    def plot_decision_accuracy_vs_coverage(
        self,
        fc_values: list[float],
        da_values: list[float],
        title: str = "Decision Accuracy vs Factor Coverage",
        save_as: str = "da_vs_fc.png",
    ) -> Path:
        """Scatter plot: DA drops when FC drops.

        Args:
            fc_values: Factor coverage per tick.
            da_values: Decision accuracy per tick (0 or 1).
            save_as: Output filename.

        Returns:
            Path to saved figure.
        """
        fig, ax = plt.subplots(figsize=(6, 5))

        fc_arr = np.array(fc_values)
        da_arr = np.array(da_values)

        # Jitter DA slightly for visibility (it's 0 or 1)
        jitter = np.random.default_rng(42).uniform(-0.03, 0.03, len(da_arr))
        da_jittered = np.clip(da_arr + jitter, -0.05, 1.05)

        ax.scatter(fc_arr, da_jittered, alpha=0.25, s=15, color="#6B4C9A")

        # Bin FC into 5 bins and show mean DA per bin
        bins = np.linspace(0, 1, 6)
        bin_centers = []
        bin_means = []
        for i in range(len(bins) - 1):
            mask = (fc_arr >= bins[i]) & (fc_arr < bins[i + 1])
            if mask.sum() > 0:
                bin_centers.append((bins[i] + bins[i + 1]) / 2)
                bin_means.append(da_arr[mask].mean())

        ax.plot(bin_centers, bin_means, "o-", color="#E74C3C", linewidth=2,
                markersize=8, label="Mean DA per FC bin", zorder=5)

        ax.set_xlabel("Factor Coverage (FC)")
        ax.set_ylabel("Decision Accuracy (DA)")
        ax.set_title(title)
        ax.set_xlim(-0.05, 1.1)
        ax.set_ylim(-0.1, 1.15)
        ax.legend(loc="lower right")

        path = self.output_dir / save_as
        fig.tight_layout()
        fig.savefig(path)
        plt.close(fig)
        logger.info("Saved: %s", path)
        return path
