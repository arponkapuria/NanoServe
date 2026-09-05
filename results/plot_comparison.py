"""
Reusable comparison plotting utility for NanoServe results/*.json files.

Usage:
    python plot_comparison.py naive kv_cache
    python plot_comparison.py naive kv_cache --title "Naive vs KV Cache" --out naive_vs_kvcache.png

Reads results/<step>.json for each step name given, plots a grouped bar
chart per metric (each run currently collapses to a single averaged bar —
extend to per-prompt / per-repeat x-axis groups once that data exists).
Requires: pip install matplotlib
"""
import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent


def load_run(step_name: str) -> tuple[str, dict]:
    path = RESULTS_DIR / f"{step_name}.json"
    with open(path) as f:
        record = json.load(f)
    return step_name, record["metrics"]


def plot_comparison(
    runs: list[tuple[str, dict]],
    out_path: Path,
    title: str = "Phase Comparison",
) -> None:
    """
    Grouped bar chart comparing multiple labeled runs.
    Each run currently contributes one averaged bar per metric.

    Panels (gated — only included if any/all runs have nonzero values):
      - TTFT, TPOT, Throughput
      - Peak Memory
      - Queue Time (any run has it — zero-vs-nonzero is itself the point)
      - Cache Hit Rate / Quality Delta / Draft Acceptance Rate
    """
    metrics = [
        ("ttft_mean", "Time To First Token (ms)", 1000),
        ("tpot_mean", "Time Per Output Token (ms)", 1000),
        ("throughput_tps", "Throughput (tok/s)", 1),
        ("peak_memory_mb", "Peak Memory (MB)", 1),
    ]

    def has_any(key):
        return any((r.get(key) or 0) for _, r in runs)

    optional_metrics = [
        ("mean_queue_time_ms", "Queue Time (ms)", 1),
        ("cache_hit_rate", "Cache Hit Rate (%)", 1),
        ("quality_delta", "Quality Delta", 1),
        ("draft_acceptance_rate", "Draft Acceptance Rate (%)", 1),
    ]
    for key, label, scale in optional_metrics:
        if has_any(key):
            metrics.append((key, label, scale))

    n_metrics = len(metrics)
    cols = 4
    rows = (n_metrics + cols - 1) // cols

    fig, axes = plt.subplots(rows, cols, figsize=(11, 4.3 * rows), squeeze=False)
    axes = axes.flatten()

    run_labels = [label for label, _ in runs]
    n_runs = len(runs)
    colors = plt.cm.Set2(range(n_runs))

    bar_width = 0.34
    gap = 0.14
    # centered positions with a real gap between bars, e.g. for 2 runs: [-0.24, 0.24]
    step = bar_width + gap
    positions = [(i - (n_runs - 1) / 2) * step for i in range(n_runs)]

    for ax, (key, label, scale) in zip(axes, metrics):
        values = [(m.get(key) or 0) * scale for _, m in runs]
        ymax = max(values) if values else 1

        for pos, run_label, value, color in zip(positions, run_labels, values, colors):
            ax.bar(pos, value, width=bar_width, color=color, edgecolor="none", zorder=3)
            ax.text(
                pos, value + ymax * 0.035,
                f"{value:,.1f}", ha="center", va="bottom",
                fontsize=9, fontweight="bold", color="#222", zorder=4,
            )

        ax.set_xticks(positions)
        ax.set_xticklabels(run_labels, fontsize=9.5)
        ax.set_xlim(-step * n_runs / 2 - 0.15, step * n_runs / 2 + 0.15)
        ax.set_ylim(0, ymax * 1.22)

        ax.set_title(label, fontsize=10.5, fontweight="bold", pad=10)
        ax.get_yaxis().set_visible(False)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)
        ax.grid(True, axis="y", alpha=0.25, zorder=0)
        ax.tick_params(axis="x", length=0)

    for ax in axes[n_metrics:]:
        ax.axis("off")

    fig.suptitle(title, fontsize=14, fontweight="bold", y=1.0)
    fig.tight_layout(rect=[0, 0, 1, 0.96], w_pad=3.0, h_pad=3.0)
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)

    print(f"Saved comparison plot to {out_path}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("steps", nargs="+", help="Step names, e.g. naive kv_cache")
    parser.add_argument("--title", default="Phase Comparison")
    parser.add_argument("--out", default="comparison.png")
    args = parser.parse_args()

    runs = [load_run(step) for step in args.steps]
    plot_comparison(runs, RESULTS_DIR / "images" / args.out, title=args.title)


if __name__ == "__main__":
    main()