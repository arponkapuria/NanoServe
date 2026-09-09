"""
Plots batch occupancy (active concurrent requests) over the course of the
continuous-batching run, showing admission, backfill, and tail-off dynamics.

Usage: python plot_occupancy.py
Reads: results/continuous_batching.json ("batched.step_trace")
Requires: pip install matplotlib
"""
import json
import itertools
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent
INPUT_PATH = RESULTS_DIR / "continuous_batching.json"
OUTPUT_PATH = RESULTS_DIR / "images" / "batch-occupancy-plot.png"


def load_trace():
    with open(INPUT_PATH) as f:
        record = json.load(f)
    trace = record["batched"]["step_trace"]
    occupancy_mean = record["batched"]["batch_occupancy_mean"]
    max_batch_size = record["config"].get("max_batch_size")  # may not be in config; falls back below
    return trace, occupancy_mean, max_batch_size


def plot_occupancy(trace: list[dict], occupancy_mean: float, max_batch_size: int | None, out_path: Path) -> None:
    # elapsed wall time per step, from cumulative step durations
    elapsed_s = list(itertools.accumulate(s["step_duration_ms"] for s in trace))
    elapsed_s = [t / 1000 for t in elapsed_s]
    active = [s["num_active"] for s in trace]

    fig, ax = plt.subplots(figsize=(9, 4.2))

    ax.step(elapsed_s, active, where="post", color="#eb8b25", linewidth=1.8, zorder=3)
    ax.fill_between(elapsed_s, active, step="post", color="#eb8b25", alpha=0.12, zorder=2)

    ax.axhline(occupancy_mean, color="#e11d48", linestyle="--", linewidth=1.3, zorder=4)
    ax.text(elapsed_s[-1] * 0.99, occupancy_mean + 0.15, f"mean {occupancy_mean:.2f}",
            ha="right", va="bottom", fontsize=9.5, color="#e11d48", fontweight="bold")

    if max_batch_size:
        ax.axhline(max_batch_size, color="#94a3b8", linestyle=":", linewidth=1.2, zorder=1)
        ax.text(elapsed_s[-1] * 0.99, max_batch_size + 0.15, f"cap {max_batch_size}",
                ha="right", va="bottom", fontsize=9, color="#64748b")

    ax.set_xlabel("Elapsed time (s)", fontsize=10.5)
    ax.set_ylabel("Active requests", fontsize=10.5)
    ax.set_ylim(0, (max_batch_size or max(active)) + 1)
    ax.set_xlim(0, elapsed_s[-1])

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="y", alpha=0.25, zorder=0)

    fig.suptitle("Batch Occupancy Over Time", fontsize=13, fontweight="bold", y=1.0)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    trace, occupancy_mean, max_batch_size = load_trace()
    plot_occupancy(trace, occupancy_mean, max_batch_size, OUTPUT_PATH)