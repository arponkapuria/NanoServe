"""
Plots the concurrent-capacity comparison: how many requests fit in the same
total KV pool budget, paged (block-based) vs naive (fixed-reservation).

Usage: python plot_capacity.py
Reads: results/continuous_batching.json ("concurrent_capacity" block)
Requires: pip install matplotlib
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent
INPUT_PATH = RESULTS_DIR / "continuous_batching.json"
OUTPUT_PATH = RESULTS_DIR / "images" / "concurrent-capacity-plot.png"


def load_capacity_data():
    with open(INPUT_PATH) as f:
        record = json.load(f)
    return record["concurrent_capacity"]


def plot_capacity(data: dict, out_path: Path) -> None:
    labels = ["Paged KV Cache", "Naive Reservation"]
    values = [data["paged_capacity"], data["naive_capacity"]]
    colors = ["#2563eb", "#94ddff"]

    fig, ax = plt.subplots(figsize=(7, 3.2))
    y_pos = [1, 0]

    ax.barh(y_pos, values, height=0.5, color=colors, zorder=3)
    for y, val in zip(y_pos, values):
        ax.text(val + max(values) * 0.02, y, f"{val} requests", ha="left", va="center",
                fontsize=10.5, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5)
    ax.set_xlabel(f"Requests that fit ({data['total_pool_tokens']}-token pool)", fontsize=10)
    ax.set_xlim(0, max(values) * 1.25)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, zorder=0)
    ax.tick_params(axis="y", length=0)

    fig.suptitle("Concurrent Capacity: Same Memory Budget", fontsize=13, fontweight="bold", y=1.05)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    data = load_capacity_data()
    plot_capacity(data, OUTPUT_PATH)