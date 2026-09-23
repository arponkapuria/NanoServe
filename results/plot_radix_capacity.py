"""
Plots concurrent-capacity gain from radix prefix sharing, from a
radix_capacity_savings_*.json file.

Usage: uv run python results/plot_radix_capacity.py [path/to/radix_capacity_savings_*.json]
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/radix_capacity_savings_radix_ramp_radix_on_schedule_on.json"
    data = json.loads(Path(path).read_text())

    labels = ["With Radix\nSharing", "Naive\n(no sharing)"]
    values = [data["shared_concurrent_capacity"], data["naive_concurrent_capacity"]]
    colors = ["#2563eb", "#94ddff"]

    fig, ax = plt.subplots(figsize=(7, 3.4))
    y_pos = [1, 0]
    ax.barh(y_pos, values, height=0.5, color=colors, zorder=3)
    for y, val in zip(y_pos, values):
        ax.text(val + max(values) * 0.02, y, f"{val:.0f} requests", ha="left", va="center",
                fontsize=10.5, fontweight="bold")

    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels, fontsize=10.5)
    pool_tokens = data["pool_blocks"] * data["block_size"]
    ax.set_xlabel(f"Requests that fit ({pool_tokens}-token pool)", fontsize=10)
    ax.set_xlim(0, max(values) * 1.25)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, zorder=0)
    ax.tick_params(axis="y", length=0)

    fig.suptitle(f"Concurrent Capacity: Radix Sharing "
                 f"({data['reduction_pct']:.0%} fewer blocks, {data['capacity_gain_x']:.1f}x capacity)",
                 fontsize=12, fontweight="bold", y=1.08)
    fig.tight_layout()

    out = Path("results/images/radix-concurrent-capacity.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=180, bbox_inches="tight", facecolor="white")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()