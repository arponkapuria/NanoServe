"""
Plots TTFT by cache-hit condition (miss/partial/full), from radix_cache_single_request.json's
timing_by_condition block.

Usage: uv run python results/plot_radix_conditions.py [path/to/radix_cache_single_request.json]
"""
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#444444", "figure.facecolor": "white", "axes.facecolor": "white",
})


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else "results/radix_cache_single_request.json"
    cond = json.loads(Path(path).read_text())["timing_by_condition"]

    keys = ["miss", "partial", "full"]
    values = [cond[k]["ttft_mean_ms"] for k in keys]
    hit_rates = [cond[k]["hit_rate_mean"] for k in keys]
    labels = [f"{k.title()}\n({hr:.0%} hit)" for k, hr in zip(keys, hit_rates)]
    colors = ["#B7BFC8", "#5FA8C9", "#2E6F95"]

    fig, ax = plt.subplots(figsize=(7, 5))
    bars = ax.bar(labels, values, color=colors, width=0.6, edgecolor="white", linewidth=0.8)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + max(values) * 0.02,
                 f"{v:.0f}ms", ha="center", fontsize=11, fontweight="medium")

    ax.set_ylabel(f"TTFT, averaged over n={cond['miss']['n']} (ms)")
    ax.set_ylim(0, max(values) * 1.25)
    ax.set_title("Radix Prefix Cache: TTFT by Cache-Hit Condition", fontsize=13.5)
    plt.tight_layout()

    out = Path("results/images/radix-ttft-by-condition.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()