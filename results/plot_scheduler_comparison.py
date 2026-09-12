"""
Plots Scheduler ON vs OFF across the four core metrics, read straight from
results/scheduler_on_full.json and results/scheduler_off_full.json.

Run from the project root:
    uv run plot_scheduler_comparison.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.spines.left": False,
    "axes.edgecolor": "#444444",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
})

COLOR_ON = "#2E6F95"
COLOR_OFF = "#B7BFC8"


def load(path: str) -> dict:
    return json.loads(Path(path).read_text())["metrics"]


def main():
    on = load("results/scheduler_on_full.json")
    off = load("results/scheduler_off_full.json")

    panels = [
        ("TTFT (mean)", on["ttft_mean"] * 1000, off["ttft_mean"] * 1000, "ms"),
        ("TPOT (mean)", on["tpot_mean"] * 1000, off["tpot_mean"] * 1000, "ms"),
        ("Throughput", on["throughput_tps"], off["throughput_tps"], "tok/s"),
        ("Peak Memory", on["peak_memory_mb"], off["peak_memory_mb"], "MB"),
    ]

    fig, axes = plt.subplots(1, 4, figsize=(14, 4.6), gridspec_kw={"wspace": 0.25})
    for ax, (title, v_on, v_off, unit) in zip(axes, panels):
        bars = ax.bar(
            ["Scheduler\nON", "Scheduler\nOFF"], [v_on, v_off],
            color=[COLOR_ON, COLOR_OFF], width=0.70, edgecolor="white", linewidth=0.8,
        )
        ax.set_title(f"{title}\n({unit})", fontsize=10, pad=-2)
        ax.set_ylim(0, max(v_on, v_off) * 1.15)
        ax.set_yticks([])
        ax.tick_params(axis="x", labelsize=10.5)
        for b, v in zip(bars, [v_on, v_off]):
            ax.text(b.get_x() + b.get_width() / 2, v + max(v_on, v_off) * 0.03,
                     f"{v:,.1f}", ha="center", fontsize=11, fontweight="medium")

    fig.suptitle("Scheduler ON vs OFF — Same 22-Request Burst", fontsize=14.5, y=1.1)
    plt.tight_layout()
    plt.savefig("results/images/scheduler-on-vs-off-plot.png", dpi=220, bbox_inches="tight")
    print("Saved: scheduler-on-vs-off-plot.png")


if __name__ == "__main__":
    main()
