"""
Plots the two-wave burst test as a horizontal timeline: one bar per admitted
request (admit -> finish), plus markers for every rejection, read straight
from results/scheduler_on_full.json — nothing hardcoded.

Run from the project root:
    uv run plot_scheduler_timeline.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

try:
    from nanoserve import settings
    MAX_QUEUE_WAIT_S = settings.MAX_QUEUE_WAIT_MS / 1000
except ImportError:
    MAX_QUEUE_WAIT_S = 2.5  # fallback if run outside the project

RESULTS_PATH = Path("results/scheduler_on_full.json")
OUT_PATH = Path("results/images/wave-admission-timeline-plot.png")

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 12,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#444444",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "grid.linestyle": "-",
})

PALETTE = {
    "early_finisher": "#F3AA3C",   # amber
    "wave1": "#2C9FE1",            # blue
    "wave1_queued": "#34D4AC",     # teal
    "wave2": "#5E2FBC",            # purple
    "queue_full": "#DE2626",       # red
    "timeout": "#B0790A",          # dark amber
    "line": "#8A8A8A",
}

LABELS = {
    "early_finisher": "Wave 1 — deliberately short (frees a slot fast)",
    "wave1": "Wave 1 — admitted immediately",
    "wave1_queued": "Wave 1 — queued, then admitted (real wait)",
    "wave2": "Wave 2 — admitted after full drain",
}


def classify(reqs: list[dict]) -> dict[int, str]:
    """Groups requests by data, not hardcoded IDs:
    - queue_time_s > 0            -> genuinely queued-then-admitted
    - lowest num_generated_tokens
      + admitted first + no wait  -> the deliberately short early finisher
    - split on the single biggest
      gap in admitted_at_s        -> wave 1 vs wave 2
    """
    by_admit = sorted(reqs, key=lambda r: r["admitted_at_s"])
    admit_times = [r["admitted_at_s"] for r in by_admit]
    gaps = np.diff(admit_times)
    split_idx = int(np.argmax(gaps)) + 1 if len(gaps) else len(by_admit)

    min_tokens = min(r["num_generated_tokens"] for r in reqs)
    groups = {}
    for i, r in enumerate(by_admit):
        rid = r["request_id"]
        if r["queue_time_s"] and r["queue_time_s"] > 0:
            groups[rid] = "wave1_queued"
        elif i < split_idx:
            groups[rid] = "early_finisher" if r["num_generated_tokens"] == min_tokens else "wave1"
        else:
            groups[rid] = "wave2"
    return groups


def main():
    data = json.loads(RESULTS_PATH.read_text())
    reqs = data["per_request"]
    rejected = data["rejected"]
    groups = classify(reqs)

    fig, ax = plt.subplots(figsize=(16, 8))

    seen = set()
    for r in sorted(reqs, key=lambda r: r["request_id"]):
        g = groups[r["request_id"]]
        label = LABELS[g] if g not in seen else None
        seen.add(g)
        ax.barh(
            r["request_id"], r["finish_time_s"] - r["admitted_at_s"],
            left=r["admitted_at_s"], height=0.75,
            color=PALETTE[g], edgecolor="white", linewidth=0.6, label=label,
        )

    max_rid = max(r["request_id"] for r in reqs)
    qf = [r for r in rejected if r["reason"] == "queue_full"]
    to = [r for r in rejected if r["reason"] == "timeout"]

    for i, r in enumerate(qf):
        y = max_rid + 2 + i * 0.65 
        ax.scatter(r["arrival_delay"], y, marker="x", color=PALETTE["queue_full"], s=70, zorder=5, linewidths=2)
    if qf:
        ax.text(qf[0]["arrival_delay"] + 0.65, max_rid + 2 + (len(qf) - 1) * 0.75 / 2,
                f"{len(qf)} rejected: queue_full\n(instant, on arrival)",
                color=PALETTE["queue_full"], fontsize=12, va="center", fontweight="medium")

    to_x = (to[0]["arrival_delay"] + MAX_QUEUE_WAIT_S) if to else None
    for i, r in enumerate(to):
        y = max_rid + 2 + len(qf) * 0.65 + 1 + i * 0.75
        ax.scatter(to_x, y, marker="x", color=PALETTE["timeout"], s=70, zorder=5, linewidths=2)
    if to:
        ax.text(to_x + 0.4, max_rid + 2 + len(qf) * 0.65 + 1 + (len(to) - 1) * 0.75 / 2,
                f"{len(to)} rejected: timeout\n(waited {MAX_QUEUE_WAIT_S:.1f}s)",
                color=PALETTE["timeout"], fontsize=12, va="center", fontweight="medium")

    wave1_finish = max(r["finish_time_s"] for r in reqs if groups[r["request_id"]] in ("wave1", "early_finisher", "wave1_queued"))
    wave2_arrivals = [r["admitted_at_s"] for r in reqs if groups[r["request_id"]] == "wave2"]
    ax.axvline(wave1_finish, color=PALETTE["line"], linestyle="--", linewidth=1.2)
    ax.text(wave1_finish + 0.3, max_rid + 1.3, f"Wave 1 fully\ndrained ({wave1_finish:.1f}s)", fontsize=10, color="#555555")

    if wave2_arrivals:
        w2_start = min(wave2_arrivals)
        ax.axvline(w2_start, color=PALETTE["line"], linestyle="--", linewidth=1.2)
        ax.text(w2_start + 0.3, max_rid + 1.3, f"Wave 2 arrives\n({w2_start:.1f}s)", fontsize=10, color="#555555")

    ax.set_xlabel("Time (s)", fontsize=12)
    ax.set_ylabel("Request ID", fontsize=12)
    ax.set_yticks(range(0, max_rid + 1))
    total_offered = data["num_requests_offered"]
    admitted = len(reqs)
    ax.set_title(
        f"Two-Wave Burst Test — Every Request's Outcome Over Time\n"
        f"({total_offered} offered: {admitted} admitted, {len(qf)} queue_full, {len(to)} timeout)",
        fontsize=13.5, pad=14,
    )
    ax.legend(loc="lower right", fontsize=10, framealpha=0.95, edgecolor="#cccccc")
    ax.set_ylim(-1.5, max_rid + 2 + max(len(qf), 1) * 0.45 + len(to) * 0.65 + 2)

    plt.tight_layout()
    plt.savefig(OUT_PATH, dpi=220, bbox_inches="tight")
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
