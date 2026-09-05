# """
# Plots the internal fragmentation waste % comparison: Paged KV Cache vs Naive
# Reservation, grouped by sequence length (Short/Medium/Long).

# Usage: python plot_fragmentation.py
# Reads: results/paged_kv_fragmentation.json (from test_fragmentation.py)
# Requires: pip install matplotlib
# """
# import json
# from pathlib import Path

# import matplotlib
# matplotlib.use("Agg")
# import matplotlib.pyplot as plt

# RESULTS_DIR = Path(__file__).resolve().parent
# INPUT_PATH = RESULTS_DIR / "paged_kv_fragmentation.json"
# OUTPUT_PATH = RESULTS_DIR / "images" / "internal-fragmentation-waste-plot.png"

# PRESET_ORDER = ["short", "medium", "long"]
# PRESET_LABELS = {"short": "Short", "medium": "Medium", "long": "Long"}


# def load_waste_data():
#     with open(INPUT_PATH) as f:
#         record = json.load(f)
#     by_preset = {r["preset"]: r for r in record["internal_fragmentation"]}
#     return [by_preset[p] for p in PRESET_ORDER]


# def plot_waste_comparison(rows, out_path: Path) -> None:
#     labels = [PRESET_LABELS[r["preset"]] for r in rows]
#     paged = [r["paged_waste_pct"] for r in rows]
#     naive = [r["naive_waste_pct"] for r in rows]
#     tokens = [r["actual_tokens"] for r in rows]

#     n_groups = len(labels)
#     bar_width = 0.3
#     gap = 0.06
#     step = bar_width + gap
#     group_centers = list(range(n_groups))

#     fig, ax = plt.subplots(figsize=(8, 5.5))

#     paged_color = "#2563eb"
#     naive_color = "#94ddff"

#     paged_x = [c - step / 2 for c in group_centers]
#     naive_x = [c + step / 2 for c in group_centers]

#     ax.bar(paged_x, paged, width=bar_width, color=paged_color, label="Paged KV Cache", zorder=3)
#     ax.bar(naive_x, naive, width=bar_width, color=naive_color, label="Naive Reservation", zorder=3)

#     for x, val in zip(paged_x, paged):
#         ax.text(x, val + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")
#     for x, val in zip(naive_x, naive):
#         ax.text(x, val + 1.5, f"{val:.1f}%", ha="center", va="bottom", fontsize=9.5, fontweight="bold")

#     xtick_labels = [f"{label}\n({tok} tok)" for label, tok in zip(labels, tokens)]
#     ax.set_xticks(group_centers)
#     ax.set_xticklabels(xtick_labels, fontsize=10)
#     ax.set_ylabel("Memory Waste (%)", fontsize=10.5)
#     ax.set_ylim(0, 105)

#     ax.spines["top"].set_visible(False)
#     ax.spines["right"].set_visible(False)
#     ax.grid(True, axis="y", alpha=0.25, zorder=0)
#     ax.tick_params(axis="x", length=0)
#     ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False, fontsize=10)

#     fig.suptitle("Internal Fragmentation: Paged vs Naive Reservation", fontsize=13, fontweight="bold", y=1.06)
#     fig.tight_layout()
#     fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
#     plt.close(fig)
#     print(f"Saved: {out_path}")


# if __name__ == "__main__":
#     rows = load_waste_data()
#     plot_waste_comparison(rows, OUTPUT_PATH)


"""
Plots the internal fragmentation waste % comparison: Paged KV Cache vs Naive
Reservation, grouped by sequence length (Short/Medium/Long).

Usage: python plot_fragmentation.py
Reads: results/paged_kv_fragmentation.json (from test_fragmentation.py)
Requires: pip install matplotlib
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS_DIR = Path(__file__).resolve().parent
INPUT_PATH = RESULTS_DIR / "paged_kv_fragmentation.json"
OUTPUT_PATH = RESULTS_DIR / "images" / "internal-fragmentation-waste-plot.png"

PRESET_ORDER = ["short", "medium", "long"]
PRESET_LABELS = {"short": "Short", "medium": "Medium", "long": "Long"}


def load_waste_data():
    with open(INPUT_PATH) as f:
        record = json.load(f)
    by_preset = {r["preset"]: r for r in record["internal_fragmentation"]}
    return [by_preset[p] for p in PRESET_ORDER]


def plot_waste_comparison(rows, out_path: Path) -> None:
    labels = [PRESET_LABELS[r["preset"]] for r in rows]
    paged = [r["paged_waste_pct"] for r in rows]
    naive = [r["naive_waste_pct"] for r in rows]
    tokens = [r["actual_tokens"] for r in rows]

    n_groups = len(labels)
    bar_height = 0.3
    gap = 0.06
    step = bar_height + gap
    group_centers = list(range(n_groups))

    fig, ax = plt.subplots(figsize=(8, 5.5))

    paged_color = "#2563eb"
    naive_color = "#94ddff"

    paged_y = [c + step / 2 for c in group_centers]
    naive_y = [c - step / 2 for c in group_centers]

    ax.barh(paged_y, paged, height=bar_height, color=paged_color, label="Paged KV Cache", zorder=3)
    ax.barh(naive_y, naive, height=bar_height, color=naive_color, label="Naive Reservation", zorder=3)

    for y, val in zip(paged_y, paged):
        ax.text(val + 1.5, y, f"{val:.1f}%", ha="left", va="center", fontsize=9.5, fontweight="bold")
    for y, val in zip(naive_y, naive):
        ax.text(val + 1.5, y, f"{val:.1f}%", ha="left", va="center", fontsize=9.5, fontweight="bold")

    ytick_labels = [f"{label} ({tok} tok)" for label, tok in zip(labels, tokens)]
    ax.set_yticks(group_centers)
    ax.set_yticklabels(ytick_labels, fontsize=10)
    ax.invert_yaxis()  # Short on top, Long on bottom
    ax.set_xlabel("Memory Waste (%)", fontsize=10.5)
    ax.set_xlim(0, 105)

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(True, axis="x", alpha=0.25, zorder=0)
    ax.tick_params(axis="y", length=0)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 1.10), ncol=2, frameon=False, fontsize=10)

    fig.suptitle("Internal Fragmentation: Paged vs Naive Reservation", fontsize=13, fontweight="bold", y=1.06)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    rows = load_waste_data()
    plot_waste_comparison(rows, OUTPUT_PATH)