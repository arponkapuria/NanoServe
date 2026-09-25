"""
Plots the step 7 (chunked prefill) results, read from results/chunked_prefill_*.json.

  1. chunked-prefill-single-long-prompt.png     TTFT (split per prefill call), TPOT, peak memory
  2. chunked-prefill-decode-stall-timeline.png  the 3 streaming requests' token gaps over time
  3. chunked-prefill-sweep.png                  worst decoder gap + long-request TTFT by chunk size

Run from the project root:
    uv run python results/plot_chunked_prefill.py
"""
import json
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False, "axes.spines.left": False,
    "axes.edgecolor": "#444444", "figure.facecolor": "white", "axes.facecolor": "white",
})

COLOR_ON = "#2E6F95"
COLOR_OFF = "#B7BFC8"
COLOR_OFF_DARK = "#7A8794"
COLOR_WINDOW = "#F4D9A6"
CHUNK_SHADES = ["#2E6F95", "#4F8DB5", "#7FAFCF", "#A9CBE0"]

RESULTS = Path("results")
OUT = RESULTS / "images"


def load(name: str) -> dict | None:
    path = RESULTS / f"chunked_prefill_{name}.json"
    return json.loads(path.read_text()) if path.exists() else None


def analyze(d: dict) -> dict:
    """Rebuild each streaming request's token timeline from its saved per-token gaps."""
    long = max(d["per_request"], key=lambda r: r["prompt_len"])
    t_start = long["admitted_at_s"]              # long prompt admitted: prefill begins
    t_join = t_start + long["ttft_s"]            # its first token: it joins the decode batch
    series = []
    for r in d["per_request"]:
        if r is long:
            continue
        t, pts = r["admitted_at_s"] + r["ttft_s"], []
        for g in r["token_times_s"]:
            t += g
            pts.append((t, g * 1000))
        series.append(pts)
    after = [g for pts in series for t, g in pts if t > t_start]
    return {"series": series, "t_start": t_start, "t_join": t_join,
            "max_after": max(after), "long_ttft_ms": long["ttft_s"] * 1000}


def bar_labels(ax, bars, values, fmt="{:,.0f}"):
    top = max(values)
    for b, v in zip(bars, values):
        ax.text(b.get_x() + b.get_width() / 2, v + top * 0.03, fmt.format(v),
                ha="center", fontsize=11, fontweight="medium")


def plot_single_long_prompt(off: dict, on: dict):
    fig, axes = plt.subplots(1, 3, figsize=(13, 4.8), gridspec_kw={"wspace": 0.3})
    labels = ["Unchunked", f"Chunked\n({on['chunk_size']})"]

    ax = axes[0]  # TTFT, stacked by prefill call
    for x, d, colors in ((0, off, [COLOR_OFF]), (1, on, CHUNK_SHADES)):
        bottom = 0
        for i, ms in enumerate(d["chunk_ms"]):
            ax.bar(x, ms, bottom=bottom, width=0.7, color=colors[i % len(colors)],
                   edgecolor="white", linewidth=1.2)
            ax.text(x, bottom + ms / 2, f"{ms:,.0f}", ha="center", va="center", fontsize=9.5,
                    color="white" if x == 1 and i < 2 else "#222222")
            bottom += ms
        ttft = d["metrics"]["ttft_mean"] * 1000
        ax.text(x, bottom + 40, f"TTFT {ttft:,.0f}", ha="center", fontsize=11, fontweight="medium")
    ax.set_xticks([0, 1]); ax.set_xticklabels(labels)
    ax.set_ylim(0, max(sum(off["chunk_ms"]), sum(on["chunk_ms"])) * 1.15)
    ax.set_yticks([])
    ax.set_title("TTFT (ms)\nsplit by prefill call", fontsize=10, pad=-2)

    for ax, key, scale, title in ((axes[1], "tpot_mean", 1000, "TPOT (ms)"),
                                  (axes[2], "peak_memory_mb", 1, "Peak Memory (MB)")):
        vals = [off["metrics"][key] * scale, on["metrics"][key] * scale]
        bars = ax.bar(labels, vals, color=[COLOR_OFF, COLOR_ON], width=0.7, edgecolor="white")
        ax.set_ylim(0, max(vals) * 1.15); ax.set_yticks([])
        ax.set_title(title, fontsize=10, pad=-2)
        bar_labels(ax, bars, vals, "{:,.1f}")

    fig.suptitle(f"Single Long Prompt ({on['prompt_len']} tokens): Unchunked vs Chunked", fontsize=14.5, y=1.04)
    path = OUT / "chunked-prefill-single-long-prompt.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    print(f"Saved: {path.name}")
    plt.close(fig)


def plot_timeline(runs: list[tuple[str, dict]]):
    x_max = max(a["t_join"] for _, a in runs) + 3
    y_max = max(g for _, a in runs for pts in a["series"] for _, g in pts) * 1.15
    fig, axes = plt.subplots(len(runs), 1, figsize=(11, 2.2 * len(runs) + 1.2), sharex=True, sharey=True)

    for i, (ax, (label, a)) in enumerate(zip(axes, runs)):
        ax.axvspan(a["t_start"], a["t_join"], color=COLOR_WINDOW, alpha=0.6, lw=0,
                   label="long prompt being prefilled" if i == 0 else None)
        color = COLOR_OFF_DARK if label == "Unchunked" else COLOR_ON
        for pts in a["series"]:
            ts, gs = zip(*pts)
            ax.plot(ts, gs, "-o", ms=3, lw=0.8, color=color, alpha=0.55)
        worst = max(((t, g) for pts in a["series"] for t, g in pts if t > a["t_start"]), key=lambda p: p[1])
        ax.annotate(f"worst: {worst[1]:,.0f} ms", xy=worst, xytext=(worst[0] + 0.5, worst[1] * 0.92),
                    fontsize=10, color="#222222", va="center",
                    arrowprops=dict(arrowstyle="-", color="#888888", lw=0.8))
        ax.set_xlim(0, x_max); ax.set_ylim(0, y_max)
        ax.text(0.01, 0.86, label, transform=ax.transAxes, fontsize=11.5, fontweight="bold")
        ax.set_ylabel("gap (ms)", fontsize=10)
        ax.grid(axis="y", color="#E5E5E5", lw=0.6)

    axes[0].legend(frameon=False, loc="upper right", fontsize=10)
    axes[-1].set_xlabel("time since start (s)")
    fig.suptitle("Streaming Requests' Time Between Tokens While a Long Prompt Arrives", fontsize=14.5, y=1.0)
    plt.tight_layout()
    path = OUT / "chunked-prefill-decode-stall-timeline.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    print(f"Saved: {path.name}")
    plt.close(fig)


def plot_sweep(runs: list[tuple[str, dict]]):
    labels = [label.replace(" ", "\n", 1) for label, _ in runs]
    colors = [COLOR_OFF if label == "Unchunked" else COLOR_ON for label, _ in runs]
    panels = [("Worst decoder gap after the\nlong prompt arrives (ms)", [a["max_after"] for _, a in runs]),
              ("Long request TTFT (ms)", [a["long_ttft_ms"] for _, a in runs])]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4.8), gridspec_kw={"wspace": 0.2})
    for ax, (title, vals) in zip(axes, panels):
        bars = ax.bar(labels, vals, color=colors, width=0.7, edgecolor="white", linewidth=0.8)
        ax.set_ylim(0, max(vals) * 1.15); ax.set_yticks([])
        ax.set_title(title, fontsize=10.5, pad=-2)
        bar_labels(ax, bars, vals)
    fig.suptitle("Chunk Size Trade-off: Shorter Freezes vs Slower Long Request", fontsize=14.5, y=1.04)
    path = OUT / "chunked-prefill-sweep.png"
    plt.savefig(path, dpi=220, bbox_inches="tight")
    print(f"Saved: {path.name}")
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)

    off, on = load("single_long_prompt_off"), load("single_long_prompt_on_c128")
    if off and on:
        plot_single_long_prompt(off, on)
    else:
        print("Skipped single-long-prompt plot (need both single_long_prompt_off and _on_c128 JSONs)")

    configs = [("Unchunked", "decode_stall_off"), ("Chunk 64", "decode_stall_on_c64"),
               ("Chunk 128", "decode_stall_on_c128"), ("Chunk 256", "decode_stall_on_c256")]
    runs = [(label, analyze(d)) for label, name in configs if (d := load(name))]
    if len(runs) >= 2:
        plot_timeline(runs)
        plot_sweep(runs)
    else:
        print("Skipped decode-stall plots (need at least the unchunked JSON plus one chunked JSON)")


if __name__ == "__main__":
    main()
