"""
Plots per-request TTFT, radix cache ON vs OFF, on the same continuous-batching ramp
workload. Aligns rows by country name (parsed from the prompt) rather than
request_id, since the scheduler may reject/timeout a different request in each run.

Usage: uv run python results/plot_radix_ramp.py [on.json] [off.json]
"""
import json
import re
import sys
from pathlib import Path

import matplotlib.pyplot as plt

plt.rcParams.update({
    "font.family": "sans-serif", "font.size": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#444444", "figure.facecolor": "white", "axes.facecolor": "white",
})


def country_of(prompt: str) -> str:
    m = re.search(r"capital of (\w+)\?", prompt)
    return m.group(1) if m else prompt[-15:]


def main():
    on_path = sys.argv[1] if len(sys.argv) > 1 else "results/radix_ramp_radix_on_schedule_on.json"
    off_path = sys.argv[2] if len(sys.argv) > 2 else "results/radix_ramp_radix_off_schedule_on.json"
    on = json.loads(Path(on_path).read_text())
    off = json.loads(Path(off_path).read_text())

    on_by_c = {country_of(r["prompt"]): r for r in on["per_request"]}
    off_by_c = {country_of(r["prompt"]): r for r in off["per_request"]}
    # Only countries admitted in BOTH runs -- one rejected-by-timeout in either run
    # has nothing to compare against and is dropped, not zero-filled.
    countries = sorted(set(on_by_c) & set(off_by_c),
                        key=lambda c: off_by_c[c].get("admitted_at_s", 0))

    ttft_on = [on_by_c[c]["ttft_s"] * 1000 for c in countries]
    ttft_off = [off_by_c[c]["ttft_s"] * 1000 for c in countries]
    hit = [on_by_c[c].get("cache_hit_tokens", 0) for c in countries]

    x = range(len(countries))
    width = 0.38
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar([i - width / 2 for i in x], ttft_off, width, label="Radix OFF", color="#B7BFC8")
    ax.bar([i + width / 2 for i in x], ttft_on, width, label="Radix ON", color="#2E6F95")
    ax.set_xticks(list(x))
    ax.set_xticklabels(countries, rotation=30, ha="right")
    ax.set_ylabel("TTFT (ms)")
    ax.set_title("Per-Request TTFT: Radix Cache ON vs OFF\n(same continuous-batching ramp, scheduler on)", fontsize=13)
    ax.legend(frameon=False)
    for i in x:
        tag = f"hit {hit[i]}tok" if hit[i] else "miss"
        ax.text(i, max(ttft_off[i], ttft_on[i]) + 8, tag, ha="center", fontsize=8, color="#555")
    plt.tight_layout()

    out = Path("results/images/radix-ttft-per-request.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out, dpi=220, bbox_inches="tight")
    print(f"Saved: {out}")


if __name__ == "__main__":
    main()