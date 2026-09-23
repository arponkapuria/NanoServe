import json
import sys
import time
from pathlib import Path

from nanoserve import settings
from nanoserve.utils import save_results

BLOCK_SIZE = settings.PAGED_KV_BLOCK_SIZE
POOL_BLOCKS = settings.PAGED_KV_NUM_BLOCKS


def blocks_needed(num_tokens: int) -> int:
    return -(-num_tokens // BLOCK_SIZE)


def main(result_path: str):
    data = json.loads(Path(result_path).read_text())
    per_request = data["per_request"]
    n = len(per_request)

    rows, total_naive, total_private = [], 0, 0
    shared_prefix_blocks = 0
    for r in per_request:
        naive = blocks_needed(r["prompt_len"] + r["num_generated_tokens"])
        cached = r.get("cached_blocks", 0)
        private = naive - cached
        total_naive += naive
        total_private += private
        shared_prefix_blocks = max(shared_prefix_blocks, cached)  # the shared chain depth, verified from real data
        rows.append({"request_id": r["request_id"], "naive_blocks": naive,
                      "cached_blocks": cached, "private_blocks": private})

    total_with_sharing = shared_prefix_blocks + total_private  # shared chain paid exactly once

    naive_avg = total_naive / n
    shared_private_avg = total_private / n
    naive_capacity = POOL_BLOCKS // naive_avg
    shared_capacity = (POOL_BLOCKS - shared_prefix_blocks) // shared_private_avg if shared_private_avg else float("inf")
    reduction = 1 - total_with_sharing / total_naive
    capacity_gain = shared_capacity / naive_capacity

    for row in rows:
        print(f"  req {row['request_id']}: naive={row['naive_blocks']} blocks  "
              f"cached={row['cached_blocks']}  private={row['private_blocks']}")
    print()
    print(f"Requests:                     {n}")
    print(f"Shared prefix (paid once):    {shared_prefix_blocks} blocks ({shared_prefix_blocks * BLOCK_SIZE} tokens)")
    print(f"Total blocks -- naive:        {total_naive}")
    print(f"Total blocks -- with radix:   {total_with_sharing}")
    print(f"Reduction:                    {reduction:.1%}")
    print()
    print(f"Pool size:                    {POOL_BLOCKS} blocks ({POOL_BLOCKS * BLOCK_SIZE} tokens)")
    print(f"Concurrent capacity, naive requests this size:      {naive_capacity:.0f}")
    print(f"Concurrent capacity, once this prefix is cached:    {shared_capacity:.0f}")
    print(f"Capacity gain:                                       {capacity_gain:.1f}x")

    record = {
        "step": "radix_capacity_savings",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "source_result_file": str(result_path),
        "device": data.get("device"), "model": data.get("model"), "config": data.get("config"),
        "block_size": BLOCK_SIZE, "pool_blocks": POOL_BLOCKS,
        "per_request": rows,
        "shared_prefix_blocks": shared_prefix_blocks,
        "total_naive_blocks": total_naive,
        "total_blocks_with_sharing": total_with_sharing,
        "reduction_pct": reduction,
        "naive_concurrent_capacity": naive_capacity,
        "shared_concurrent_capacity": shared_capacity,
        "capacity_gain_x": capacity_gain,
    }
    out_path = save_results(f"radix_capacity_savings_{Path(result_path).stem}", record, settings.RESULTS_DIR)
    print(f"\nSaved: {out_path}")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "results/radix_cache_continuous_batching.json")