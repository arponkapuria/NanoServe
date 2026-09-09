import statistics
import time
from dataclasses import asdict

from nanoserve import settings, utils
from nanoserve.config import EngineConfig
from nanoserve.engine import NanoServeEngine


def run_sequential(engine, requests):
    utils.reset_memory_stats()
    utils.sync()
    start = time.perf_counter()
    ttfts, tpots, total_tokens = [], [], 0
    per_request = []
    for req in requests:
        result = engine.generate(req["prompt"], req["max_new_tokens"])
        ttfts.append(result["metrics"].ttft_mean)
        tpots.append(result["metrics"].tpot_mean)
        total_tokens += result["num_new_tokens"]
        per_request.append({
            "prompt": req["prompt"], "generated_text": result["text"],
            "num_generated_tokens": result["num_new_tokens"],
        })
    total_wall = time.perf_counter() - start
    return {
        "ttft_mean": statistics.mean(ttfts), "ttft_p99": utils.percentile(ttfts, 99),
        "tpot_mean": statistics.mean(tpots), "tpot_p99": utils.percentile(tpots, 99),
        "throughput_tps": total_tokens / total_wall,
        "peak_memory_mb": utils.peak_memory_mb(),
        "batch_occupancy_mean": 1.0,
        "total_wall_s": total_wall, "total_tokens": total_tokens,
        "per_request": per_request,
    }


def main():
    config = EngineConfig(
        use_kv_cache=True, use_paged_kv=True, use_continuous_batching=True,
        use_scheduler=False, use_radix_cache=False, use_chunked_prefill=False,
    )
    engine = NanoServeEngine(config)
    requests = settings.CONTINUOUS_BATCH_REQUESTS

    # Warmup: one throwaway pass through each path before measuring, 
    # same reasoning as benchmark.py's --warmup-runs (first MPS call per new shape is inflated).
    engine.generate(requests[0]["prompt"], max_new_tokens=10)
    engine.run_continuous_batch(requests)

    seq = run_sequential(engine, requests)
    batch_result = engine.run_continuous_batch(requests)
    bm = batch_result["metrics"]

    total_pool_tokens = settings.PAGED_KV_NUM_BLOCKS * settings.PAGED_KV_BLOCK_SIZE
    avg_tokens = statistics.mean([r["num_generated_tokens"] + r["prompt_len"] for r in batch_result["per_request"]])
    naive_capacity = total_pool_tokens // settings.NAIVE_MAX_RESERVED_TOKENS
    paged_capacity = int(total_pool_tokens // avg_tokens)

    print("\nSequential (step 3 path, one request at a time) -----------------------------------")
    print()
    print(f"TTFT mean/p99: {seq['ttft_mean']*1000:.1f}/{seq['ttft_p99']*1000:.1f} ms")
    print(f"TPOT mean/p99: {seq['tpot_mean']*1000:.1f}/{seq['tpot_p99']*1000:.1f} ms")
    print(f"Throughput:    {seq['throughput_tps']:.2f} tok/s")
    print(f"Peak mem:      {seq['peak_memory_mb']:.1f} MB")

    print("\nContinuous batching -----------------------------------")
    print()
    print(f"TTFT mean/p99: {bm.ttft_mean*1000:.1f}/{bm.ttft_p99*1000:.1f} ms")
    print(f"TPOT mean/p99: {bm.tpot_mean*1000:.1f}/{bm.tpot_p99*1000:.1f} ms")
    print(f"Throughput:    {bm.throughput_tps:.2f} tok/s")
    print(f"Peak mem:      {bm.peak_memory_mb:.1f} MB")
    print(f"Batch occupancy mean: {bm.batch_occupancy_mean:.2f}")

    print("\nConcurrent capacity (same pool: {} tokens total) --------------------------".format(total_pool_tokens))
    print()
    print(f"Naive fixed-reservation ({settings.NAIVE_MAX_RESERVED_TOKENS} tok/req): fits {naive_capacity} requests")
    print(f"Paged (avg {avg_tokens:.0f} tok/req actually used): fits ~{paged_capacity} requests")

    record = {
        "step": "continuous_batching",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(engine.device), "model": settings.MODEL_NAME,
        "config": asdict(config),
        "sequential": seq,
        "batched": {**asdict(bm), "num_requests": batch_result["num_requests"],
                    "total_tokens": batch_result["total_tokens"], "total_wall_s": batch_result["total_wall_s"],
                    "per_request": batch_result["per_request"], "step_trace": batch_result["step_trace"]},
        "concurrent_capacity": {"total_pool_tokens": total_pool_tokens, "naive_capacity": naive_capacity,
                                 "paged_capacity": paged_capacity, "avg_tokens_per_request": avg_tokens},
    }
    path = utils.save_results("continuous_batching", record, settings.RESULTS_DIR)

    print()
    durations = [s["step_duration_ms"] for batch_result["step_trace"] in [batch_result["step_trace"]] for s in batch_result["step_trace"]]
    slow_steps = [s for s in batch_result["step_trace"] if s["step_duration_ms"] > 2 * statistics.median(durations)]
    print(f"\nStep trace: {len(durations)} steps, median {statistics.median(durations):.1f}ms, "
          f"{len(slow_steps)} steps >2x median")
    for s in slow_steps[:10]:
        print(f"  step {s['step_index']}: {s['step_duration_ms']:.1f}ms, "
              f"blocks_needed={s['blocks_needed']}, active={s['num_active']}")

    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()