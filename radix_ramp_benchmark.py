import argparse
import time
from dataclasses import asdict
import statistics

from nanoserve import settings
from nanoserve.config import EngineConfig
from nanoserve.engine import NanoServeEngine
from nanoserve.utils import save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-radix-cache", action="store_true",
                         help="Enable radix prefix caching. Off = step 4 behavior on this workload, for comparison.")
    parser.add_argument("--use-scheduler", action="store_true",
                         help="Enable the scheduler (backpressure + timeout) on top of continuous batching.")
    args = parser.parse_args()

    requests = settings.RADIX_RAMP_REQUESTS

    config = EngineConfig(
        use_kv_cache=True, use_paged_kv=True, use_continuous_batching=True,
        use_scheduler=args.use_scheduler, use_radix_cache=args.use_radix_cache, use_chunked_prefill=False,
    )
    engine = NanoServeEngine(config)

    engine.generate("Warm up the accelerator before timing.", max_new_tokens=4)
    engine.run_continuous_batch([{"prompt": "Also warm up the batched decode path.",
                                   "max_new_tokens": 4, "arrival_delay": 0.0}])

    result = engine.run_continuous_batch(requests)
    m = result["metrics"]

    print(f"Radix cache:         {'on' if args.use_radix_cache else 'off'}")
    print(f"Scheduler:           {'on' if args.use_scheduler else 'off'}")
    print(f"Requests offered:    {len(requests)}")
    print(f"Admitted + finished: {result['num_requests']}")
    print(f"Rejected (queue_full): {m.rejected_queue_full}")
    print(f"Rejected (timeout):    {m.rejected_timeout}")
    print(f"Mean queue time:     {m.mean_queue_time_ms:.1f} ms")
    print(f"TTFT mean/p99:       {m.ttft_mean*1000:.1f}/{(m.ttft_p99 or 0)*1000:.1f} ms")
    print(f"TPOT mean/p99:       {m.tpot_mean*1000:.1f}/{(m.tpot_p99 or 0)*1000:.1f} ms")
    print(f"Throughput:          {m.throughput_tps:.2f} tok/s")
    print(f"Peak mem:            {m.peak_memory_mb:.1f} MB")
    print(f"Occupancy mean:      {m.batch_occupancy_mean:.2f}")
    if args.use_radix_cache:
        print(f"Cache hit rate:      {(m.cache_hit_rate or 0):.1%}")
        print(f"Radix tree stats:    {engine.radix_cache.stats()}")
    print()
    for r in sorted(result["per_request"], key=lambda x: x["request_id"]):
        print(f"  req {r['request_id']}: prompt_len={r['prompt_len']:3d}  "
              f"cache_hit={r.get('cache_hit_tokens', 0):3d}  cached_blocks={r.get('cached_blocks', 0)}  "
              f"ttft={r['ttft_s']*1000:.1f}ms  text={r['generated_text'][:35]!r}")

    durations = [s["step_duration_ms"] for s in result["step_trace"]]
    slow_steps = [s for s in result["step_trace"] if s["step_duration_ms"] > 2 * statistics.median(durations)]
    print(f"\nStep trace: {len(durations)} steps, median {statistics.median(durations):.1f}ms, "
          f"{len(slow_steps)} steps >2x median")

    step_name = f"radix_ramp_{'radix_on' if args.use_radix_cache else 'radix_off'}_{'schedule_on' if args.use_scheduler else 'schedule_off'}"
    record = {
        "step": step_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(engine.device), "model": settings.MODEL_NAME,
        "config": asdict(config),
        "num_requests_offered": len(requests),
        "metrics": asdict(m),
        "per_request": result["per_request"],
        "step_trace": result["step_trace"],
        "rejected": result["rejected"],
        "tree_stats": engine.radix_cache.stats() if args.use_radix_cache else None,
        "total_wall_s": result["total_wall_s"],
    }
    path = save_results(step_name, record, settings.RESULTS_DIR)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()