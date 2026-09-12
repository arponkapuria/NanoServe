import argparse
import time
from dataclasses import asdict

from nanoserve import settings
from nanoserve.config import EngineConfig
from nanoserve.engine import NanoServeEngine
from nanoserve.utils import save_results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--use-scheduler", action="store_true",
                         help="Enable the scheduler (backpressure + timeout). Off = step 4 unbounded-queue behavior, for comparison.")
    parser.add_argument("--wave", choices=["wave1", "full"], default="full",
                         help="wave1: just the initial burst, used to measure real drain time before setting Wave 2's arrival delay. full: both waves.")
    args = parser.parse_args()

    requests = settings.SCHEDULER_BURST_WAVE1 if args.wave == "wave1" else settings.SCHEDULER_BURST_REQUESTS

    config = EngineConfig(
        use_kv_cache=True, use_paged_kv=True, use_continuous_batching=True,
        use_scheduler=args.use_scheduler,
        use_radix_cache=False, use_chunked_prefill=False,
    )
    engine = NanoServeEngine(config)

    # Warmup: throwaway single-request pass through both codepaths this step touches
    # (generate + run_continuous_batch), same reasoning as continuous_batch_benchmark.py's
    # warmup. Uses one request, not the burst list — warming up with the full burst
    # would trigger real (discarded) rejections and cost as much time as the measured run.
    engine.generate(requests[0]["prompt"], max_new_tokens=10)
    engine.run_continuous_batch([{**requests[0], "arrival_delay": 0.0}])

    result = engine.run_continuous_batch(requests)
    m = result["metrics"]

    print()
    print(f"Scheduler:             {'on' if args.use_scheduler else 'off'}")
    print(f"Requests offered:      {len(requests)}")
    print(f"Admitted + finished:   {result['num_requests']}")
    print(f"Rejected (queue_full): {m.rejected_queue_full}")
    print(f"Rejected (timeout):    {m.rejected_timeout}")
    print(f"Mean queue time:       {m.mean_queue_time_ms:.1f} ms")
    print(f"TTFT (mean/p99):       {m.ttft_mean*1000:.1f} / {(m.ttft_p99 or 0)*1000:.1f} ms")
    print(f"TPOT (mean/p99):       {m.tpot_mean*1000:.1f} / {(m.tpot_p99 or 0)*1000:.1f} ms")
    print(f"Throughput:            {m.throughput_tps:.2f} tok/s")
    print(f"Peak mem:              {m.peak_memory_mb:.1f} MB")
    print(f"Occupancy mean:        {m.batch_occupancy_mean:.2f}")
    print(f"Total wall:            {result['total_wall_s']:.2f}s")
    print()

    step_name = f"scheduler_{'on' if args.use_scheduler else 'off'}_{args.wave}"
    record = {
        "step": step_name,
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(engine.device), "model": settings.MODEL_NAME,
        "config": asdict(config),
        "num_requests_offered": len(requests),
        "metrics": asdict(m),
        "rejected": result["rejected"],
        "per_request": result["per_request"],
        "total_wall_s": result["total_wall_s"],
    }
    path = save_results(step_name, record, settings.RESULTS_DIR)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()