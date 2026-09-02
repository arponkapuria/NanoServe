import argparse
import time
from dataclasses import asdict

from nanoserve import settings
from nanoserve.config import AggregateMetrics, EngineConfig
from nanoserve.engine import NanoServeEngine
from nanoserve.utils import save_results


def build_prompt(preset: str) -> str:
    p = settings.BENCHMARK_PROMPTS[preset]
    if isinstance(p, dict):  # prefix_shared -> single-request test uses first suffix
        return p["prefix"] + p["suffixes"][0]
    return p


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--preset", default="short", choices=list(settings.BENCHMARK_PROMPTS))
    parser.add_argument("--num-runs", type=int, default=3)
    parser.add_argument("--gen-len", type=int, default=None)
    parser.add_argument("--warmup-runs", type=int, default=1)
    parser.add_argument("--use-kv-cache", action="store_true")
    args = parser.parse_args()

    prompt = build_prompt(args.preset)
    gen_len = args.gen_len or settings.GEN_LEN_DEFAULTS[args.preset]

    config = EngineConfig(
        use_kv_cache=args.use_kv_cache, use_paged_kv=False, use_continuous_batching=False,
        use_scheduler=False, use_radix_cache=False, use_chunked_prefill=False,
    )

    engine = NanoServeEngine(config)

    for _ in range(args.warmup_runs):
        engine.generate(prompt, max_new_tokens=gen_len)

    runs = [engine.generate(prompt, max_new_tokens=gen_len) for _ in range(args.num_runs)]
    all_metrics = [asdict(r["metrics"]) for r in runs]

    def avg(key):
        return sum(x[key] for x in all_metrics) / len(all_metrics)

    m = AggregateMetrics(
        ttft_mean=avg("ttft_mean"),
        tpot_mean=avg("tpot_mean"),
        throughput_tps=avg("throughput_tps"),
        peak_memory_mb=avg("peak_memory_mb"),
    )
    result = runs[-1]  # generated_text/num_new_tokens for the record (deterministic under greedy)

    print(f"TTFT:       {m.ttft_mean * 1000:.1f} ms")
    print(f"TPOT:       {m.tpot_mean * 1000:.1f} ms")
    print(f"Throughput: {m.throughput_tps:.2f} tok/s")
    print(f"Peak mem:   {m.peak_memory_mb:.1f} MB")
    print(f"New tokens: {result['num_new_tokens']}")

    step_name = "kv_cache" if args.use_kv_cache else "naive"

    record = {
        "step": step_name, 
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(engine.device), 
        "model": settings.MODEL_NAME,
        "config": asdict(config), 
        "preset": args.preset, 
        "gen_len": gen_len,
        "num_runs": args.num_runs, 
        "warmup_runs": args.warmup_runs,
        "metrics": asdict(m),
        "per_run_metrics": all_metrics,
        "prompt": prompt,
        "generated_text": result["text"],
    }
    path = save_results(step_name, record, settings.RESULTS_DIR)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()