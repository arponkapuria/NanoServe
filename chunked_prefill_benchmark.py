"""Step 7 tests. One config per invocation, flag-driven like the other benchmarks; each
run saves its own results/chunked_prefill_<test>_<off|on_cN>.json for diffing.

  --test single-long-prompt   one long prompt alone: correctness + overhead
  --test decode-stall         3 streaming requests + a long prompt arriving mid-run

  uv run python chunked_prefill_benchmark.py --test single-long-prompt
  uv run python chunked_prefill_benchmark.py --test single-long-prompt --use-chunked-prefill
  uv run python chunked_prefill_benchmark.py --test decode-stall
  uv run python chunked_prefill_benchmark.py --test decode-stall --use-chunked-prefill [--chunk-size 64]
"""
import argparse
import json
import statistics
import time
from dataclasses import asdict

from nanoserve import settings
from nanoserve.config import EngineConfig
from nanoserve.engine import NanoServeEngine
from nanoserve.utils import percentile, save_results

LONG = settings.BENCHMARK_PROMPTS["long"]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", choices=["single-long-prompt", "decode-stall"], required=True)
    parser.add_argument("--use-chunked-prefill", action="store_true",
                         help="Enable chunked prefill. Off = whole-prompt prefill (steps 4-6 behavior), for comparison.")
    parser.add_argument("--chunk-size", type=int, default=settings.PREFILL_CHUNK_SIZE,
                         help="Tokens per prefill chunk (multiple of the KV block size). Ignored when the flag is off.")
    parser.add_argument("--num-runs", type=int, default=3, help="Timed runs for single-long-prompt (decode-stall is single-shot).")
    parser.add_argument("--stall-ms", type=float, default=500.0,
                         help="decode-stall: a decoder gap above this counts as a stall (fixed, same for every config).")
    args = parser.parse_args()

    # Radix stays OFF on purpose: with it on, the warmup would cache the long prompt and
    # later runs would skip most of the prefill this step is measuring.
    config = EngineConfig(
        use_kv_cache=True, use_paged_kv=True, use_continuous_batching=True,
        use_scheduler=False, use_radix_cache=False, use_chunked_prefill=args.use_chunked_prefill,
    )
    engine = NanoServeEngine(config)

    # Warmup: same convention as the other benchmarks, plus one throwaway long-prompt pass
    # so this mode's chunk shapes are warm before timing.
    engine.generate("Warm up the accelerator before timing.", max_new_tokens=4)
    engine.run_continuous_batch([{"prompt": LONG, "max_new_tokens": 2, "arrival_delay": 0.0}],
                                 chunk_size=args.chunk_size)

    mode = f"on_c{args.chunk_size}" if args.use_chunked_prefill else "off"
    step_name = f"chunked_prefill_{args.test.replace('-', '_')}_{mode}"
    record = {
        "step": step_name, "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(engine.device), "model": settings.MODEL_NAME, "config": asdict(config),
        "chunk_size": args.chunk_size if args.use_chunked_prefill else None,
    }

    print()
    print(f"Chunked prefill:     {'on (chunk=' + str(args.chunk_size) + ')' if args.use_chunked_prefill else 'off'}")
    if args.test == "single-long-prompt":
        record.update(run_single_long_prompt(engine, args))
    else:
        record.update(run_decode_stall(engine, args))

    path = save_results(step_name, record, settings.RESULTS_DIR)
    print(f"\nSaved: {path}")


def run_single_long_prompt(engine, args):
    req = [{"prompt": LONG, "max_new_tokens": 48, "arrival_delay": 0.0}]
    runs = [engine.run_continuous_batch(req, chunk_size=args.chunk_size) for _ in range(args.num_runs)]
    all_metrics = [asdict(r["metrics"]) for r in runs]

    def avg(key):
        return sum(m[key] for m in all_metrics) / len(all_metrics)

    last = runs[-1]
    pr = last["per_request"][0]
    chunk_ms = [round(t["duration_ms"], 1) for t in last["prefill_trace"]]
    print(f"Prompt length:       {pr['prompt_len']} tokens")
    print(f"Prefill chunks:      {len(chunk_ms)}  {chunk_ms} ms each")
    print(f"TTFT:                {avg('ttft_mean') * 1000:.1f} ms")
    print(f"TPOT:                {avg('tpot_mean') * 1000:.1f} ms")
    print(f"Peak mem:            {max(m['peak_memory_mb'] for m in all_metrics):.1f} MB")

    identical = None
    baseline = settings.RESULTS_DIR / "chunked_prefill_single_long_prompt_off.json"
    if args.use_chunked_prefill:
        if baseline.exists():
            base_text = json.loads(baseline.read_text())["generated_text"]
            identical = base_text == pr["generated_text"]
            print(f"Text == unchunked:   {identical}")
            if not identical:
                a, b = base_text, pr["generated_text"]
                i = next((k for k in range(min(len(a), len(b))) if a[k] != b[k]), min(len(a), len(b)))
                print(f"  first difference at char {i}:\n  unchunked: {a[max(0, i - 30):i + 40]!r}\n  chunked:   {b[max(0, i - 30):i + 40]!r}")
        else:
            print("Text == unchunked:   (run once without --use-chunked-prefill to enable this check)")
    return {
        "num_runs": args.num_runs, "prompt_len": pr["prompt_len"], "chunk_ms": chunk_ms,
        "metrics": {k: avg(k) for k in ("ttft_mean", "tpot_mean", "throughput_tps")} |
                   {"peak_memory_mb": max(m["peak_memory_mb"] for m in all_metrics)},
        "generated_text": pr["generated_text"], "text_identical_to_unchunked": identical,
        "prefill_trace": last["prefill_trace"],
    }


def run_decode_stall(engine, args):
    result = engine.run_continuous_batch(settings.DECODE_STALL_REQUESTS, chunk_size=args.chunk_size)
    m = result["metrics"]
    long_req = next(x for x in result["per_request"] if x["prompt"] == LONG)
    decoders = [x for x in result["per_request"] if x["prompt"] != LONG]
    gaps = [g * 1000 for d in decoders for g in d["token_times_s"]]  # ms, decoders only
    stalls = sum(1 for g in gaps if g > args.stall_ms)

    print(f"Long prompt length:  {long_req['prompt_len']} tokens, "
          f"{len(result['prefill_trace']) - len(decoders)} prefill chunk(s)")
    print(f"Long request TTFT:   {long_req['ttft_s'] * 1000:.1f} ms")
    print(f"Decoder gap mean/p99/max: {statistics.mean(gaps):.1f}/{percentile(gaps, 99):.1f}/{max(gaps):.1f} ms")
    print(f"Decoder gaps >{args.stall_ms:.0f}ms:  {stalls}")
    print(f"Throughput:          {m.throughput_tps:.2f} tok/s")
    print(f"Peak mem:            {m.peak_memory_mb:.1f} MB")
    print(f"Occupancy mean:      {m.batch_occupancy_mean:.2f}")
    print(f"Total wall:          {result['total_wall_s']:.2f}s")
    return {
        "stall_ms": args.stall_ms, "metrics": asdict(m),
        "decoder_gaps_ms": {"mean": statistics.mean(gaps), "p99": percentile(gaps, 99), "max": max(gaps), "stalls": stalls},
        "long_ttft_ms": long_req["ttft_s"] * 1000, "per_request": result["per_request"],
        "prefill_trace": result["prefill_trace"], "total_wall_s": result["total_wall_s"],
    }


if __name__ == "__main__":
    main()
