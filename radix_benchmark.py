import time
from dataclasses import asdict

from nanoserve import settings
from nanoserve.config import EngineConfig
from nanoserve.engine import NanoServeEngine
from nanoserve.utils import save_results

NUM_TIMING_RUNS = 5

MISS_TEMPLATES = [
    "You are assistant Alpha. The weather today is cold and rainy outside.",
    "You are assistant Bravo. My favorite hobby is playing chess on weekends.",
    "You are assistant Charlie. The stock market fluctuated wildly this month.",
    "You are assistant Delta. She planted tomatoes and basil in her garden.",
    "You are assistant Echo. The museum exhibit featured ancient Egyptian artifacts.",
]  # each diverges from token 0 -- no two of these can ever share a block with each
   # other or with the real prefix, so hit_rate is guaranteed 0 on every call


def _time_condition(engine, make_prompt, gen_len: int, n: int = NUM_TIMING_RUNS) -> dict:
    """Run `n` prompts under one cache condition, return averaged core-5 metrics +
    hit rate + forwarded-token count. `make_prompt(i)` must land in the same
    condition every call -- if any two calls in the same group accidentally share a
    full block, later calls will hit on earlier ones (the miss-condition bug fixed
    last round)."""
    ttfts, tpots, throughputs, peak_mems, hit_rates, forwarded = [], [], [], [], [], []
    for i in range(n):
        r = engine.generate(make_prompt(i), max_new_tokens=gen_len)
        m = r["metrics"]
        ttfts.append(m.ttft_mean)
        tpots.append(m.tpot_mean)
        throughputs.append(m.throughput_tps)
        peak_mems.append(m.peak_memory_mb)
        hit_rates.append(r["cache_hit_tokens"] / r["prompt_len"])
        forwarded.append(r["prompt_len"] - r["cache_hit_tokens"])
    return {
        "ttft_mean_ms": sum(ttfts) / n * 1000,
        "tpot_mean_ms": sum(tpots) / n * 1000,
        "throughput_tps_mean": sum(throughputs) / n,
        "peak_memory_mb_mean": sum(peak_mems) / n,
        "hit_rate_mean": sum(hit_rates) / n,
        "forwarded_tokens_mean": sum(forwarded) / n,
        "n": n,
    }


def main():
    prefix = settings.BENCHMARK_PROMPTS["prefix_shared"]["prefix"]
    suffixes = settings.BENCHMARK_PROMPTS["prefix_shared"]["suffixes"]
    gen_len = settings.GEN_LEN_DEFAULTS["prefix_shared"]

    config = EngineConfig(
        use_kv_cache=True, use_paged_kv=True, use_radix_cache=True,
        use_continuous_batching=False, use_scheduler=False, use_chunked_prefill=False,
    )
    engine = NanoServeEngine(config)

    engine.generate("Warm up the accelerator before timing.", max_new_tokens=4)  # absorb first-call MPS overhead; off-prefix so it doesn't pollute the tree below

    # --- Demo: 3 distinct suffixes sharing `prefix` (cross-request sharing), then a
    # verbatim repeat of request 0 (the full-match / max_m edge case). Single-shot
    # numbers here are illustrative of the tree growing, not a timing measurement. ---
    prompts = [prefix + s for s in suffixes] + [prefix + suffixes[0]]

    results = []
    for i, prompt in enumerate(prompts):
        r = engine.generate(prompt, max_new_tokens=gen_len)
        hit, total = r["cache_hit_tokens"], r["prompt_len"]
        print(f"[{i}] prompt_len={total:3d}  cache_hit={hit:3d} ({hit/total:.0%})  "
              f"TTFT={r['metrics'].ttft_mean*1000:.1f}ms  text={r['text'][:40]!r}")
        results.append({
            "prompt": prompt, "prompt_len": total, "cache_hit_tokens": hit,
            "ttft_ms": r["metrics"].ttft_mean * 1000, "tpot_ms": r["metrics"].tpot_mean * 1000,
            "throughput_tps": r["metrics"].throughput_tps, "peak_memory_mb": r["metrics"].peak_memory_mb,
            "text": r["text"],
        })  

    total_hit = sum(x["cache_hit_tokens"] for x in results)
    total_len = sum(x["prompt_len"] for x in results)
    print(f"\nAggregate cache_hit_rate: {total_hit/total_len:.1%}")
    print(f"Radix tree stats: {engine.radix_cache.stats()}")

    # --- Actual TTFT: average several runs per cache condition. Each miss-condition
    # call uses a DIFFERENT template (diverges from token 0), so it's guaranteed 0%
    # hit every time -- reusing one template with only a trailing digit varying was
    # the bug in the previous version (later calls hit on the first call's blocks). ---
    miss = _time_condition(
        engine, lambda i: MISS_TEMPLATES[i % len(MISS_TEMPLATES)] + f" What is the capital of planet number {i}?", gen_len)
    partial = _time_condition(
        engine, lambda i: prefix + f" What is the capital of country number {i}?", gen_len)
    full_prompt = prefix + suffixes[0]
    engine.generate(full_prompt, max_new_tokens=gen_len)  # prime it once, not counted
    full = _time_condition(engine, lambda i: full_prompt, gen_len)

    for name, c in [("Miss", miss), ("Partial", partial), ("Full", full)]:
        print(f"{name:8s} TTFT={c['ttft_mean_ms']:7.1f}ms  TPOT={c['tpot_mean_ms']:6.1f}ms  "
              f"throughput={c['throughput_tps_mean']:5.2f}tok/s  peak_mem={c['peak_memory_mb_mean']:7.1f}MB  "
              f"hit_rate={c['hit_rate_mean']:.0%}  forwarded={c['forwarded_tokens_mean']:.1f}tok  (n={c['n']})")

    record = {
        "step": "radix_cache_single_request",
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "device": str(engine.device), "model": settings.MODEL_NAME,
        "config": asdict(config), "gen_len": gen_len,
        "num_runs": NUM_TIMING_RUNS, "warmup_runs": 1,
        "per_request": results, "cache_hit_rate": total_hit / total_len,
        "tree_stats": engine.radix_cache.stats(),
        "timing_by_condition": {"miss": miss, "partial": partial, "full": full},
    }
    path = save_results("radix_cache_single_request", record, settings.RESULTS_DIR)
    print(f"Saved: {path}")


if __name__ == "__main__":
    main()