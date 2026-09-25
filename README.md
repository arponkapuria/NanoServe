<div align="center">

# 👾 NanoServe

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org) [![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.45+-yellow)](https://huggingface.co/docs/transformers) [![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A tiny LLM inference engine, build incrementally, phase by phase — every core serving optimization implemented from scratch, benchmarked, and unified into a production-style serving system.**

[Motivation](#motivation) · [Development Phases](#development-phases) · [Project Structure](#project-structure) · [Getting Started](#getting-started)

</div>

---

## Motivation

Serving engines like vLLM, SGLang, and TensorRT-LLM are full of optimizations that look arbitrary from the outside — paged KV caches, continuous batching, speculative decoding, kernel fusion. Reading their source without having felt the problems these solve firsthand means memorizing a list of tricks instead of understanding a system.

**NanoServe** exists to fix that: each serving topic is implemented by hand, in isolation, and measured — so the next optimization's necessity is obvious from the numbers, not just asserted. Once every phase below has been built and benchmarked this way, they get unified into a single, cohesive inference engine — one product, assembled from parts whose purpose has actually been proven rather than copied.

## Development Phases

| # | Topic | Description | Status | Writeup |
|---|---|---|:---:|---|
| 0 | Naive Serving | Single-request baseline with no cache — every decode step recomputes the whole sequence from scratch. Establishes the numbers everything else is measured against. | ✅ | [Blog](https://arponkapuria.github.io/blogs/posts/nanoserve-naive-decode-to-kv-caching/) |
| 1 | KV Caching | Prefill once, cache Keys/Values, feed only the newest token per decode step instead of recomputing the whole sequence. ~4x lower TPOT and ~3.5x higher throuput than naive on this hardware. | ✅ | [Blog](https://arponkapuria.github.io/blogs/posts/nanoserve-naive-decode-to-kv-caching/) |
| 2 | KV Cache Memory Management | PagedAttention (vLLM's technique) — block-based KV storage (fixed-size blocks + block table + free list) with gather-based attention standing in for vLLM's fused CUDA kernel, unavailable on MPS. Eliminates internal/external fragmentation at speed parity with plain KV cache. | ✅ | [Blog](https://arponkapuria.github.io/blogs/posts/nanoserve-paged-kv-cache/) |
| 3 | Continuous batching | Iteration-level scheduler that shares the accelerator across multiple concurrent requests instead of serving one at a time — admits/evicts requests every decode step via a shared, multi-tenant paged KV cache. +31.8% system throughput over sequential serving; 36 vs. 4 concurrent requests fit in the same memory budget compared to naive fixed-reservation. | ✅ | [Blog](https://arponkapuria.github.io/blogs/posts/nanoserve-continuous-batching/) |
| 4 | Scheduler | FCFS admission queue with fixed-depth backpressure and per-request timeouts on top of continuous batching's admit-when-free policy. Tested with a two-wave burst — with a scheduler-off control run reproducing continuous batching's unbounded fallback exactly. | ✅ | [Blog](https://arponkapuria.github.io/blogs/posts/nanoserve-scheduler/) |
| 5 | Prefix Caching | Block-aligned RadixAttention (SGLang's prefix-caching technique) built on the paged allocator — skips prefill (only) for tokens a prior request with the same prefix already computed, sharing physical KV blocks instead of duplicating them. 73.7% cache hit rate on a shared-prefix ramp test; -62.5% KV blocks used, 3.47x concurrent capacity (36 → 125 requests) vs. no sharing; TTFT -37.3% aggregate. | ✅ | [Blog](https://arponkapuria.github.io/blogs/posts/nanoserve-prefix-caching/) |
| 6 | Chunked Prefill | Splits a long prompt's prefill into fixed-size chunks (default 128 tokens) interleaved one-per-iteration with other requests' decode steps, decode-first — instead of one uninterrupted forward pass blocking the whole batch. On a 3-streaming-request + 1 long-prompt (433 tokens) scenario: worst decoder-side gap -49% (1,574ms → 801ms), at the cost of +116% TTFT for the long request itself and ~-5% throughput. Single-request path unaffected (flag scoped to continuous batching only). | ✅ | [Blog](https://arponkapuria.github.io/blogs/posts/nanoserve-chunked-prefill/) |


Each step is documented (blogs) as it's built, so the project doubles as a running record of what was done and why, not just a finished artifact.

## Project Structure

```
NanoServe/
├── src/nanoserve/
│   ├── config.py                       # EngineConfig feature flags + AggregateMetrics dataclass
│   ├── settings.py                     # device detection, model/dtype config, MAX_BATCH_SIZE, MAX_QUEUE_DEPTH, MAX_QUEUE_WAIT_MS, PREFILL_CHUNK_SIZE, benchmark request presets
│   ├── engine.py                       # naive decode, kv cache, paged_kv_cache, radix-aware prefill, chunked prefill (single-request + continuous-batch admission)
│   ├── naive_baseline.py               # naive reservation allocator + simulated contiguous arena, for fragmentation comparison
│   ├── paged_cache.py                  # PagedKVPool (block allocator), PagedKVCache (single-request), BatchedDecodeCache (multi-tenant) — both radix-cache aware (cached_blocks split, ref-count release)
│   ├── scheduler.py                    # Scheduler: FCFS queue, fixed-depth backpressure, fixed-wait timeout — pure state-transition class, no clock/model calls
│   ├── radix_cache.py                  # RadixCache/RadixNode: block-aligned prefix trie, refcounted, longest-prefix match + insert
│   └── utils.py                        # device-agnostic memory tracking
│
├── results/
│   ├── plot_comparison.py              # grouped-bar metric comparison across any two runs
│   ├── plot_fragmentation.py           # internal fragmentation, waste comparison
│   ├── plot_capacity.py                # concurrent-capacity comparison (paged vs naive, same memory budget)
│   ├── plot_occupancy.py               # batch occupancy over time, from continuous batching's step trace
│   ├── plot_scheduler_comparison.py    # scheduler on vs off, four core metrics
│   ├── plot_scheduler_timeline.py      # two-wave burst outcome timeline, data-derived grouping (no hardcoded IDs)
│   ├── plot_radix_conditions.py        # TTFT by cache-hit condition (miss/partial/full), single-request test
│   ├── plot_radix_ramp.py              # per-request TTFT, radix on vs off, continuous-batching ramp
│   ├── plot_radix_capacity.py          # concurrent-capacity gain from prefix sharing
│   ├── plot_chunked_prefill.py         # single-long-prompt overhead, decode-stall token-gap timeline, chunk-size sweep
│   └── images/                         # generated plots
│
├── benchmark.py                        # benchmarking and compare phases
├── continuous_batch_benchmark.py       # sequential vs. continuous-batched run, same requests, same pool
├── scheduler_benchmark.py              # two-wave burst test: scheduler on vs off, backpressure + timeout + recovery
├── test_fragmentation.py               # internal + external fragmentation tests, paged vs naive kv_cache
├── radix_benchmark.py                  # radix caching benchmarking - single-request (miss/partial/full hit conditions)
├── radix_ramp_benchmark.py             # continuous-batching ramp: radix cache on/off, --use-scheduler flag
├── radix_capacity_savings.py           # block/capacity-savings analysis from a saved ramp result
├── chunked_prefill_benchmark.py        # single-long-prompt overhead test + 3-request decode-stall test, --use-chunked-prefill / --chunk-size
└── notes/                              # writeups explaining phases/results
```

## Getting Started

### Prerequisites
- `Python 3.12+`
- [`uv`](https://docs.astral.sh/uv/getting-started/installation/)
- `CUDA GPU, Apple Silicon (MPS), or CPU`

### Install

```bash
git clone https://github.com/arponkapuria/NanoServe.git
cd NanoServe
uv sync
```

### Run a benchmark

Prompt presets (`short`, `medium`, `long`, `prefix_shared`) are defined in `settings.py`, each with its own `max_new_tokens` length.

**Naive Decode**
  
```bash
# Naive decode (baseline)
uv run python benchmark.py --preset medium  
```

**Key Value Caching**

```bash
# KV cache
uv run python benchmark.py --preset medium --use-kv-cache 
``` 

**KV Cache Memory Management**

```bash
# KV cache + PagedAttention
uv run python benchmark.py --preset medium --use-paged-kv

# Fragmentation tests (internal + external, paged vs naive)
uv run python test_fragmentation.py 
```

**Continuous Batching**

```bash
# Continuous batching (sequential vs. batched, same requests, same pool)
uv run python continuous_batch_benchmark.py
```

**Scheduler**

```bash
# Scheduler (two-wave burst: admits, backpressure, timeouts, clean recovery)
uv run scheduler_benchmark.py --wave full --use-scheduler
 
# Scheduler-off control, same requests (continuous batching's unbounded fallback)
uv run scheduler_benchmark.py --wave full
```

**Prefix Caching**

```bash
# Prefix Cache + RadixAttention — single-request (miss/partial/full hit conditions)
uv run python radix_benchmark.py

# Prefix Cache + RadixAttention — continuous-batching ramp, on vs off
uv run python radix_ramp_benchmark.py --use-radix-cache --use-scheduler
uv run python radix_ramp_benchmark.py --use-scheduler

# Block/capacity savings from a saved ramp result
uv run python radix_capacity_savings.py results/radix_ramp_radix_on_schedule_on.json
```

**Chunked Prefill**

```bash
# Chunked prefill — single long prompt alone (overhead + correctness)
uv run python chunked_prefill_benchmark.py --test single-long-prompt
uv run python chunked_prefill_benchmark.py --test single-long-prompt --use-chunked-prefill
 
# Chunked prefill — 3 streaming requests + a long prompt arriving mid-run (the stall test)
uv run python chunked_prefill_benchmark.py --test decode-stall
uv run python chunked_prefill_benchmark.py --test decode-stall --use-chunked-prefill [--chunk-size 64]
 
# Plots for both tests above
uv run python results/plot_chunked_prefill.py
```

Metrics and plots are written to `results/`.

---

<div align="center">

**Built by [Arpon Kapuria](https://arponkapuria.github.io/)**

</div>