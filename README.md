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
| 0 | Naive serving | Single-request baseline with no cache — every decode step recomputes the whole sequence from scratch. Establishes the numbers everything else is measured against. | ✅ | [Blog](https://arponkapuria.github.io/blogs/posts/nanoserve-naive-decode-to-kv-caching/) |
| 1 | KV caching | Prefill once, cache Keys/Values, feed only the newest token per decode step instead of recomputing the whole sequence. ~4x lower TPOT and ~3.5x higher throuput than naive on this hardware. | ✅ | [Blog](https://arponkapuria.github.io/blogs/posts/nanoserve-naive-decode-to-kv-caching/) |
| 2 | Paged KV cache | Block-based KV storage (fixed-size blocks + block table + free list) with gather-based attention, replacing vLLM's fused CUDA kernel — unavailable on MPS. Eliminates internal/external fragmentation at speed parity with plain KV cache. | ✅ | [Blog](https://arponkapuria.github.io/blogs/posts/nanoserve-paged-kv-cache/) |
| 3 | Continuous batching | Iteration-level scheduler that shares the accelerator across multiple concurrent requests instead of serving one at a time — admits/evicts requests every decode step via a shared, multi-tenant paged KV cache. +31.8% system throughput over sequential serving; 36 vs. 4 concurrent requests fit in the same memory budget compared to naive fixed-reservation. | ✅ | [Blog](https://arponkapuria.github.io/blogs/posts/nanoserve-continuous-batching/) |
| 4 | Scheduler | FCFS admission queue with fixed-depth backpressure and per-request timeouts on top of continuous batching's admit-when-free policy. Tested with a two-wave burst — with a scheduler-off control run reproducing continuous batching's unbounded fallback exactly. | ✅ | [Blog](https://arponkapuria.github.io/blogs/posts/nanoserve-scheduler/) |
| 5 | Radix Cache | - | ⏳ | — |
| 6 | Chunked Prefill | - | ⏳ | — |
| 7 | Quantization | - | ⏳ | — |
| 8 | Speculative Decoding | - | ⏳ | — |
| 9 | Load Testing | - | ⏳ | — |
| 10 | Comparison | - | ⏳ | — |

Each step is documented (blogs) as it's built, so the project doubles as a running record of what was done and why, not just a finished artifact.

## Project Structure

```
NanoServe/
├── src/nanoserve/
│   ├── config.py                       # EngineConfig feature flags + AggregateMetrics dataclass
│   ├── settings.py                     # device detection, model/dtype config, MAX_BATCH_SIZE, MAX_QUEUE_DEPTH, MAX_QUEUE_WAIT_MS, benchmark request presets
│   ├── engine.py                       # naive decode, kv cache, paged_kv_cache, run_continuous_batch (iteration-level scheduler, scheduler-aware admission)
│   ├── scheduler.py                    # Scheduler: FCFS queue, fixed-depth backpressure, fixed-wait timeout — pure state-transition class, no clock/model calls
│   ├── naive_baseline.py               # naive reservation allocator + simulated contiguous arena, for fragmentation comparison
│   ├── paged_cache.py                  # PagedKVPool (block allocator), PagedKVCache (single-request), BatchedDecodeCache (multi-tenant, shared decode batch)
│   └── utils.py                        # device-agnostic memory tracking
│
├── results/
│   ├── naive.json
│   ├── kv_cache.json
│   ├── paged_kv.json
│   ├── paged_kv_fragmentation.json
│   ├── continuous_batching.json
│   ├── scheduler_on_wave1.json
│   ├── scheduler_on_full.json
│   ├── scheduler_off_full.json
│   ├── plot_comparison.py              # grouped-bar metric comparison across any two runs
│   ├── plot_fragmentation.py           # internal fragmentation, waste comparison
│   ├── plot_capacity.py                # concurrent-capacity comparison (paged vs naive, same memory budget)
│   ├── plot_occupancy.py               # batch occupancy over time, from continuous batching's step trace
│   ├── plot_scheduler_comparison.py    # scheduler on vs off, four core metrics
│   ├── plot_scheduler_timeline.py      # two-wave burst outcome timeline, data-derived grouping (no hardcoded IDs)
│   └── images/                         # generated plots
│
├── benchmark.py                        # benchmarking and compare phases
├── continuous_batch_benchmark.py       # sequential vs. continuous-batched run, same requests, same pool
├── scheduler_benchmark.py              # two-wave burst test: scheduler on vs off, backpressure + timeout + recovery
├── test_fragmentation.py               # internal + external fragmentation tests, paged vs naive kv_cache
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

```bash
# Naive decode (baseline)
uv run python benchmark.py --preset medium  
```

```bash
# KV cache
uv run python benchmark.py --preset medium --use-kv-cache 
``` 

```bash
# Paged KV cache
uv run python benchmark.py --preset medium --use-paged-kv

# Fragmentation tests (internal + external, paged vs naive)
uv run python test_fragmentation.py 
```

```bash
# Continuous batching (sequential vs. batched, same requests, same pool)
uv run python continuous_batch_benchmark.py
```

```bash
# Scheduler (two-wave burst: admits, backpressure, timeouts, clean recovery)
uv run scheduler_benchmark.py --wave full --use-scheduler
 
# Scheduler-off control, same requests (continuous batching's unbounded fallback)
uv run scheduler_benchmark.py --wave full
```

Metrics and plots are written to `results/`.

---

<div align="center">

**Built by [Arpon Kapuria](https://arponkapuria.github.io/)**

</div>