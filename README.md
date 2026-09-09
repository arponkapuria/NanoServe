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
| 3 | Continuous batching | - | ⏳ | — |
| 4 | Scheduler | - | ⏳ | — |
| 5 | Radix Cache | - | ⏳ | — |
| 6 | Chunked Prefill | - | ⏳ | — |
| 7 | Quantization | - | ⏳ | — |
| 8 | Speculative Decoding | - | ⏳ | — |
| 9 | Load Testing | - | ⏳ | — |
| 10 | Comparison | - | ⏳ | — |



<!-- | 3 | Continuous batching | Replaces one-request-at-a-time serving with iteration-level batching; measures the shift in TTFT, TPOT, and throughput under concurrent load. | `WIP` | — |
| 4 | Request scheduling | Adds a request queue with priorities, backpressure, timeouts, and cancellation so the server degrades predictably under overload instead of falling over. | ⏳ | — |
| 5 | Paged KV cache | Replaces contiguous KV allocation with block-based paged allocation; measures the fragmentation problem it fixes. | ⏳ | — |
| 6 | Quantization | Surveys int8 / int4 / fp8 approaches and implements one, measuring the resulting speed/memory/quality trade-off. | ⏳ | — |
| 7 | Speculative decoding | Implements draft-model speculative decoding and measures when — and whether — it actually speeds up generation at this model size. | ⏳ | — |
| 8 | Observability | Exposes a `/metrics` endpoint with TTFT, inter-token latency, throughput, accelerator utilization, KV cache usage, and queue time in real time. | ⏳ | — |
| 9 | Load testing at scale | Throws real concurrent traffic at the engine, finds the point where throughput stops scaling, and explains why using the observability data. | ⏳ | — |
| 10 | Comparison vs. vLLM / SGLang / TensorRT-LLM | Studies their scheduling, KV cache, quantization, and speculative decoding design choices; writes up how MicroServe's approach differs and why. | ⏳ | — |
| 11 | Accelerator path optimization | CUDA graphs, kernel fusion, custom attention kernels, and CPU/GPU sync tuning — on both CUDA and Apple Silicon. | ⏳ | — |
| 12 | Advanced memory strategies | CPU KV cache offloading and prefix caching, beyond the basic paged implementation from Phase 5. | ⏳ | — |
| 13 | Distributed serving | Multi-GPU serving, distributed inference, prefill/decode disaggregation, and request routing across multiple engine instances. | ⏳ | — |
| 14 | UI | A usable front end on top of the engine, once the core serving logic is stable enough to be worth presenting. | ⏳ | — |
| 15 | Unify into one product | Assembles every phase above into a single, cohesive inference engine — one product built from parts already proven in isolation. | ⏳ | — | -->

Each step is documented (blogs) as it's built, so the project doubles as a running record of what was done and why, not just a finished artifact.

## Project Structure

```
NanoServe/
├── src/nanoserve/
│   ├── config.py                   # EngineConfig feature flags + AggregateMetrics dataclass
│   ├── settings.py                 # device detection, model/dtype config, MAX_BATCH_SIZE, benchmark request presets
│   ├── engine.py                   # naive decode, kv cache, paged_kv_cache, run_continuous_batch (iteration-level scheduler)
│   ├── naive_baseline.py           # naive reservation allocator + simulated contiguous arena, for fragmentation comparison
│   ├── paged_cache.py              # PagedKVPool (block allocator), PagedKVCache (single-request), BatchedDecodeCache (multi-tenant, shared decode batch)
│   └── utils.py                    # device-agnostic memory tracking
│
├── results/
│   ├── naive.json
│   ├── kv_cache.json
│   ├── paged_kv.json
│   ├── paged_kv_fragmentation.json
│   ├── continuous_batching.json
│   ├── plot_comparison.py          # grouped-bar metric comparison across any two runs
│   ├── plot_fragmentation.py
│   ├── plot_capacity.py            # concurrent-capacity comparison (paged vs naive, same memory budget)
│   ├── plot_occupancy.py           # batch occupancy over time, from continuous batching's step trace
│   └── images/                     # generated plots
│
├── benchmark.py                    # benchmarking and compare phases
├── continuous_batch_bench.py       # sequential vs. continuous-batched run, same requests, same pool
├── test_fragmentation.py           # internal + external fragmentation tests, paged vs naive kv_cache
└── notes/                          # writeups explaining phases/results
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
uv run python continuous_batch_bench.py
```

Metrics and plots are written to `results/`.

---

<div align="center">

**Built by [Arpon Kapuria](https://arponkapuria.github.io/)**

</div>