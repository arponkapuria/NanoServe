<div align="center">

# 👾 NanoServe

[![Python](https://img.shields.io/badge/Python-3.12+-3776AB?logo=python&logoColor=white)](https://python.org) [![PyTorch](https://img.shields.io/badge/PyTorch-2.6+-EE4C2C?logo=pytorch&logoColor=white)](https://pytorch.org) [![Transformers](https://img.shields.io/badge/🤗%20Transformers-4.45+-yellow)](https://huggingface.co/docs/transformers) [![FastAPI](https://img.shields.io/badge/FastAPI-0.111+-009688?logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com) [![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**A from-scratch LLM inference engine, engineered incrementally, phase by phase — every core serving optimization implemented from scratch, rigorously benchmarked, and unified into a production-style serving system.**

[Motivation](#motivation) · [Development Phases](#development-phases) · [Project Structure](#project-structure) · [Getting Started](#getting-started)

</div>

---

## Motivation

Serving engines like vLLM, SGLang, and TensorRT-LLM are full of optimizations that look arbitrary from the outside — paged KV caches, continuous batching, speculative decoding, kernel fusion. Reading their source without having felt the problems these solve firsthand means memorizing a list of tricks instead of understanding a system.

**NanoServe** exists to fix that: each serving topic is implemented by hand, in isolation, and measured — so the next optimization's necessity is obvious from the numbers, not just asserted. Once every phase below has been built and benchmarked this way, they get unified into a single, cohesive inference engine — one product, assembled from parts whose purpose has actually been proven rather than copied.

## Development Phases

| # | Topic | Description | Status | Writeup |
|---|---|---|:---:|---|
| 1 | Naive serving | Single-request baseline with no cache — every decode step recomputes the whole sequence from scratch. Establishes the numbers everything else is measured against. | ✅ | [Blog](https://arponkapuria.github.io/blogs/posts/nanoserve-naive-decode-to-kv-caching/) |
| 2 | KV caching | Hand-rolled key/value cache so each decode step only computes the new token instead of the entire sequence; measures why decode becomes memory-bandwidth bound once redundant compute is gone. | ✅ | [Blog](https://arponkapuria.github.io/blogs/posts/nanoserve-naive-decode-to-kv-caching/) |
| 3 | Paged KV cache | - | `WIP` | — |
| 4 | Continuous batching | - | ⏳ | — |
| 5 | Scheduler | - | ⏳ | — |
| 6 | Radix Cache | - | ⏳ | — |
| 7 | Chunked Prefill | - | ⏳ | — |
| 8 | Quantization | - | ⏳ | — |
| 9 | Speculative Decoding | - | ⏳ | — |
| 10 | Load Testing | - | ⏳ | — |
| 11 | Comparison | - | ⏳ | — |



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
MicroServe/
├── src/microserve/
│   ├── config.py           # metrics and engine configs
│   ├── settings            # device detection, model/dtype config
│   ├── engine.py           # naive decode, kv cache
│   └── utils.py            # device-agnostic memory tracking
│
├── results/
│   ├── naive.json
│   ├── kv_cache.json
│   ├── plot_comparison.py
│   └── images/             # generated plots
│
├── benchmark.py            # benchmarking and compare phases
└── notes/                  # writeups explaining phases/results
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

There are different prompt configurations in `settings.py` and based on the selection prompt, `max_tokens` length changes. 

```bash

# Naive baseline
uv run python benchmark.py --preset medium  

# KV cache
uv run python benchmark.py --preset medium --use-kv-cache  
```

Metrics and plots are written to `results/`.

---

<div align="center">

**Built by [Arpon Kapuria](https://arponkapuria.github.io/)**

</div>