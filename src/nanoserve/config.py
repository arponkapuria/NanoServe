from dataclasses import dataclass


@dataclass
class EngineConfig:
    use_kv_cache: bool = True
    use_paged_kv: bool = True
    use_continuous_batching: bool = True
    use_scheduler: bool = True
    use_radix_cache: bool = True
    use_chunked_prefill: bool = True
    use_quantization: bool = False
    use_speculative_decoding: bool = False


@dataclass
class AggregateMetrics:
    ttft_mean: float
    tpot_mean: float
    throughput_tps: float
    peak_memory_mb: float
    saturation_rps: float | None = None
    mean_queue_time_ms: float | None = None
    cache_hit_rate: float | None = None
    quality_delta: float | None = None
    draft_acceptance_rate: float | None = None
    ttft_p99: float | None = None
    tpot_p99: float | None = None
    batch_occupancy_mean: float | None = None