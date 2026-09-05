"""
Fragmentation tests: paged KV cache vs. naive contiguous allocation,
on the two waste modes paging is designed to fix.

Usage: uv run python test_fragmentation.py
"""
import time

from nanoserve import settings
from nanoserve.config import EngineConfig
from nanoserve.engine import NanoServeEngine
from nanoserve.naive_baseline import ContiguousArena, measure_reservation_waste_mb
from nanoserve.paged_cache import PagedKVPool
from nanoserve.utils import save_results


def run_internal_fragmentation_test(engine) -> list[dict]:
    print("\n=== Internal Fragmentation: Naive Reservation vs. Paged KV Cache ===")
    results = []
    for preset in ["short", "medium", "long"]:
        prompt_data = settings.BENCHMARK_PROMPTS[preset]
        prompt = prompt_data if isinstance(prompt_data, str) else prompt_data["prefix"] + prompt_data["suffixes"][0]
        result = engine.generate(prompt, max_new_tokens=settings.GEN_LEN_DEFAULTS[preset])
        actual_tokens = result["total_tokens"]

        block_size = settings.PAGED_KV_BLOCK_SIZE
        blocks_held = -(-actual_tokens // block_size)  # ceil division
        paged_waste_pct = 100 * (blocks_held * block_size - actual_tokens) / (blocks_held * block_size)

        mc = engine.model.config
        head_dim = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)
        _, naive_waste_pct = measure_reservation_waste_mb(
            num_layers=mc.num_hidden_layers, num_kv_heads=mc.num_key_value_heads,
            head_dim=head_dim, max_reserved_tokens=settings.NAIVE_MAX_RESERVED_TOKENS,
            actual_tokens=actual_tokens, device=engine.device,
        )

        print(f"{preset:8s} actual={actual_tokens:4d} tok | paged waste={paged_waste_pct:5.1f}% | naive waste={naive_waste_pct:5.1f}%")
        results.append({"preset": preset, "actual_tokens": actual_tokens,
                         "paged_waste_pct": round(paged_waste_pct, 2), "naive_waste_pct": round(naive_waste_pct, 2)})
    return results


def run_external_fragmentation_test() -> dict:
    print("\n=== External Fragmentation: Contiguous Arena vs. Paged Block Pool ===")
    # Adversarial pattern: alloc A(4), alloc B(3), free A, alloc C(6).
    # Total free after freeing A is 7 (>= 6 needed) but the 4 A freed and the
    # 3 never-touched units aren't adjacent (B sits between them) — a
    # contiguous allocator can't merge them into one usable span.
    total_units = 10

    pool = PagedKVPool(num_layers=1, num_kv_heads=1, head_dim=1, block_size=1,
                        num_blocks=total_units, device="cpu", dtype=None)
    a = pool.allocate(4)
    pool.allocate(3)
    pool.release(a)
    try:
        pool.allocate(6)
        paged_success = True
    except RuntimeError:
        paged_success = False

    arena = ContiguousArena(total_size=total_units)
    a_off = arena.allocate(4)
    arena.allocate(3)
    arena.free(a_off, 4)
    naive_success = arena.allocate(6) is not None

    print("Pattern: alloc A(4), alloc B(3), free A, alloc C(6) — total free=7, needed=6")
    print(f"  Paged block pool : {'SUCCESS' if paged_success else 'FAILED'} (blocks don't need to be adjacent)")
    print(f"  Naive contiguous : {'SUCCESS' if naive_success else 'FAILED'} (largest span=4 < 6, despite 7 total free)")
    return {"paged_success": paged_success, "naive_success": naive_success}


def main():
    config = EngineConfig(use_kv_cache=True, use_paged_kv=True, use_continuous_batching=False,
                           use_scheduler=False, use_radix_cache=False, use_chunked_prefill=False)
    engine = NanoServeEngine(config)

    record = {
        "step": "paged_kv_fragmentation", "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "internal_fragmentation": run_internal_fragmentation_test(engine),
        "external_fragmentation": run_external_fragmentation_test(),
    }
    path = save_results("paged_kv_fragmentation", record, settings.RESULTS_DIR)
    print(f"\nSaved: {path}")


if __name__ == "__main__":
    main()