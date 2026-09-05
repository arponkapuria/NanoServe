"""Naive-side baselines paged KV cache is designed to beat. Not a competing
engine — just the two allocation strategies, isolated and measured/simulated
on their own terms, for direct comparison against the real PagedKVPool."""

import torch

from . import utils


def measure_reservation_waste_mb(num_layers, num_kv_heads, head_dim, max_reserved_tokens, actual_tokens, device):
    """Internal fragmentation, REAL measurement: allocate a real contiguous KV
    buffer sized for the worst case (max_reserved_tokens), measure its real
    memory cost, free it. Returns (allocated_mb, waste_pct)."""
    utils.reset_memory_stats()
    before = utils.peak_memory_mb()

    shape = (num_layers, 2, num_kv_heads, max_reserved_tokens, head_dim)  # 2 = K and V
    buf = torch.zeros(shape, device=device, dtype=torch.float16)
    utils.sync()
    allocated_mb = utils.peak_memory_mb() - before
    del buf

    waste_pct = 100 * (1 - actual_tokens / max_reserved_tokens)
    return allocated_mb, waste_pct


class ContiguousArena:
    """External fragmentation, SIMULATED: a first-fit contiguous allocator over
    an abstract size range. No GPU memory involved on purpose — external
    fragmentation is a bookkeeping property of the allocation policy, not a
    hardware fact, so this is the standard way to test it."""

    def __init__(self, total_size: int):
        self.free_spans = [(0, total_size)]  # (offset, size), sorted by offset

    def allocate(self, size: int) -> int | None:
        """Returns an offset, or None if no SINGLE free span is big enough —
        even if total free space would suffice. That gap is external fragmentation."""
        for i, (offset, span_size) in enumerate(self.free_spans):
            if span_size >= size:
                self.free_spans[i] = (offset + size, span_size - size)
                if self.free_spans[i][1] == 0:
                    self.free_spans.pop(i)
                return offset
        return None

    def free(self, offset: int, size: int) -> None:
        self.free_spans.append((offset, size))
        self.free_spans.sort()
        merged = []
        for span in self.free_spans:
            if merged and merged[-1][0] + merged[-1][1] == span[0]:
                merged[-1] = (merged[-1][0], merged[-1][1] + span[1])
            else:
                merged.append(span)
        self.free_spans = merged