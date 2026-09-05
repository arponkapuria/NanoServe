import torch
from transformers.cache_utils import Cache

class PagedKVPool:
    """Engine-owned physical block storage + free list. Created once, shared across every sequential request — this is what makes block reuse (and the fragmentation test) possible."""

    def __init__(self, num_layers, num_kv_heads, head_dim, block_size, num_blocks, device, dtype):
        self.block_size = block_size
        shape = (num_layers, num_blocks, num_kv_heads, block_size, head_dim)
        self.k_pool = torch.zeros(shape, device=device, dtype=dtype)
        self.v_pool = torch.zeros_like(self.k_pool)
        self.free_blocks = list(range(num_blocks))

    def allocate(self, n: int) -> list[int]:
        """Allocate n blocks from the free list. Returns a list of allocated block indices."""

        if len(self.free_blocks) < n:
            raise RuntimeError(f"KV pool exhausted: need {n} blocks, {len(self.free_blocks)} free")

        return [self.free_blocks.pop() for _ in range(n)]

    def release(self, block_indices: list[int]):
        """Release blocks back to the free list."""

        self.free_blocks.extend(block_indices)


class PagedKVCache(Cache):
    """Per-request view into a shared PagedKVPool. Implements the same `update()` interface HF's built-in DynamicCache does, so the model calls it without knowing paging is happening underneath."""

    def __init__(self, pool: PagedKVPool):
        self.pool = pool
        self.block_table: list[int] = []
        self.num_tokens = 0          # tokens committed from PRIOR steps
        self._step_new_tokens = 0    # new tokens being written THIS step
        self.layers: list = []  

    def begin_step(self, num_new_tokens: int) -> None:
        """Call once per forward pass, before invoking the model. Allocates new
        blocks if needed, and precomputes this step's read/write indices ONCE —
        update() below reuses them across all 28 layers instead of recomputing
        per layer."""
        self._step_new_tokens = num_new_tokens
        block_size = self.pool.block_size
        device = self.pool.k_pool.device

        start, end = self.num_tokens, self.num_tokens + num_new_tokens
        needed_blocks = -(-end // block_size)
        if needed_blocks > len(self.block_table):
            self.block_table += self.pool.allocate(needed_blocks - len(self.block_table))

        table = torch.tensor(self.block_table, device=device)
        positions = torch.arange(start, end, device=device)
        self._write_phys = table[positions // block_size]   # (n_new,) — vectorized, no per-token loop
        self._write_slot = positions % block_size            # (n_new,)
        self._read_phys = table[:needed_blocks]               # blocks to gather for attention
        self._n_blocks = needed_blocks
        self._end = end

    def commit_step(self) -> None:
        self.num_tokens += self._step_new_tokens
        self._step_new_tokens = 0

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        block_size = self.pool.block_size

        # Vectorized write: ALL new tokens in one shot, not a python loop.
        self.pool.k_pool[layer_idx, self._write_phys, :, self._write_slot, :] = key_states[0].permute(1, 0, 2)
        self.pool.v_pool[layer_idx, self._write_phys, :, self._write_slot, :] = value_states[0].permute(1, 0, 2)

        k = self.pool.k_pool[layer_idx].index_select(0, self._read_phys)
        v = self.pool.v_pool[layer_idx].index_select(0, self._read_phys)

        num_kv_heads, head_dim = k.shape[1], k.shape[3]
        k = k.permute(1, 0, 2, 3).reshape(1, num_kv_heads, self._n_blocks * block_size, head_dim)[:, :, :self._end, :]
        v = v.permute(1, 0, 2, 3).reshape(1, num_kv_heads, self._n_blocks * block_size, head_dim)[:, :, :self._end, :]
        return k, v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self.num_tokens

    def get_mask_sizes(self, cache_position, layer_idx: int = 0) -> tuple[int, int]:
        """Override the base Cache implementation, which expects a self.layers list of
        per-layer cache objects we don't use — we track length ourselves instead."""
        
        query_length = cache_position.shape[0] if torch.is_tensor(cache_position) else int(cache_position)
        kv_length = self.get_seq_length() + query_length
        kv_offset = 0
        return kv_length, kv_offset

    def get_max_cache_shape(self) -> int | None:
        return None  # unbounded — the pool enforces the real limit, not the mask logic

    def free(self) -> None:
        """Release all blocks back to the pool. Call this when the request is done."""

        self.pool.release(self.block_table)
        self.block_table = []
        self.num_tokens = 0

