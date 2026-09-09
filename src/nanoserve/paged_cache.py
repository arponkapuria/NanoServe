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


class BatchedDecodeCache(Cache):
    """Decode-phase cache shared by multiple concurrent requests. Each request keeps
    its own block_table + num_tokens (handed over from its PagedKVCache right after
    prefill). Every decode step batches whichever requests are currently active into
    one forward call — this is the actual mechanism continuous batching relies on."""

    def __init__(self, pool: PagedKVPool):
        self.pool = pool
        self.requests: dict[int, dict] = {}
        self.active_ids: list[int] = []

    def add_request(self, req_id: int, block_table: list[int], num_tokens: int) -> None:
        device = self.pool.k_pool.device
        bt_tensor = torch.tensor(block_table, dtype=torch.long, device=device)
        self.requests[req_id] = {
            "block_table": list(block_table), "block_table_tensor": bt_tensor, "num_tokens": num_tokens,
        }

    def remove_request(self, req_id: int) -> None:
        state = self.requests.pop(req_id)
        self.pool.release(state["block_table"])

    def begin_step(self, active_ids: list[int]) -> torch.Tensor:
        """Allocates this step's new-token block for each active request, and builds
        the batched write/read indices ONCE — same 'compute once, reuse across 28
        layers' pattern as step 3's begin_step, now with a batch dimension. Returns
        position_ids since each request sits at a different absolute position."""
        self.active_ids = active_ids
        block_size = self.pool.block_size
        device = self.pool.k_pool.device

        write_phys, write_slot, positions = [], [], []
        max_len = 0
        for rid in active_ids:
            st = self.requests[rid]
            pos = st["num_tokens"]
            needed_blocks = -(-(pos + 1) // block_size)
            if needed_blocks > len(st["block_table"]):
                new_blocks = self.pool.allocate(needed_blocks - len(st["block_table"]))
                st["block_table"] += new_blocks
                new_tensor = torch.tensor(new_blocks, dtype=torch.long, device=device)
                st["block_table_tensor"] = torch.cat([st["block_table_tensor"], new_tensor])
            write_phys.append(st["block_table"][pos // block_size])
            write_slot.append(pos % block_size)
            positions.append(pos)
            max_len = max(max_len, pos + 1)

        self._write_phys = torch.tensor(write_phys, device=device)
        self._write_slot = torch.tensor(write_slot, device=device)
        self._max_len = max_len

        blocks_needed = -(-max_len // block_size)
        table = torch.zeros((len(active_ids), blocks_needed), dtype=torch.long, device=device)
        for i, rid in enumerate(active_ids):
            bt_tensor = self.requests[rid]["block_table_tensor"][:blocks_needed]
            table[i, :bt_tensor.shape[0]] = bt_tensor
        self._read_table = table

        real_lens = torch.tensor([self.requests[rid]["num_tokens"] + 1 for rid in active_ids], device=device)
        idx = torch.arange(blocks_needed * block_size, device=device).unsqueeze(0)
        pad_mask = (idx < real_lens.unsqueeze(1))[:, :max_len]  # (batch, max_len) True=real token

        neg_inf = torch.finfo(torch.float16).min
        self._attn_bias = torch.zeros((len(active_ids), 1, 1, max_len), dtype=torch.float16, device=device)
        self._attn_bias.masked_fill_(~pad_mask.view(len(active_ids), 1, 1, max_len), neg_inf)

        return torch.tensor(positions, device=device).unsqueeze(1)  # (batch, 1)

    def commit_step(self) -> None:
        for rid in self.active_ids:
            self.requests[rid]["num_tokens"] += 1

    def update(self, key_states, value_states, layer_idx, cache_kwargs=None):
        block_size = self.pool.block_size
        batch = len(self.active_ids)

        self.pool.k_pool[layer_idx, self._write_phys, :, self._write_slot, :] = key_states[:, :, 0, :]
        self.pool.v_pool[layer_idx, self._write_phys, :, self._write_slot, :] = value_states[:, :, 0, :]

        k = self.pool.k_pool[layer_idx][self._read_table]   # (batch, blocks, kv_heads, block_size, head_dim)
        v = self.pool.v_pool[layer_idx][self._read_table]
        num_kv_heads, head_dim = k.shape[2], k.shape[4]
        blocks_needed = self._read_table.shape[1]
        k = k.permute(0, 2, 1, 3, 4).reshape(batch, num_kv_heads, blocks_needed * block_size, head_dim)[:, :, :self._max_len, :]
        v = v.permute(0, 2, 1, 3, 4).reshape(batch, num_kv_heads, blocks_needed * block_size, head_dim)[:, :, :self._max_len, :]
        return k, v

    def get_seq_length(self, layer_idx: int = 0) -> int:
        return self._max_len

    def get_mask_sizes(self, cache_position, layer_idx: int = 0) -> tuple[int, int]:
        return self._max_len, 0

    def get_max_cache_shape(self) -> int | None:
        return None