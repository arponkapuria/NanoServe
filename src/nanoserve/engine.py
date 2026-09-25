import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import settings, utils
from .config import AggregateMetrics, EngineConfig
from .paged_cache import PagedKVCache, PagedKVPool, BatchedDecodeCache
from .scheduler import Scheduler
from .radix_cache import RadixCache


class NanoServeEngine:
    def __init__(self, config: EngineConfig | None = None):
        self.config = config or EngineConfig()
        self.device = (
            torch.device(settings.DEVICE_OVERRIDE)
            if settings.DEVICE_OVERRIDE
            else utils.get_device()
        )

        self.tokenizer = AutoTokenizer.from_pretrained(settings.MODEL_NAME)
        self.model = AutoModelForCausalLM.from_pretrained(
            settings.MODEL_NAME, dtype=torch.float16
        ).to(self.device)

        self.pool = None
        self.radix_cache = None

        if self.config.use_radix_cache:
            if not self.config.use_paged_kv:
                raise ValueError("use_radix_cache requires use_paged_kv")
            self.radix_cache = RadixCache(settings.PAGED_KV_BLOCK_SIZE)

        if self.config.use_chunked_prefill and not self.config.use_paged_kv:
            raise ValueError("use_chunked_prefill requires use_paged_kv")

        if self.config.use_paged_kv:
            mc = self.model.config
            head_dim = getattr(mc, "head_dim", mc.hidden_size // mc.num_attention_heads)
            self.pool = PagedKVPool(
                num_layers=mc.num_hidden_layers,
                num_kv_heads=mc.num_key_value_heads,
                head_dim=head_dim,
                block_size=settings.PAGED_KV_BLOCK_SIZE,
                num_blocks=settings.PAGED_KV_NUM_BLOCKS,
                device=self.device,
                dtype=torch.float16,
            )

        self.model.eval()

    def _build_input_ids(self, prompt: str) -> torch.Tensor:
        messages = [{"role": "user", "content": prompt}]
        text = self.tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
            enable_thinking=False,
        )
        return self.tokenizer(text, return_tensors="pt").input_ids.to(self.device)

    def generate(self, prompt: str, max_new_tokens: int) -> dict:
        if self.config.use_radix_cache:
            return self._generate_radix_kv(prompt, max_new_tokens)
        if self.config.use_paged_kv:
            return self._generate_paged_kv(prompt, max_new_tokens)
        if self.config.use_kv_cache:
            return self._generate_kv_cache(prompt, max_new_tokens)
        return self._generate_naive(prompt, max_new_tokens)

    @torch.no_grad()
    def _generate_naive(self, prompt: str, max_new_tokens: int) -> dict:
        """No cache — full sequence recomputed every step. Step 1 baseline."""

        input_ids = self._build_input_ids(prompt)
        generated = input_ids

        utils.reset_memory_stats()
        utils.sync()
        start = time.perf_counter()

        first_token_time = None
        token_times = []

        for _ in range(max_new_tokens):
            step_start = time.perf_counter()
            logits = self.model(generated).logits
            next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
            utils.sync()
            step_end = time.perf_counter()

            if first_token_time is None:
                first_token_time = step_end - start
            else:
                token_times.append(step_end - step_start)

            generated = torch.cat([generated, next_token], dim=1)
            if next_token.item() == self.tokenizer.eos_token_id:
                break

        total_time = time.perf_counter() - start
        return self._build_result(generated, input_ids, first_token_time, token_times, total_time)

    @torch.no_grad()
    def _generate_kv_cache(self, prompt: str, max_new_tokens: int) -> dict:
        """Prefill once, then feed only the newest token each step, reusing past_key_values."""

        input_ids = self._build_input_ids(prompt)
        generated = input_ids

        utils.reset_memory_stats()
        utils.sync()
        start = time.perf_counter()

        outputs = self.model(input_ids, use_cache=True)
        past_key_values = outputs.past_key_values
        next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
        utils.sync()
        first_token_time = time.perf_counter() - start
        generated = torch.cat([generated, next_token], dim=1)

        token_times = []
        for _ in range(max_new_tokens - 1):
            if next_token.item() == self.tokenizer.eos_token_id:
                break
            step_start = time.perf_counter()
            outputs = self.model(next_token, past_key_values=past_key_values, use_cache=True)
            past_key_values = outputs.past_key_values
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            utils.sync()
            token_times.append(time.perf_counter() - step_start)
            generated = torch.cat([generated, next_token], dim=1)

        total_time = time.perf_counter() - start
        return self._build_result(generated, input_ids, first_token_time, token_times, total_time)

    @torch.no_grad()
    def _generate_paged_kv(self, prompt: str, max_new_tokens: int) -> dict:
        """Paged KV cache: block-based storage, gather-based attention (no fused
        kernel on MPS — see project notes on why)."""

        input_ids = self._build_input_ids(prompt)
        cache = PagedKVCache(self.pool)

        utils.reset_memory_stats()
        utils.sync()
        start = time.perf_counter()

        try: 
            seq_len = input_ids.shape[1]
            cache.begin_step(seq_len)
            cache_position = torch.arange(0, seq_len, device=self.device)
            outputs = self.model(input_ids, past_key_values=cache, use_cache=True, cache_position=cache_position)
            cache.commit_step()
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            utils.sync()
            first_token_time = time.perf_counter() - start
            generated = torch.cat([input_ids, next_token], dim=1)

            token_times = []
            for _ in range(max_new_tokens - 1):
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                step_start = time.perf_counter()
                cache.begin_step(1)
                cache_position = torch.tensor([cache.num_tokens], device=self.device)
                outputs = self.model(next_token, past_key_values=cache, use_cache=True, cache_position=cache_position)
                cache.commit_step()
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                utils.sync()
                token_times.append(time.perf_counter() - step_start)
                generated = torch.cat([generated, next_token], dim=1)

            total_time = time.perf_counter() - start
            return self._build_result(generated, input_ids, first_token_time, token_times, total_time)
        finally:
            cache.free()

    @torch.no_grad()
    def _generate_radix_kv(self, prompt: str, max_new_tokens: int) -> dict:
        """Paged KV + block-aligned prefix caching. Matches the prompt's complete
        blocks against the shared RadixCache; only the unmatched remainder (plus any
        ragged, non-full-block tail) is actually prefilled. Newly-computed complete
        blocks are inserted into the tree for future requests. Generation and the
        ragged tail always stay private — never cached (prompt-only cache)."""

        input_ids = self._build_input_ids(prompt)
        prompt_tokens = input_ids[0].tolist()
        seq_len = input_ids.shape[1]
        block_size = settings.PAGED_KV_BLOCK_SIZE

        cache = PagedKVCache(self.pool)

        utils.reset_memory_stats()
        utils.sync()
        start = time.perf_counter()

        matched_block_ids, matched_nodes = self.radix_cache.match(prompt_tokens)
        # Logits (unlike KV) are never cached — always leave >=1 token to actually
        # forward, even on a verbatim-repeat request that matches 100% of the prompt.
        max_m = (seq_len - 1) // block_size
        m = min(len(matched_block_ids), max_m)
        matched_block_ids, matched_nodes = matched_block_ids[:m], matched_nodes[:m]
        matched_len = m * block_size
        self.radix_cache.touch(matched_nodes)

        cache.block_table = list(matched_block_ids)
        cache.num_tokens = matched_len
        cache._radix_path = list(matched_nodes)

        try:
            remainder_len = seq_len - matched_len
            cache.begin_step(remainder_len)
            cache_position = torch.arange(matched_len, seq_len, device=self.device)
            outputs = self.model(
                input_ids[:, matched_len:],
                past_key_values=cache,
                use_cache=True,
                cache_position=cache_position,
            )
            cache.commit_step()
            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            utils.sync()
            first_token_time = time.perf_counter() - start
            generated = torch.cat([input_ids, next_token], dim=1)

            # Insert newly-computed COMPLETE blocks only. The ragged tail is skipped
            # on purpose: it's about to receive generated tokens, and a block another
            # request may match into must never be mutated afterward.
            num_complete_blocks = seq_len // block_size
            new_cacheable = num_complete_blocks - m
            if new_cacheable > 0:
                new_block_ids = cache.block_table[m:m + new_cacheable]
                parent = matched_nodes[-1] if matched_nodes else self.radix_cache.root
                cache._radix_path += self.radix_cache.insert(prompt_tokens, parent, m, new_block_ids)
            cache._cached_blocks = m + new_cacheable

            token_times = []
            for _ in range(max_new_tokens - 1):
                if next_token.item() == self.tokenizer.eos_token_id:
                    break
                step_start = time.perf_counter()
                cache.begin_step(1)
                cache_position = torch.tensor([cache.num_tokens], device=self.device)
                outputs = self.model(next_token, past_key_values=cache, use_cache=True, cache_position=cache_position)
                cache.commit_step()
                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                utils.sync()
                token_times.append(time.perf_counter() - step_start)
                generated = torch.cat([generated, next_token], dim=1)

            total_time = time.perf_counter() - start
            result = self._build_result(generated, input_ids, first_token_time, token_times, total_time)
            result["prompt_len"] = seq_len
            result["cache_hit_tokens"] = matched_len
            return result
        finally:
            cache.free(radix_cache=self.radix_cache)

    @torch.no_grad()
    def run_continuous_batch(self, requests: list[dict], chunk_size: int | None = None) -> dict:
        """Orca/vLLM-style iteration-level continuous batching: each request prefills
        (existing single-request PagedKVCache path), then joins a shared BatchedDecodeCache
        decode batch (capped at settings.MAX_BATCH_SIZE) until EOS or max_new_tokens.
        Batch membership is re-evaluated every decode step.

        When self.config.use_scheduler is True, arrivals pass through a Scheduler first:
        an explicit FCFS wait queue with fixed-depth backpressure (settings.MAX_QUEUE_DEPTH,
        rejected reason 'queue_full') and a fixed per-request wait timeout
        (settings.MAX_QUEUE_WAIT_MS, rejected reason 'timeout') before being admitted.
        When False, falls back to step 4's original behavior exactly: admit immediately
        whenever a slot is free, unbounded queue, no rejections.

        When self.config.use_chunked_prefill is True (step 7), a prompt is prefilled in
        chunks of `chunk_size` tokens (default settings.PREFILL_CHUNK_SIZE), ONE chunk per
        loop iteration and only AFTER that iteration's decode step (decode-first). Active
        requests therefore never wait longer than one chunk for their next token. One prompt
        is prefilled at a time and holds a batch slot while doing so. When False, the whole
        prompt is prefilled in a single forward pass, exactly as in steps 4-6."""

        chunked = self.config.use_chunked_prefill
        block_size = settings.PAGED_KV_BLOCK_SIZE
        chunk_size = chunk_size or settings.PREFILL_CHUNK_SIZE
        if chunked and chunk_size % block_size:
            raise ValueError(f"chunk_size {chunk_size} must be a multiple of block size {block_size}")

        decode_cache = BatchedDecodeCache(self.pool)
        pending = sorted(requests, key=lambda r: r["arrival_delay"])
        active: dict[int, dict] = {}
        finished: dict[int, dict] = {}
        rejected: list[dict] = []
        waiting_no_scheduler: list[dict] = []  # only used when scheduler disabled
        prefilling: dict | None = None         # step 7: the ONE request currently mid-prefill
        prefill_trace: list[dict] = []         # step 7: one entry per chunk forward pass
        next_id = 0
        occupancy_samples = []

        scheduler = (
            Scheduler(settings.MAX_QUEUE_DEPTH, settings.MAX_QUEUE_WAIT_MS / 1000)
            if self.config.use_scheduler else None
        )

        utils.reset_memory_stats()
        utils.sync()
        wall_start = time.perf_counter()

        def intake(now):
            """Move any request whose arrival_delay has elapsed out of `pending`. With
            a scheduler this is the moment backpressure is checked; without one it goes
            straight onto the old unbounded waiting list (step 4 behavior)."""
            nonlocal pending
            still_waiting = []
            for req in pending:
                if req["arrival_delay"] > now:
                    still_waiting.append(req)
                    continue
                if scheduler is not None:
                    decision = scheduler.submit(req, now)
                    if decision == "rejected_queue_full":
                        rejected.append({"prompt": req["prompt"], "reason": "queue_full",
                                          "arrival_delay": req["arrival_delay"]})
                else:
                    waiting_no_scheduler.append(req)
            pending = still_waiting

        def start_prefill(req, queue_time_s):
            """Step 7 (1/2): set a request up for prefill — tokenize, radix match, create its
            cache — but run NO model code yet. advance_prefill() consumes it chunk by chunk."""
            nonlocal next_id, prefilling
            rid = next_id
            next_id += 1

            input_ids = self._build_input_ids(req["prompt"])
            prompt_tokens = input_ids[0].tolist()
            seq_len = input_ids.shape[1]
            cache = PagedKVCache(self.pool)

            matched_nodes: list = []
            matched_len = 0
            if self.radix_cache is not None:
                matched_block_ids, matched_nodes = self.radix_cache.match(prompt_tokens)
                max_m = (seq_len - 1) // block_size
                m = min(len(matched_block_ids), max_m)
                matched_block_ids, matched_nodes = matched_block_ids[:m], matched_nodes[:m]
                matched_len = m * block_size
                self.radix_cache.touch(matched_nodes)
                cache.block_table = list(matched_block_ids)
                cache.num_tokens = matched_len

            prefilling = {
                "rid": rid, "req": req, "input_ids": input_ids, "prompt_tokens": prompt_tokens,
                "seq_len": seq_len, "cache": cache, "matched_nodes": matched_nodes,
                "matched_len": matched_len, "next_pos": matched_len, "chunk_index": 0,
                "queue_time_s": queue_time_s, "t_admit": time.perf_counter(),
            }

        def advance_prefill():
            """Step 7 (2/2): run ONE chunk of the in-flight prefill (the entire remainder when
            chunking is off). Chunk k attends to chunks 0..k-1 already sitting in the paged
            cache — same mechanism as the radix remainder path. On the LAST chunk: sample token
            1, cache the prompt's complete blocks in the radix tree, hand the request to the
            decode batch."""
            nonlocal prefilling
            p = prefilling
            cache, seq_len, start = p["cache"], p["seq_len"], p["next_pos"]
            end = min(start + chunk_size, seq_len) if chunked else seq_len

            cache.begin_step(end - start)
            cache_position = torch.arange(start, end, device=self.device)
            t0 = time.perf_counter()
            outputs = self.model(
                p["input_ids"][:, start:end], past_key_values=cache,
                use_cache=True, cache_position=cache_position,
            )
            cache.commit_step()
            utils.sync()
            prefill_trace.append({
                "request_id": p["rid"], "chunk_index": p["chunk_index"], "start": start,
                "end": end, "duration_ms": (time.perf_counter() - t0) * 1000,
            })
            p["chunk_index"] += 1
            p["next_pos"] = end
            if end < seq_len:
                return  # more chunks to go; the decode batch gets its turn first

            # ---- last chunk: everything below is the old admit_one tail, unchanged ----
            rid, req, matched_nodes = p["rid"], p["req"], p["matched_nodes"]
            ttft = time.perf_counter() - p["t_admit"]  # admission -> first token, incl. interleaved decode steps

            radix_path = list(matched_nodes)
            cached_blocks = len(matched_nodes)
            if self.radix_cache is not None:
                num_complete_blocks = seq_len // block_size
                m = len(matched_nodes)
                new_cacheable = num_complete_blocks - m
                if new_cacheable > 0:
                    new_block_ids = cache.block_table[m:m + new_cacheable]
                    parent = matched_nodes[-1] if matched_nodes else self.radix_cache.root
                    radix_path += self.radix_cache.insert(p["prompt_tokens"], parent, m, new_block_ids)
                cached_blocks = m + new_cacheable

            next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
            decode_cache.add_request(rid, cache.block_table, cache.num_tokens,
                                      cached_blocks=cached_blocks, radix_path=radix_path)
            active[rid] = {
                "prompt": req["prompt"], "prompt_len": seq_len, "max_new_tokens": req["max_new_tokens"],
                "next_token": next_token, "num_generated": 1,
                "ttft": ttft, "token_times": [], "last_time": time.perf_counter(),
                "generated_ids": [next_token.clone()],
                "done": next_token.item() == self.tokenizer.eos_token_id,
                "queue_time_s": p["queue_time_s"],
                "cache_hit_tokens": p["matched_len"],
                "cached_blocks": cached_blocks,
            }
            prefilling = None
            if active[rid]["done"]:
                active[rid]["finish_time"] = time.perf_counter() - wall_start
                finished[rid] = active.pop(rid)
                decode_cache.remove_request(rid, radix_cache=self.radix_cache)

        def admit(req, queue_time_s):
            start_prefill(req, queue_time_s)
            if not chunked:  # chunking off: whole prompt now, in one pass (steps 4-6 behavior)
                advance_prefill()

        def slots_free():
            if prefilling is not None:  # step 7: one prefill at a time; it already holds a slot
                return 0
            free = settings.MAX_BATCH_SIZE - len(active)
            return min(free, 1) if chunked else free

        def try_admit():
            now = time.perf_counter() - wall_start
            intake(now)
            if scheduler is not None:
                for entry in scheduler.expire(now):
                    rejected.append({"prompt": entry["request"]["prompt"], "reason": "timeout",
                                      "arrival_delay": entry["request"]["arrival_delay"]})
                for entry in scheduler.pull(now, slots_free()):
                    admit(entry["request"], entry["queue_time_s"])
            else:
                while waiting_no_scheduler and slots_free() > 0:
                    admit(waiting_no_scheduler.pop(0), queue_time_s=0.0)

        def queued_depth():
            return scheduler.depth if scheduler is not None else len(waiting_no_scheduler)

        try_admit()
        step_index = 0  # REMOVE: for debugging, can be removed later
        step_trace = []
        while active or pending or queued_depth() or prefilling is not None:
            if not active and prefilling is None:
                now = time.perf_counter() - wall_start
                if queued_depth() == 0 and pending:
                    wait = pending[0]["arrival_delay"] - now
                    if wait > 0:
                        time.sleep(wait)
                try_admit()
                continue

            if active:
                active_ids = list(active.keys())
                occupancy_samples.append(len(active_ids))
                batched_tokens = torch.cat([active[rid]["next_token"] for rid in active_ids], dim=0)

                position_ids = decode_cache.begin_step(active_ids)
                blocks_needed = decode_cache._read_table.shape[1]
                cache_position = torch.tensor([decode_cache._max_len - 1], device=self.device)
                attn_mask = decode_cache._attn_bias

                utils.sync()
                step_t0 = time.perf_counter()

                outputs = self.model(
                    batched_tokens, past_key_values=decode_cache, use_cache=True,
                    cache_position=cache_position, position_ids=position_ids, attention_mask=attn_mask,
                )
                utils.sync()
                step_duration = time.perf_counter() - step_t0
                decode_cache.commit_step()

                step_trace.append({
                    "step_index": step_index, "num_active": len(active_ids),
                    "blocks_needed": blocks_needed, "max_len": decode_cache._max_len,
                    "step_duration_ms": step_duration * 1000,
                })
                step_index += 1

                now = time.perf_counter()

                next_tokens = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                for i, rid in enumerate(active_ids):
                    st = active[rid]
                    st["token_times"].append(now - st["last_time"])
                    st["last_time"] = now
                    tok = next_tokens[i:i + 1]
                    st["next_token"] = tok
                    st["generated_ids"].append(tok.clone())
                    st["num_generated"] += 1
                    is_eos = tok.item() == self.tokenizer.eos_token_id
                    if is_eos or st["num_generated"] >= st["max_new_tokens"]:
                        st["done"] = True

                for rid in list(active.keys()):
                    if active[rid]["done"]:
                        active[rid]["finish_time"] = time.perf_counter() - wall_start
                        finished[rid] = active.pop(rid)
                        decode_cache.remove_request(rid, radix_cache=self.radix_cache)

            if prefilling is not None:  # step 7: decode-first — one prefill chunk AFTER the decode step
                advance_prefill()

            try_admit()

        total_wall = time.perf_counter() - wall_start
        all_ttft = [r["ttft"] for r in finished.values()]
        all_step_times = [t for r in finished.values() for t in r["token_times"]]
        total_tokens = sum(r["num_generated"] for r in finished.values())

        per_request = []
        for rid, r in finished.items():
            text = self.tokenizer.decode(torch.cat(r["generated_ids"], dim=1)[0], skip_special_tokens=True)
            per_request.append({
                "request_id": rid, "prompt": r["prompt"], "generated_text": text,
                "num_generated_tokens": r["num_generated"], "prompt_len": r["prompt_len"],
                "finish_time_s": r["finish_time"], "ttft_s": r["ttft"],
                "admitted_at_s": r["finish_time"] - sum(r["token_times"]) - r["ttft"],
                "queue_time_s": r.get("queue_time_s", 0.0),
                "cache_hit_tokens": r.get("cache_hit_tokens", 0),
                "cached_blocks": r.get("cached_blocks", 0),
                "token_times_s": r["token_times"],  # step 7: per-token gaps, for stall analysis
            })

        queue_times = [r["queue_time_s"] for r in finished.values() if r.get("queue_time_s") is not None]
        rejected_queue_full = sum(1 for r in rejected if r["reason"] == "queue_full")
        rejected_timeout = sum(1 for r in rejected if r["reason"] == "timeout")

        cache_hit_rate = None
        if self.config.use_radix_cache:
            total_prompt_tokens = sum(r["prompt_len"] for r in finished.values())
            total_hit_tokens = sum(r.get("cache_hit_tokens", 0) for r in finished.values())
            cache_hit_rate = (total_hit_tokens / total_prompt_tokens) if total_prompt_tokens else 0.0

        metrics = AggregateMetrics(
            ttft_mean=sum(all_ttft) / len(all_ttft),
            tpot_mean=sum(all_step_times) / len(all_step_times) if all_step_times else 0.0,
            throughput_tps=total_tokens / total_wall,
            peak_memory_mb=utils.peak_memory_mb(),
            ttft_p99=utils.percentile(all_ttft, 99),
            tpot_p99=utils.percentile(all_step_times, 99),
            tpot_max=max(all_step_times) if all_step_times else 0.0,
            batch_occupancy_mean=sum(occupancy_samples) / len(occupancy_samples) if occupancy_samples else 0.0,
            mean_queue_time_ms=(sum(queue_times) / len(queue_times) * 1000) if queue_times else 0.0,
            rejected_queue_full=rejected_queue_full,
            rejected_timeout=rejected_timeout,
            cache_hit_rate=cache_hit_rate,
        )
        return {
            "metrics": metrics,
            "num_requests": len(finished),
            "total_tokens": total_tokens,
            "total_wall_s": total_wall,
            "per_request": per_request,
            "step_trace": step_trace,
            "prefill_trace": prefill_trace,
            "chunk_size": chunk_size if chunked else None,
            "rejected": rejected,
        }

    def _build_result(self, generated, input_ids, first_token_time, token_times, total_time) -> dict:
        num_new = generated.shape[1] - input_ids.shape[1]
        metrics = AggregateMetrics(
            ttft_mean=first_token_time,
            tpot_mean=(sum(token_times) / len(token_times)) if token_times else 0.0,
            throughput_tps=num_new / total_time if total_time > 0 else 0.0,
            peak_memory_mb=utils.peak_memory_mb(),
        )
        return {
            "text": self.tokenizer.decode(generated[0, input_ids.shape[1]:], skip_special_tokens=True),
            "metrics": metrics,
            "num_new_tokens": num_new,
            "total_tokens": generated.shape[1],   # prompt + generated — what the cache actually holds
        }