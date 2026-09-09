import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import settings, utils
from .config import AggregateMetrics, EngineConfig
from .paged_cache import PagedKVCache, PagedKVPool, BatchedDecodeCache


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
    def run_continuous_batch(self, requests: list[dict]) -> dict:
        """Orca/vLLM-style iteration-level continuous batching: each request prefills
        sequentially when admitted (existing single-request PagedKVCache path), then
        joins a shared BatchedDecodeCache decode batch (capped at settings.MAX_BATCH_SIZE)
        until EOS or max_new_tokens. Batch membership is re-evaluated every decode step."""

        decode_cache = BatchedDecodeCache(self.pool)
        pending = sorted(requests, key=lambda r: r["arrival_delay"])
        active: dict[int, dict] = {}
        finished: dict[int, dict] = {}
        next_id = 0
        occupancy_samples = []

        utils.reset_memory_stats()
        utils.sync()
        wall_start = time.perf_counter()

        def try_admit():
            nonlocal next_id
            while pending and len(active) < settings.MAX_BATCH_SIZE:
                elapsed = time.perf_counter() - wall_start
                if pending[0]["arrival_delay"] > elapsed:
                    break
                req = pending.pop(0)
                rid = next_id
                next_id += 1

                input_ids = self._build_input_ids(req["prompt"])
                cache = PagedKVCache(self.pool)
                seq_len = input_ids.shape[1]
                cache.begin_step(seq_len)
                cache_position = torch.arange(0, seq_len, device=self.device)
                t0 = time.perf_counter()
                outputs = self.model(input_ids, past_key_values=cache, use_cache=True, cache_position=cache_position)
                cache.commit_step()
                utils.sync()
                ttft = time.perf_counter() - t0

                next_token = outputs.logits[:, -1, :].argmax(dim=-1, keepdim=True)
                decode_cache.add_request(rid, cache.block_table, cache.num_tokens)
                active[rid] = {
                    "prompt": req["prompt"], "prompt_len": seq_len, "max_new_tokens": req["max_new_tokens"],
                    "next_token": next_token, "num_generated": 1,
                    "ttft": ttft, "token_times": [], "last_time": time.perf_counter(),
                    "generated_ids": [next_token.clone()],
                    "done": next_token.item() == self.tokenizer.eos_token_id,
                }
                if active[rid]["done"]:
                    active[rid]["finish_time"] = time.perf_counter() - wall_start
                    finished[rid] = active.pop(rid)
                    decode_cache.remove_request(rid)

        try_admit()
        step_index = 0  # REMOVE: for debugging, can be removed later
        step_trace = []
        while active or pending:
            if not active:
                wait = pending[0]["arrival_delay"] - (time.perf_counter() - wall_start)
                if wait > 0:
                    time.sleep(wait)
                try_admit()
                continue

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
                    decode_cache.remove_request(rid)

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
                "finish_time_s": r["finish_time"],
            })

        metrics = AggregateMetrics(
            ttft_mean=sum(all_ttft) / len(all_ttft),
            tpot_mean=sum(all_step_times) / len(all_step_times) if all_step_times else 0.0,
            throughput_tps=total_tokens / total_wall,
            peak_memory_mb=utils.peak_memory_mb(),
            ttft_p99=utils.percentile(all_ttft, 99),
            tpot_p99=utils.percentile(all_step_times, 99),
            batch_occupancy_mean=sum(occupancy_samples) / len(occupancy_samples),
        )
        return {
            "metrics": metrics,
            "num_requests": len(finished),
            "total_tokens": total_tokens,
            "total_wall_s": total_wall,
            "per_request": per_request,
            "step_trace": step_trace,
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