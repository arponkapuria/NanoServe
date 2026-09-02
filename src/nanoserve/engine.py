import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from . import settings, utils
from .config import AggregateMetrics, EngineConfig


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
            settings.MODEL_NAME, torch_dtype=torch.float16
        ).to(self.device)
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
        }