from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-1.7B"
DEVICE_OVERRIDE = None  # e.g. "cpu" to force off MPS
PAGED_KV_BLOCK_SIZE = 16
PAGED_KV_NUM_BLOCKS = 256
NAIVE_MAX_RESERVED_TOKENS = 1024  # what a naive engine would reserve per sequence upfront

PROJECT_ROOT = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_ROOT / ".cache"
RESULTS_DIR = PROJECT_ROOT / "results"

# Sized ahead for the full build order, not just step 1:
# short/medium  -> single-request baselines (steps 1-3, 8, 9)
# long          -> exceeds one chunk, for chunked-prefill splitting (step 7)
# prefix_shared -> common prefix + distinct suffixes, for radix cache (step 6)
BENCHMARK_PROMPTS = {
    "short": "Explain what a hash table is and why it's useful, in one paragraph.",
    "medium": (
        "Write a detailed explanation of how binary search trees work, "
        "including insertion, deletion, and lookup, and discuss their "
        "time complexity in the average and worst case. Use an example "
        "to illustrate when rebalancing is needed."
    ),
    "long": (
        "Write a comprehensive technical overview of how modern operating "
        "systems manage virtual memory, covering paging, page tables, TLBs, "
        "and page replacement policies. " * 15
    ),
    "prefix_shared": {
        "prefix": "You are a helpful assistant. Background: the user is a software engineer. ",
        "suffixes": [
            "What is the capital of France?",
            "What is the capital of Japan?",
            "What is the capital of Brazil?",
        ],
    },
}

GEN_LEN_DEFAULTS = {
    "short": 50,
    "medium": 128,
    "long": 128,
    "prefix_shared": 32,
}

# cap on concurrent decode-batch membership. 
# Bump on bigger hardware (along with PAGED_KV_NUM_BLOCKS) 
MAX_BATCH_SIZE = 6

CONTINUOUS_BATCH_REQUESTS = [
    {"prompt": BENCHMARK_PROMPTS["short"], "max_new_tokens": GEN_LEN_DEFAULTS["short"], "arrival_delay": 0.0},
    {"prompt": BENCHMARK_PROMPTS["medium"], "max_new_tokens": GEN_LEN_DEFAULTS["medium"], "arrival_delay": 0.3},
    {"prompt": BENCHMARK_PROMPTS["short"], "max_new_tokens": GEN_LEN_DEFAULTS["short"], "arrival_delay": 0.6},
    {"prompt": BENCHMARK_PROMPTS["medium"], "max_new_tokens": GEN_LEN_DEFAULTS["medium"], "arrival_delay": 0.9},
    {"prompt": BENCHMARK_PROMPTS["short"], "max_new_tokens": GEN_LEN_DEFAULTS["short"], "arrival_delay": 1.2},
    {"prompt": BENCHMARK_PROMPTS["medium"], "max_new_tokens": GEN_LEN_DEFAULTS["medium"], "arrival_delay": 1.5},
    # backfill: dynamically timed against real finish times (21.65s / 22.39s / 22.39s)
    {"prompt": BENCHMARK_PROMPTS["short"], "max_new_tokens": GEN_LEN_DEFAULTS["short"], "arrival_delay": 21.8},
    {"prompt": BENCHMARK_PROMPTS["short"], "max_new_tokens": GEN_LEN_DEFAULTS["short"], "arrival_delay": 22.5},
    {"prompt": BENCHMARK_PROMPTS["short"], "max_new_tokens": GEN_LEN_DEFAULTS["short"], "arrival_delay": 22.6},
]