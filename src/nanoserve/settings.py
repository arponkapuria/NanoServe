from pathlib import Path

MODEL_NAME = "Qwen/Qwen3-1.7B"
DEVICE_OVERRIDE = None  # e.g. "cpu" to force off MPS

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