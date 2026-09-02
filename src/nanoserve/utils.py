import json
import time
from pathlib import Path

import torch


def get_device() -> torch.device:
    acc = torch.accelerator.current_accelerator()
    return torch.device(acc.type) if acc is not None else torch.device("cpu")


def sync() -> None:
    if torch.accelerator.is_available():
        torch.accelerator.synchronize()


def reset_memory_stats() -> None:
    if torch.accelerator.is_available():
        torch.accelerator.reset_peak_memory_stats()


def peak_memory_mb() -> float:
    if not torch.accelerator.is_available():
        return 0.0
    return torch.accelerator.max_memory_allocated() / (1024 ** 2)


def save_results(step_name: str, record: dict, results_dir: Path) -> Path:
    results_dir.mkdir(parents=True, exist_ok=True)
    path = results_dir / f"{step_name}.json"
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)
    return path