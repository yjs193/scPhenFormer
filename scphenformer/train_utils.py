"""Shared training helpers."""

from __future__ import annotations

import math
import os
import random

import numpy as np
import torch


def require_path(value: str, name: str) -> str:
    if not value:
        raise ValueError(f"Please provide --{name.replace('_', '-')}")
    return value


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def cosine_learning_rate(
    epoch: float, epochs: int, maximum: float, minimum: float, warmup_epochs: int
) -> float:
    if epochs < 1 or not 0 <= warmup_epochs <= epochs or not 0 < minimum <= maximum:
        raise ValueError("Require epochs >= warmup >= 0 and 0 < minimum_lr <= maximum_lr")
    if not 0 <= epoch <= epochs:
        raise ValueError("Training progress must be between zero and the number of epochs")
    if warmup_epochs and epoch <= warmup_epochs:
        return minimum + (maximum - minimum) * epoch / warmup_epochs
    progress = (epoch - warmup_epochs) / max(1, epochs - warmup_epochs)
    return minimum + 0.5 * (maximum - minimum) * (1.0 + math.cos(math.pi * progress))


def move(batch: dict, device: torch.device) -> dict:
    return {
        key: value.to(device, non_blocking=True) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

def classification_metrics(labels: list[int], predictions: list[int], number_of_classes: int) -> dict[str, float]:
    from sklearn.metrics import accuracy_score, f1_score

    return {
        "accuracy": float(accuracy_score(labels, predictions)),
        "macro_f1": float(f1_score(labels, predictions, labels=list(range(number_of_classes)), average="macro", zero_division=0)),
    }


def distributed_device() -> tuple[torch.device, int, int]:
    import torch.distributed as dist

    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    rank = int(os.environ.get("RANK", "0"))
    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)
        device = torch.device("cuda", local_rank)
    else:
        device = torch.device("cpu")
    if world_size > 1:
        dist.init_process_group("nccl" if device.type == "cuda" else "gloo")
    return device, rank, world_size
