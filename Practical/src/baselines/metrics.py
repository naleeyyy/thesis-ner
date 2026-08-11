"""Tiny shared helpers for baseline inference."""

from __future__ import annotations

import torch


def resolve_device() -> torch.device:
    """Pick the best available inference device: CUDA > MPS > CPU."""
    if torch.cuda.is_available():
        return torch.device("cuda:0")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def wikiann_ids_to_bio(label_names: list[str], ner_tag_ids: list[int]) -> list[str]:
    return [label_names[i] for i in ner_tag_ids]
