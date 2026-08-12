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


def first_subword_tags(
    word_ids: list[int | None],
    pred_ids: list[int],
    id2label: dict[int, str],
    n_tokens: int,
    remap=None,
) -> list[str]:
    """One BIO tag per word: whatever the word's *first* subword predicted.

    Shared by the frozen-checkpoint baselines and the fine-tuned model so both are scored
    the same way. If they diverged here, "my model beats Kushtrim's 0.925" would be
    comparing two different measurements rather than two models.

    Words with no surviving subword (only possible when a long sentence is truncated)
    keep `O`. `remap` is the per-model cross-tagset function; None means identity.
    """
    tags = ["O"] * n_tokens
    seen: set[int] = set()
    for sub_idx, wid in enumerate(word_ids):
        if wid is None or wid in seen or wid >= n_tokens:
            continue
        seen.add(wid)
        tag = id2label[pred_ids[sub_idx]]
        tags[wid] = remap(tag) if remap else tag
    return tags
