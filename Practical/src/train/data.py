"""Load and encode token-classification data from either source.

Two sources have to look identical downstream: the hand-annotated gold JSONL and
WikiANN-sq. WikiANN is what proves the training pipeline works before any gold data
exists, and it stays useful afterwards as supplementary training data — the gold set's
irreplaceable job is being a clean test set, not bulk training signal.

Both arrive as `Example(tokens, tags)` with WikiANN's tagset, so nothing downstream needs
to know where a sentence came from.
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

from ..annotate.llm_label import spans_to_bio

LABELS = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
LABEL2ID = {label: i for i, label in enumerate(LABELS)}
ID2LABEL = dict(enumerate(LABELS))

# Subword positions that are not the first of their word. Cross-entropy ignores this
# index, so the loss is computed on exactly one position per word — matching how
# predictions are read back at eval time.
IGNORE_INDEX = -100


@dataclass
class Example:
    tokens: list[str]
    tags: list[str]
    source: str = ""

    def __post_init__(self):
        if len(self.tokens) != len(self.tags):
            raise ValueError(f"{len(self.tokens)} tokens but {len(self.tags)} tags")


def load_gold(path: Path, mode: str = "full") -> list[Example]:
    """Adjudicated gold JSONL → examples.

    `mode` picks the boundary convention: `full` tags the whole phrase
    (`Stacioni i Bramit`), `head` only the proper-name core (`Bramit`). Training and
    evaluating under both is the point of having annotated both.
    """
    out: list[Example] = []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            if rec.get("junk"):
                continue
            tokens = rec["tokens"]
            tags = spans_to_bio(rec["spans"], len(tokens), mode)
            out.append(Example(tokens, tags, source=rec.get("id", "gold")))
    return out


def load_wikiann(split: str) -> list[Example]:
    from datasets import load_dataset

    ds = load_dataset("unimelb-nlp/wikiann", "sq", split=split)
    names = ds.features["ner_tags"].feature.names
    return [
        Example(row["tokens"], [names[i] for i in row["ner_tags"]], source=f"wikiann/{split}")
        for row in ds
    ]


def split_examples(
    examples: list[Example], dev_frac: float, test_frac: float, seed: int
) -> tuple[list[Example], list[Example], list[Example]]:
    """Shuffle once with a fixed seed, then slice.

    The split is deliberately independent of the training seed: every seed must see the
    same test set, or the run-to-run spread mixes model variance with data variance and
    stops meaning anything.
    """
    shuffled = list(examples)
    random.Random(seed).shuffle(shuffled)
    n = len(shuffled)
    n_test = int(round(n * test_frac))
    n_dev = int(round(n * dev_frac))
    return shuffled[n_test + n_dev :], shuffled[n_test : n_test + n_dev], shuffled[:n_test]


def encode(example: Example, tokenizer, max_length: int = 256) -> dict:
    """Tokenize into subwords and align one label per word to its first subword."""
    enc = tokenizer(
        example.tokens,
        is_split_into_words=True,
        truncation=True,
        max_length=max_length,
    )
    word_ids = enc.word_ids()
    labels: list[int] = []
    seen: set[int] = set()
    for wid in word_ids:
        if wid is None or wid in seen:
            labels.append(IGNORE_INDEX)
        else:
            seen.add(wid)
            labels.append(LABEL2ID[example.tags[wid]])
    enc["labels"] = labels
    return dict(enc)


def collate(batch: list[dict], pad_token_id: int) -> dict:
    """Pad a batch to its longest member; labels pad with the ignore index."""
    import torch

    width = max(len(item["input_ids"]) for item in batch)
    out: dict[str, list[list[int]]] = {"input_ids": [], "attention_mask": [], "labels": []}
    for item in batch:
        pad = width - len(item["input_ids"])
        out["input_ids"].append(item["input_ids"] + [pad_token_id] * pad)
        out["attention_mask"].append(item["attention_mask"] + [0] * pad)
        out["labels"].append(item["labels"] + [IGNORE_INDEX] * pad)
    return {k: torch.tensor(v) for k, v in out.items()}
