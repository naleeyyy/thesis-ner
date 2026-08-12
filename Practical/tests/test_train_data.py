"""Tests for training data loading and label alignment.

The alignment is the part that silently corrupts a run: if labels drift from their words
by one subword, the model still trains and still scores plausibly, just on the wrong
targets.
"""

from __future__ import annotations

import json
from collections import UserDict

import pytest

from src.train.data import (
    IGNORE_INDEX,
    LABEL2ID,
    Example,
    collate,
    encode,
    load_gold,
    split_examples,
)


class FakeEncoding(UserDict):
    """Mimics transformers' BatchEncoding: dict-like, plus a `.word_ids()` method."""

    def __init__(self, data, word_ids):
        super().__init__(data)
        self._word_ids = word_ids

    def word_ids(self, batch_index: int = 0):
        return self._word_ids


class FakeTokenizer:
    """Splits each word into one subword per character, so alignment is checkable by eye."""

    pad_token_id = 0

    def __call__(self, tokens, is_split_into_words=True, truncation=True, max_length=256, **kw):
        ids, word_ids = [], []
        for i, tok in enumerate(tokens):
            for _ in tok:
                ids.append(len(ids) + 1)
                word_ids.append(i)
        ids, word_ids = ids[:max_length], word_ids[:max_length]
        return FakeEncoding({"input_ids": ids, "attention_mask": [1] * len(ids)}, word_ids)


def test_example_rejects_mismatched_lengths():
    with pytest.raises(ValueError, match="tokens but"):
        Example(["a", "b"], ["O"])


def test_encode_labels_only_the_first_subword_of_each_word():
    ex = Example(["ab", "c"], ["B-LOC", "O"])
    enc = encode(ex, FakeTokenizer())
    # 'ab' -> two subwords: first carries B-LOC, second is ignored.
    assert enc["labels"] == [LABEL2ID["B-LOC"], IGNORE_INDEX, LABEL2ID["O"]]


def test_encode_produces_one_label_per_subword():
    ex = Example(["abc", "de"], ["B-PER", "I-PER"])
    enc = encode(ex, FakeTokenizer())
    assert len(enc["labels"]) == len(enc["input_ids"])


def test_encode_truncation_does_not_desynchronise_labels():
    ex = Example(["aaaa", "bbbb"], ["B-LOC", "I-LOC"])
    enc = encode(ex, FakeTokenizer(), max_length=5)
    assert len(enc["labels"]) == len(enc["input_ids"]) == 5
    assert enc["labels"][0] == LABEL2ID["B-LOC"]


def test_collate_pads_labels_with_the_ignore_index():
    batch = [
        {"input_ids": [1, 2], "attention_mask": [1, 1], "labels": [0, 1]},
        {"input_ids": [3], "attention_mask": [1], "labels": [2]},
    ]
    out = collate(batch, pad_token_id=0)
    assert out["input_ids"].tolist() == [[1, 2], [3, 0]]
    assert out["labels"].tolist() == [[0, 1], [2, IGNORE_INDEX]]
    assert out["attention_mask"].tolist() == [[1, 1], [1, 0]]


def test_split_is_deterministic_and_partitions_everything():
    examples = [Example([f"w{i}"], ["O"], source=str(i)) for i in range(100)]
    a = split_examples(examples, 0.15, 0.25, seed=7)
    b = split_examples(examples, 0.15, 0.25, seed=7)
    assert [len(x) for x in a] == [60, 15, 25]
    assert [e.source for e in a[2]] == [e.source for e in b[2]], "same seed, same test set"
    seen = {e.source for part in a for e in part}
    assert len(seen) == 100, "every example lands in exactly one split"


def test_split_seed_is_independent_of_training_seed():
    # Every training seed must see the same test set, or run-to-run spread mixes
    # model variance with data variance.
    examples = [Example([f"w{i}"], ["O"], source=str(i)) for i in range(50)]
    t1 = split_examples(examples, 0.2, 0.2, seed=1)[2]
    t2 = split_examples(examples, 0.2, 0.2, seed=1)[2]
    assert [e.source for e in t1] == [e.source for e in t2]


def test_load_gold_reads_spans_and_skips_junk(tmp_path):
    path = tmp_path / "gold.jsonl"
    rows = [
        {"id": "s1", "tokens": ["Stacioni", "i", "Bramit", "ka"],
         "spans": [{"start": 0, "end": 2, "head_start": 2, "head_end": 2, "type": "LOC"}],
         "junk": False},
        {"id": "s2", "tokens": ["broken", "x"], "spans": [], "junk": True},
    ]
    path.write_text("\n".join(json.dumps(r) for r in rows))

    full = load_gold(path, "full")
    assert len(full) == 1, "junk-flagged sentences are excluded from training"
    assert full[0].tags == ["B-LOC", "I-LOC", "I-LOC", "O"]

    head = load_gold(path, "head")
    assert head[0].tags == ["O", "O", "B-LOC", "O"], "head mode narrows the span"
