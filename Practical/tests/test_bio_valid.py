"""BIO sanity check for any labeled JSONL under data/labeled/.

Catches two common annotation bugs:
  1. `I-X` not preceded by `B-X` or `I-X` of the same type.
  2. `tokens` and `ner_tags` of different length.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

DATA_DIR = Path(__file__).resolve().parents[1] / "data" / "labeled"
ALLOWED_TAGS = {"O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"}


def _iter_labeled_files():
    if not DATA_DIR.exists():
        return []
    return sorted(DATA_DIR.glob("*.jsonl"))


@pytest.mark.parametrize("path", _iter_labeled_files(), ids=lambda p: p.name)
def test_bio_valid(path: Path):
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not lines:
        pytest.skip(f"{path.name} is empty")
    for rec in lines:
        tokens = rec["tokens"]
        tags = rec["ner_tags"]
        assert len(tokens) == len(tags), f"{rec['id']}: tokens/tags length mismatch"
        assert all(t in ALLOWED_TAGS for t in tags), f"{rec['id']}: unknown tag(s) in {tags}"
        last_type = None
        for i, t in enumerate(tags):
            if t == "O":
                last_type = None
                continue
            prefix, etype = t.split("-", 1)
            if prefix == "B":
                last_type = etype
            elif prefix == "I":
                assert last_type == etype, (
                    f"{rec['id']}: I-{etype} at index {i} without preceding B-{etype}"
                )
                last_type = etype


def test_no_labeled_files_is_ok():
    """If nothing's been labeled yet, just verify the directory exists."""
    assert DATA_DIR.exists()
