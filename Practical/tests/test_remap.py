"""Unit tests for the cross-tagset remap in the baseline runner.

`_remap_tag` already bit us once: Kushtrim's id2label has the bare string
`MISC` at index 0 (instead of `O`), which crashed the original split-based
implementation. These tests pin the tolerant behaviour.
"""

from __future__ import annotations

import pytest

from src.baselines.metrics import wikiann_ids_to_bio
from src.baselines.run_baselines import _remap_tag

IDENTITY = {"PER": "PER", "ORG": "ORG", "LOC": "LOC"}


@pytest.mark.parametrize(
    ("tag", "label_map", "expected"),
    [
        ("O", IDENTITY, "O"),
        ("B-PER", IDENTITY, "B-PER"),
        ("I-LOC", IDENTITY, "I-LOC"),
        # explicit drop of a cross-tagset type
        ("B-MISC", {**IDENTITY, "MISC": "O"}, "O"),
        ("I-DATE", {**IDENTITY, "DATE": "O"}, "O"),
        # type absent from the map is dropped too
        ("B-EVENT", IDENTITY, "O"),
        # rename
        ("B-GPE", {**IDENTITY, "GPE": "LOC"}, "B-LOC"),
        ("I-GPE", {**IDENTITY, "GPE": "LOC"}, "I-LOC"),
        # malformed tags (Kushtrim's bare `MISC` at id 0) degrade to O
        ("MISC", IDENTITY, "O"),
        ("X-PER", IDENTITY, "O"),
        ("", IDENTITY, "O"),
    ],
)
def test_remap_tag(tag: str, label_map: dict[str, str], expected: str):
    assert _remap_tag(tag, label_map) == expected


def test_wikiann_ids_to_bio():
    names = ["O", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]
    assert wikiann_ids_to_bio(names, [0, 1, 2, 5]) == ["O", "B-PER", "I-PER", "B-LOC"]
