"""Tests for NER error classification.

The categories must be mutually exclusive and jointly exhaustive: every mismatch between
gold and prediction has to land in exactly one bucket, or the percentages in the report
will not add up and the analysis will quietly mislead.
"""

from __future__ import annotations

from src.baselines.errors import classify, overlaps


def kinds(gold, pred):
    return sorted(e["kind"] for e in classify(gold, pred))


def test_exact_match_produces_no_errors():
    tags = ["B-LOC", "I-LOC", "O"]
    assert classify(tags, tags) == []


def test_missed_entity():
    assert kinds(["B-LOC", "O"], ["O", "O"]) == ["missed"]


def test_spurious_entity():
    assert kinds(["O", "O"], ["B-LOC", "O"]) == ["spurious"]


def test_type_error_keeps_the_extent():
    errs = classify(["B-LOC", "I-LOC"], ["B-ORG", "I-ORG"])
    assert [e["kind"] for e in errs] == ["type"]
    assert errs[0]["gold"][1:] == errs[0]["pred"][1:], "same tokens, different type"


def test_boundary_error_when_extents_differ():
    assert kinds(["B-LOC", "I-LOC", "O"], ["B-LOC", "O", "O"]) == ["boundary"]


def test_a_type_error_is_not_reported_as_a_boundary_error():
    """With both a same-extent and a ragged candidate, the same-extent one must win."""
    gold = ["B-LOC", "I-LOC", "O", "O"]
    pred = ["B-ORG", "I-ORG", "O", "O"]
    assert kinds(gold, pred) == ["type"]


def test_non_overlapping_spans_are_missed_plus_spurious_not_boundary():
    gold = ["B-LOC", "O", "O", "O"]
    pred = ["O", "O", "B-LOC", "O"]
    assert kinds(gold, pred) == ["missed", "spurious"]


def test_every_mismatch_is_categorised_exactly_once():
    gold = ["B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG"]
    pred = ["B-PER", "O", "O", "B-ORG", "O", "O"]
    errs = classify(gold, pred)
    # 3 gold, 2 predicted, no exact matches. Both predictions get partnered, so there is
    # no spurious span: PER is a boundary error, LOC->ORG a type error, ORG unmatched.
    assert sorted(e["kind"] for e in errs) == ["boundary", "missed", "type"]


def test_all_four_categories_can_occur_together():
    gold = ["B-PER", "I-PER", "O", "B-LOC", "O", "B-ORG", "O", "O"]
    pred = ["B-PER", "O", "O", "B-ORG", "O", "O", "O", "B-LOC"]
    errs = classify(gold, pred)
    assert sorted(e["kind"] for e in errs) == ["boundary", "missed", "spurious", "type"]


def test_one_prediction_cannot_absorb_two_gold_spans():
    gold = ["B-LOC", "O", "B-LOC", "O"]
    pred = ["B-LOC", "I-LOC", "I-LOC", "O"]
    errs = classify(gold, pred)
    assert sum(1 for e in errs if e["kind"] == "missed") == 1, "second gold span must not vanish"


def test_overlaps_is_type_agnostic():
    assert overlaps(("LOC", 0, 2), ("ORG", 2, 4))
    assert not overlaps(("LOC", 0, 1), ("LOC", 2, 3))
