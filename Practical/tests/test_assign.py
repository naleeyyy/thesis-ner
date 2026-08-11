"""Tests for batch assignment.

The failure this guards against is silent: across 20+ batches, a sentence handed out
twice inflates apparent agreement, and a sentence never handed out is simply lost. Both
look like normal data at collection time.
"""

from __future__ import annotations

import pytest

from src.annotate.assign import assigned_ids, ids_for, reserve

POOL = [{"id": f"sq-{i:04d}", "tokens": ["a", "b"]} for i in range(100)]


def _reserve(ledger, assignees, n, batch, overlap=False, condition="assisted", seed=1):
    return reserve(POOL, ledger, assignees, n, batch, condition, overlap, seed)


def test_reserves_the_requested_count():
    chosen, rows = _reserve([], ["ana"], 50, "r1")
    assert len(chosen) == 50
    assert len(rows) == 50


def test_a_second_batch_never_reissues_the_first_batch_sentences():
    _, rows1 = _reserve([], ["ana"], 50, "r1")
    chosen2, _ = _reserve(rows1, ["blerim"], 50, "r1")
    assert not (assigned_ids(rows1) & {c["id"] for c in chosen2})


def test_the_pool_is_exhausted_exactly():
    ledger = []
    for i, who in enumerate(["a", "b"]):
        _, rows = _reserve(ledger, [who], 50, f"r{i}")
        ledger += rows
    assert len(assigned_ids(ledger)) == 100


def test_running_out_of_pool_fails_loudly():
    _, rows = _reserve([], ["ana"], 100, "r1")
    with pytest.raises(SystemExit, match="only 0 unassigned"):
        _reserve(rows, ["blerim"], 50, "r2")


def test_overlap_gives_every_assignee_the_same_sentences():
    people = ["ana", "blerim", "drita"]
    chosen, rows = _reserve([], people, 20, "overlap", overlap=True)
    assert len(chosen) == 20
    assert len(rows) == 60, "one ledger row per (sentence, annotator)"
    for who in people:
        assert ids_for(rows, who) == {c["id"] for c in chosen}


def test_overlap_still_blocks_those_sentences_from_later_unique_batches():
    _, overlap_rows = _reserve([], ["ana", "blerim"], 20, "ov", overlap=True)
    chosen, _ = _reserve(overlap_rows, ["ana"], 30, "r1")
    assert not (assigned_ids(overlap_rows) & {c["id"] for c in chosen})


def test_condition_is_recorded_for_the_anchoring_experiment():
    _, rows = _reserve([], ["ana"], 10, "ov-b", overlap=True, condition="scratch")
    assert {r["condition"] for r in rows} == {"scratch"}
    assert {r["overlap"] for r in rows} == {True}


def test_same_batch_label_is_reproducible_for_a_given_seed():
    a, _ = _reserve([], ["ana"], 10, "r1", seed=7)
    b, _ = _reserve([], ["ana"], 10, "r1", seed=7)
    assert [x["id"] for x in a] == [x["id"] for x in b]


def test_chosen_batch_is_sorted_by_id():
    chosen, _ = _reserve([], ["ana"], 25, "r1")
    ids = [c["id"] for c in chosen]
    assert ids == sorted(ids)
