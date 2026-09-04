"""Tests for majority-vote adjudication.

The round-trip property matters most: `bio_to_spans` is the inverse of `spans_to_bio`,
and a bug there would silently corrupt every multiply-annotated sentence in the gold set.
"""

from __future__ import annotations

from src.annotate.adjudicate import bio_to_spans, majority_tags
from src.annotate.llm_label import spans_to_bio


def span(start, end, typ="LOC"):
    return {"start": start, "end": end, "head_start": start, "head_end": end, "type": typ}


def test_round_trip_single_and_multi_token():
    for spans in ([span(0, 1)], [span(2, 2, "PER"), span(4, 6, "ORG")], []):
        tags = spans_to_bio(spans, 8, "full")
        got = [(s["start"], s["end"], s["type"]) for s in bio_to_spans(tags)]
        assert got == [(s["start"], s["end"], s["type"]) for s in spans]


def test_adjacent_same_type_entities_do_not_merge():
    # Two B- tags in a row are two entities; merging them would silently join
    # neighbouring names into one span.
    assert len(bio_to_spans(["B-LOC", "B-LOC", "O"])) == 2


def test_orphan_i_tag_is_treated_as_a_start():
    # Malformed input from a vote can produce I- without B-. Recovering the entity is
    # more conservative than dropping it.
    got = bio_to_spans(["O", "I-PER", "O"])
    assert got == [{"start": 1, "end": 1, "type": "PER"}]


def test_majority_vote_picks_the_plurality():
    tags, disputed = majority_tags([["B-LOC", "O"], ["B-LOC", "O"], ["O", "O"]])
    assert tags == ["B-LOC", "O"]
    assert disputed == [0]


def test_tie_resolves_to_O_and_is_flagged():
    # An even split means no majority reading exists; inventing one would manufacture
    # agreement nobody expressed.
    tags, disputed = majority_tags([["B-LOC"], ["O"]])
    assert tags == ["O"]
    assert disputed == [0]


def test_unanimous_agreement_reports_no_dispute():
    tags, disputed = majority_tags([["B-ORG", "I-ORG"]] * 4)
    assert tags == ["B-ORG", "I-ORG"]
    assert disputed == []
