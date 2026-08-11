"""Tests for inter-annotator agreement.

The κ implementations are checked against hand-computable cases and against the textbook
worked example, because a subtly wrong chance-correction term produces plausible-looking
numbers that would go straight into the thesis unnoticed.
"""

from __future__ import annotations

import math

import pytest

from src.annotate.agreement import (
    cohen_kappa,
    complete_subset,
    disagreements,
    fleiss_kappa,
    group,
    junk_agreement,
    span_f1,
    summarize,
    token_ratings,
)

TOKENS = ["Stacioni", "i", "Bramit", "ka", "lidhje", "me", "Tuluzën", "."]


def rec(sid, annotator, spans, tokens=None, junk=False):
    return {
        "id": sid,
        "tokens": tokens or TOKENS,
        "spans": spans,
        "annotator": annotator,
        "junk": junk,
        "flags": ["junk"] if junk else [],
    }


def span(start, end, etype="LOC", head=None):
    hs, he = head or (start, end)
    return {"start": start, "end": end, "head_start": hs, "head_end": he, "type": etype}


# --------------------------------------------------------------------- kappa maths


def test_fleiss_perfect_agreement_with_two_categories():
    ratings = [["A", "A"], ["B", "B"], ["A", "A"], ["B", "B"]]
    assert fleiss_kappa(ratings) == pytest.approx(1.0)


def test_fleiss_unanimous_single_category_returns_one_not_nan():
    # P_e == 1 makes kappa formally undefined; NaN here would poison the whole report.
    assert fleiss_kappa([["O", "O"], ["O", "O"]]) == 1.0


def test_fleiss_total_disagreement_is_negative():
    ratings = [["A", "B"], ["B", "A"], ["A", "B"], ["B", "A"]]
    assert fleiss_kappa(ratings) < 0


def test_fleiss_matches_the_textbook_worked_example():
    # Fleiss (1971) 10 subjects x 14 raters, 5 categories; published kappa = 0.210.
    table = [
        [0, 0, 0, 0, 14], [0, 2, 6, 4, 2], [0, 0, 3, 5, 6], [0, 3, 9, 2, 0],
        [2, 2, 8, 1, 1], [7, 7, 0, 0, 0], [3, 2, 6, 3, 0], [2, 5, 3, 2, 2],
        [6, 5, 2, 1, 0], [0, 2, 2, 3, 7],
    ]
    ratings = [
        [str(cat) for cat, n in enumerate(row) for _ in range(n)] for row in table
    ]
    assert fleiss_kappa(ratings) == pytest.approx(0.210, abs=0.002)


def test_fleiss_needs_at_least_two_raters():
    assert math.isnan(fleiss_kappa([["A"], ["B"]]))


def test_cohen_perfect_and_chance():
    assert cohen_kappa(["A", "B", "A"], ["A", "B", "A"]) == pytest.approx(1.0)
    # Both annotators always say A: no variance, so no disagreement to correct for.
    assert cohen_kappa(["A", "A"], ["A", "A"]) == 1.0


def test_cohen_matches_a_hand_computed_case():
    # 2x2: agree on 20+15, disagree on 5+10. p_o = 0.70, p_e = 0.50 -> kappa = 0.40
    a = ["Y"] * 25 + ["N"] * 25
    b = ["Y"] * 20 + ["N"] * 5 + ["Y"] * 10 + ["N"] * 15
    assert cohen_kappa(a, b) == pytest.approx(0.40, abs=1e-9)


# ------------------------------------------------------------------- data assembly


def test_group_collects_annotators_per_sentence():
    grouped = group([rec("s1", "ana", []), rec("s1", "blerim", []), rec("s2", "ana", [])])
    assert set(grouped["s1"].by_annotator) == {"ana", "blerim"}
    assert set(grouped["s2"].by_annotator) == {"ana"}


def test_complete_subset_excludes_partially_annotated_sentences():
    # Fleiss needs a constant rater count; mixing these in changes what it measures.
    grouped = group([rec("s1", "ana", []), rec("s1", "blerim", []), rec("s2", "ana", [])])
    subset = complete_subset(grouped, ["ana", "blerim"])
    assert [s.sentence_id for s in subset] == ["s1"]


def test_token_ratings_shape():
    grouped = group([rec("s1", "ana", []), rec("s1", "blerim", [])])
    rows = token_ratings(complete_subset(grouped, ["ana", "blerim"]), ["ana", "blerim"], "full")
    assert len(rows) == len(TOKENS)
    assert all(len(r) == 2 for r in rows)


# ----------------------------------------------------------------- the two views


def test_full_and_head_views_can_disagree_differently():
    """Two annotators agreeing on the phrase but differing on the head.

    Full-span agreement is perfect; head agreement is not. Reporting only one view would
    hide exactly the boundary question the dual annotation exists to answer.
    """
    a = rec("s1", "ana", [span(0, 2, "LOC", head=(2, 2))])
    b = rec("s1", "blerim", [span(0, 2, "LOC", head=(0, 2))])
    grouped = group([a, b])
    full = summarize(grouped, ["ana", "blerim"], "full")
    head = summarize(grouped, ["ana", "blerim"], "head")
    assert full["pct_tokens_disputed"] == 0.0
    assert head["pct_tokens_disputed"] > 0.0


def test_span_f1_is_one_for_identical_annotations():
    a = rec("s1", "ana", [span(6, 6)])
    b = rec("s1", "blerim", [span(6, 6)])
    sentences = complete_subset(group([a, b]), ["ana", "blerim"])
    assert span_f1(sentences, "ana", "blerim", "full") == pytest.approx(1.0)


def test_span_f1_is_zero_when_spans_do_not_match():
    a = rec("s1", "ana", [span(0, 2)])
    b = rec("s1", "blerim", [span(6, 6)])
    sentences = complete_subset(group([a, b]), ["ana", "blerim"])
    assert span_f1(sentences, "ana", "blerim", "full") == pytest.approx(0.0)


def test_span_f1_is_harsher_than_token_kappa_on_a_boundary_slip():
    # One token off: token-level agreement stays high, span-level F1 collapses to 0.
    a = rec("s1", "ana", [span(0, 2)])
    b = rec("s1", "blerim", [span(0, 1)])
    grouped = group([a, b])
    view = summarize(grouped, ["ana", "blerim"], "full")
    assert view["pairwise"]["ana vs blerim"]["span_f1"] == 0.0
    assert view["pairwise"]["ana vs blerim"]["cohen_kappa"] > 0.5


# ------------------------------------------------------------------ disagreements


def test_disagreements_are_ranked_worst_first():
    grouped = group([
        rec("s1", "ana", [span(6, 6)]), rec("s1", "blerim", [span(6, 6)]),
        rec("s2", "ana", [span(0, 2)]), rec("s2", "blerim", []),
    ])
    rows = disagreements(complete_subset(grouped, ["ana", "blerim"]), ["ana", "blerim"], "full")
    assert [d.sentence_id for d in rows] == ["s2"], "agreed sentences are omitted"
    assert rows[0].n_tokens_disputed == 3


def test_disagreement_detail_names_the_token_and_both_labels():
    grouped = group([rec("s1", "ana", [span(6, 6)]), rec("s1", "blerim", [])])
    rows = disagreements(complete_subset(grouped, ["ana", "blerim"]), ["ana", "blerim"], "full")
    item = rows[0].detail[0]
    assert item["token"] == "Tuluzën"
    assert item["tags"] == {"ana": "B-LOC", "blerim": "O"}


# -------------------------------------------------------------------- junk flag


def test_junk_agreement_counts_unanimous_and_partial_flags():
    grouped = group([
        rec("s1", "ana", [], junk=True), rec("s1", "blerim", [], junk=True),
        rec("s2", "ana", [], junk=True), rec("s2", "blerim", [], junk=False),
    ])
    out = junk_agreement(grouped, ["ana", "blerim"])
    assert out["flagged_by_someone"] == 2
    assert out["flagged_by_everyone"] == 1


def test_summarize_reports_nothing_when_no_sentence_is_shared():
    grouped = group([rec("s1", "ana", []), rec("s2", "blerim", [])])
    assert summarize(grouped, ["ana", "blerim"], "full")["n_sentences"] == 0
