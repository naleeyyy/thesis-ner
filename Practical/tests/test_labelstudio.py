"""Round-trip tests for the Label Studio bridge.

The one property that must never break: tokens → task → annotation → tokens returns the
identical spans. A half-token drift here produces gold labels that look plausible and are
wrong, which no downstream check would catch.
"""

from __future__ import annotations

import json

import pytest

from src.annotate.labelstudio import (
    HEAD_LABEL,
    chars_to_tokens,
    from_annotation,
    load_export,
    to_task,
    token_offsets,
)

TOKENS = ["Stacioni", "i", "Bramit", "ka", "lidhje", "me", "Tuluzën", ","]

RECORD = {
    "id": "sq-0000",
    "tokens": TOKENS,
    "source_url": "https://sq.wikipedia.org/wiki/Bram",
    "llm_spans": [
        {"start": 0, "end": 2, "head_start": 2, "head_end": 2, "type": "LOC",
         "surface": "Stacioni i Bramit", "head_surface": "Bramit"},
        {"start": 6, "end": 6, "head_start": 6, "head_end": 6, "type": "LOC",
         "surface": "Tuluzën", "head_surface": "Tuluzën"},
    ],
}


def _annotation(result):
    return {"result": result, "completed_by": {"email": "a@example.com"}, "lead_time": 12.5}


def _region(start, end, label, text):
    return {
        "from_name": "label", "to_name": "text", "type": "labels",
        "value": {"start": start, "end": end, "text": text, "labels": [label]},
    }


# ------------------------------------------------------------------------- offsets


def test_token_offsets_reconstructs_text_and_spans():
    text, spans = token_offsets(["a", "bb", "c"])
    assert text == "a bb c"
    assert spans == [(0, 1), (2, 4), (5, 6)]
    for tok, (s, e) in zip(["a", "bb", "c"], spans, strict=True):
        assert text[s:e] == tok


def test_offsets_survive_non_ascii():
    # Albanian diacritics are multi-byte in UTF-8 but single Python characters; Label
    # Studio counts the same way, so this must line up exactly.
    text, spans = token_offsets(["Tuluzën", "Shqipërisë"])
    assert text[spans[0][0] : spans[0][1]] == "Tuluzën"
    assert text[spans[1][0] : spans[1][1]] == "Shqipërisë"


def test_chars_to_tokens_exact_and_whitespace_tolerant():
    _, spans = token_offsets(TOKENS)
    assert chars_to_tokens(spans[0][0], spans[2][1], spans) == (0, 2)
    # dragged one character past the end, into the following space
    assert chars_to_tokens(spans[0][0], spans[2][1] + 1, spans) == (0, 2)


def test_chars_to_tokens_rejects_a_split_token():
    _, spans = token_offsets(TOKENS)
    assert chars_to_tokens(spans[0][0], spans[0][1] - 2, spans) is None


def test_chars_to_tokens_rejects_an_empty_selection():
    _, spans = token_offsets(TOKENS)
    assert chars_to_tokens(0, 0, spans) is None


# --------------------------------------------------------------------- task export


def test_assisted_task_carries_predictions_and_scratch_does_not():
    assisted = to_task(RECORD, with_predictions=True)
    scratch = to_task(RECORD, with_predictions=False)
    assert assisted["data"]["text"] == scratch["data"]["text"], "conditions must be comparable"
    assert "predictions" not in scratch
    assert len(assisted["predictions"][0]["result"]) == 3  # 2 entities + 1 HEAD


def test_head_region_emitted_only_where_it_narrows_the_span():
    labels = [
        r["value"]["labels"][0] for r in to_task(RECORD, True)["predictions"][0]["result"]
    ]
    # Tuluzën's head equals its full span, so it gets no HEAD region.
    assert labels.count(HEAD_LABEL) == 1


def test_prediction_offsets_select_the_intended_text():
    task = to_task(RECORD, True)
    text = task["data"]["text"]
    for region in task["predictions"][0]["result"]:
        v = region["value"]
        assert text[v["start"] : v["end"]] == v["text"]


# -------------------------------------------------------------------- round trip


def test_round_trip_preserves_spans_exactly():
    task = to_task(RECORD, True)
    regions = task["predictions"][0]["result"]
    record, issues = from_annotation(task, _annotation(regions), TOKENS)
    assert issues == []
    got = [(s["start"], s["end"], s["head_start"], s["head_end"], s["type"]) for s in record["spans"]]
    want = [(s["start"], s["end"], s["head_start"], s["head_end"], s["type"]) for s in RECORD["llm_spans"]]
    assert got == want


def test_round_trip_recovers_surfaces():
    task = to_task(RECORD, True)
    record, _ = from_annotation(task, _annotation(task["predictions"][0]["result"]), TOKENS)
    assert record["spans"][0]["surface"] == "Stacioni i Bramit"
    assert record["spans"][0]["head_surface"] == "Bramit"


def test_entity_without_head_defaults_to_full_span():
    _, spans = token_offsets(TOKENS)
    regions = [_region(spans[0][0], spans[2][1], "LOC", "Stacioni i Bramit")]
    record, issues = from_annotation({"data": {"sentence_id": "x"}}, _annotation(regions), TOKENS)
    assert issues == []
    span = record["spans"][0]
    assert (span["head_start"], span["head_end"]) == (span["start"], span["end"])


def test_annotator_and_lead_time_are_captured():
    # Time per sentence is one of the assisted-vs-scratch measurements, so losing it
    # here would quietly remove a result from the thesis.
    task = to_task(RECORD, True)
    record, _ = from_annotation(task, _annotation(task["predictions"][0]["result"]), TOKENS)
    assert record["annotator"] == "a@example.com"
    assert record["lead_time_s"] == 12.5


# ------------------------------------------------------------------ sentence flags


def _choices(*values):
    return {"from_name": "flags", "to_name": "text", "type": "choices",
            "value": {"choices": list(values)}}


def test_junk_flag_is_captured():
    record, issues = from_annotation(
        {"data": {"sentence_id": "x"}}, _annotation([_choices("junk")]), TOKENS
    )
    assert record["junk"] is True
    assert record["flags"] == ["junk"]
    assert issues == []


def test_no_flags_means_not_junk():
    task = to_task(RECORD, True)
    record, _ = from_annotation(task, _annotation(task["predictions"][0]["result"]), TOKENS)
    assert record["junk"] is False
    assert record["flags"] == []


def test_unsure_flag_does_not_mark_junk():
    record, _ = from_annotation(
        {"data": {"sentence_id": "x"}}, _annotation([_choices("unsure")]), TOKENS
    )
    assert record["junk"] is False
    assert "unsure" in record["flags"]


def test_junk_with_entities_is_contradictory_and_reported():
    _, spans = token_offsets(TOKENS)
    regions = [_region(spans[6][0], spans[6][1], "LOC", "Tuluzën"), _choices("junk")]
    record, issues = from_annotation({"data": {"sentence_id": "x"}}, _annotation(regions), TOKENS)
    assert record["junk"] is True
    assert any("flagged junk" in i.reason for i in issues)


# ------------------------------------------------------------------ error reporting


def test_misaligned_span_is_reported_not_snapped():
    _, spans = token_offsets(TOKENS)
    regions = [_region(spans[0][0], spans[0][1] - 3, "LOC", "Stacio")]
    record, issues = from_annotation({"data": {"sentence_id": "x"}}, _annotation(regions), TOKENS)
    assert record["spans"] == []
    assert "token boundaries" in issues[0].reason


def test_overlapping_entities_are_rejected():
    _, spans = token_offsets(TOKENS)
    regions = [
        _region(spans[0][0], spans[2][1], "LOC", "Stacioni i Bramit"),
        _region(spans[2][0], spans[2][1], "ORG", "Bramit"),
    ]
    record, issues = from_annotation({"data": {"sentence_id": "x"}}, _annotation(regions), TOKENS)
    assert len(record["spans"]) == 1
    assert any("overlapping" in i.reason for i in issues)


def test_orphan_head_is_reported():
    _, spans = token_offsets(TOKENS)
    regions = [_region(spans[6][0], spans[6][1], HEAD_LABEL, "Tuluzën")]
    _, issues = from_annotation({"data": {"sentence_id": "x"}}, _annotation(regions), TOKENS)
    assert any("not inside any entity" in i.reason for i in issues)


def test_unknown_label_is_reported():
    _, spans = token_offsets(TOKENS)
    regions = [_region(spans[6][0], spans[6][1], "MISC", "Tuluzën")]
    record, issues = from_annotation({"data": {"sentence_id": "x"}}, _annotation(regions), TOKENS)
    assert record["spans"] == []
    assert "MISC" in issues[0].reason


# --------------------------------------------------------------------- export file


def test_load_export_handles_multiple_annotators_and_skips_cancelled(tmp_path):
    task = to_task(RECORD, True)
    regions = task["predictions"][0]["result"]
    task["annotations"] = [
        {"result": regions, "completed_by": {"email": "a@x.com"}},
        {"result": regions, "completed_by": {"email": "b@x.com"}},
        {"result": [], "completed_by": {"email": "c@x.com"}, "was_cancelled": True},
    ]
    path = tmp_path / "export.json"
    path.write_text(json.dumps([task]), encoding="utf-8")

    result = load_export(path, {"sq-0000": TOKENS})
    assert len(result.records) == 2, "one record per annotator, cancelled skipped"
    assert {r["annotator"] for r in result.records} == {"a@x.com", "b@x.com"}


def test_load_export_reports_an_unknown_sentence_id(tmp_path):
    path = tmp_path / "export.json"
    path.write_text(json.dumps([{"data": {"sentence_id": "nope"}, "annotations": []}]), encoding="utf-8")
    result = load_export(path, {"sq-0000": TOKENS})
    assert result.records == []
    assert "not found" in result.issues[0].reason


@pytest.mark.parametrize("tokens", [["a"], ["a", "b"], ["Tuluzën", ",", "Karkasonën"]])
def test_round_trip_is_lossless_for_a_full_sentence_span(tokens):
    _, spans = token_offsets(tokens)
    regions = [_region(spans[0][0], spans[-1][1], "LOC", " ".join(tokens))]
    record, issues = from_annotation({"data": {"sentence_id": "x"}}, _annotation(regions), tokens)
    assert issues == []
    assert (record["spans"][0]["start"], record["spans"][0]["end"]) == (0, len(tokens) - 1)
