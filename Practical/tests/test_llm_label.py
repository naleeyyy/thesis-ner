"""Unit tests for span validation and BIO conversion.

These run without touching the API. The point is that a malformed model response can
never produce a malformed BIO sequence downstream — `validate_spans` is the only thing
standing between the LLM and the annotation tool.
"""

from __future__ import annotations

import pytest

from src.annotate.llm_label import (
    cache_key,
    spans_to_bio,
    usd_cost,
    validate_spans,
)
from src.annotate.prompt import render_tokens

TOKENS = ["Universiteti", "i", "Prishtinës", "ndodhet", "në", "Kosovë", "."]


def _spans(*triples):
    """Spans whose head equals the full span — the no-common-noun case."""
    return [
        {"start": s, "end": e, "head_start": s, "head_end": e, "type": t, "text": ""}
        for s, e, t in triples
    ]


def test_accepts_a_clean_multi_token_span():
    result = validate_spans(_spans((0, 2, "ORG"), (5, 5, "LOC")), TOKENS)
    assert not result.rejections
    assert [s["surface"] for s in result.spans] == ["Universiteti i Prishtinës", "Kosovë"]


def test_rejects_out_of_range_index():
    result = validate_spans(_spans((5, 99, "LOC")), TOKENS)
    assert result.spans == []
    assert result.rejections[0].reason == "index out of range"


def test_rejects_inverted_span():
    result = validate_spans(_spans((4, 1, "LOC")), TOKENS)
    assert [r.reason for r in result.rejections] == ["start after end"]


def test_rejects_unknown_type():
    # MISC is the tag most likely to leak in — WikiANN has no MISC and neither do we.
    result = validate_spans(_spans((0, 0, "MISC")), TOKENS)
    assert result.spans == []
    assert "MISC" in result.rejections[0].reason


def test_rejects_non_integer_index():
    result = validate_spans([{"start": "0", "end": 1, "type": "ORG", "text": ""}], TOKENS)
    assert [r.reason for r in result.rejections] == ["non-integer index"]


def test_first_span_wins_an_overlap():
    result = validate_spans(_spans((0, 2, "ORG"), (2, 2, "LOC")), TOKENS)
    assert len(result.spans) == 1
    assert result.spans[0]["type"] == "ORG"
    assert result.rejections[0].reason == "overlaps an earlier span"


def test_surface_mismatch_is_counted_but_span_is_kept():
    # Indices are authoritative; a wrong `text` signals the model miscounted, which we
    # want to measure rather than silently correct.
    spans = [{"start": 5, "end": 5, "type": "LOC", "text": "Shqipëri"}]
    result = validate_spans(spans, TOKENS)
    assert len(result.spans) == 1
    assert result.surface_mismatches == 1


def test_matching_surface_is_not_a_mismatch():
    spans = [{"start": 5, "end": 5, "type": "LOC", "text": "Kosovë"}]
    assert validate_spans(spans, TOKENS).surface_mismatches == 0


def test_spans_to_bio():
    spans = validate_spans(_spans((0, 2, "ORG"), (5, 5, "LOC")), TOKENS).spans
    assert spans_to_bio(spans, len(TOKENS)) == [
        "B-ORG", "I-ORG", "I-ORG", "O", "O", "B-LOC", "O",
    ]


def test_spans_to_bio_empty_sentence_is_all_o():
    assert spans_to_bio([], 4) == ["O", "O", "O", "O"]


# --------------------------------------------------------- nested head / full views


def _nested(start, end, head_start, head_end, etype="ORG"):
    return [
        {
            "start": start, "end": end,
            "head_start": head_start, "head_end": head_end,
            "type": etype, "text": "",
        }
    ]


def test_head_is_preserved_and_surfaced():
    # `Universiteti i Prishtinës` — full phrase, head is the proper name alone.
    result = validate_spans(_nested(0, 2, 2, 2), TOKENS)
    span = result.spans[0]
    assert span["surface"] == "Universiteti i Prishtinës"
    assert span["head_surface"] == "Prishtinës"
    assert result.head_fallbacks == 0


def test_the_two_views_differ_exactly_where_the_head_is_narrower():
    spans = validate_spans(_nested(0, 2, 2, 2), TOKENS).spans
    assert spans_to_bio(spans, len(TOKENS), "full")[:3] == ["B-ORG", "I-ORG", "I-ORG"]
    assert spans_to_bio(spans, len(TOKENS), "head")[:3] == ["O", "O", "B-ORG"]


@pytest.mark.parametrize(
    ("head_start", "head_end"),
    [
        (1, 0),   # inverted
        (0, 5),   # extends past the full span
        (4, 4),   # entirely outside the full span
        (None, 2),  # missing
    ],
)
def test_bad_head_falls_back_to_full_span_without_losing_the_entity(head_start, head_end):
    result = validate_spans(_nested(0, 2, head_start, head_end), TOKENS)
    assert len(result.spans) == 1, "the entity must survive a bad head"
    assert result.head_fallbacks == 1
    span = result.spans[0]
    assert (span["head_start"], span["head_end"]) == (span["start"], span["end"])


def test_head_view_is_also_well_formed_bio_after_fallback():
    spans = validate_spans(_nested(0, 2, 9, 9), TOKENS).spans
    assert spans_to_bio(spans, len(TOKENS), "head")[:3] == ["B-ORG", "I-ORG", "I-ORG"]


def test_spans_to_bio_rejects_an_unknown_mode():
    with pytest.raises(ValueError, match="mode"):
        spans_to_bio([], 3, mode="heads")


def test_bio_output_is_always_well_formed():
    """Whatever the model returns, the emitted tags are a valid BIO sequence."""
    garbage = [
        {"start": -1, "end": 2, "type": "LOC", "text": ""},
        {"start": 0, "end": 1, "type": "PER", "text": ""},
        {"start": 1, "end": 3, "type": "ORG", "text": ""},  # overlaps the PER
        {"start": 3, "end": 3, "type": "MISC", "text": ""},
    ]
    tags = spans_to_bio(validate_spans(garbage, TOKENS).spans, len(TOKENS))
    assert len(tags) == len(TOKENS)
    for i, tag in enumerate(tags):
        if tag.startswith("I-"):
            prev = tags[i - 1]
            assert prev in (tag, "B-" + tag[2:]), f"orphan {tag} at {i}"


def test_cache_key_is_sensitive_to_tokens_and_model():
    a = cache_key("claude-opus-5", TOKENS)
    assert a == cache_key("claude-opus-5", TOKENS)
    assert a != cache_key("claude-opus-5", TOKENS[:-1])
    assert a != cache_key("claude-sonnet-5", TOKENS)


def test_render_tokens_numbers_from_zero():
    rendered = render_tokens(["a", "b"])
    assert "0\ta" in rendered and "1\tb" in rendered
    assert "(2 tokens)" in rendered


def test_usd_cost_prices_cache_reads_cheaper_than_fresh_input():
    fresh = usd_cost(
        "claude-opus-5",
        {"input_tokens": 1000, "output_tokens": 0,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0},
    )
    cached = usd_cost(
        "claude-opus-5",
        {"input_tokens": 0, "output_tokens": 0,
         "cache_creation_input_tokens": 0, "cache_read_input_tokens": 1000},
    )
    assert cached == fresh * 0.1


def test_promotional_rate_applies_before_the_end_date():
    import datetime

    from src.annotate.llm_label import rates_for

    assert rates_for("claude-sonnet-5", datetime.date(2026, 8, 11)) == (2.0, 10.0)


def test_list_rate_resumes_after_the_promotion_ends():
    import datetime

    from src.annotate.llm_label import rates_for

    assert rates_for("claude-sonnet-5", datetime.date(2026, 9, 1)) == (3.0, 15.0)


def test_models_without_a_promotion_always_use_list_rate():
    import datetime

    from src.annotate.llm_label import rates_for

    assert rates_for("claude-opus-5", datetime.date(2026, 8, 11)) == (5.0, 25.0)


def test_reported_cost_matches_the_billed_dashboard_figure():
    """Regression for the ~33% overstatement the list-rate table produced on Sonnet 5."""
    usage = {
        "input_tokens": 548, "output_tokens": 369,
        "cache_creation_input_tokens": 1452, "cache_read_input_tokens": 5808,
    }
    import datetime

    from src.annotate.llm_label import rates_for

    assert rates_for("claude-sonnet-5", datetime.date(2026, 8, 11)) == (2.0, 10.0)
    # $0.0096 billed vs $0.0144 under the old list-rate table.
    assert usd_cost("claude-sonnet-5", usage) == pytest.approx(0.0096, abs=5e-4)


def test_usd_cost_unknown_model_is_zero_not_an_error():
    assert usd_cost("some-future-model", {
        "input_tokens": 100, "output_tokens": 100,
        "cache_creation_input_tokens": 0, "cache_read_input_tokens": 0,
    }) == 0.0
