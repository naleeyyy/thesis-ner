"""Unit tests for the bootstrap CI / paired significance machinery.

The main risk here is silent disagreement with seqeval: if `sentence_counts` extracted
spans differently, every reported F1 and CI would drift from the headline metric without
anything crashing. `test_matches_seqeval` pins that.
"""

from __future__ import annotations

import numpy as np
import pytest
from seqeval.metrics import f1_score

from src.baselines import stats


def test_sentence_counts_exact_match():
    preds = [["B-PER", "I-PER", "O"]]
    refs = [["B-PER", "I-PER", "O"]]
    assert stats.sentence_counts(preds, refs).tolist() == [[1, 0, 0]]


def test_sentence_counts_boundary_error_is_both_fp_and_fn():
    # Predicting a one-token PER where gold has a two-token PER is not a partial credit
    # case: the span simply does not match, so it counts once as FP and once as FN.
    preds = [["B-PER", "O"]]
    refs = [["B-PER", "I-PER"]]
    assert stats.sentence_counts(preds, refs).tolist() == [[0, 1, 1]]


def test_sentence_counts_filters_by_entity_type():
    preds = [["B-PER", "B-LOC"]]
    refs = [["B-PER", "B-LOC"]]
    assert stats.sentence_counts(preds, refs, entity_type="LOC").tolist() == [[1, 0, 0]]
    assert stats.sentence_counts(preds, refs, entity_type="ORG").tolist() == [[0, 0, 0]]


def test_prf_from_counts():
    counts = np.array([[3, 1, 1]])
    precision, recall, f1 = stats.prf_from_counts(counts)
    assert precision == pytest.approx(0.75)
    assert recall == pytest.approx(0.75)
    assert f1 == pytest.approx(0.75)


def test_prf_from_counts_empty_is_zero_not_nan():
    precision, recall, f1 = stats.prf_from_counts(np.zeros((3, 3), dtype=np.int64))
    assert (precision, recall, f1) == (0.0, 0.0, 0.0)


def test_matches_seqeval():
    """Count-based micro-F1 must equal seqeval's, or the CIs describe a different metric."""
    preds = [
        ["B-PER", "I-PER", "O", "B-LOC"],
        ["O", "B-ORG", "O"],
        ["B-LOC", "O"],
        ["O", "O"],
    ]
    refs = [
        ["B-PER", "I-PER", "O", "B-ORG"],
        ["O", "B-ORG", "I-ORG"],
        ["B-LOC", "O"],
        ["B-PER", "O"],
    ]
    _, _, f1 = stats.prf_from_counts(stats.sentence_counts(preds, refs))
    assert f1 == pytest.approx(f1_score(refs, preds))


def test_resample_indices_is_seeded():
    a = stats.resample_indices(50, 16, seed=7)
    b = stats.resample_indices(50, 16, seed=7)
    c = stats.resample_indices(50, 16, seed=8)
    assert np.array_equal(a, b)
    assert not np.array_equal(a, c)
    assert a.shape == (16, 50)
    assert a.min() >= 0 and a.max() < 50


def test_bootstrap_ci_brackets_point_estimate():
    rng = np.random.default_rng(0)
    # 200 sentences, roughly 70% of gold spans recovered.
    counts = np.stack(
        [rng.choice([1, 0], size=200, p=[0.7, 0.3]), np.zeros(200), rng.choice([0, 1], size=200, p=[0.7, 0.3])],
        axis=1,
    ).astype(np.int64)
    _, _, point = stats.prf_from_counts(counts)
    indices = stats.resample_indices(len(counts), 500, seed=1)
    lo, hi = stats.percentile_ci(stats.bootstrap_f1(counts, indices))
    assert lo < point < hi


def test_bootstrap_f1_chunking_is_transparent():
    rng = np.random.default_rng(3)
    counts = rng.integers(0, 4, size=(80, 3), dtype=np.int64)
    indices = stats.resample_indices(80, 300, seed=2)
    assert np.allclose(
        stats.bootstrap_f1(counts, indices, chunk=7),
        stats.bootstrap_f1(counts, indices, chunk=1000),
    )


def test_paired_bootstrap_detects_a_large_gap():
    # Model A gets every span, model B gets none — as separated as it gets.
    good = np.tile(np.array([1, 0, 0], dtype=np.int64), (150, 1))
    bad = np.tile(np.array([0, 1, 1], dtype=np.int64), (150, 1))
    indices = stats.resample_indices(150, 500, seed=4)
    result = stats.paired_bootstrap("good", good, "bad", bad, indices)
    assert result.delta == pytest.approx(1.0)
    assert result.significant
    assert result.p_value < 0.01


def test_paired_bootstrap_finds_no_gap_between_identical_models():
    rng = np.random.default_rng(5)
    counts = rng.integers(0, 3, size=(120, 3), dtype=np.int64)
    indices = stats.resample_indices(120, 500, seed=6)
    result = stats.paired_bootstrap("a", counts, "b", counts.copy(), indices)
    assert result.delta == pytest.approx(0.0)
    assert not result.significant
    assert result.p_value == pytest.approx(1.0)
    # Pairing is what collapses this interval: unpaired, the two CIs would both be wide.
    assert result.ci_low == 0.0 and result.ci_high == 0.0
