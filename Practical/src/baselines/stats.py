"""Bootstrap confidence intervals and paired significance tests for span-level F1.

Why bootstrap and not multiple seeds: baseline inference over a frozen checkpoint is
deterministic (`logits.argmax(-1)`), so re-running with a different seed reproduces the
predictions exactly and seed variance is identically zero. The quantity that *is*
uncertain is the test set — WikiANN-sq test is a 1000-sentence sample of Albanian
Wikipedia, and a different sample would have given a different F1. Resampling sentences
with replacement estimates that sampling variability directly.

Model comparisons use a *paired* bootstrap: both models are scored on the same resampled
sentence indices, so the shared difficulty of a resample cancels out of the difference.
That is substantially more powerful than checking whether two independent CIs overlap,
which is the usual mistake.

Entities never cross sentence boundaries, so micro-averaged P/R/F1 over any subset of
sentences is exactly the sum of that subset's per-sentence (tp, fp, fn) rows. Caching
those rows once turns each of the thousands of resamples into an integer sum.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from seqeval.metrics.sequence_labeling import get_entities

# Number of resamples. 2000 is comfortably above the ~1000 needed for stable 95%
# percentile bounds, and the whole bootstrap still runs in well under a second.
DEFAULT_RESAMPLES = 2000
DEFAULT_ALPHA = 0.05
DEFAULT_SEED = 12345


def sentence_counts(
    preds: list[list[str]],
    refs: list[list[str]],
    entity_type: str | None = None,
) -> np.ndarray:
    """Per-sentence (tp, fp, fn) span counts, shape (n_sentences, 3).

    Spans are extracted with seqeval's own `get_entities` so these counts agree with
    the headline seqeval metrics rather than approximating them. Pass `entity_type`
    to restrict to one class (per-entity F1).
    """
    rows = np.zeros((len(refs), 3), dtype=np.int64)
    for i, (pred, ref) in enumerate(zip(preds, refs, strict=True)):
        pset = set(get_entities(pred))
        gset = set(get_entities(ref))
        if entity_type is not None:
            pset = {e for e in pset if e[0] == entity_type}
            gset = {e for e in gset if e[0] == entity_type}
        rows[i] = (len(pset & gset), len(pset - gset), len(gset - pset))
    return rows


def prf_from_counts(counts: np.ndarray) -> tuple[float, float, float]:
    """Micro precision, recall, F1 from a (n, 3) count matrix or a single (3,) row."""
    tp, fp, fn = counts.sum(axis=0) if counts.ndim == 2 else counts
    precision = float(tp / (tp + fp)) if tp + fp else 0.0
    recall = float(tp / (tp + fn)) if tp + fn else 0.0
    denom = 2 * tp + fp + fn
    f1 = float(2 * tp / denom) if denom else 0.0
    return precision, recall, f1


def resample_indices(n_sentences: int, n_resamples: int, seed: int) -> np.ndarray:
    """Sentence indices for each bootstrap resample, shape (n_resamples, n_sentences).

    Generated once per run and shared across every model so comparisons stay paired,
    and seeded so a re-run reproduces the intervals exactly.
    """
    rng = np.random.default_rng(seed)
    return rng.integers(0, n_sentences, size=(n_resamples, n_sentences), dtype=np.int64)


def bootstrap_f1(counts: np.ndarray, indices: np.ndarray, chunk: int = 256) -> np.ndarray:
    """F1 under each resample, shape (n_resamples,).

    Chunked because `counts[indices]` materialises (n_resamples, n_sentences, 3) in one
    go, which is hundreds of MB at realistic sizes.
    """
    out = np.empty(len(indices), dtype=np.float64)
    for start in range(0, len(indices), chunk):
        block = indices[start : start + chunk]
        sums = counts[block].sum(axis=1)  # (block, 3)
        tp, fp, fn = sums[:, 0], sums[:, 1], sums[:, 2]
        denom = 2 * tp + fp + fn
        out[start : start + len(block)] = np.where(denom > 0, 2 * tp / np.maximum(denom, 1), 0.0)
    return out


def percentile_ci(samples: np.ndarray, alpha: float = DEFAULT_ALPHA) -> tuple[float, float]:
    """Two-sided percentile confidence interval."""
    lo, hi = np.percentile(samples, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return float(lo), float(hi)


@dataclass(frozen=True)
class Comparison:
    """Paired-bootstrap comparison of two models on the same test sentences."""

    model_a: str
    model_b: str
    f1_a: float
    f1_b: float
    delta: float           # f1_a - f1_b on the full test set
    ci_low: float          # CI on the *difference*, not on either F1
    ci_high: float
    p_value: float
    significant: bool

    def to_dict(self) -> dict:
        return asdict(self)


def paired_bootstrap(
    name_a: str,
    counts_a: np.ndarray,
    name_b: str,
    counts_b: np.ndarray,
    indices: np.ndarray,
    alpha: float = DEFAULT_ALPHA,
) -> Comparison:
    """Test whether model A's F1 differs from model B's on the same test set.

    Two-sided percentile bootstrap: the p-value is twice the smaller tail mass of the
    resampled difference distribution on either side of zero. Equivalently, the result
    is significant at `alpha` exactly when the CI on the difference excludes zero.
    """
    deltas = bootstrap_f1(counts_a, indices) - bootstrap_f1(counts_b, indices)
    _, _, f1_a = prf_from_counts(counts_a)
    _, _, f1_b = prf_from_counts(counts_b)
    lo, hi = percentile_ci(deltas, alpha)
    tail = min(float((deltas <= 0).mean()), float((deltas >= 0).mean()))
    p_value = min(2.0 * tail, 1.0)
    return Comparison(
        model_a=name_a,
        model_b=name_b,
        f1_a=f1_a,
        f1_b=f1_b,
        delta=f1_a - f1_b,
        ci_low=lo,
        ci_high=hi,
        p_value=p_value,
        significant=bool(lo > 0 or hi < 0),
    )
