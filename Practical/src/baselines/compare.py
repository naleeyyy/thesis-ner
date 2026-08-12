"""Compare two saved prediction files with a paired bootstrap.

`0.919 vs 0.925` on its own says nothing — the question is whether the gap survives
resampling the test set. This pairs the two models on the same resampled sentences, so
the shared difficulty of a resample cancels out of the difference and much smaller gaps
become detectable than two independent confidence intervals would suggest.

Both files must be predictions over the *same* test sentences in the same order — the
gold column is checked, and a mismatch is an error rather than a silent comparison of two
different test sets.

    python -m src.baselines.compare results/predictions/a.jsonl results/predictions/b.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from . import stats


def load_predictions(path: Path) -> tuple[list[list[str]], list[list[str]]]:
    preds, golds = [], []
    with Path(path).open(encoding="utf-8") as fh:
        for line in fh:
            if not line.strip():
                continue
            rec = json.loads(line)
            preds.append(rec["pred"])
            golds.append(rec["gold"])
    return preds, golds


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("a", type=Path, help="First prediction JSONL.")
    p.add_argument("b", type=Path, help="Second prediction JSONL.")
    p.add_argument("--alpha", type=float, default=stats.DEFAULT_ALPHA)
    p.add_argument("--resamples", type=int, default=stats.DEFAULT_RESAMPLES)
    p.add_argument("--seed", type=int, default=stats.DEFAULT_SEED)
    args = p.parse_args()

    preds_a, gold_a = load_predictions(args.a)
    preds_b, gold_b = load_predictions(args.b)

    if len(gold_a) != len(gold_b):
        raise SystemExit(f"different lengths: {len(gold_a)} vs {len(gold_b)}")
    if gold_a != gold_b:
        n_diff = sum(1 for x, y in zip(gold_a, gold_b, strict=True) if x != y)
        raise SystemExit(
            f"the two files disagree about the gold labels on {n_diff} sentences — "
            "they are not predictions over the same test set, so a paired test is invalid"
        )

    counts_a = stats.sentence_counts(preds_a, gold_a)
    counts_b = stats.sentence_counts(preds_b, gold_b)
    indices = stats.resample_indices(len(gold_a), args.resamples, args.seed)
    result = stats.paired_bootstrap(
        args.a.stem, counts_a, args.b.stem, counts_b, indices, args.alpha
    )

    conf = round((1 - args.alpha) * 100)
    print(f"{len(gold_a)} sentences, {args.resamples} resamples\n")
    print(f"  {result.model_a:34s} F1 {result.f1_a:.4f}")
    print(f"  {result.model_b:34s} F1 {result.f1_b:.4f}")
    print(f"\n  difference {result.delta:+.4f}   {conf}% CI "
          f"[{result.ci_low:+.4f}, {result.ci_high:+.4f}]   p = "
          f"{'< 0.001' if result.p_value < 0.001 else format(result.p_value, '.3f')}")
    if result.significant:
        better = result.model_a if result.delta > 0 else result.model_b
        print(f"\n  {better} is better; the interval excludes zero.")
    else:
        print("\n  Not distinguishable: the interval on the difference includes zero.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
