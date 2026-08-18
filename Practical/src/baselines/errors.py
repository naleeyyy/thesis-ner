"""Classify NER errors into the categories that suggest different fixes.

A single F1 says a model is wrong; it does not say *how*. These four categories do, and
they point at different remedies:

- **missed**      gold span with no overlapping prediction — a recall failure
- **spurious**    predicted span overlapping no gold span — a precision failure
- **type**        same token extent, wrong entity type — the tagset is being confused
- **boundary**    overlapping extents that do not match exactly — the span edges are wrong

The distinction matters for the write-up. Boundary and type errors mean the model found
the entity and mislabelled its edges or class, which annotation guidelines can address.
Missed and spurious errors mean it did not find it at all, which is a modelling or data
problem. Reporting them together hides which one you have.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path

from seqeval.metrics.sequence_labeling import get_entities

CATEGORIES = ("missed", "spurious", "type", "boundary")


def overlaps(a: tuple, b: tuple) -> bool:
    """True when two spans share at least one token, ignoring their types."""
    return a[1] <= b[2] and b[1] <= a[2]


def classify(gold_tags: list[str], pred_tags: list[str]) -> list[dict]:
    """Errors for one sentence. Exact matches produce nothing."""
    gold = list(get_entities(gold_tags))
    pred = list(get_entities(pred_tags))

    exact = set(gold) & set(pred)
    gold_left = [g for g in gold if g not in exact]
    pred_left = [p for p in pred if p not in exact]

    errors: list[dict] = []
    matched_pred: set[int] = set()

    for g in gold_left:
        partners = [
            (i, p) for i, p in enumerate(pred_left) if i not in matched_pred and overlaps(g, p)
        ]
        if not partners:
            errors.append({"kind": "missed", "gold": g, "pred": None})
            continue
        # Prefer a partner with identical extent (a pure type error) over a ragged one,
        # so a type confusion is never reported as a boundary problem.
        same_extent = [(i, p) for i, p in partners if (p[1], p[2]) == (g[1], g[2])]
        i, p = (same_extent or partners)[0]
        matched_pred.add(i)
        kind = "type" if (p[1], p[2]) == (g[1], g[2]) else "boundary"
        errors.append({"kind": kind, "gold": g, "pred": p})

    for i, p in enumerate(pred_left):
        if i not in matched_pred:
            errors.append({"kind": "spurious", "gold": None, "pred": p})
    return errors


def analyse(path: Path) -> dict:
    rows = [json.loads(line) for line in path.open(encoding="utf-8") if line.strip()]
    kinds: Counter[str] = Counter()
    by_type: dict[str, Counter[str]] = {}
    confusions: Counter[tuple[str, str]] = Counter()
    boundary_shapes: Counter[str] = Counter()
    n_gold = 0

    for row in rows:
        n_gold += len(get_entities(row["gold"]))
        for err in classify(row["gold"], row["pred"]):
            kinds[err["kind"]] += 1
            entity_type = (err["gold"] or err["pred"])[0]
            by_type.setdefault(entity_type, Counter())[err["kind"]] += 1
            if err["kind"] == "type":
                confusions[(err["gold"][0], err["pred"][0])] += 1
            elif err["kind"] == "boundary":
                g, p = err["gold"], err["pred"]
                # Did the prediction cover more tokens than gold, fewer, or shift?
                gold_len, pred_len = g[2] - g[1] + 1, p[2] - p[1] + 1
                shape = "too wide" if pred_len > gold_len else (
                    "too narrow" if pred_len < gold_len else "shifted"
                )
                boundary_shapes[shape] += 1

    return {
        "model": path.stem,
        "n_sentences": len(rows),
        "n_gold_entities": n_gold,
        "n_errors": sum(kinds.values()),
        "by_kind": dict(kinds),
        "by_entity_type": {k: dict(v) for k, v in by_type.items()},
        "type_confusions": {f"{a}->{b}": n for (a, b), n in confusions.most_common()},
        "boundary_shapes": dict(boundary_shapes),
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("predictions", type=Path, nargs="+")
    p.add_argument("--out", type=Path, default=None, help="Write the full breakdown as JSON.")
    args = p.parse_args()

    reports = [analyse(path) for path in args.predictions]

    header = f"{'model':24s} {'errors':>7} " + "".join(f"{c:>10s}" for c in CATEGORIES)
    print(header)
    for r in reports:
        row = f"{r['model']:24s} {r['n_errors']:>7d} "
        for c in CATEGORIES:
            n = r["by_kind"].get(c, 0)
            share = 100 * n / max(r["n_errors"], 1)
            row += f"{n:>6d}{share:>3.0f}%"
        print(row)

    print("\nerrors per gold entity, by entity type:")
    for r in reports:
        parts = []
        for etype in ("PER", "ORG", "LOC"):
            total = sum(r["by_entity_type"].get(etype, {}).values())
            parts.append(f"{etype} {total:4d}")
        print(f"  {r['model']:24s} " + "  ".join(parts))

    print("\nmost common type confusions (gold -> predicted):")
    for r in reports:
        top = list(r["type_confusions"].items())[:3]
        print(f"  {r['model']:24s} " + ("  ".join(f"{k} {v}" for k, v in top) or "none"))

    print("\nboundary error shapes:")
    for r in reports:
        s = r["boundary_shapes"]
        print(f"  {r['model']:24s} " + "  ".join(f"{k} {v}" for k, v in sorted(s.items())))

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(reports, indent=2))
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
