"""Inter-annotator agreement, and the disagreement list that drives adjudication.

Reports three things, because no single number answers the question:

- **Token-level κ** (Fleiss across all annotators, Cohen pairwise) — the standard,
  chance-corrected measure. Read it knowing that most tokens are `O` and every annotator
  agrees on those, so the absolute value flatters NER; the *comparison* between conditions
  is what carries information, not the number on its own.
- **Span-level pairwise F1** — treats one annotator as the reference and scores the other
  exactly as a model would be scored. Harsher than κ and much closer to what actually
  matters downstream.
- **A ranked disagreement report** — the sentences to adjudicate first.

Everything is computed twice, once per boundary convention (`full` and `head`), because
the point of recording both was to keep that choice open.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path

from ..baselines import stats
from .llm_label import spans_to_bio


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


# ------------------------------------------------------------------------ kappa maths


def fleiss_kappa(ratings: list[list[str]]) -> float:
    """Fleiss' κ over items rated by a fixed number of annotators.

    `ratings[i]` is the list of labels the annotators gave item i — same length for every
    item. Returns 1.0 for unanimous agreement on a single category, where κ is formally
    undefined (P_e == 1): with one category in play there is no disagreement to correct
    for, and returning NaN there would poison an otherwise meaningful report.
    """
    if not ratings:
        return float("nan")
    n_raters = len(ratings[0])
    if n_raters < 2:
        return float("nan")

    categories = sorted({label for row in ratings for label in row})
    n_items = len(ratings)

    # P_i: observed pairwise agreement within item i.
    agreement_sum = 0.0
    category_totals: Counter[str] = Counter()
    for row in ratings:
        counts = Counter(row)
        category_totals.update(counts)
        agreement_sum += (sum(c * c for c in counts.values()) - n_raters) / (
            n_raters * (n_raters - 1)
        )
    p_bar = agreement_sum / n_items

    total = n_items * n_raters
    p_e = sum((category_totals[c] / total) ** 2 for c in categories)

    if p_e >= 1.0:
        return 1.0
    return (p_bar - p_e) / (1 - p_e)


def cohen_kappa(a: list[str], b: list[str]) -> float:
    """Cohen's κ between two annotators over paired items."""
    if not a:
        return float("nan")
    n = len(a)
    observed = sum(1 for x, y in zip(a, b, strict=True) if x == y) / n

    count_a, count_b = Counter(a), Counter(b)
    expected = sum(
        (count_a[c] / n) * (count_b[c] / n) for c in set(count_a) | set(count_b)
    )
    if expected >= 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


# --------------------------------------------------------------------- data assembly


@dataclass
class Annotated:
    """All annotations of one sentence, keyed by annotator."""

    sentence_id: str
    tokens: list[str]
    by_annotator: dict[str, dict] = field(default_factory=dict)

    def tags(self, annotator: str, mode: str) -> list[str]:
        rec = self.by_annotator[annotator]
        return spans_to_bio(rec["spans"], len(self.tokens), mode)


def group(records: list[dict]) -> dict[str, Annotated]:
    out: dict[str, Annotated] = {}
    for rec in records:
        sid = rec["id"]
        if sid not in out:
            out[sid] = Annotated(sid, rec["tokens"])
        out[sid].by_annotator[rec.get("annotator") or "unknown"] = rec
    return out


def complete_subset(
    grouped: dict[str, Annotated], annotators: list[str]
) -> list[Annotated]:
    """Sentences every named annotator labelled.

    Fleiss' κ needs a constant number of raters per item, so mixing partially-overlapping
    sentences in would silently change what the statistic means.
    """
    wanted = set(annotators)
    return [a for a in grouped.values() if wanted <= set(a.by_annotator)]


# ------------------------------------------------------------------------- reporting


def token_ratings(
    sentences: list[Annotated], annotators: list[str], mode: str
) -> list[list[str]]:
    """One row per token: the tag each annotator gave it."""
    rows: list[list[str]] = []
    for sent in sentences:
        per_annotator = [sent.tags(who, mode) for who in annotators]
        for position in range(len(sent.tokens)):
            rows.append([tags[position] for tags in per_annotator])
    return rows


def span_f1(sentences: list[Annotated], a: str, b: str, mode: str) -> float:
    """Span-level F1 scoring annotator `b` against annotator `a` as reference."""
    refs = [s.tags(a, mode) for s in sentences]
    preds = [s.tags(b, mode) for s in sentences]
    _, _, f1 = stats.prf_from_counts(stats.sentence_counts(preds, refs))
    return f1


@dataclass
class Disagreement:
    sentence_id: str
    n_tokens_disputed: int
    detail: list[dict]


def disagreements(
    sentences: list[Annotated], annotators: list[str], mode: str
) -> list[Disagreement]:
    """Every token where annotators differ, ranked worst sentence first."""
    out: list[Disagreement] = []
    for sent in sentences:
        tags = {who: sent.tags(who, mode) for who in annotators}
        detail = []
        for i, token in enumerate(sent.tokens):
            assigned = {who: tags[who][i] for who in annotators}
            if len(set(assigned.values())) > 1:
                detail.append({"index": i, "token": token, "tags": assigned})
        if detail:
            out.append(Disagreement(sent.sentence_id, len(detail), detail))
    out.sort(key=lambda d: -d.n_tokens_disputed)
    return out


def summarize(
    grouped: dict[str, Annotated], annotators: list[str], mode: str
) -> dict:
    sentences = complete_subset(grouped, annotators)
    if not sentences:
        return {"mode": mode, "n_sentences": 0, "note": "no sentence has all annotators"}

    ratings = token_ratings(sentences, annotators, mode)
    pairwise = {}
    for a, b in combinations(annotators, 2):
        col_a = [r[annotators.index(a)] for r in ratings]
        col_b = [r[annotators.index(b)] for r in ratings]
        pairwise[f"{a} vs {b}"] = {
            "cohen_kappa": round(cohen_kappa(col_a, col_b), 4),
            "span_f1": round(span_f1(sentences, a, b, mode), 4),
        }

    disputed = disagreements(sentences, annotators, mode)
    return {
        "mode": mode,
        "n_sentences": len(sentences),
        "n_tokens": len(ratings),
        "annotators": annotators,
        "fleiss_kappa": round(fleiss_kappa(ratings), 4),
        "pairwise": pairwise,
        "n_sentences_with_disagreement": len(disputed),
        "pct_tokens_disputed": round(
            100 * sum(d.n_tokens_disputed for d in disputed) / max(len(ratings), 1), 2
        ),
    }


def junk_agreement(grouped: dict[str, Annotated], annotators: list[str]) -> dict:
    """Do annotators agree on which sentences are broken?

    Disagreement here usually means the sentence is *hard* rather than mangled, so it is
    worth surfacing separately instead of folding into the label statistics.
    """
    sentences = complete_subset(grouped, annotators)
    if not sentences:
        return {"n_sentences": 0}
    ratings = [
        [str(bool(s.by_annotator[who].get("junk"))) for who in annotators]
        for s in sentences
    ]
    flagged = sum(1 for r in ratings if "True" in r)
    unanimous = sum(1 for r in ratings if len(set(r)) == 1 and r[0] == "True")
    return {
        "n_sentences": len(sentences),
        "fleiss_kappa": round(fleiss_kappa(ratings), 4),
        "flagged_by_someone": flagged,
        "flagged_by_everyone": unanimous,
    }


# ------------------------------------------------------------------------------- CLI


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="inp", type=Path, required=True, help="Collected JSONL.")
    p.add_argument("--out", type=Path, default=None, help="Write the full report as JSON.")
    p.add_argument(
        "--disagreements",
        type=Path,
        default=None,
        help="Write the ranked adjudication list as JSONL.",
    )
    p.add_argument(
        "--annotator",
        action="append",
        default=None,
        help="Restrict to these annotators (repeatable). Default: everyone present.",
    )
    args = p.parse_args()

    records = read_jsonl(args.inp)
    grouped = group(records)

    present = sorted({r.get("annotator") or "unknown" for r in records})
    annotators = args.annotator or present
    missing = [a for a in annotators if a not in present]
    if missing:
        raise SystemExit(f"not in the data: {', '.join(missing)}\navailable: {', '.join(present)}")
    if len(annotators) < 2:
        raise SystemExit(f"need at least 2 annotators, found {len(annotators)}: {present}")

    counts = Counter(r.get("annotator") or "unknown" for r in records)
    print(f"{len(records)} annotations of {len(grouped)} sentences")
    for who in annotators:
        print(f"  {who:28s} {counts[who]:5d}")

    report = {
        "n_annotations": len(records),
        "n_sentences": len(grouped),
        "per_annotator": dict(counts),
        "junk": junk_agreement(grouped, annotators),
        "views": {mode: summarize(grouped, annotators, mode) for mode in ("full", "head")},
    }

    for mode in ("full", "head"):
        view = report["views"][mode]
        print(f"\n--- {mode} spans ---")
        if not view["n_sentences"]:
            print("  no sentence annotated by all of them")
            continue
        print(f"  {view['n_sentences']} sentences, {view['n_tokens']} tokens")
        print(f"  Fleiss kappa      {view['fleiss_kappa']:.4f}")
        print(f"  tokens disputed   {view['pct_tokens_disputed']:.2f}%")
        for pair, vals in view["pairwise"].items():
            print(f"    {pair:34s} kappa {vals['cohen_kappa']:.4f}   span-F1 {vals['span_f1']:.4f}")

    junk = report["junk"]
    if junk.get("n_sentences"):
        print(
            f"\njunk flag: kappa {junk['fleiss_kappa']:.4f}   "
            f"{junk['flagged_by_someone']} flagged by someone, "
            f"{junk['flagged_by_everyone']} by everyone"
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8")
        print(f"\nreport → {args.out}")

    if args.disagreements:
        sentences = complete_subset(grouped, annotators)
        rows = disagreements(sentences, annotators, "full")
        args.disagreements.parent.mkdir(parents=True, exist_ok=True)
        with args.disagreements.open("w", encoding="utf-8") as fh:
            for d in rows:
                tokens = grouped[d.sentence_id].tokens
                fh.write(
                    json.dumps(
                        {
                            "id": d.sentence_id,
                            "text": " ".join(tokens),
                            "n_tokens_disputed": d.n_tokens_disputed,
                            "detail": d.detail,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
        print(f"{len(rows)} sentences to adjudicate → {args.disagreements}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
