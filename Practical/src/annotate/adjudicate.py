"""Collapse multiply-annotated sentences into one gold record per sentence.

Most sentences carry exactly one annotation and pass through untouched. The shared
sentences carry one per annotator and have to be resolved. This module does that by
token-level majority vote under each boundary convention, which is a defensible
automatic baseline and, importantly, a reproducible one --- but it is not the same
as a human adjudicating the hard cases, and the two should not be confused in the
write-up. `--report` lists every sentence where the vote was not unanimous, which is
the queue a human should work through.

A tie is resolved toward `O`. Ties happen when annotators split evenly on a token,
which means there is no majority reading to adopt; inventing one by picking the
alphabetically-first label would manufacture agreement that nobody expressed.

    python -m src.annotate.adjudicate --in data/labeled/collected/all.jsonl \
        --out data/labeled/gold.jsonl --report results/adjudication.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

from .llm_label import spans_to_bio


def bio_to_spans(tags: list[str]) -> list[dict]:
    """Inverse of spans_to_bio for a single flat tag view."""
    spans: list[dict] = []
    start, typ = None, None
    for i, tag in enumerate(tags + ["O"]):
        if tag.startswith("B-") or (tag == "O" and start is not None) or tag.startswith("I-"):
            if tag.startswith("I-") and start is not None and tag[2:] == typ:
                continue
            if start is not None:
                spans.append({"start": start, "end": i - 1, "type": typ})
                start, typ = None, None
        if tag.startswith("B-"):
            start, typ = i, tag[2:]
        elif tag.startswith("I-") and start is None:
            # An I- tag with no preceding B- is a malformed sequence; treat it as a start
            # rather than dropping the entity, which is the more conservative repair.
            start, typ = i, tag[2:]
    return spans


def majority_tags(per_annotator: list[list[str]]) -> tuple[list[str], list[int]]:
    """Token-level majority vote. Returns the tags and the indices that were disputed."""
    n = len(per_annotator[0])
    out, disputed = [], []
    for i in range(n):
        votes = Counter(tags[i] for tags in per_annotator)
        top = votes.most_common()
        if len(top) > 1 and top[0][1] == top[1][1]:
            out.append("O")  # tie -> no majority reading exists
            disputed.append(i)
        else:
            out.append(top[0][0])
            if len(top) > 1:
                disputed.append(i)
    return out, disputed


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--report", type=Path, default=None, help="Markdown list of disputed sentences.")
    p.add_argument(
        "--drop-junk",
        action="store_true",
        help="Omit sentences any annotator flagged as junk (default: keep, flag preserved).",
    )
    args = p.parse_args()

    by_sentence: dict[str, list[dict]] = defaultdict(list)
    for line in args.inp.open(encoding="utf-8"):
        if line.strip():
            rec = json.loads(line)
            by_sentence[rec["id"]].append(rec)

    gold, disputed_rows = [], []
    n_voted = n_junk = 0
    for sid, recs in sorted(by_sentence.items()):
        tokens = recs[0]["tokens"]
        junk_votes = sum(1 for r in recs if r.get("junk"))
        is_junk = junk_votes > len(recs) / 2

        if len(recs) == 1:
            out = dict(recs[0])
            out["n_annotators"] = 1
        else:
            n_voted += 1
            # Vote separately per view, then rebuild spans carrying both boundaries.
            full, d_full = majority_tags([spans_to_bio(r["spans"], len(tokens), "full") for r in recs])
            head, d_head = majority_tags([spans_to_bio(r["spans"], len(tokens), "head") for r in recs])
            full_spans = bio_to_spans(full)
            head_spans = bio_to_spans(head)
            merged = []
            for s in full_spans:
                # Attach the head that falls inside this full span, else default to it.
                inside = [h for h in head_spans if h["start"] >= s["start"] and h["end"] <= s["end"]]
                h = inside[0] if inside else s
                merged.append(
                    {
                        "start": s["start"],
                        "end": s["end"],
                        "head_start": h["start"],
                        "head_end": h["end"],
                        "type": s["type"],
                        "text": " ".join(tokens[s["start"] : s["end"] + 1]),
                    }
                )
            out = {
                "id": sid,
                "tokens": tokens,
                "spans": merged,
                "source_url": recs[0].get("source_url"),
                "junk": is_junk,
                "n_annotators": len(recs),
                "annotators": sorted(r["annotator"] for r in recs),
            }
            if d_full or d_head:
                disputed_rows.append(
                    {
                        "id": sid,
                        "tokens": tokens,
                        "n_annotators": len(recs),
                        "disputed_full": d_full,
                        "disputed_head": d_head,
                        "junk_votes": junk_votes,
                    }
                )

        out["junk"] = is_junk
        if is_junk:
            n_junk += 1
            if args.drop_junk:
                continue
        gold.append(out)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in gold:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_ent = sum(len(r["spans"]) for r in gold)
    print(f"{len(by_sentence)} sentences -> {len(gold)} gold records ({n_ent} entities)")
    print(f"  {n_voted} resolved by majority vote, {len(disputed_rows)} of them non-unanimous")
    print(f"  {n_junk} flagged junk by a majority" + (" (dropped)" if args.drop_junk else " (kept)"))

    if args.report and disputed_rows:
        lines = [
            "# Sentences needing human adjudication",
            "",
            f"{len(disputed_rows)} of {n_voted} multiply-annotated sentences had at least one",
            "token where annotators disagreed. Majority vote resolved them automatically;",
            "these are the ones a human should confirm.",
            "",
        ]
        for row in disputed_rows:
            toks = row["tokens"]
            marked = " ".join(
                f"**{t}**" if i in set(row["disputed_full"]) | set(row["disputed_head"]) else t
                for i, t in enumerate(toks)
            )
            lines += [
                f"## `{row['id']}` ({row['n_annotators']} annotators)",
                "",
                marked,
                "",
                f"- disputed tokens (full): {row['disputed_full']}",
                f"- disputed tokens (head): {row['disputed_head']}",
                f"- junk votes: {row['junk_votes']}",
                "",
            ]
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text("\n".join(lines), encoding="utf-8")
        print(f"  adjudication queue -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
