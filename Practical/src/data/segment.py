"""Tokenize + sentence-segment Albanian Wikipedia extracts with stanza.

Input  JSONL: {"title", "extract", "source_url", ...}
Output JSONL: {"id", "tokens": [...], "text", "source_url", "title"}

Filters to sentences with 8-40 tokens, drops sentences mangled by extraction or
tokenization (see `src.data.quality`), shuffles with a fixed seed, and keeps the first
`--sample`.

The quality filter runs *before* the sample is taken, so `--sample N` yields N usable
sentences rather than N minus however many were broken. Dropped sentences are counted by
reason and can be written out with `--rejects` for review — the drop rate is a reportable
data-quality number, so it is worth not throwing away.
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

import stanza
from tqdm import tqdm

from .quality import sentence_issues

MIN_TOKENS = 8
MAX_TOKENS = 40


def build_pipeline() -> stanza.Pipeline:
    """Load the Albanian tokenizer, downloading the model on first use."""
    try:
        return stanza.Pipeline(lang="sq", processors="tokenize", verbose=False)
    except Exception:
        stanza.download("sq", processors="tokenize", verbose=False)
        return stanza.Pipeline(lang="sq", processors="tokenize", verbose=False)


def template_shape(text: str) -> str:
    """Collapse digits so near-identical stub sentences share one key.

    Albanian Wikipedia has many generated geography stubs that differ only in their
    numbers — `Ka sipërfaqe prej 43 .45 km² , dhe gjendet në lartësi mbidetare 400 m .`
    appears 15 times with different figures.
    """
    return re.sub(r"\d+", "#", text)


def deduplicate(
    candidates: list[dict], max_per_template: int
) -> tuple[list[dict], list[dict]]:
    """Drop exact repeats, and cap how many members of one template family survive.

    Template sentences are not useless — they are genuine negatives, and a model has to
    learn that a sentence can contain no entities at all. But fifteen near-identical
    copies buy nothing over three, and each one costs an annotation slot and a
    pre-labeling call. Capping keeps the negatives diverse instead of repetitive.
    """
    seen: set[str] = set()
    per_shape: dict[str, int] = {}
    kept: list[dict] = []
    dropped: list[dict] = []

    for rec in candidates:
        text = " ".join(rec["tokens"])
        if text in seen:
            dropped.append({**rec, "issues": ["exact-duplicate"]})
            continue
        seen.add(text)

        shape = template_shape(text)
        count = per_shape.get(shape, 0)
        if count >= max_per_template:
            dropped.append({**rec, "issues": ["template-repeat"]})
            continue
        per_shape[shape] = count + 1
        kept.append(rec)

    return kept, dropped


def iter_articles(path: Path):
    with path.open(encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--sample", type=int, default=150)
    p.add_argument("--seed", type=int, default=20260605)
    p.add_argument(
        "--rejects", type=Path, default=None, help="Optional JSONL of filtered-out sentences."
    )
    p.add_argument(
        "--no-filter", action="store_true", help="Skip the quality filter (for comparison)."
    )
    p.add_argument(
        "--max-per-template",
        type=int,
        default=3,
        help="Cap on near-identical stub sentences differing only in their numbers.",
    )
    args = p.parse_args()

    nlp = build_pipeline()

    candidates: list[dict] = []
    rejects: list[dict] = []
    n_length_filtered = 0
    reasons: dict[str, int] = {}
    articles = list(iter_articles(args.inp))
    for art in tqdm(articles, desc="segmenting"):
        doc = nlp(art["extract"])
        for sent in doc.sentences:
            tokens = [w.text for w in sent.words]
            if not MIN_TOKENS <= len(tokens) <= MAX_TOKENS:
                n_length_filtered += 1
                continue
            rec = {
                "tokens": tokens,
                "text": sent.text,
                "source_url": art.get("source_url"),
                "title": art.get("title"),
            }
            issues = [] if args.no_filter else sentence_issues(tokens)
            if issues:
                for reason in issues:
                    reasons[reason] = reasons.get(reason, 0) + 1
                rejects.append({**rec, "issues": issues})
                continue
            candidates.append(rec)

    n_before_dedup = len(candidates)
    candidates, duplicates = deduplicate(candidates, args.max_per_template)
    rejects.extend(duplicates)
    for rec in duplicates:
        reason = rec["issues"][0]
        reasons[reason] = reasons.get(reason, 0) + 1

    rng = random.Random(args.seed)
    rng.shuffle(candidates)
    chosen = candidates[: args.sample]
    for i, rec in enumerate(chosen):
        rec["id"] = f"sq-{i:04d}"

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as f:
        for rec in chosen:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.rejects and rejects:
        args.rejects.parent.mkdir(parents=True, exist_ok=True)
        with args.rejects.open("w", encoding="utf-8") as f:
            for rec in rejects:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    total = len(candidates) + len(rejects)
    print(
        f"{len(articles)} articles → {total + n_length_filtered} sentences "
        f"({n_length_filtered} outside {MIN_TOKENS}-{MAX_TOKENS} tokens)"
    )
    if total:
        print(f"  {len(rejects)}/{total} dropped ({len(rejects)/total*100:.1f}%)")
        for reason, n in sorted(reasons.items(), key=lambda kv: -kv[1]):
            print(f"      {n:5d}  {reason}")
        print(f"  {n_before_dedup - len(candidates)} of those were duplicates or templates")
    print(f"  kept {len(chosen)} of {len(candidates)} usable (target {args.sample}) → {args.out}")
    if len(chosen) < args.sample:
        print(f"  WARNING: {args.sample - len(chosen)} short of target; sample more articles")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
