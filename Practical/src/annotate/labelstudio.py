"""Convert between our token-level JSONL and Label Studio's character-offset format.

Label Studio annotates character spans over a text string; everything downstream here is
token-indexed BIO. The conversion has to be exactly lossless in both directions, because a
half-token drift produces labels that look plausible and are wrong — the worst possible
failure for gold data. Two decisions buy that guarantee:

**The displayed text is `" ".join(tokens)`, not the original sentence.** Stanza splits
punctuation off (`Tuluzën` `,`), so the original string and the token list disagree about
spacing. Rebuilding the text from the tokens means character offsets map back to token
indices by construction. The cost is cosmetic — annotators see a space before commas —
and it is worth paying for an alignment that cannot drift.

**The head is marked only when it differs from the full span.** Most entities are bare
names where the two coincide, so requiring both every time would double the clicks for no
information. Annotators mark the entity, then add a `HEAD` region inside it only for the
`X i Y` cases; on import, an entity with no nested head defaults to head == full span.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

from .prompt import ENTITY_TYPES

HEAD_LABEL = "HEAD"


def token_offsets(tokens: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Rebuild the display text and each token's [start, end) character span."""
    text_parts: list[str] = []
    spans: list[tuple[int, int]] = []
    cursor = 0
    for i, tok in enumerate(tokens):
        if i:
            text_parts.append(" ")
            cursor += 1
        spans.append((cursor, cursor + len(tok)))
        text_parts.append(tok)
        cursor += len(tok)
    return "".join(text_parts), spans


def chars_to_tokens(
    start: int, end: int, spans: list[tuple[int, int]]
) -> tuple[int, int] | None:
    """Map a character span to inclusive token indices, or None if it doesn't align.

    Leading/trailing whitespace in the selection is tolerated (annotators drag past a word
    constantly). A selection cutting *into* a token is not — that is a real misalignment
    and is reported rather than snapped, so it can be fixed at the source.
    """
    covered = [i for i, (s, e) in enumerate(spans) if s < end and e > start]
    if not covered:
        return None
    first, last = covered[0], covered[-1]
    if start > spans[first][0] or end < spans[last][1]:
        return None  # selection splits a token
    return first, last


# ------------------------------------------------------------------ export to tasks


def to_task(record: dict, with_predictions: bool, model_version: str = "") -> dict:
    """One Label Studio task from one pre-labeled record.

    `with_predictions=False` produces the from-scratch condition: identical text, no
    suggestions. Keeping both conditions on one code path is what makes them comparable.
    """
    tokens = record["tokens"]
    text, spans = token_offsets(tokens)

    task = {
        "data": {
            "text": text,
            "sentence_id": record.get("id"),
            "source_url": record.get("source_url"),
        }
    }
    if not with_predictions:
        return task

    results = []
    for span in record.get("llm_spans", []):
        s_char = spans[span["start"]][0]
        e_char = spans[span["end"]][1]
        results.append(
            {
                "from_name": "label",
                "to_name": "text",
                "type": "labels",
                "value": {
                    "start": s_char,
                    "end": e_char,
                    "text": text[s_char:e_char],
                    "labels": [span["type"]],
                },
            }
        )
        # Only emit a HEAD region where it actually narrows the span — otherwise the
        # annotator sees a redundant second box on every single entity.
        if (span["head_start"], span["head_end"]) != (span["start"], span["end"]):
            hs = spans[span["head_start"]][0]
            he = spans[span["head_end"]][1]
            results.append(
                {
                    "from_name": "label",
                    "to_name": "text",
                    "type": "labels",
                    "value": {
                        "start": hs,
                        "end": he,
                        "text": text[hs:he],
                        "labels": [HEAD_LABEL],
                    },
                }
            )

    task["predictions"] = [{"model_version": model_version, "result": results}]
    return task


# ---------------------------------------------------------------- import from export


@dataclass
class ImportIssue:
    sentence_id: str
    reason: str
    detail: str = ""


@dataclass
class ImportResult:
    records: list[dict] = field(default_factory=list)
    issues: list[ImportIssue] = field(default_factory=list)


def _regions(annotation: dict) -> list[dict]:
    return [r for r in annotation.get("result", []) if r.get("type") == "labels"]


def _flags(annotation: dict) -> list[str]:
    """Sentence-level checkboxes — `junk`, `unsure`.

    Junk is flagged rather than deleted: the mangle rate is a reportable data-quality
    number, a silently dropped sentence cannot be audited, and two annotators disagreeing
    about junk usually means the sentence is hard rather than broken.
    """
    out: list[str] = []
    for result in annotation.get("result", []):
        if result.get("type") == "choices":
            out.extend(result.get("value", {}).get("choices", []))
    return sorted(set(out))


def from_annotation(
    task: dict, annotation: dict, tokens: list[str]
) -> tuple[dict, list[ImportIssue]]:
    """Turn one Label Studio annotation back into token-indexed spans.

    Heads are attached by containment: a HEAD region nested inside an entity becomes that
    entity's head. An entity with no nested head keeps head == full span.
    """
    sid = task.get("data", {}).get("sentence_id", "?")
    _, spans = token_offsets(tokens)
    issues: list[ImportIssue] = []

    entities: list[dict] = []
    heads: list[tuple[int, int]] = []

    for region in _regions(annotation):
        value = region["value"]
        labels = value.get("labels") or []
        if not labels:
            continue
        label = labels[0]
        mapped = chars_to_tokens(value["start"], value["end"], spans)
        if mapped is None:
            issues.append(
                ImportIssue(sid, "span does not align to token boundaries", repr(value.get("text")))
            )
            continue
        lo, hi = mapped
        if label == HEAD_LABEL:
            heads.append((lo, hi))
        elif label in ENTITY_TYPES:
            entities.append({"start": lo, "end": hi, "type": label})
        else:
            issues.append(ImportIssue(sid, f"unknown label {label!r}"))

    entities.sort(key=lambda e: e["start"])

    # Reject overlapping entities outright — BIO cannot represent them, and silently
    # dropping one would hide a genuine annotator error.
    claimed: set[int] = set()
    kept: list[dict] = []
    for ent in entities:
        positions = set(range(ent["start"], ent["end"] + 1))
        if positions & claimed:
            issues.append(
                ImportIssue(sid, "overlapping entity spans", f"{ent['start']}-{ent['end']}")
            )
            continue
        claimed |= positions
        kept.append(ent)

    for ent in kept:
        nested = [h for h in heads if h[0] >= ent["start"] and h[1] <= ent["end"]]
        if len(nested) > 1:
            issues.append(ImportIssue(sid, "multiple HEAD regions in one entity"))
        lo, hi = nested[0] if nested else (ent["start"], ent["end"])
        ent["head_start"], ent["head_end"] = lo, hi
        ent["surface"] = " ".join(tokens[ent["start"] : ent["end"] + 1])
        ent["head_surface"] = " ".join(tokens[lo : hi + 1])

    orphans = [
        h
        for h in heads
        if not any(h[0] >= e["start"] and h[1] <= e["end"] for e in kept)
    ]
    for h in orphans:
        issues.append(ImportIssue(sid, "HEAD region not inside any entity", f"{h[0]}-{h[1]}"))

    flags = _flags(annotation)
    if "junk" in flags and kept:
        issues.append(
            ImportIssue(sid, "flagged junk but also has entity spans", f"{len(kept)} spans")
        )

    record = {
        "id": sid,
        "tokens": tokens,
        "spans": kept,
        "flags": flags,
        "junk": "junk" in flags,
        "source_url": task.get("data", {}).get("source_url"),
        "annotator": _annotator_of(annotation),
        "lead_time_s": annotation.get("lead_time"),
    }
    return record, issues


def _annotator_of(annotation: dict) -> str | None:
    """Label Studio reports `completed_by` as an id or an expanded user object."""
    who = annotation.get("completed_by")
    if isinstance(who, dict):
        return who.get("email") or who.get("username") or str(who.get("id"))
    return None if who is None else str(who)


def load_export(path: Path, tokens_by_id: dict[str, list[str]]) -> ImportResult:
    """Read a Label Studio JSON export into one record per (sentence, annotator).

    Tokens come from the original JSONL rather than being re-derived from the displayed
    text — the source of truth for tokenization is the segmenter, not the annotation tool.
    """
    tasks = json.loads(Path(path).read_text(encoding="utf-8"))
    result = ImportResult()

    for task in tasks:
        sid = task.get("data", {}).get("sentence_id")
        tokens = tokens_by_id.get(sid)
        if tokens is None:
            result.issues.append(ImportIssue(str(sid), "sentence id not found in source JSONL"))
            continue
        annotations = [a for a in task.get("annotations", []) if not a.get("was_cancelled")]
        if not annotations:
            result.issues.append(ImportIssue(str(sid), "no completed annotation"))
            continue
        for annotation in annotations:
            record, issues = from_annotation(task, annotation, tokens)
            result.records.append(record)
            result.issues.extend(issues)

    return result


# ------------------------------------------------------------------------------- CLI


def _read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def main() -> int:
    import argparse

    p = argparse.ArgumentParser(description="Label Studio import/export for Albanian NER.")
    sub = p.add_subparsers(dest="cmd", required=True)

    exp = sub.add_parser("tasks", help="Build a Label Studio task file from pre-labeled JSONL.")
    exp.add_argument("--in", dest="inp", type=Path, required=True)
    exp.add_argument("--out", type=Path, required=True)
    exp.add_argument(
        "--condition",
        choices=["assisted", "scratch"],
        default="assisted",
        help="'assisted' ships the LLM spans as editable predictions; 'scratch' ships the "
        "same sentences with none. This flag is the whole anchoring experiment.",
    )
    exp.add_argument("--limit", type=int, default=None)
    exp.add_argument("--model-version", default="", help="Recorded on each prediction.")

    imp = sub.add_parser("collect", help="Convert a Label Studio JSON export back to JSONL.")
    imp.add_argument("--export", type=Path, required=True, help="Label Studio JSON export.")
    imp.add_argument("--source", type=Path, required=True, help="JSONL the tasks were built from.")
    imp.add_argument("--out", type=Path, required=True)

    args = p.parse_args()

    if args.cmd == "tasks":
        records = _read_jsonl(args.inp)
        if args.limit:
            records = records[: args.limit]
        with_preds = args.condition == "assisted"
        tasks = [to_task(r, with_preds, args.model_version) for r in records]
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(tasks, ensure_ascii=False, indent=2), encoding="utf-8")
        n_pred = sum(len(t.get("predictions", [{}])[0].get("result", [])) for t in tasks if with_preds)
        print(f"{len(tasks)} tasks ({args.condition}) → {args.out}")
        if with_preds:
            print(f"  {n_pred} pre-annotated regions")
        return 0

    source = {r["id"]: r["tokens"] for r in _read_jsonl(args.source)}
    result = load_export(args.export, source)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in result.records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")

    n_junk = sum(1 for r in result.records if r["junk"])
    n_unsure = sum(1 for r in result.records if "unsure" in r["flags"])
    print(f"{len(result.records)} annotations → {args.out}")
    print(f"  {n_junk} flagged junk, {n_unsure} flagged unsure")
    by_annotator: dict[str, int] = {}
    for rec in result.records:
        key = rec.get("annotator") or "unknown"
        by_annotator[key] = by_annotator.get(key, 0) + 1
    for who, n in sorted(by_annotator.items(), key=lambda kv: -kv[1]):
        print(f"  {who}: {n}")

    if result.issues:
        print(f"\n{len(result.issues)} issues:", flush=True)
        counts: dict[str, int] = {}
        for issue in result.issues:
            counts[issue.reason] = counts.get(issue.reason, 0) + 1
        for reason, n in sorted(counts.items(), key=lambda kv: -kv[1]):
            print(f"  {n:4d}  {reason}")
        print("\nfirst few:")
        for issue in result.issues[:5]:
            print(f"  {issue.sentence_id}: {issue.reason} {issue.detail}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
