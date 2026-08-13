"""Hand out batches of sentences to annotators, without losing track of who has what.

The campaign runs in batches of ~50 rather than one large allocation per person: a 50-
sentence ask gets accepted and finished, a 200-sentence ask often gets accepted and
abandoned. The cost of that is bookkeeping — a dozen people over several rounds is 20+
batches, and doing it by hand guarantees that some sentence is handed out twice while
another is never annotated at all.

This module keeps an append-only ledger of every assignment and reserves only sentences
nobody has seen. It also emits the batch as a JSONL subset, which is the input to
pre-labeling — so you only ever pay to pre-label sentences somebody is about to annotate.

    pool  →  assign reserve  →  llm_label  →  labelstudio tasks  →  Label Studio

Overlap batches are the exception: `--overlap` hands the *same* sentences to several
people on purpose, because that is what inter-annotator agreement is computed from.
"""

from __future__ import annotations

import argparse
import datetime
import json
import random
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_LEDGER = REPO_ROOT / "data" / "interim" / "assignments.jsonl"


def read_jsonl(path: Path) -> list[dict]:
    with Path(path).open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def load_ledger(path: Path) -> list[dict]:
    return read_jsonl(path) if Path(path).exists() else []


def append_ledger(path: Path, rows: list[dict]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with Path(path).open("a", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row, ensure_ascii=False) + "\n")


def assigned_ids(ledger: list[dict]) -> set[str]:
    """Sentences already handed to somebody — never reissued to a different batch."""
    return {row["sentence_id"] for row in ledger}


def ids_for(ledger: list[dict], assignee: str) -> set[str]:
    return {row["sentence_id"] for row in ledger if row["assignee"] == assignee}


def reserve(
    pool: list[dict],
    ledger: list[dict],
    assignees: list[str],
    n: int,
    batch: str,
    condition: str,
    overlap: bool,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    """Pick `n` unseen sentences and record them against each assignee.

    Returns (sentences, new ledger rows). For an overlap batch every assignee gets the
    same sentences; otherwise the batch belongs to the single assignee.
    """
    taken = assigned_ids(ledger)
    available = [rec for rec in pool if rec["id"] not in taken]
    if len(available) < n:
        raise SystemExit(
            f"only {len(available)} unassigned sentences left in the pool, need {n}. "
            "Sample more articles or lower --n."
        )

    rng = random.Random(seed)
    chosen = rng.sample(available, n)
    chosen.sort(key=lambda r: r["id"])

    stamp = datetime.datetime.now(datetime.UTC).isoformat(timespec="seconds")
    rows = [
        {
            "sentence_id": rec["id"],
            "assignee": who,
            "batch": batch,
            "condition": condition,
            "overlap": overlap,
            "assigned_at": stamp,
        }
        for rec in chosen
        for who in assignees
    ]
    return chosen, rows


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="cmd", required=True)

    res = sub.add_parser("reserve", help="Reserve a batch of unseen sentences.")
    res.add_argument("--pool", type=Path, required=True)
    res.add_argument("--out", type=Path, required=True, help="JSONL subset to pre-label.")
    res.add_argument("--assignee", action="append", required=True, help="Repeatable.")
    res.add_argument("--n", type=int, default=50)
    res.add_argument("--batch", required=True, help="Label for this round, e.g. 'r1'.")
    res.add_argument("--condition", choices=["assisted", "scratch"], default="assisted")
    res.add_argument(
        "--overlap",
        action="store_true",
        help="Give every assignee the SAME sentences (this is what IAA is computed from). "
        "Without it, --assignee must name exactly one person.",
    )
    res.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    res.add_argument("--seed", type=int, default=None)

    bat = sub.add_parser(
        "batches",
        help="Generate numbered batches, each mixing a shared overlap set with unique "
        "sentences. Hand out batch numbers; the ledger tracks the rest.",
    )
    bat.add_argument("--pool", type=Path, required=True)
    bat.add_argument("--out-dir", type=Path, required=True)
    bat.add_argument("--count", type=int, default=10, help="How many batches to create.")
    bat.add_argument("--size", type=int, default=100, help="Sentences per batch.")
    bat.add_argument(
        "--overlap",
        type=int,
        default=20,
        help="Sentences shared by EVERY batch, for inter-annotator agreement. Embedding "
        "them means agreement needs no separate project and no extra assignment.",
    )
    bat.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    bat.add_argument("--seed", type=int, default=20260813)

    st = sub.add_parser("status", help="Show pool usage and per-annotator counts.")
    st.add_argument("--pool", type=Path, required=True)
    st.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)

    args = p.parse_args()
    pool = read_jsonl(args.pool)
    ledger = load_ledger(args.ledger)

    if args.cmd == "batches":
        unique_per_batch = args.size - args.overlap
        if unique_per_batch <= 0:
            raise SystemExit("--overlap must be smaller than --size")
        needed = args.overlap + args.count * unique_per_batch
        free = len(pool) - len(assigned_ids(ledger))
        if free < needed:
            raise SystemExit(f"need {needed} unassigned sentences, only {free} left")

        shared, shared_rows = reserve(
            pool, ledger, ["shared-overlap"], args.overlap, "overlap", "both", True, args.seed
        )
        append_ledger(args.ledger, shared_rows)
        ledger += shared_rows

        args.out_dir.mkdir(parents=True, exist_ok=True)
        print(f"{args.overlap} shared sentences + {unique_per_batch} unique per batch\n")
        for i in range(1, args.count + 1):
            label = f"{i:02d}"
            # Odd batches carry the model's suggestions, even ones do not. Since every
            # batch contains the same shared sentences, that alternation yields the same
            # sentences annotated both ways — the anchoring experiment — with no extra
            # bookkeeping for whoever hands the batches out.
            condition = "assisted" if i % 2 else "scratch"
            unique, rows = reserve(
                pool, ledger, [f"batch-{label}"], unique_per_batch, label,
                condition, False, args.seed + i,
            )
            append_ledger(args.ledger, rows)
            ledger += rows

            records = sorted(shared + unique, key=lambda r: r["id"])
            out = args.out_dir / f"batch_{label}_{condition}.jsonl"
            with out.open("w", encoding="utf-8") as fh:
                for rec in records:
                    fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
            print(f"  batch {label}  {condition:9s} {len(records):4d} sentences → {out.name}")

        remaining = len(pool) - len(assigned_ids(ledger))
        print(f"\n{remaining} sentences still unassigned")
        return 0

    if args.cmd == "status":
        taken = assigned_ids(ledger)
        print(f"pool {len(pool)}   assigned {len(taken)}   remaining {len(pool) - len(taken)}")
        by_person: dict[str, int] = {}
        by_batch: dict[str, int] = {}
        for row in ledger:
            by_person[row["assignee"]] = by_person.get(row["assignee"], 0) + 1
            key = f"{row['batch']} ({row['condition']})"
            by_batch[key] = by_batch.get(key, 0) + 1
        if by_person:
            print("\nper annotator:")
            for who, n in sorted(by_person.items(), key=lambda kv: -kv[1]):
                print(f"  {who:24s} {n:5d}")
            print("\nper batch (annotation slots, not unique sentences):")
            for b, n in sorted(by_batch.items()):
                print(f"  {b:24s} {n:5d}")
        return 0

    if not args.overlap and len(args.assignee) != 1:
        raise SystemExit("multiple --assignee requires --overlap; otherwise pass exactly one")

    seed = args.seed if args.seed is not None else abs(hash(args.batch)) % (2**31)
    chosen, rows = reserve(
        pool, ledger, args.assignee, args.n, args.batch, args.condition,
        args.overlap, seed,
    )

    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as fh:
        for rec in chosen:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    append_ledger(args.ledger, rows)

    who = ", ".join(args.assignee)
    kind = "overlap" if args.overlap else "unique"
    print(f"reserved {len(chosen)} sentences ({kind}, {args.condition}) for {who}")
    print(f"  batch '{args.batch}' → {args.out}")
    print(f"  ledger += {len(rows)} rows → {args.ledger}")
    remaining = len(pool) - len(assigned_ids(ledger + rows))
    print(f"  {remaining} sentences still unassigned")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
