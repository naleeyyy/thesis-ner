"""Per-annotator quality control over collected annotations.

Agreement statistics compare annotators on the *shared* sentences only, which is a
20-sentence window. An annotator can look fine there and still apply the protocol
inconsistently across their own 100. This module checks each annotator's whole output
for the failure modes that agreement cannot see.

The one that matters most here is silent non-use of a label. Under the dual-boundary
convention a head is only marked where it is narrower than the full span, so an
annotator who never marks one produces records that are indistinguishable, field by
field, from an annotator who judged every span to be its own head. The difference only
appears as a rate: across a multi-token entity pool, peers narrow 10--49% of the time,
so a rate of zero over dozens of opportunities is a protocol failure rather than a
run of judgements.

    python -m src.annotate.qc --in data/labeled/collected/all.jsonl
"""

from __future__ import annotations

import argparse
import json
import statistics as st
import sys
from collections import defaultdict
from pathlib import Path


def binomial_zero_prob(rate: float, n: int) -> float:
    """P(0 successes in n trials at `rate`) — how surprising an all-zero run is."""
    return (1.0 - rate) ** n


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    p.add_argument("--in", dest="inp", type=Path, required=True)
    p.add_argument("--out", type=Path, default=None, help="Write the report as JSON.")
    p.add_argument(
        "--min-opportunities",
        type=int,
        default=20,
        help="Below this many multi-token entities, a zero rate is not yet evidence.",
    )
    args = p.parse_args()

    per: dict[str, dict] = defaultdict(
        lambda: {"sentences": 0, "entities": 0, "multi": 0, "narrowed": 0, "times": [], "junk": 0}
    )
    for line in args.inp.open(encoding="utf-8"):
        if not line.strip():
            continue
        rec = json.loads(line)
        a = rec.get("annotator") or "unknown"
        d = per[a]
        d["sentences"] += 1
        d["junk"] += 1 if rec.get("junk") else 0
        if rec.get("lead_time_s"):
            d["times"].append(rec["lead_time_s"])
        for s in rec["spans"]:
            d["entities"] += 1
            if s["end"] > s["start"]:
                d["multi"] += 1
                if (s.get("head_start"), s.get("head_end")) != (s["start"], s["end"]):
                    d["narrowed"] += 1

    rates = [
        d["narrowed"] / d["multi"]
        for d in per.values()
        if d["multi"] >= args.min_opportunities and d["narrowed"] > 0
    ]
    floor = min(rates) if rates else 0.0

    rows, flagged = [], []
    for a, d in sorted(per.items(), key=lambda kv: kv[0]):
        rate = d["narrowed"] / d["multi"] if d["multi"] else float("nan")
        row = {
            "annotator": a,
            "sentences": d["sentences"],
            "entities": d["entities"],
            "multi_token": d["multi"],
            "narrowed": d["narrowed"],
            "narrow_rate": None if d["multi"] == 0 else round(rate, 4),
            "median_time_s": round(st.median(d["times"]), 1) if d["times"] else None,
            "junk_flagged": d["junk"],
        }
        # Flag only when there were enough chances for zero to mean something.
        if d["multi"] >= args.min_opportunities and d["narrowed"] == 0 and floor > 0:
            row["flag"] = "never marked a head"
            row["p_if_lowest_peer_rate"] = float(f"{binomial_zero_prob(floor, d['multi']):.3g}")
            flagged.append(a)
        rows.append(row)

    print(f"{'ann':>6} {'sents':>6} {'ents':>5} {'multi':>6} {'narrow':>7} {'rate':>7} {'med s':>7}")
    for r in rows:
        rate = "  n/a  " if r["narrow_rate"] is None else f"{100 * r['narrow_rate']:6.1f}%"
        note = f"   <-- {r['flag']} (p={r['p_if_lowest_peer_rate']:.1e})" if "flag" in r else ""
        print(
            f"{r['annotator']:>6} {r['sentences']:6d} {r['entities']:5d} {r['multi_token']:6d} "
            f"{r['narrowed']:7d} {rate} {r['median_time_s'] or 0:7.1f}{note}"
        )

    if flagged:
        print(
            f"\n{len(flagged)} annotator(s) flagged: {', '.join(flagged)}.\n"
            "Their head annotations default to the full span, so head-view results computed\n"
            "over their sentences are biased toward full-span behaviour. Either exclude them\n"
            "from head-view analysis or have the affected sentences re-annotated."
        )

    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(
            json.dumps({"annotators": rows, "flagged": flagged}, indent=2) + "\n", encoding="utf-8"
        )
        print(f"\n-> {args.out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
