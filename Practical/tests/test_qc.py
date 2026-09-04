"""Tests for per-annotator quality control.

The flagging rule has to survive two ways of being wrong: staying silent when an
annotator genuinely skipped a label, and crying wolf when they simply had few chances
to use it.
"""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

from src.annotate.qc import binomial_zero_prob

REPO = Path(__file__).resolve().parents[1]


def rec(annotator, spans, sid="s1", n_tokens=10):
    return {
        "id": sid,
        "annotator": annotator,
        "tokens": [f"t{i}" for i in range(n_tokens)],
        "spans": spans,
        "junk": False,
        "lead_time_s": 5.0,
    }


def span(start, end, head_start=None, head_end=None):
    return {
        "start": start,
        "end": end,
        "head_start": start if head_start is None else head_start,
        "head_end": end if head_end is None else head_end,
        "type": "LOC",
    }


def run_qc(tmp_path, records, extra=()):
    src = tmp_path / "in.jsonl"
    src.write_text("\n".join(json.dumps(r) for r in records), encoding="utf-8")
    out = tmp_path / "qc.json"
    subprocess.run(
        [sys.executable, "-m", "src.annotate.qc", "--in", str(src), "--out", str(out), *extra],
        cwd=REPO,
        check=True,
        capture_output=True,
    )
    return json.loads(out.read_text())


def test_binomial_zero_prob_shrinks_with_more_opportunities():
    assert binomial_zero_prob(0.1, 1) == 0.9
    assert binomial_zero_prob(0.1, 50) < 1e-2
    assert binomial_zero_prob(0.0, 100) == 1.0


def test_annotator_who_never_narrows_is_flagged(tmp_path):
    # One annotator narrows heads; the other never does, over many opportunities.
    records = []
    for i in range(30):
        records.append(rec("good", [span(0, 2, 2, 2)], sid=f"g{i}"))
        records.append(rec("lazy", [span(0, 2)], sid=f"l{i}"))
    report = run_qc(tmp_path, records)
    assert report["flagged"] == ["lazy"]
    by = {r["annotator"]: r for r in report["annotators"]}
    assert by["lazy"]["narrow_rate"] == 0.0
    assert by["good"]["narrow_rate"] == 1.0


def test_too_few_opportunities_is_not_flagged(tmp_path):
    # Three multi-token entities and no narrowing is not evidence of anything.
    records = [rec("good", [span(0, 2, 2, 2)], sid=f"g{i}") for i in range(30)]
    records += [rec("sparse", [span(0, 2)], sid=f"s{i}") for i in range(3)]
    report = run_qc(tmp_path, records)
    assert report["flagged"] == []


def test_single_token_entities_are_not_opportunities(tmp_path):
    # A one-token entity cannot have a narrower head, so it must not count toward
    # the denominator — otherwise every annotator looks negligent.
    records = [rec("good", [span(0, 2, 2, 2)], sid=f"g{i}") for i in range(30)]
    records += [rec("shorty", [span(1, 1)], sid=f"s{i}") for i in range(50)]
    report = run_qc(tmp_path, records)
    by = {r["annotator"]: r for r in report["annotators"]}
    assert by["shorty"]["multi_token"] == 0
    assert by["shorty"]["narrow_rate"] is None
    assert report["flagged"] == []
