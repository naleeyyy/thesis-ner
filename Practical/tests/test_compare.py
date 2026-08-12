"""Tests for the paired model comparison.

The guard that matters: refusing to compare predictions made over different test sets.
Silently pairing mismatched files would produce a confident, meaningless p-value.
"""

from __future__ import annotations

import json
import subprocess
import sys


def write(path, rows):
    path.write_text("\n".join(json.dumps(r) for r in rows), encoding="utf-8")


def run(a, b):
    return subprocess.run(
        [sys.executable, "-m", "src.baselines.compare", str(a), str(b)],
        capture_output=True, text=True,
    )


def test_identical_models_are_not_distinguishable(tmp_path):
    rows = [{"id": i, "gold": ["B-LOC", "O"], "pred": ["B-LOC", "O"]} for i in range(30)]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(a, rows)
    write(b, rows)
    out = run(a, b)
    assert out.returncode == 0, out.stderr
    assert "Not distinguishable" in out.stdout


def test_a_large_gap_is_detected(tmp_path):
    gold = ["B-LOC", "O"]
    good = [{"id": i, "gold": gold, "pred": gold} for i in range(40)]
    bad = [{"id": i, "gold": gold, "pred": ["O", "O"]} for i in range(40)]
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(a, good)
    write(b, bad)
    out = run(a, b)
    assert out.returncode == 0, out.stderr
    assert "is better" in out.stdout


def test_mismatched_gold_is_rejected(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(a, [{"id": 0, "gold": ["B-LOC"], "pred": ["B-LOC"]}])
    write(b, [{"id": 0, "gold": ["B-ORG"], "pred": ["B-ORG"]}])
    out = run(a, b)
    assert out.returncode != 0
    assert "not predictions over the same test set" in out.stderr


def test_different_lengths_are_rejected(tmp_path):
    a, b = tmp_path / "a.jsonl", tmp_path / "b.jsonl"
    write(a, [{"id": 0, "gold": ["O"], "pred": ["O"]}])
    write(b, [{"id": i, "gold": ["O"], "pred": ["O"]} for i in range(2)])
    out = run(a, b)
    assert out.returncode != 0
    assert "different lengths" in out.stderr
