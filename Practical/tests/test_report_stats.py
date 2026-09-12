"""Tests for the report-statistics module.

The permutation test is the machinery behind the anchoring result, so it is checked
against cases with known answers rather than trusted because it printed something
plausible once.
"""

from __future__ import annotations

from src.analysis.report_stats import exact_permutation


def test_perfectly_separated_groups_hit_the_floor():
    # Every assisted value exceeds every scratch value: the observed split is the single
    # most extreme of the 252, so p must equal exactly 1/252.
    a = [0.9, 0.91, 0.92, 0.93, 0.94]
    b = [0.5, 0.51, 0.52, 0.53, 0.54]
    r = exact_permutation(a, b)
    assert r["n_splits"] == 252
    assert r["p_one_tailed"] == r["floor_one_tailed"] == round(1 / 252, 5)
    assert r["p_two_tailed"] == round(2 / 252, 5)


def test_identical_groups_are_not_significant():
    r = exact_permutation([1.0, 2.0, 3.0], [1.0, 2.0, 3.0])
    assert r["difference"] == 0.0
    # A zero difference is matched or exceeded by at least half of all splits.
    assert r["p_one_tailed"] >= 0.5


def test_direction_matters_for_one_tailed():
    a = [1.0, 1.0, 1.0]
    b = [2.0, 2.0, 2.0]
    r = exact_permutation(a, b)
    # a is smaller than b, so a difference this large in a's favour never occurs.
    assert r["p_one_tailed"] == 1.0
    assert r["p_two_tailed"] == round(2 / 20, 5)


def test_split_count_is_n_choose_k():
    r = exact_permutation([1, 2, 3, 4], [5, 6])
    assert r["n_splits"] == 15  # 6 choose 4
