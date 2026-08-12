"""Tests for active-learning selection.

The selection logic is where a silent bug does the most damage: if a strategy ever
re-selects an already-labelled sentence, the "budget" on the x-axis of the learning curve
stops meaning what the plot claims, and the comparison is void.
"""

from __future__ import annotations

import random

import pytest

from src.train.active import select


def test_random_selects_the_requested_count_from_the_unlabelled_set():
    unlabelled = list(range(50))
    picked = select("random", unlabelled, 10, random.Random(1))
    assert len(picked) == 10
    assert set(picked) <= set(unlabelled)
    assert len(set(picked)) == 10, "no duplicates within a round"


def test_random_is_reproducible_for_a_given_seed():
    a = select("random", list(range(50)), 10, random.Random(7))
    b = select("random", list(range(50)), 10, random.Random(7))
    assert a == b


def test_uncertainty_takes_the_highest_scoring_candidates():
    scores = [0.0] * 10
    scores[3], scores[7], scores[1] = 0.9, 0.8, 0.7
    picked = select("uncertainty", list(range(10)), 3, random.Random(0), scores)
    assert picked == [3, 7, 1]


def test_uncertainty_ignores_scores_of_already_labelled_items():
    # index 0 has the highest score but is not a candidate
    scores = [9.9, 0.5, 0.4, 0.3]
    picked = select("uncertainty", [1, 2, 3], 2, random.Random(0), scores)
    assert 0 not in picked
    assert picked == [1, 2]


def test_requesting_more_than_available_returns_everything_once():
    unlabelled = [4, 5, 6]
    for strategy in ("random", "uncertainty"):
        picked = select(strategy, unlabelled, 10, random.Random(0), [0.0] * 10)
        assert sorted(picked) == [4, 5, 6]


def test_uncertainty_without_scores_is_an_error():
    with pytest.raises(ValueError, match="needs scores"):
        select("uncertainty", [1, 2], 1, random.Random(0))


def test_unknown_strategy_is_rejected():
    with pytest.raises(ValueError, match="unknown strategy"):
        select("magic", [1, 2], 1, random.Random(0))


def test_a_full_simulated_run_never_reveals_a_sentence_twice():
    """Mirrors the acquire loop: the union of all rounds must have no repeats."""
    pool_size, rng = 60, random.Random(3)
    labelled = rng.sample(range(pool_size), 10)
    unlabelled = [i for i in range(pool_size) if i not in set(labelled)]
    for _ in range(4):
        scores = [rng.random() for _ in range(pool_size)]
        picked = select("uncertainty", unlabelled, 10, rng, scores)
        assert not (set(picked) & set(labelled)), "re-selected an already-labelled sentence"
        labelled.extend(picked)
        picked_set = set(picked)
        unlabelled = [i for i in unlabelled if i not in picked_set]
    assert len(labelled) == len(set(labelled)) == 50
    assert len(unlabelled) == 10
