"""Tests for the training-data ablation.

The property that makes the ablation valid is that only the *training* data changes —
the test set must be byte-identical across conditions. If it drifted, the comparison
would silently be measuring two different benchmarks.
"""

from __future__ import annotations

import pytest

from src.train.ablation import CONDITIONS, training_set
from src.train.data import Example, split_examples

GOLD = [Example([f"g{i}"], ["O"], source=f"gold-{i}") for i in range(10)]
WIKI = [Example([f"w{i}"], ["O"], source=f"wiki-{i}") for i in range(20)]


def test_gold_only_uses_no_wikiann():
    train = training_set("gold_only", GOLD, WIKI)
    assert len(train) == 10
    assert all(e.source.startswith("gold-") for e in train)


def test_wikiann_only_uses_no_gold():
    train = training_set("wikiann_only", GOLD, WIKI)
    assert len(train) == 20
    assert all(e.source.startswith("wiki-") for e in train)


def test_combined_is_the_union():
    train = training_set("gold_plus_wikiann", GOLD, WIKI)
    assert len(train) == 30
    assert {e.source for e in train} == {e.source for e in GOLD + WIKI}


def test_conditions_do_not_mutate_the_inputs():
    before = list(GOLD)
    training_set("gold_plus_wikiann", GOLD, WIKI)
    training_set("gold_only", GOLD, WIKI)
    assert GOLD == before, "a condition modified the shared gold list in place"


def test_unknown_condition_is_rejected():
    with pytest.raises(ValueError, match="unknown condition"):
        training_set("magic", GOLD, WIKI)


@pytest.mark.parametrize("condition", CONDITIONS)
def test_every_condition_shares_one_test_set(condition):
    """The whole ablation rests on this: the test split never varies with the condition."""
    gold_train, dev, test = split_examples(GOLD, 0.2, 0.2, seed=42)
    baseline_ids = [e.source for e in test]
    # Building any condition's training set must leave the split untouched.
    training_set(condition, gold_train, WIKI)
    _, _, test_again = split_examples(GOLD, 0.2, 0.2, seed=42)
    assert [e.source for e in test_again] == baseline_ids


def test_test_set_never_leaks_into_training():
    gold_train, dev, test = split_examples(GOLD, 0.2, 0.2, seed=42)
    test_ids = {e.source for e in test}
    for condition in CONDITIONS:
        train_ids = {e.source for e in training_set(condition, gold_train, WIKI)}
        assert not (train_ids & test_ids), f"{condition} trains on test sentences"
