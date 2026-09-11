from __future__ import annotations

from sklearn.model_selection import GroupShuffleSplit


def test_grouped_split_keeps_replicates_together() -> None:
    groups = ["a", "a", "b", "b"]
    train, test = next(GroupShuffleSplit(n_splits=1, test_size=0.5, random_state=1).split([[0], [1], [2], [3]], groups=groups))
    assert {groups[i] for i in train}.isdisjoint({groups[i] for i in test})
