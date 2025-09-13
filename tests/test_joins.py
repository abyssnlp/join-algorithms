import pytest
from dataclasses import dataclass
from join_algorithms.hash_join import HashJoinAlgorithm
from join_algorithms.sort_merge_join import SortMergeJoinAlgorithm
from join_algorithms.parallel_hash_join import ParallelHashJoinAlgorithm

from join_algorithms.base import BaseDataset


@dataclass(frozen=True)
class A:
    id: int
    name: str


@dataclass(frozen=True)
class B:
    id: int
    value: float


@dataclass(frozen=True)
class AB:
    id: int
    name: str
    value: float


@pytest.mark.parametrize(
    "JoinClass",
    [
        HashJoinAlgorithm[A, B, AB],
        SortMergeJoinAlgorithm[A, B, AB],
        ParallelHashJoinAlgorithm[A, B, AB],
    ],
)
def test_basic_join(JoinClass):
    dataset1 = BaseDataset[A](
        rows=[
            A(1, "Alice"),
            A(2, "Bob"),
            A(2, "Bobby"),
            A(3, "Charlie"),
        ]
    )

    dataset2 = BaseDataset[B](
        rows=[
            B(4, 100.0),
            B(2, 200.0),
            B(3, 300.0),
            B(2, 250.0),
        ]
    )

    joiner = JoinClass[A, B, AB]()
    result = joiner.join(dataset1, dataset2, build_key_idx=0, probe_key_idx=0)

    expected_rows = [
        AB(2, "Bob", 200.0),
        AB(2, "Bob", 250.0),
        AB(2, "Bobby", 200.0),
        AB(2, "Bobby", 250.0),
        AB(3, "Charlie", 300.0),
    ]

    assert len(result.rows) == len(expected_rows)
    for row in expected_rows:
        assert row in result.rows


@pytest.mark.parametrize(
    "JoinClass",
    [
        HashJoinAlgorithm[A, B, AB],
        SortMergeJoinAlgorithm[A, B, AB],
        ParallelHashJoinAlgorithm[A, B, AB],
    ],
)
def test_empty_datasets(JoinClass):
    dataset1 = BaseDataset[A](rows=[])
    dataset2 = BaseDataset[B](rows=[])

    joiner = JoinClass[A, B, AB]()
    result = joiner.join(dataset1, dataset2, build_key_idx=0, probe_key_idx=0)

    assert result.rows == []


@pytest.mark.parametrize(
    "JoinClass",
    [
        HashJoinAlgorithm[A, B, AB],
        SortMergeJoinAlgorithm[A, B, AB],
        ParallelHashJoinAlgorithm[A, B, AB],
    ],
)
def test_invalid_key_index(JoinClass):
    dataset1 = BaseDataset[A](rows=[A(1, "Alice")])
    dataset2 = BaseDataset[B](rows=[B(1, 100.0)])

    joiner = JoinClass[A, B, AB]()
    with pytest.raises(IndexError):
        joiner.join(dataset1, dataset2, build_key_idx=5, probe_key_idx=0)

    with pytest.raises(IndexError):
        joiner.join(dataset1, dataset2, build_key_idx=0, probe_key_idx=5)
