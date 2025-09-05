from typing import TypeVar
from dataclasses import astuple
from distributed_join.algorithms.base import BaseAlgorithm, BaseDataset

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class GraceSortMergeJoinAlgorithm(BaseAlgorithm[T, U, V]):
    algorithm_name = "Grace Sort-Merge Join"
