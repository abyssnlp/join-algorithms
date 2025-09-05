from typing import TypeVar
from dataclasses import astuple
from join_algorithms.base import BaseAlgorithm, BaseDataset

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class ExternalSortMergeAlgorithm(BaseAlgorithm[T, U, V]):
    algorithm_name = "External Sort-Merge Join"

    def __init__(self):
        super().__init__()
        self._result_type = self._extract_result_type()

    def join(
        self,
        dataset1: BaseDataset[T],
        dataset2: BaseDataset[U],
        build_key_idx: int,
        probe_key_idx: int,
    ) -> BaseDataset[V]: ...
