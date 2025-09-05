from typing import TypeVar
from collections import defaultdict
from dataclasses import astuple
from join_algorithms.algorithms.base import BaseAlgorithm, BaseDataset

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


class HashJoinAlgorithm(BaseAlgorithm[T, U, V]):
    algorithm_name = "Hash Join"

    def __init__(self):
        super().__init__()
        self.hash_table = defaultdict(list)
        self._result_type = self._extract_result_type()

    def join(
        self,
        dataset1: BaseDataset[T],
        dataset2: BaseDataset[U],
        build_key_idx: int,
        probe_key_idx: int,
    ) -> BaseDataset[V]:
        self.hash_table.clear()
        # build phase
        for row in dataset1:
            key = astuple(row)[build_key_idx]
            self.hash_table[key].append(row)

        # probe phase
        joined_rows = []
        for row in dataset2:
            key = astuple(row)[probe_key_idx]
            if key in self.hash_table:
                for match_row in self.hash_table[key]:
                    combined_tuple = self._combine_rows(match_row, row, probe_key_idx)

                    if self._result_type:
                        result_obj = self._result_type(*combined_tuple)
                    else:
                        result_obj = combined_tuple
                    joined_rows.append(result_obj)

        return BaseDataset[V](rows=joined_rows)

    @property
    def get_hash_table(self):
        return self.hash_table


if __name__ == "__main__":
    from dataclasses import dataclass

    @dataclass(slots=True, frozen=True)
    class A:
        id: int
        name: str

    @dataclass(slots=True, frozen=True)
    class B:
        id: int
        value: float

    @dataclass(slots=True, frozen=True)
    class C:
        id: int
        name: str
        value: float

    hash_join = HashJoinAlgorithm[A, B, C]()
    print(hash_join.algorithm_name)
    print(
        hash_join.join(
            BaseDataset[A](
                rows=[
                    A(1, "Alice"),
                    A(2, "Bob"),
                    A(3, "Charlie"),
                ]
            ),
            BaseDataset[B](
                rows=[
                    B(1, 100.0),
                    B(2, 200.0),
                    B(4, 400.0),
                ]
            ),
            build_key_idx=0,
            probe_key_idx=0,
        )
    )
    print(hash_join.get_hash_table)
