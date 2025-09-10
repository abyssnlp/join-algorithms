from typing import TypeVar, ClassVar, Any, Dict, Protocol
from dataclasses import astuple
from join_algorithms.base import BaseAlgorithm, BaseDataset


class DataClassProtocol(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]


T = TypeVar("T", bound=DataClassProtocol)
U = TypeVar("U", bound=DataClassProtocol)
V = TypeVar("V", bound=DataClassProtocol)


class SortMergeJoinAlgorithm(BaseAlgorithm[T, U, V]):
    algorithm_name = "Sort Merge Join"

    def __init__(self):
        super().__init__()
        self._result_type = self._extract_result_type()

    def join(
        self,
        dataset1: BaseDataset[T],
        dataset2: BaseDataset[U],
        build_key_idx: int,
        probe_key_idx: int,
    ) -> BaseDataset[V]:
        # sort phase
        sorted_dataset1 = sorted(dataset1, key=lambda row: astuple(row)[build_key_idx])
        sorted_dataset2 = sorted(dataset2, key=lambda row: astuple(row)[probe_key_idx])

        # merge phase
        i, j = 0, 0
        joined_rows = []

        while i < len(sorted_dataset1) and j < len(sorted_dataset2):
            row1 = sorted_dataset1[i]
            row2 = sorted_dataset2[j]
            key1 = astuple(row1)[build_key_idx]
            key2 = astuple(row2)[probe_key_idx]

            if key1 < key2:
                i += 1
            elif key1 > key2:
                j += 1
            else:
                current_key = key1

                # get all matching rows in dataset1 and dataset2
                i_start = i
                j_start = j

                while (
                    i < len(sorted_dataset1)
                    and astuple(sorted_dataset1[i])[build_key_idx] == current_key
                ):
                    i += 1

                while (
                    j < len(sorted_dataset2)
                    and astuple(sorted_dataset2[j])[probe_key_idx] == current_key
                ):
                    j += 1

                i_end = i
                j_end = j

                # cartesian
                for row1_idx in range(i_start, i_end):
                    for row2_idx in range(j_start, j_end):
                        row1 = sorted_dataset1[row1_idx]
                        row2 = sorted_dataset2[row2_idx]

                        combined_tuple = self._combine_rows(row1, row2, probe_key_idx)
                        result_obj = self._create_result_object(combined_tuple)
                        joined_rows.append(result_obj)

        return BaseDataset[V](rows=joined_rows)


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

    algorithm = SortMergeJoinAlgorithm[A, B, C]()
    result_dataset = algorithm.join(
        dataset1, dataset2, build_key_idx=0, probe_key_idx=0
    )

    print(result_dataset)
