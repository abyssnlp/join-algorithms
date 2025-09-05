from abc import ABC, abstractmethod
from dataclasses import dataclass, astuple
from typing import Generic, TypeVar, Sequence, get_args

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")


@dataclass(frozen=True)
class BaseDataset(Generic[T]):
    rows: Sequence[T]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


class BaseAlgorithm(ABC, Generic[T, U, V]):
    algorithm_name: str

    def _extract_result_type(self):
        """
        Extract the result type V from the generic parameters.
        """
        origin = getattr(self, "__orig_class__", None)
        if origin is not None:
            args = get_args(origin)
            if len(args) >= 3:
                return args[2]
        return None

    @abstractmethod
    def join(
        self,
        dataset1: BaseDataset[T],
        dataset2: BaseDataset[U],
        build_key_idx: int,
        probe_key_idx: int,
    ) -> BaseDataset[V]:
        """
        Perform a join between two datasets on specified key indices.

        Args:
            dataset1: The dataset to build the hash table from.
            dataset2: The dataset to probe against the hash table.
            build_key_idx: The index of the key in dataset1 to build the hash table on
            probe_key_idx: The index of the key in dataset2 to probe against the hash table

        Returns:
            A new dataset containing the joined rows.
        """
        pass

    def _combine_rows(self, row1: T, row2: U, probe_key_idx: int) -> V:
        row1_tuple = astuple(row1)
        row2_tuple = astuple(row2)
        row2_without_key = row2_tuple[:probe_key_idx] + row2_tuple[probe_key_idx + 1 :]
        return row1_tuple + row2_without_key
