from abc import ABC, abstractmethod
from dataclasses import dataclass, astuple
from typing import (
    Generic,
    TypeVar,
    Sequence,
    get_args,
    ClassVar,
    Any,
    Dict,
    Protocol,
    get_origin,
    Optional,
)


class DataClassProtocol(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]


T = TypeVar("T", bound=DataClassProtocol)
U = TypeVar("U", bound=DataClassProtocol)
V = TypeVar("V", bound=DataClassProtocol)


@dataclass(frozen=True)
class BaseDataset(Generic[T]):
    rows: Sequence[T]

    def __len__(self) -> int:
        return len(self.rows)

    def __iter__(self):
        return iter(self.rows)


class BaseAlgorithm(ABC, Generic[T, U, V]):
    algorithm_name: str

    def __init__(self) -> None:
        self._result_type: Optional[type] = None

    @classmethod
    def __class_getitem__(cls, params):
        class ParameterizedAlgorithm(cls):
            _type_params = params if isinstance(params, tuple) else (params,)

            def __init__(self) -> None:
                super().__init__()
                if len(self._type_params) >= 3:
                    self._result_type = self._type_params[2]

        return ParameterizedAlgorithm

    def _extract_result_type(self):
        """
        Extract the result type V from the generic parameters.
        """

        if hasattr(self, "_type_params") and len(getattr(self, "_type_params")) >= 3:
            return getattr(self, "_type_params")[2]

        if hasattr(self, "__orig_class__"):
            origin = getattr(self, "__orig_class__")
            args = get_args(origin)
            if len(args) >= 3:
                return args[2]

        for base in getattr(self.__class__, "__orig_bases__", []):
            if get_origin(base) is BaseAlgorithm:
                args = get_args(base)
                if len(args) >= 3:
                    return args[2]

        return None

    def _set_result_type(self):
        if self._result_type is None:
            self._result_type = self._extract_result_type()

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

    def _combine_rows(self, row1: T, row2: U, probe_key_idx: int) -> tuple:
        row1_tuple = astuple(row1)
        row2_tuple = astuple(row2)
        row2_without_key = row2_tuple[:probe_key_idx] + row2_tuple[probe_key_idx + 1 :]
        return row1_tuple + row2_without_key

    def _create_result_object(self, combined_tuple: tuple) -> V:
        self._set_result_type()

        if self._result_type and not isinstance(self._result_type, TypeVar):
            try:
                return self._result_type(*combined_tuple)
            except TypeError as e:
                raise TypeError(
                    f"Error creating result object of type {self._result_type} with data {combined_tuple}: {e}"
                ) from e
        return combined_tuple  # type: ignore
