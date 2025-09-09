import os
import heapq
import pickle
import uuid
from typing import TypeVar, Final, List, Iterator, ClassVar, Any, Dict, Protocol
from dataclasses import astuple
from join_algorithms.base import BaseAlgorithm, BaseDataset
from join_algorithms.sort_merge_join import SortMergeJoinAlgorithm


class DataClassProtocol(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]


T = TypeVar("T", bound=DataClassProtocol)
U = TypeVar("U", bound=DataClassProtocol)
V = TypeVar("V", bound=DataClassProtocol)


class ExternalSortMergeAlgorithm(BaseAlgorithm[T, U, V]):
    MEMORY_LIMIT: Final[int] = 10  # number of rows to hold in-memory
    TMP_DIR: Final[str] = os.path.join(os.getcwd(), "temp")
    algorithm_name = "External Sort-Merge Join"

    def __init__(self):
        super().__init__()
        self._result_type = self._extract_result_type()
        self.sort_merge_joiner = SortMergeJoinAlgorithm[T, U, V]()
        self.temp_files = []

    def _write_sorted_run(self, rows: list) -> str:
        temp_file = os.path.join(
            self.TMP_DIR,
            f"sorted_run_{len(self.temp_files)}_{uuid.uuid4().hex[:8]}.tmp",
        )

        with open(temp_file, "wb") as f:
            pickle.dump(rows, f)
        return temp_file

    def _read_sorted_run(self, file_path: str) -> Iterator[Any]:
        with open(file_path, "rb") as f:
            rows = pickle.load(f)
            for row in rows:
                yield row

    def _merge_sorted_runs(self, temp_files: List[str], key_idx: int) -> Iterator[Any]:
        if not temp_files:
            return iter([])

        iterators = [self._read_sorted_run(f) for f in temp_files]
        heap = []

        for i, it in enumerate(iterators):
            try:
                record = next(it)
                heapq.heappush(heap, (astuple(record)[key_idx], i, record))
            except StopIteration:
                pass

        while heap:
            _, run_idx, record = heapq.heappop(heap)
            yield record
            try:
                next_record = next(iterators[run_idx])
                heapq.heappush(
                    heap, (astuple(next_record)[key_idx], run_idx, next_record)
                )
            except StopIteration:
                pass

    def _external_sort(self, dataset: BaseDataset, key_idx: int):
        temp_files = []
        buffer = []

        for row in dataset:
            buffer.append(row)

            if len(buffer) >= self.MEMORY_LIMIT:
                buffer.sort(key=lambda r: astuple(r)[key_idx])
                temp_file = self._write_sorted_run(buffer)
                temp_files.append(temp_file)
                buffer = []

        if buffer:
            buffer.sort(key=lambda r: astuple(r)[key_idx])
            temp_file = self._write_sorted_run(buffer)
            temp_files.append(temp_file)

        self.temp_files.extend(temp_files)
        return temp_files

    def join(
        self,
        dataset1: BaseDataset[T],
        dataset2: BaseDataset[U],
        build_key_idx: int,
        probe_key_idx: int,
    ) -> BaseDataset[V]:
        try:
            temp_files1 = self._external_sort(dataset1, build_key_idx)
            temp_files2 = self._external_sort(dataset2, probe_key_idx)
            sorted_dataset1 = self._merge_sorted_runs(temp_files1, build_key_idx)
            sorted_dataset2 = self._merge_sorted_runs(temp_files2, probe_key_idx)

            # ideally we'd use iterators throughout, but the sort-merge join implementation
            # expects BaseDataset inputs, so we convert the iterators to lists here.
            return self.sort_merge_joiner.join(
                BaseDataset[T](rows=list(sorted_dataset1)),  # type: ignore
                BaseDataset[U](rows=list(sorted_dataset2)),  # type: ignore
                build_key_idx,
                probe_key_idx,
            )
        finally:
            for f in self.temp_files:
                try:
                    if os.path.exists(f):
                        os.remove(f)
                except Exception as e:
                    print(f"Error cleaning up file {f}: {e}")
            self.temp_files = []


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

    data1 = [A(i, f"name_{i}") for i in range(21)]
    data2 = [B(i, float(i) * 1.5) for i in range(15, 35)]

    external_sort_merge = ExternalSortMergeAlgorithm[A, B, C]()
    print(external_sort_merge.algorithm_name)
    print(
        external_sort_merge.join(
            BaseDataset[A](rows=data1), BaseDataset[B](rows=data2), 0, 0
        )
    )
