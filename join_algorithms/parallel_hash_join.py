import multiprocessing as mp
from typing import TypeVar, Final, ClassVar, Hashable, Any, Dict, Protocol
from dataclasses import astuple
from join_algorithms.base import BaseAlgorithm, BaseDataset
from join_algorithms.hash_join import HashJoinAlgorithm


class DataClassProtocol(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]


T = TypeVar("T", bound=DataClassProtocol)
U = TypeVar("U", bound=DataClassProtocol)
V = TypeVar("V", bound=DataClassProtocol)


class ParallelHashJoinAlgorithm(BaseAlgorithm[T, U, V]):
    NUM_WORKERS: Final[int] = max(1, mp.cpu_count() - 1)
    algorithm_name = "Parallel Hash Join"

    def __init__(self) -> None:
        super().__init__()
        self.hash_joiner = HashJoinAlgorithm[T, U, V]()

    def _worker_join(
        self,
        worker_id: int,
        dataset1: BaseDataset[T],
        dataset2_chunk: BaseDataset[U],
        build_key_idx: int,
        probe_key_idx: int,
    ) -> BaseDataset[V]:
        """
        Worker function to perform hash join on partitions of the datasets.
        We can also partition before sending to workers so each worker doesn't need to scan entire datasets.
        If we pre-partition, we scan and send only relevant partitions to each worker.
        """
        a_partition, b_partition = [], []

        for row in dataset1:
            if hash(astuple(row)[build_key_idx]) % self.NUM_WORKERS == worker_id:
                a_partition.append(row)

        for row in dataset2_chunk:
            if hash(astuple(row)[probe_key_idx]) % self.NUM_WORKERS == worker_id:
                b_partition.append(row)

        print(
            f"Worker {worker_id} processing {len(a_partition)} rows from dataset1 and {len(b_partition)} rows from dataset2."
        )

        a_dataset = BaseDataset[T](rows=a_partition)
        b_dataset = BaseDataset[U](rows=b_partition)
        return self.hash_joiner.join(a_dataset, b_dataset, build_key_idx, probe_key_idx)

    def join(
        self,
        dataset1: BaseDataset[T],
        dataset2: BaseDataset[U],
        build_key_idx: int,
        probe_key_idx: int,
    ) -> BaseDataset[V]:

        with mp.Pool(processes=self.NUM_WORKERS) as pool:
            args = [
                (worker_id, dataset1, dataset2, build_key_idx, probe_key_idx)
                for worker_id in range(self.NUM_WORKERS)
            ]
            results = pool.starmap(self._worker_join, args)

        joined_rows = []
        for result in results:
            joined_rows.extend(result.rows)

        print("Sample joined rows:")
        print("\n".join(map(str, joined_rows[:10])))
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
    class AB:
        id: int
        name: str
        value: float

    dataset1 = BaseDataset[A](rows=[A(i, f"name_{i}") for i in range(1000)])
    dataset2 = BaseDataset[B](rows=[B(i, float(i) * 1.5) for i in range(500, 600)])
    parallel_hash_join = ParallelHashJoinAlgorithm[A, B, AB]()
    print(parallel_hash_join.algorithm_name)
    result_dataset = parallel_hash_join.join(
        dataset1, dataset2, build_key_idx=0, probe_key_idx=0
    )
    print(f"Joined {len(result_dataset.rows)} rows.")
