from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import TypeVar, Final, ClassVar, Any, Dict, Protocol
from dataclasses import astuple
from join_algorithms.base import BaseAlgorithm, BaseDataset
from join_algorithms.hash_join import HashJoinAlgorithm
from join_algorithms.config import DEFAULT_CONFIG


class DataClassProtocol(Protocol):
    __dataclass_fields__: ClassVar[Dict[str, Any]]


T = TypeVar("T", bound=DataClassProtocol)
U = TypeVar("U", bound=DataClassProtocol)
V = TypeVar("V", bound=DataClassProtocol)


class ParallelHashJoinAlgorithm(BaseAlgorithm[T, U, V]):
    NUM_WORKERS: Final[int] = DEFAULT_CONFIG.PARALLEL_WORKERS
    algorithm_name = "Parallel Hash Join"

    def __init__(self) -> None:
        super().__init__()

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
        if hasattr(self, "_type_params") and len(getattr(self, "_type_params")) >= 3:
            params = getattr(self, "_type_params")
            hash_joiner_class = HashJoinAlgorithm[params[0], params[1], params[2]]
            hash_joiner = hash_joiner_class()
        else:
            hash_joiner = HashJoinAlgorithm()
            hash_joiner._result_type = self._result_type

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
        return hash_joiner.join(a_dataset, b_dataset, build_key_idx, probe_key_idx)

    def join(
        self,
        dataset1: BaseDataset[T],
        dataset2: BaseDataset[U],
        build_key_idx: int,
        probe_key_idx: int,
    ) -> BaseDataset[V]:
        joined_rows = []

        with ThreadPoolExecutor(max_workers=self.NUM_WORKERS) as executor:
            futures = [
                executor.submit(
                    self._worker_join,
                    worker_id,
                    dataset1,
                    dataset2,
                    build_key_idx,
                    probe_key_idx,
                )
                for worker_id in range(self.NUM_WORKERS)
            ]
            for future in as_completed(futures):
                try:
                    worker_result = future.result()
                    joined_rows.extend(worker_result.rows)
                except Exception as e:
                    print(f"Worker encountered an error: {e}")
                    raise

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
    print(result_dataset)
    print(f"Joined {len(result_dataset.rows)} rows.")
