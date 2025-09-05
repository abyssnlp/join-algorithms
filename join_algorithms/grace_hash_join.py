import os
from typing import TypeVar, Final, Hashable
from collections import defaultdict
from dataclasses import astuple
from join_algorithms.base import BaseAlgorithm, BaseDataset
from join_algorithms.hash_join import HashJoinAlgorithm

T = TypeVar("T")
U = TypeVar("U")
V = TypeVar("V")
PartitionKey = TypeVar("PartitionKey", bound=Hashable)


class GraceHashJoinAlgorithm(BaseAlgorithm[T, U, V]):
    NUM_PARTITIONS: Final[int] = 5
    TMP_DIR: Final[str] = os.path.join(os.getcwd(), "temp")
    algorithm_name = "Grace Hash Join"

    def __init__(self):
        super().__init__()
        self.hash_joiner = HashJoinAlgorithm[T, U, V]()
        os.makedirs(self.TMP_DIR, exist_ok=True)
        self._result_type = self._extract_result_type()

    def _hash_function(self, key: PartitionKey) -> int:
        return hash(key) % self.NUM_PARTITIONS

    def _partition_datasets(
        self,
        dataset1: BaseDataset[T],
        dataset2: BaseDataset[U],
        build_key_idx: int,
        probe_key_idx: int,
    ):
        partition_files1 = [
            open(os.path.join(self.TMP_DIR, f"partition1_{i}.tmp"), "w+")
            for i in range(self.NUM_PARTITIONS)
        ]
        partition_files2 = [
            open(os.path.join(self.TMP_DIR, f"partition2_{i}.tmp"), "w+")
            for i in range(self.NUM_PARTITIONS)
        ]

        for row in dataset1:
            key = astuple(row)[build_key_idx]
            part_key = self._hash_function(key)

            partition_files1[part_key].write(f"{str(row)}\n")

        for row in dataset2:
            key = astuple(row)[probe_key_idx]
            part_key = self._hash_function(key)

            partition_files2[part_key].write(f"{str(row)}\n")

        return partition_files1, partition_files2

    def join(
        self,
        dataset1: BaseDataset[T],
        dataset2: BaseDataset[U],
        build_key_idx: int,
        probe_key_idx: int,
    ) -> BaseDataset[V]:
        try:
            partition_files1, partition_files2 = self._partition_datasets(
                dataset1, dataset2, build_key_idx, probe_key_idx
            )

            joined_rows = []

            for i in range(self.NUM_PARTITIONS):
                partition_files1[i].seek(0)
                partition_files2[i].seek(0)

                # dangerous; only for demo purposes
                build_side = [eval(row.strip()) for row in partition_files1[i]]
                probe_side = [eval(row.strip()) for row in partition_files2[i]]

                partial_joined = self.hash_joiner.join(
                    BaseDataset[T](rows=build_side),
                    BaseDataset[U](rows=probe_side),
                    build_key_idx,
                    probe_key_idx,
                ).rows
                joined_rows.extend(partial_joined)
            return BaseDataset[V](rows=joined_rows)

        finally:
            for f in partition_files1 + partition_files2:
                try:
                    f.close()
                    os.remove(f.name)
                except Exception as e:
                    print(f"Error cleaning up file {f.name}: {e}")


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

    grace_hash_join = GraceHashJoinAlgorithm[A, B, C]()
    print(grace_hash_join.algorithm_name)
    print(
        grace_hash_join.join(
            BaseDataset[A](
                rows=[
                    A("abc123", "Alice"),
                    A("def123", "Bob"),
                    A("po1k23", "Charlie"),
                    A("asd13214", "David"),
                    A("kmo9000", "Eve"),
                ]
            ),
            BaseDataset[B](
                rows=[
                    B("po1k23", 300.0),
                    B("asd13214", 400.0),
                    B("kmo9000", 500.0),
                    B("kmo9000", 550.0),
                    B("imoi8989", 600.0),
                    B("iomoqw12", 700.0),
                ]
            ),
            build_key_idx=0,
            probe_key_idx=0,
        )
    )
