import os
import multiprocessing as mp
from dataclasses import dataclass
from typing import Final


@dataclass(frozen=True)
class JoinConfig:
    EXTERNAL_SORT_MEMORY_LIMIT: Final[int] = 10
    GRACE_HASH_PARTITIONS: Final[int] = 5
    PARALLEL_WORKERS: Final[int] = max(1, mp.cpu_count() - 1)
    TEMP_DIR: Final[str] = os.path.join(os.getcwd(), "temp")


DEFAULT_CONFIG: Final[JoinConfig] = JoinConfig()
