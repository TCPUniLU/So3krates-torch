from abc import abstractmethod
from typing import Iterator, List, Optional

import torch.utils.data

from .atomic_data import AtomicData
from .cache import DiskCache
from .utils import Configuration
from so3krates_torch.tools.utils import AtomicNumberTable


class BaseAtomicDataset(torch.utils.data.Dataset):
    """Abstract map-style dataset that produces AtomicData objects.

    Subclasses implement _load_config(idx) to return a Configuration
    for a given index. Graph construction (neighbor lists) is deferred
    to __getitem__ and optionally cached to disk.
    """

    def __init__(
        self,
        cutoff: float,
        cutoff_lr: Optional[float] = None,
        z_table: Optional[AtomicNumberTable] = None,
        heads: Optional[List[str]] = None,
        cache: Optional[DiskCache] = None,
    ):
        super().__init__()
        self.cutoff = cutoff
        self.cutoff_lr = cutoff_lr
        self.z_table = z_table or AtomicNumberTable(
            [int(z) for z in range(1, 119)]
        )
        self.heads = heads
        self.cache = cache

    @abstractmethod
    def __len__(self) -> int: ...

    @abstractmethod
    def _load_config(self, idx: int) -> Configuration: ...

    def _build_graph(self, config: Configuration) -> AtomicData:
        return AtomicData.from_config(
            config,
            z_table=self.z_table,
            cutoff=self.cutoff,
            cutoff_lr=self.cutoff_lr,
            heads=self.heads,
        )

    def __getitem__(self, idx: int) -> AtomicData:
        if self.cache is not None:
            cached = self.cache.load(idx)
            if cached is not None:
                return cached
        config = self._load_config(idx)
        data = self._build_graph(config)
        if self.cache is not None:
            self.cache.save(idx, data)
        return data

    def iter_configs(self) -> Iterator[Configuration]:
        """Yield Configuration objects without graph construction.

        Useful for compute_average_E0s which needs Configurations
        but not full AtomicData with neighbor lists.
        """
        for idx in range(len(self)):
            yield self._load_config(idx)
