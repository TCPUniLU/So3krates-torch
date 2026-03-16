import logging
from typing import List, Optional

from ase.io import read as ase_read

from .utils import Configuration, KeySpecification, config_from_atoms
from so3krates_torch.tools.utils import AtomicNumberTable

from .base import BaseAtomicDataset
from .cache import DiskCache

logger = logging.getLogger(__name__)


class XYZDataset(BaseAtomicDataset):
    """Map-style dataset for XYZ / ExtXYZ files.

    Two modes of operation controlled by the ``lazy`` parameter:

    - **Mode A** (``lazy=False``, default): All ASE Atoms objects are
      loaded into memory at init, but graph construction (neighbor
      lists) is deferred to ``__getitem__``.
    - **Mode B** (``lazy=True``): Only a frame count is determined at
      init. Individual frames are read from disk on demand via
      ``ase.io.read(path, index=idx)``.
    """

    def __init__(
        self,
        file_path: str,
        cutoff: float,
        cutoff_lr: Optional[float] = None,
        z_table: Optional[AtomicNumberTable] = None,
        heads: Optional[List[str]] = None,
        head_name: Optional[str] = None,
        key_specification: Optional[KeySpecification] = None,
        cache: Optional[DiskCache] = None,
        lazy: bool = False,
    ):
        super().__init__(
            cutoff=cutoff,
            cutoff_lr=cutoff_lr,
            z_table=z_table,
            heads=heads,
            cache=cache,
        )
        self._file_path = file_path
        self._head_name = head_name
        self._key_specification = (
            key_specification
            if key_specification is not None
            else KeySpecification()
        )
        self._lazy = lazy
        self._atoms_list = None
        self._length = None

        if lazy:
            self._length = self._count_frames()
            logger.info(
                f"XYZDataset (lazy): {self._length} frames "
                f"from {file_path}"
            )
        else:
            self._atoms_list = ase_read(file_path, index=":")
            self._length = len(self._atoms_list)
            logger.info(
                f"XYZDataset: loaded {self._length} frames "
                f"from {file_path}"
            )

    def _count_frames(self) -> int:
        """Count frames without loading full data into memory."""
        from ase.io import iread

        count = 0
        for _ in iread(self._file_path):
            count += 1
        return count

    def __len__(self) -> int:
        return self._length

    def _load_config(self, idx: int) -> Configuration:
        if self._lazy:
            atoms = ase_read(self._file_path, index=idx)
        else:
            atoms = self._atoms_list[idx]
        return config_from_atoms(
            atoms,
            key_specification=self._key_specification,
            head_name=self._head_name,
        )
