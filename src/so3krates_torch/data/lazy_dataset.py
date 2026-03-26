"""
Lazy Dataset for on-the-fly loading and preprocessing of raw HDF5 data.

Each __getitem__ reads one structure from raw HDF5 and computes its
neighbor list on the fly. Use with DataLoader(num_workers>0) for
parallel preprocessing that overlaps with training.
"""

import json
import logging

import h5py
import torch

from so3krates_torch.data.atomic_data import AtomicData
from so3krates_torch.data.utils import (
    KeySpecification,
    config_from_atoms,
)
from so3krates_torch.tools.utils import AtomicNumberTable


class LazyAtomicDataset(torch.utils.data.Dataset):
    """Lazily loads and preprocesses raw HDF5 structures.

    Each __getitem__ reads one structure from raw HDF5 and computes
    its neighbor list on the fly. Designed for use with
    DataLoader(num_workers>0) so that multiple workers preprocess
    in parallel while the GPU trains.

    Args:
        hdf5_path: Path to raw HDF5 file.
        r_max: Short-range cutoff for neighbor lists.
        r_max_lr: Long-range cutoff (None to skip).
        keyspec: KeySpecification for property mapping.
        indices: Optional subset of indices (for train/val split).
        z_table: AtomicNumberTable (default: all 118 elements).
        heads: List of head names for multi-head models.
    """

    def __init__(
        self,
        hdf5_path: str,
        r_max: float,
        r_max_lr: float,
        keyspec: KeySpecification,
        indices=None,
        z_table: AtomicNumberTable = None,
        heads: list = None,
    ):
        self.hdf5_path = hdf5_path
        self.r_max = r_max
        self.r_max_lr = r_max_lr
        self.keyspec = keyspec
        self.z_table = z_table or AtomicNumberTable(
            [int(z) for z in range(1, 119)]
        )
        self.heads = heads or ["Default"]

        # Read metadata (fast — no data loading)
        with h5py.File(hdf5_path, "r") as f:
            self.num_configs = int(f.attrs["num_configs"])
            # Read stored keyspec if available and none provided
            if (
                not keyspec.info_keys
                and not keyspec.arrays_keys
                and "keyspec_info" in f.attrs
            ):
                info_keys = json.loads(f.attrs["keyspec_info"])
                arrays_keys = json.loads(f.attrs["keyspec_arrays"])
                self.keyspec = KeySpecification(
                    info_keys=info_keys,
                    arrays_keys=arrays_keys,
                )

        self.indices = (
            indices if indices is not None else list(range(self.num_configs))
        )

        logging.info(
            f"LazyAtomicDataset: {len(self.indices)} structures "
            f"from {hdf5_path} (r_max={r_max}, r_max_lr={r_max_lr})"
        )

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> AtomicData:
        actual_idx = self.indices[idx]

        # Each worker opens its own file handle (safe for
        # concurrent reads)
        from so3krates_torch.data.hdf5_utils import (
            load_atoms_from_hdf5,
        )

        atoms = load_atoms_from_hdf5(self.hdf5_path, index=actual_idx)
        config = config_from_atoms(atoms, key_specification=self.keyspec)
        return AtomicData.from_config(
            config,
            z_table=self.z_table,
            cutoff=float(self.r_max),
            cutoff_lr=self.r_max_lr,
            heads=self.heads,
        )
