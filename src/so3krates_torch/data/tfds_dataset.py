import logging
from typing import List, Optional

import numpy as np

try:
    import tensorflow_datasets as tfds

    HAS_TFDS = True
except ImportError:
    HAS_TFDS = False

from ase import Atoms

from .utils import Configuration, KeySpecification, config_from_atoms
from so3krates_torch.tools.utils import AtomicNumberTable

from .base import BaseAtomicDataset
from .cache import DiskCache

logger = logging.getLogger(__name__)


class TFDSDataset(BaseAtomicDataset):
    """Map-style dataset for TensorFlow Datasets (TFDS) format.

    Requires ``tensorflow-datasets`` to be installed. The TFDS
    dataset is loaded once and indexed by integer position.

    The expected TFDS schema should contain at minimum:
    - ``atomic_numbers``: int tensor [n_atoms]
    - ``positions``: float tensor [n_atoms, 3]

    Optional fields (mapped to ASE info/arrays):
    - ``energy``: float scalar
    - ``forces``: float tensor [n_atoms, 3]
    - ``cell``: float tensor [3, 3]
    - ``pbc``: bool tensor [3]
    - ``stress``: float tensor [6] (Voigt notation)
    - ``dipole``: float tensor [3]
    - ``charges``: float tensor [n_atoms]
    - ``hirshfeld_ratios``: float tensor [n_atoms]
    """

    # Map from TFDS field names to ASE info/arrays keys
    INFO_FIELD_MAP = {
        "energy": "REF_energy",
        "stress": "REF_stress",
        "dipole": "REF_dipole",
    }
    ARRAYS_FIELD_MAP = {
        "forces": "REF_forces",
        "charges": "REF_charges",
        "hirshfeld_ratios": "REF_hirsh_ratios",
    }

    def __init__(
        self,
        dataset_name: str,
        split: str,
        cutoff: float,
        cutoff_lr: Optional[float] = None,
        data_dir: Optional[str] = None,
        z_table: Optional[AtomicNumberTable] = None,
        heads: Optional[List[str]] = None,
        head_name: Optional[str] = None,
        key_specification: Optional[KeySpecification] = None,
        cache: Optional[DiskCache] = None,
    ):
        if not HAS_TFDS:
            raise ImportError(
                "TFDSDataset requires tensorflow-datasets. "
                "Install with: pip install tensorflow-datasets"
            )
        super().__init__(
            cutoff=cutoff,
            cutoff_lr=cutoff_lr,
            z_table=z_table,
            heads=heads,
            cache=cache,
        )
        self._head_name = head_name
        self._key_specification = (
            key_specification
            if key_specification is not None
            else KeySpecification()
        )

        ds = tfds.load(dataset_name, split=split, data_dir=data_dir)
        # Materialize as a list for random access
        self._examples = list(ds.as_numpy_iterator())
        self._length = len(self._examples)
        logger.info(
            f"TFDSDataset: {self._length} examples from "
            f"{dataset_name}[{split}]"
        )

    @staticmethod
    def _tfds_to_atoms(example: dict) -> Atoms:
        """Convert a TFDS example dict to an ASE Atoms object."""
        positions = np.array(example["positions"], dtype=np.float64)
        atomic_numbers = np.array(example["atomic_numbers"], dtype=np.int32)

        cell = None
        pbc = False
        if "cell" in example and example["cell"] is not None:
            cell = np.array(example["cell"], dtype=np.float64)
            pbc = True
        if "pbc" in example and example["pbc"] is not None:
            pbc = np.array(example["pbc"], dtype=bool)

        atoms = Atoms(
            numbers=atomic_numbers,
            positions=positions,
            cell=cell,
            pbc=pbc,
        )

        # Store target properties in ASE info/arrays
        for tfds_key, ase_key in TFDSDataset.INFO_FIELD_MAP.items():
            if tfds_key in example and example[tfds_key] is not None:
                val = example[tfds_key]
                atoms.info[ase_key] = (
                    float(val)
                    if np.ndim(val) == 0
                    else np.array(val, dtype=np.float64)
                )

        for tfds_key, ase_key in TFDSDataset.ARRAYS_FIELD_MAP.items():
            if tfds_key in example and example[tfds_key] is not None:
                atoms.arrays[ase_key] = np.array(
                    example[tfds_key], dtype=np.float64
                )

        return atoms

    def __len__(self) -> int:
        return self._length

    def _load_config(self, idx: int) -> Configuration:
        example = self._examples[idx]
        atoms = self._tfds_to_atoms(example)
        return config_from_atoms(
            atoms,
            key_specification=self._key_specification,
            head_name=self._head_name,
        )
