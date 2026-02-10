"""
HDF5 Data Loading and Preprocessing Utilities

This module provides comprehensive HDF5 support for atomic structure data:
- Raw HDF5: Store atomic structures in efficient binary format
- Preprocessed HDF5: Cache fully preprocessed AtomicData with neighbor lists
"""

import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import ase
import ase.io
import h5py
import numpy as np
import torch

from so3krates_torch.data.atomic_data import AtomicData
from so3krates_torch.data.utils import (
    Configuration,
    Configurations,
    KeySpecification,
)
from so3krates_torch.tools.utils import AtomicNumberTable

# Format version for compatibility tracking
RAW_HDF5_FORMAT_VERSION = "1.0"
PREPROCESSED_HDF5_FORMAT_VERSION = "1.0"


# ============================================================================
# Raw HDF5 Format (Atomic Structures)
# ============================================================================


def save_atoms_to_hdf5(
    atoms_list: List[ase.Atoms],
    output_path: str,
    key_specification: Optional[KeySpecification] = None,
    description: Optional[str] = None,
) -> None:
    """
    Save ASE Atoms list to raw HDF5.

    Args:
        atoms_list: List of ASE Atoms objects
        output_path: Path to output HDF5 file
        key_specification: Optional key specification for properties
        description: Optional dataset description
    """
    if key_specification is None:
        key_specification = KeySpecification()

    with h5py.File(output_path, "w") as f:
        # Write metadata
        f.attrs["format_version"] = RAW_HDF5_FORMAT_VERSION
        f.attrs["format_type"] = "raw"
        f.attrs["num_configs"] = len(atoms_list)
        f.attrs["timestamp"] = datetime.now().isoformat()
        if description:
            f.attrs["description"] = description

        # Write each configuration
        for i, atoms in enumerate(atoms_list):
            group = f.create_group(f"config_{i}")
            _write_atoms_to_hdf5_group(
                group, atoms, key_specification
            )

    logging.info(
        f"Saved {len(atoms_list)} configurations to raw HDF5: "
        f"{output_path}"
    )


def _write_atoms_to_hdf5_group(
    group: h5py.Group,
    atoms: ase.Atoms,
    key_spec: KeySpecification,
) -> None:
    """Write single ASE Atoms object to HDF5 group."""
    # Basic atomic structure
    group["atomic_numbers"] = atoms.get_atomic_numbers().astype(np.int32)
    group["positions"] = atoms.get_positions().astype(np.float64)
    group["cell"] = np.array(atoms.get_cell()).astype(np.float64)
    group["pbc"] = np.array(atoms.get_pbc(), dtype=bool)

    # Properties subgroup
    properties_grp = group.create_group("properties")

    # Store info dict properties
    for prop_name, key in key_spec.info_keys.items():
        if key in atoms.info:
            value = atoms.info[key]
            if value is not None:
                properties_grp[prop_name] = value

    # Store arrays dict properties
    for prop_name, key in key_spec.arrays_keys.items():
        if key in atoms.arrays:
            value = atoms.arrays[key]
            if value is not None:
                properties_grp[prop_name] = value

    # Store config metadata if present
    if "config_type" in atoms.info:
        group["config_type"] = str(atoms.info["config_type"])
    if "head" in atoms.info:
        group["head"] = str(atoms.info["head"])

    # Store weights if present
    weights_grp = group.create_group("property_weights")
    if "energy_weight" in atoms.info:
        weights_grp["energy"] = atoms.info["energy_weight"]
    if "forces_weight" in atoms.info:
        weights_grp["forces"] = atoms.info["forces_weight"]
    if "stress_weight" in atoms.info:
        weights_grp["stress"] = atoms.info["stress_weight"]
    if "config_weight" in atoms.info:
        group["weight"] = atoms.info["config_weight"]


def load_atoms_from_hdf5(
    hdf5_path: str,
    index: Optional[Union[int, slice, List[int]]] = None,
) -> Union[ase.Atoms, List[ase.Atoms]]:
    """
    Load ASE Atoms from raw HDF5 (mimics ase.io.read).

    Args:
        hdf5_path: Path to HDF5 file
        index: Index, slice, or list of indices to load (None = all)

    Returns:
        Single Atoms object or list of Atoms objects
    """
    with h5py.File(hdf5_path, "r") as f:
        num_configs = f.attrs["num_configs"]

        # Parse index
        if index is None:
            indices = list(range(num_configs))
        elif isinstance(index, int):
            indices = [index]
        elif isinstance(index, slice):
            indices = list(range(*index.indices(num_configs)))
        else:
            indices = list(index)

        # Load atoms
        atoms_list = []
        for i in indices:
            group = f[f"config_{i}"]
            atoms = _read_atoms_from_hdf5_group(group)
            atoms_list.append(atoms)

    # Return single or list
    if isinstance(index, int):
        return atoms_list[0]
    return atoms_list


def _read_atoms_from_hdf5_group(group: h5py.Group) -> ase.Atoms:
    """Read single ASE Atoms object from HDF5 group."""
    atomic_numbers = np.array(group["atomic_numbers"])
    positions = np.array(group["positions"])
    cell = np.array(group["cell"])
    pbc = np.array(group["pbc"])

    atoms = ase.Atoms(
        numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        pbc=pbc,
    )

    # Read properties
    if "properties" in group:
        for key in group["properties"].keys():
            value = np.array(group["properties"][key])
            # Store as info or arrays depending on shape
            if value.ndim == 0 or (
                value.ndim == 1 and len(value) == 1
            ):
                atoms.info[key] = value.item() if value.ndim == 0 else value
            else:
                if len(value) == len(atoms):
                    atoms.arrays[key] = value
                else:
                    atoms.info[key] = value

    # Read metadata
    if "config_type" in group:
        atoms.info["config_type"] = group["config_type"][()].decode()
    if "head" in group:
        atoms.info["head"] = group["head"][()].decode()
    if "weight" in group:
        atoms.info["config_weight"] = float(group["weight"][()])

    # Read property weights
    if "property_weights" in group:
        for key in group["property_weights"].keys():
            atoms.info[f"{key}_weight"] = float(
                group["property_weights"][key][()]
            )

    return atoms


def configs_from_hdf5(
    hdf5_path: str,
    key_specification: Optional[KeySpecification] = None,
    head_name: str = "Default",
) -> List[Configuration]:
    """
    Load Configuration objects directly from raw HDF5.

    Args:
        hdf5_path: Path to HDF5 file
        key_specification: Key specification for properties
        head_name: Head name for configurations

    Returns:
        List of Configuration objects
    """
    if key_specification is None:
        key_specification = KeySpecification()

    # Load atoms first
    atoms_list = load_atoms_from_hdf5(hdf5_path, index=None)

    # Import here to avoid circular dependency
    from so3krates_torch.tools.utils import create_configs_from_list

    # Create configurations using existing utility
    configs = create_configs_from_list(
        atoms_list,
        key_specification,
        head_name=head_name,
    )

    return configs


# ============================================================================
# Preprocessed HDF5 Format (AtomicData with neighbor lists)
# ============================================================================


def save_preprocessed_hdf5(
    data_list: List[AtomicData],
    output_path: str,
    r_max: float,
    r_max_lr: Optional[float],
    z_table: AtomicNumberTable,
    description: Optional[str] = None,
) -> None:
    """
    Save preprocessed AtomicData list to HDF5.

    Args:
        data_list: List of AtomicData objects
        output_path: Path to output HDF5 file
        r_max: Short-range cutoff used for neighbor lists
        r_max_lr: Long-range cutoff (optional)
        z_table: Atomic number table
        description: Optional dataset description
    """
    with h5py.File(output_path, "w") as f:
        # Write metadata
        f.attrs["format_version"] = PREPROCESSED_HDF5_FORMAT_VERSION
        f.attrs["format_type"] = "preprocessed"
        f.attrs["num_configs"] = len(data_list)
        f.attrs["r_max"] = r_max
        if r_max_lr is not None:
            f.attrs["r_max_lr"] = r_max_lr
        f.attrs["z_table"] = z_table.zs
        f.attrs["timestamp"] = datetime.now().isoformat()
        if description:
            f.attrs["description"] = description

        # Compute average number of neighbors
        total_neighbors = sum(
            data.edge_index.shape[1] for data in data_list
        )
        total_atoms = sum(data.num_nodes for data in data_list)
        avg_num_neighbors = total_neighbors / max(total_atoms, 1)
        f.attrs["avg_num_neighbors"] = avg_num_neighbors

        # Write each configuration
        for i, data in enumerate(data_list):
            group = f.create_group(f"config_{i}")
            _write_atomic_data_to_hdf5_group(group, data)

    logging.info(
        f"Saved {len(data_list)} preprocessed configurations to HDF5: "
        f"{output_path}"
    )


def _write_atomic_data_to_hdf5_group(
    group: h5py.Group, data: AtomicData
) -> None:
    """Write single AtomicData to HDF5 group."""
    # Basic structure
    group["num_nodes"] = data.num_nodes
    group["positions"] = data.positions.cpu().numpy()
    group["atomic_numbers"] = data.atomic_numbers.cpu().numpy()
    group["node_attrs"] = data.node_attrs.cpu().numpy()
    if data.cell is not None:
        group["cell"] = data.cell.cpu().numpy()

    # Weights and metadata
    if data.weight is not None:
        group["weight"] = data.weight.cpu().numpy()
    if data.head is not None:
        group["head"] = data.head.cpu().numpy()

    # Short-range edges
    edges_grp = group.create_group("edges")
    edges_grp["edge_index"] = data.edge_index.cpu().numpy()
    edges_grp["shifts"] = data.shifts.cpu().numpy()
    edges_grp["unit_shifts"] = data.unit_shifts.cpu().numpy()

    # Long-range edges (optional)
    if (
        data.edge_index_lr is not None
        and data.edge_index_lr.numel() > 0
    ):
        edges_lr_grp = group.create_group("edges_lr")
        edges_lr_grp["edge_index_lr"] = (
            data.edge_index_lr.cpu().numpy()
        )
        edges_lr_grp["shifts_lr"] = data.shifts_lr.cpu().numpy()
        edges_lr_grp["unit_shifts_lr"] = (
            data.unit_shifts_lr.cpu().numpy()
        )

    # Properties
    props_grp = group.create_group("properties")
    if data.energy is not None:
        props_grp["energy"] = data.energy.cpu().numpy()
    if data.forces is not None:
        props_grp["forces"] = data.forces.cpu().numpy()
    if data.stress is not None:
        props_grp["stress"] = data.stress.cpu().numpy()
    if data.virials is not None:
        props_grp["virials"] = data.virials.cpu().numpy()
    if data.dipole is not None:
        props_grp["dipole"] = data.dipole.cpu().numpy()
    if data.charges is not None:
        props_grp["charges"] = data.charges.cpu().numpy()
    if data.hirshfeld_ratios is not None:
        props_grp["hirshfeld_ratios"] = (
            data.hirshfeld_ratios.cpu().numpy()
        )

    # Property weights
    weights_grp = group.create_group("weights")
    if data.energy_weight is not None:
        weights_grp["energy_weight"] = data.energy_weight.cpu().numpy()
    if data.forces_weight is not None:
        weights_grp["forces_weight"] = data.forces_weight.cpu().numpy()
    if data.stress_weight is not None:
        weights_grp["stress_weight"] = data.stress_weight.cpu().numpy()
    if data.virials_weight is not None:
        weights_grp["virials_weight"] = (
            data.virials_weight.cpu().numpy()
        )
    if data.dipole_weight is not None:
        weights_grp["dipole_weight"] = data.dipole_weight.cpu().numpy()
    if data.charges_weight is not None:
        weights_grp["charges_weight"] = (
            data.charges_weight.cpu().numpy()
        )
    if data.hirshfeld_ratios_weight is not None:
        weights_grp["hirshfeld_ratios_weight"] = (
            data.hirshfeld_ratios_weight.cpu().numpy()
        )


def _read_atomic_data_from_hdf5_group(
    group: h5py.Group, z_table: AtomicNumberTable
) -> AtomicData:
    """Read single AtomicData from HDF5 group."""
    # Basic structure
    num_nodes = int(group["num_nodes"][()])
    positions = torch.from_numpy(np.array(group["positions"]))
    atomic_numbers = torch.from_numpy(np.array(group["atomic_numbers"]))
    node_attrs = torch.from_numpy(np.array(group["node_attrs"]))
    cell = (
        torch.from_numpy(np.array(group["cell"]))
        if "cell" in group
        else None
    )

    # Weights and metadata
    weight = (
        torch.from_numpy(np.array(group["weight"]))
        if "weight" in group
        else torch.tensor(1.0)
    )
    head = (
        torch.from_numpy(np.array(group["head"]))
        if "head" in group
        else torch.tensor(0, dtype=torch.long)
    )

    # Short-range edges
    edges_grp = group["edges"]
    edge_index = torch.from_numpy(np.array(edges_grp["edge_index"]))
    shifts = torch.from_numpy(np.array(edges_grp["shifts"]))
    unit_shifts = torch.from_numpy(np.array(edges_grp["unit_shifts"]))

    # Long-range edges (optional)
    edge_index_lr = None
    shifts_lr = None
    unit_shifts_lr = None
    if "edges_lr" in group:
        edges_lr_grp = group["edges_lr"]
        edge_index_lr = torch.from_numpy(
            np.array(edges_lr_grp["edge_index_lr"])
        )
        shifts_lr = torch.from_numpy(np.array(edges_lr_grp["shifts_lr"]))
        unit_shifts_lr = torch.from_numpy(
            np.array(edges_lr_grp["unit_shifts_lr"])
        )

    # Properties
    props_grp = group["properties"]
    energy = (
        torch.from_numpy(np.array(props_grp["energy"]))
        if "energy" in props_grp
        else None
    )
    forces = (
        torch.from_numpy(np.array(props_grp["forces"]))
        if "forces" in props_grp
        else None
    )
    stress = (
        torch.from_numpy(np.array(props_grp["stress"]))
        if "stress" in props_grp
        else None
    )
    virials = (
        torch.from_numpy(np.array(props_grp["virials"]))
        if "virials" in props_grp
        else None
    )
    dipole = (
        torch.from_numpy(np.array(props_grp["dipole"]))
        if "dipole" in props_grp
        else None
    )
    charges = (
        torch.from_numpy(np.array(props_grp["charges"]))
        if "charges" in props_grp
        else None
    )
    hirshfeld_ratios = (
        torch.from_numpy(np.array(props_grp["hirshfeld_ratios"]))
        if "hirshfeld_ratios" in props_grp
        else None
    )

    # Property weights
    weights_grp = group["weights"]
    energy_weight = (
        torch.from_numpy(np.array(weights_grp["energy_weight"]))
        if "energy_weight" in weights_grp
        else torch.tensor(1.0)
    )
    forces_weight = (
        torch.from_numpy(np.array(weights_grp["forces_weight"]))
        if "forces_weight" in weights_grp
        else torch.tensor(1.0)
    )
    stress_weight = (
        torch.from_numpy(np.array(weights_grp["stress_weight"]))
        if "stress_weight" in weights_grp
        else torch.tensor(1.0)
    )
    virials_weight = (
        torch.from_numpy(np.array(weights_grp["virials_weight"]))
        if "virials_weight" in weights_grp
        else torch.tensor(1.0)
    )
    dipole_weight = (
        torch.from_numpy(np.array(weights_grp["dipole_weight"]))
        if "dipole_weight" in weights_grp
        else None
    )
    charges_weight = (
        torch.from_numpy(np.array(weights_grp["charges_weight"]))
        if "charges_weight" in weights_grp
        else torch.tensor(1.0)
    )
    hirshfeld_ratios_weight = (
        torch.from_numpy(
            np.array(weights_grp["hirshfeld_ratios_weight"])
        )
        if "hirshfeld_ratios_weight" in weights_grp
        else torch.tensor(1.0)
    )

    # Create AtomicData
    return AtomicData(
        edge_index=edge_index,
        node_attrs=node_attrs,
        atomic_numbers=atomic_numbers,
        positions=positions,
        shifts=shifts,
        unit_shifts=unit_shifts,
        cell=cell,
        weight=weight,
        head=head,
        energy_weight=energy_weight,
        forces_weight=forces_weight,
        stress_weight=stress_weight,
        virials_weight=virials_weight,
        dipole_weight=dipole_weight,
        charges_weight=charges_weight,
        hirshfeld_ratios_weight=hirshfeld_ratios_weight,
        forces=forces,
        energy=energy,
        stress=stress,
        virials=virials,
        dipole=dipole,
        hirshfeld_ratios=hirshfeld_ratios,
        charges=charges,
        elec_temp=None,
        edge_index_lr=edge_index_lr,
        shifts_lr=shifts_lr,
        unit_shifts_lr=unit_shifts_lr,
    )


class PreprocessedHDF5Dataset(torch.utils.data.Dataset):
    """
    PyTorch Dataset for lazy-loading preprocessed HDF5.

    Args:
        hdf5_path: Path to preprocessed HDF5 file
        validate_cutoffs: Validate r_max matches file metadata
        expected_r_max: Expected short-range cutoff
        expected_r_max_lr: Expected long-range cutoff
    """

    def __init__(
        self,
        hdf5_path: str,
        validate_cutoffs: bool = True,
        expected_r_max: Optional[float] = None,
        expected_r_max_lr: Optional[float] = None,
    ):
        self.hdf5_path = hdf5_path

        # Read metadata
        with h5py.File(hdf5_path, "r") as f:
            self.num_configs = f.attrs["num_configs"]
            self.r_max = f.attrs["r_max"]
            self.r_max_lr = (
                f.attrs["r_max_lr"] if "r_max_lr" in f.attrs else None
            )
            z_table_zs = f.attrs["z_table"]
            self.z_table = AtomicNumberTable(z_table_zs.tolist())
            self.metadata = dict(f.attrs)

        # Validate cutoffs
        if validate_cutoffs:
            if (
                expected_r_max is not None
                and abs(self.r_max - expected_r_max) > 1e-6
            ):
                raise ValueError(
                    f"r_max mismatch: file has {self.r_max}, "
                    f"expected {expected_r_max}"
                )
            if expected_r_max_lr is not None:
                if self.r_max_lr is None:
                    raise ValueError(
                        f"expected r_max_lr={expected_r_max_lr} but "
                        f"file has no long-range cutoff"
                    )
                if abs(self.r_max_lr - expected_r_max_lr) > 1e-6:
                    raise ValueError(
                        f"r_max_lr mismatch: file has {self.r_max_lr}, "
                        f"expected {expected_r_max_lr}"
                    )

        logging.info(
            f"Loaded preprocessed HDF5 dataset: {hdf5_path} "
            f"({self.num_configs} configs, r_max={self.r_max}, "
            f"r_max_lr={self.r_max_lr})"
        )

    def __len__(self) -> int:
        return self.num_configs

    def __getitem__(self, idx: int) -> AtomicData:
        """Lazy load single AtomicData from HDF5."""
        with h5py.File(self.hdf5_path, "r") as f:
            group = f[f"config_{idx}"]
            return _read_atomic_data_from_hdf5_group(group, self.z_table)


# ============================================================================
# Format Detection and Validation
# ============================================================================


def detect_file_format(path: str) -> str:
    """
    Detect format: 'xyz', 'hdf5_raw', or 'hdf5_preprocessed'.

    Args:
        path: Path to file

    Returns:
        Format string
    """
    path_obj = Path(path)

    # Check extension
    if path_obj.suffix.lower() == ".xyz":
        return "xyz"

    if path_obj.suffix.lower() in [".h5", ".hdf5"]:
        # Check HDF5 metadata
        try:
            with h5py.File(path, "r") as f:
                if "format_type" in f.attrs:
                    format_type = f.attrs["format_type"]
                    if isinstance(format_type, bytes):
                        format_type = format_type.decode()
                    if format_type == "raw":
                        return "hdf5_raw"
                    elif format_type == "preprocessed":
                        return "hdf5_preprocessed"

                # Fallback: check for r_max (indicates preprocessed)
                if "r_max" in f.attrs:
                    return "hdf5_preprocessed"
                else:
                    return "hdf5_raw"
        except Exception as e:
            logging.warning(
                f"Could not read HDF5 metadata from {path}: {e}"
            )
            return "hdf5_raw"

    # Unknown format
    raise ValueError(f"Unsupported file format: {path}")


def validate_preprocessed_hdf5(
    hdf5_path: str,
    expected_r_max: Optional[float] = None,
    expected_r_max_lr: Optional[float] = None,
) -> None:
    """
    Validate that HDF5 file contains preprocessed data.

    Args:
        hdf5_path: Path to HDF5 file
        expected_r_max: Expected short-range cutoff
        expected_r_max_lr: Expected long-range cutoff

    Raises:
        ValueError: If file is not preprocessed format or cutoffs
                    don't match
    """
    with h5py.File(hdf5_path, "r") as f:
        # Check format type
        if "format_type" not in f.attrs:
            raise ValueError(
                f"{hdf5_path} is missing format_type metadata"
            )

        format_type = f.attrs["format_type"]
        if isinstance(format_type, bytes):
            format_type = format_type.decode()

        if format_type != "preprocessed":
            raise ValueError(
                f"{hdf5_path} is not preprocessed format "
                f"(format_type={format_type})"
            )

        # Check for r_max
        if "r_max" not in f.attrs:
            raise ValueError(
                f"{hdf5_path} is missing r_max metadata "
                f"(not a valid preprocessed file)"
            )

        r_max = f.attrs["r_max"]

        # Validate r_max
        if expected_r_max is not None:
            if abs(r_max - expected_r_max) > 1e-6:
                raise ValueError(
                    f"r_max mismatch in {hdf5_path}: "
                    f"file has {r_max}, expected {expected_r_max}"
                )

        # Validate r_max_lr
        if expected_r_max_lr is not None:
            if "r_max_lr" not in f.attrs:
                raise ValueError(
                    f"Expected r_max_lr={expected_r_max_lr} but "
                    f"{hdf5_path} has no r_max_lr metadata"
                )
            r_max_lr = f.attrs["r_max_lr"]
            if abs(r_max_lr - expected_r_max_lr) > 1e-6:
                raise ValueError(
                    f"r_max_lr mismatch in {hdf5_path}: "
                    f"file has {r_max_lr}, expected {expected_r_max_lr}"
                )

        # Check structure (verify it has edge_index)
        if f.attrs["num_configs"] > 0:
            group = f["config_0"]
            if "edges" not in group:
                raise ValueError(
                    f"{hdf5_path} is missing 'edges' group "
                    f"(not a valid preprocessed file)"
                )
            if "edge_index" not in group["edges"]:
                raise ValueError(
                    f"{hdf5_path} is missing 'edges/edge_index' "
                    f"(not a valid preprocessed file)"
                )

    logging.info(
        f"Validated preprocessed HDF5: {hdf5_path} "
        f"(r_max={r_max})"
    )
