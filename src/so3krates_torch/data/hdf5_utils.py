"""
HDF5 Data Loading and Preprocessing Utilities

This module provides comprehensive HDF5 support for atomic structure data:
- Raw HDF5 v2.0: Columnar layout for fast saving of large datasets
- Preprocessed HDF5: Cache fully preprocessed AtomicData with neighbor lists

Raw HDF5 v2.0 layout
---------------------
All structures are stored as flat arrays with a CSR-style offset index
instead of one HDF5 group per structure.  This avoids O(N) HDF5 object
creations and is orders of magnitude faster for datasets with millions of
structures.

On-disk structure::

    / (attrs)  format_version="2.0", format_type="raw", num_configs, ...
    /n_atoms          [num_configs]       int32
    /offsets          [num_configs+1]     int64  (CSR cumsum of n_atoms)
    /atomic_numbers   [total_atoms]       int32
    /positions        [total_atoms, 3]    float64
    /cell             [num_configs, 3, 3] float64
    /pbc              [num_configs, 3]    bool
    /config_metadata/
        config_type   [num_configs]       str
        head          [num_configs]       str
        weight        [num_configs]       float64  (NaN if absent)
    /property_weights/
        energy        [num_configs]       float64  (NaN if absent)
        forces        [num_configs]       float64  (NaN if absent)
        stress        [num_configs]       float64  (NaN if absent)
    /properties/info/{key}    [num_configs, ...]  float64  (NaN if absent)
    /properties/arrays/{key}  [total_atoms, ...]  float64  (NaN if absent)

Reading a single structure at index i::

    start, end = offsets[i], offsets[i+1]
    atomic_numbers_i = atomic_numbers[start:end]
    positions_i      = positions[start:end, :]
    cell_i           = cell[i]
"""

import itertools
import json
import logging
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple, Union

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
    config_from_atoms,
    compute_average_E0s,
)
from so3krates_torch.tools.utils import (
    AtomicNumberTable,
    create_configs_from_list,
)

# Format version for compatibility tracking
RAW_HDF5_FORMAT_VERSION = "2.0"
PREPROCESSED_HDF5_FORMAT_VERSION = "1.0"

_HDF5_CHUNK = 65536  # chunk size (rows) for all raw v2.0 datasets


# ============================================================================
# Raw HDF5 Format v2.0 — Save
# ============================================================================


def save_atoms_to_hdf5(
    atoms_iter: Union[List[ase.Atoms], Iterable[ase.Atoms]],
    output_path: str,
    key_specification: Optional[KeySpecification] = None,
    description: Optional[str] = None,
    batch_size: int = 100_000,
) -> None:
    """Save ASE Atoms to raw HDF5 using the columnar v2.0 layout.

    Uses flat arrays + a CSR-style offset index instead of one HDF5 group
    per structure.  For large datasets (millions of structures) this is
    dramatically faster because it creates O(1) HDF5 objects regardless of
    dataset size.

    Supports streaming: ``atoms_iter`` may be a generator so that peak RAM
    usage scales with ``batch_size``, not total dataset size.

    Args:
        atoms_iter: List or any iterable (including generators) of ASE
            Atoms objects.
        output_path: Path to the output HDF5 file.
        key_specification: Optional key specification for properties.
        description: Optional dataset description string.
        batch_size: Number of structures processed per write batch.
            Controls peak RAM for generator inputs.
    """
    if key_specification is None:
        key_specification = KeySpecification()

    total_hint = (
        len(atoms_iter) if hasattr(atoms_iter, "__len__") else None
    )

    with h5py.File(output_path, "w") as f:
        f.attrs["format_version"] = RAW_HDF5_FORMAT_VERSION
        f.attrs["format_type"] = "raw"
        f.attrs["timestamp"] = datetime.now().isoformat()
        if description:
            f.attrs["description"] = description
        f.attrs["keyspec_info"] = json.dumps(
            key_specification.info_keys
        )
        f.attrs["keyspec_arrays"] = json.dumps(
            key_specification.arrays_keys
        )

        total_configs, total_atoms = _stream_atoms_to_hdf5_v2(
            f,
            iter(atoms_iter),
            key_specification,
            batch_size,
            total_hint,
        )

        f.attrs["num_configs"] = total_configs
        f.attrs["num_atoms"] = total_atoms

        # Build CSR offset index from n_atoms
        if total_configs > 0:
            n_atoms_all = f["n_atoms"][:]
            offsets = np.concatenate(
                [[0], np.cumsum(n_atoms_all).astype(np.int64)]
            )
            f.create_dataset("offsets", data=offsets)
        else:
            f.create_dataset(
                "offsets", data=np.array([0], dtype=np.int64)
            )

    logging.info(
        f"Saved {total_configs} configurations to raw HDF5 v2.0: "
        f"{output_path}"
    )


def _remap_ase_reserved_keys(
    batch: List[ase.Atoms], key_spec: KeySpecification
) -> None:
    """Populate atoms.info/arrays for ASE-reserved keys.

    Since ASE >= 3.23.0b1, ``energy``, ``forces``, and ``stress`` read from
    extxyz may be stored as calculator results rather than in
    ``atoms.info`` / ``atoms.arrays``.  This mirrors the workaround in
    ``load_from_xyz`` so that raw-mode streaming also captures these values.
    Mutates batch in-place.
    """
    energy_key = key_spec.info_keys.get("energy")
    forces_key = key_spec.arrays_keys.get("forces")
    stress_key = key_spec.info_keys.get("stress")

    for atoms in batch:
        if energy_key == "energy" and energy_key not in atoms.info:
            try:
                atoms.info["energy"] = atoms.get_potential_energy()
            except Exception:
                atoms.info["energy"] = None
        if forces_key == "forces" and forces_key not in atoms.arrays:
            try:
                atoms.arrays["forces"] = atoms.get_forces()
            except Exception:
                atoms.arrays["forces"] = None
        if stress_key == "stress" and stress_key not in atoms.info:
            try:
                atoms.info["stress"] = atoms.get_stress()
            except Exception:
                atoms.info["stress"] = None


def _detect_prop_meta(
    batch: List[ase.Atoms],
    key_spec: KeySpecification,
) -> Dict:
    """Detect which properties are present in the first batch.

    Returns:
        dict mapping ase_key -> (is_per_atom, item_shape, dtype)
        Only keys found in at least one structure are included.
    """
    meta: Dict[str, Tuple[bool, tuple, type]] = {}

    # Info keys — per-config (scalar or small array)
    for _prop_name, key in key_spec.info_keys.items():
        for a in batch:
            if key in a.info and a.info[key] is not None:
                val = np.asarray(a.info[key], dtype=np.float64)
                item_shape = val.shape  # () for scalar, (3,) for dipole
                meta[key] = (False, item_shape, np.float64)
                break

    # Arrays keys — per-atom
    for _prop_name, key in key_spec.arrays_keys.items():
        for a in batch:
            if key in a.arrays and a.arrays[key] is not None:
                val = np.asarray(a.arrays[key], dtype=np.float64)
                # (n_atoms,) → item_shape ()
                # (n_atoms, 3) → item_shape (3,)
                item_shape = val.shape[1:] if val.ndim > 1 else ()
                meta[key] = (True, item_shape, np.float64)
                break

    return meta


def _collect_batch_arrays(
    batch: List[ase.Atoms],
    prop_meta: Dict,
    batch_n_atoms: np.ndarray,
) -> Dict[str, np.ndarray]:
    """Build property arrays for a batch, filling NaN for missing entries."""
    result: Dict[str, np.ndarray] = {}
    batch_total_atoms = int(batch_n_atoms.sum())
    B = len(batch)

    for key, (is_per_atom, item_shape, dtype) in prop_meta.items():
        if is_per_atom:
            arr = np.full(
                (batch_total_atoms,) + item_shape, np.nan, dtype=dtype
            )
            offset = 0
            for a, n in zip(batch, batch_n_atoms):
                n = int(n)
                if key in a.arrays and a.arrays[key] is not None:
                    val = np.asarray(a.arrays[key], dtype=dtype)
                    arr[offset : offset + n] = val.reshape(
                        (n,) + item_shape
                    )
                offset += n
        else:
            # item_shape=() → arr shape (B,); (6,) → shape (B, 6)
            arr = np.full(
                (B,) + item_shape if item_shape else (B,),
                np.nan,
                dtype=dtype,
            )
            for i, a in enumerate(batch):
                if key in a.info and a.info[key] is not None:
                    val = np.asarray(a.info[key], dtype=dtype)
                    if item_shape:
                        arr[i] = val.reshape(item_shape)
                    else:
                        arr[i] = float(val)

        result[key] = arr

    return result


def _append_dataset(ds: h5py.Dataset, new_data) -> None:
    """Resize a chunked HDF5 dataset along axis 0 and append new data."""
    n = len(new_data)
    old = ds.shape[0]
    ds.resize(old + n, axis=0)
    ds[old : old + n] = new_data


def _stream_atoms_to_hdf5_v2(
    f: h5py.File,
    atoms_iter: Iterable[ase.Atoms],
    key_spec: KeySpecification,
    batch_size: int,
    total_hint: Optional[int],
) -> Tuple[int, int]:
    """Stream atoms into an open HDF5 file using columnar v2.0 layout.

    Returns:
        (total_configs, total_atoms)
    """
    str_dt = h5py.string_dtype(encoding="utf-8")
    c = _HDF5_CHUNK

    total_configs = 0
    total_atoms = 0
    prop_meta: Optional[Dict] = None
    datasets_created = False
    start_time = time.time()

    batch_iter = iter(
        lambda: list(itertools.islice(atoms_iter, batch_size)), []
    )

    for batch in batch_iter:
        B = len(batch)
        batch_n_atoms = np.array(
            [len(a) for a in batch], dtype=np.int32
        )
        bat = int(batch_n_atoms.sum())  # total atoms in batch

        # --- Mandatory per-atom arrays ---
        atomic_numbers = np.concatenate(
            [a.get_atomic_numbers().astype(np.int32) for a in batch]
        )
        positions = np.concatenate(
            [a.get_positions().astype(np.float64) for a in batch]
        )

        # --- Mandatory per-config arrays ---
        cell = np.stack(
            [np.array(a.get_cell(), dtype=np.float64) for a in batch]
        )
        pbc = np.stack(
            [np.array(a.get_pbc(), dtype=bool) for a in batch]
        )

        # --- Config metadata ---
        config_type = [a.info.get("config_type", "") for a in batch]
        head_vals = [a.info.get("head", "") for a in batch]
        weight = np.array(
            [a.info.get("config_weight", np.nan) for a in batch],
            dtype=np.float64,
        )

        # --- Property weights ---
        energy_weight = np.array(
            [a.info.get("energy_weight", np.nan) for a in batch],
            dtype=np.float64,
        )
        forces_weight = np.array(
            [a.info.get("forces_weight", np.nan) for a in batch],
            dtype=np.float64,
        )
        stress_weight = np.array(
            [a.info.get("stress_weight", np.nan) for a in batch],
            dtype=np.float64,
        )

        # --- Remap ASE-reserved keys (energy/forces/stress) ---
        _remap_ase_reserved_keys(batch, key_spec)

        # --- Detect property shapes from first batch ---
        if prop_meta is None:
            prop_meta = _detect_prop_meta(batch, key_spec)

        prop_arrays = _collect_batch_arrays(
            batch, prop_meta, batch_n_atoms
        )

        # --- Create all datasets on first batch ---
        if not datasets_created:
            f.create_dataset(
                "n_atoms",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=(c,),
            )
            f.create_dataset(
                "atomic_numbers",
                shape=(0,),
                maxshape=(None,),
                dtype=np.int32,
                chunks=(c,),
            )
            f.create_dataset(
                "positions",
                shape=(0, 3),
                maxshape=(None, 3),
                dtype=np.float64,
                chunks=(c, 3),
            )
            f.create_dataset(
                "cell",
                shape=(0, 3, 3),
                maxshape=(None, 3, 3),
                dtype=np.float64,
                chunks=(c, 3, 3),
            )
            f.create_dataset(
                "pbc",
                shape=(0, 3),
                maxshape=(None, 3),
                dtype=bool,
                chunks=(c, 3),
            )

            meta_grp = f.require_group("config_metadata")
            meta_grp.create_dataset(
                "config_type",
                shape=(0,),
                maxshape=(None,),
                dtype=str_dt,
                chunks=(c,),
            )
            meta_grp.create_dataset(
                "head",
                shape=(0,),
                maxshape=(None,),
                dtype=str_dt,
                chunks=(c,),
            )
            meta_grp.create_dataset(
                "weight",
                shape=(0,),
                maxshape=(None,),
                dtype=np.float64,
                chunks=(c,),
            )

            pw_grp = f.require_group("property_weights")
            for pw_name in ("energy", "forces", "stress"):
                pw_grp.create_dataset(
                    pw_name,
                    shape=(0,),
                    maxshape=(None,),
                    dtype=np.float64,
                    chunks=(c,),
                )

            for key, (is_per_atom, item_shape, dtype) in prop_meta.items():
                grp_path = (
                    "properties/arrays" if is_per_atom
                    else "properties/info"
                )
                grp = f.require_group(grp_path)
                # (c,) + () = (c,);  (c,) + (3,) = (c, 3)
                ds_chunks = (c,) + item_shape
                grp.create_dataset(
                    key,
                    shape=(0,) + item_shape,
                    maxshape=(None,) + item_shape,
                    dtype=dtype,
                    chunks=ds_chunks,
                )

            datasets_created = True

        # --- Append batch to datasets ---
        _append_dataset(f["n_atoms"], batch_n_atoms)
        _append_dataset(f["atomic_numbers"], atomic_numbers)
        _append_dataset(f["positions"], positions)
        _append_dataset(f["cell"], cell)
        _append_dataset(f["pbc"], pbc)
        _append_dataset(f["config_metadata/config_type"], config_type)
        _append_dataset(f["config_metadata/head"], head_vals)
        _append_dataset(f["config_metadata/weight"], weight)
        _append_dataset(f["property_weights/energy"], energy_weight)
        _append_dataset(f["property_weights/forces"], forces_weight)
        _append_dataset(f["property_weights/stress"], stress_weight)

        for key, (is_per_atom, _shape, _dtype) in prop_meta.items():
            if key in prop_arrays:
                subgrp = (
                    "properties/arrays"
                    if is_per_atom
                    else "properties/info"
                )
                _append_dataset(
                    f[f"{subgrp}/{key}"], prop_arrays[key]
                )

        total_configs += B
        total_atoms += bat

        elapsed = time.time() - start_time
        rate = total_configs / elapsed if elapsed > 0 else 0
        hint = f"/{total_hint}" if total_hint else ""
        logging.info(
            f"Progress: {total_configs}{hint} configs "
            f"({rate:.0f} configs/s)"
        )

    return total_configs, total_atoms


# ============================================================================
# Raw HDF5 Format v2.0 — Load
# ============================================================================


def load_atoms_from_hdf5(
    hdf5_path: str,
    index: Optional[Union[int, slice, List[int]]] = None,
) -> Union[ase.Atoms, List[ase.Atoms]]:
    """Load ASE Atoms from raw HDF5 v2.0 (mimics ase.io.read).

    Args:
        hdf5_path: Path to HDF5 file.
        index: Index, slice, or list of indices to load (None = all).

    Returns:
        Single Atoms object or list of Atoms objects.
    """
    with h5py.File(hdf5_path, "r") as f:
        num_configs = int(f.attrs["num_configs"])

        if index is None:
            return _load_all_atoms_v2(f, num_configs)

        if isinstance(index, int):
            idx = index if index >= 0 else num_configs + index
            return _load_single_atom_v2(f, idx)

        if isinstance(index, slice):
            indices = list(range(*index.indices(num_configs)))
        else:
            indices = list(index)

        return [_load_single_atom_v2(f, i) for i in indices]


def _load_all_atoms_v2(
    f: h5py.File, num_configs: int
) -> List[ase.Atoms]:
    """Load all structures at once (one I/O call per flat array)."""
    if num_configs == 0:
        return []

    offsets = f["offsets"][:]
    atomic_numbers_all = f["atomic_numbers"][:]
    positions_all = f["positions"][:]
    cell_all = f["cell"][:]
    pbc_all = f["pbc"][:]

    cfg_meta = f["config_metadata"]
    config_types = cfg_meta["config_type"][:]
    head_vals = cfg_meta["head"][:]
    weights = cfg_meta["weight"][:]

    pw = f["property_weights"]
    energy_weights = pw["energy"][:]
    forces_weights = pw["forces"][:]
    stress_weights = pw["stress"][:]

    info_props: Dict[str, np.ndarray] = {}
    if "properties/info" in f:
        for key in f["properties/info"].keys():
            info_props[key] = f[f"properties/info/{key}"][:]

    arrays_props: Dict[str, np.ndarray] = {}
    if "properties/arrays" in f:
        for key in f["properties/arrays"].keys():
            arrays_props[key] = f[f"properties/arrays/{key}"][:]

    atoms_list = []
    for i in range(num_configs):
        start = int(offsets[i])
        end = int(offsets[i + 1])

        a = ase.Atoms(
            numbers=atomic_numbers_all[start:end],
            positions=positions_all[start:end],
            cell=cell_all[i],
            pbc=pbc_all[i],
        )
        _fill_atoms_metadata(
            a,
            i,
            start,
            end,
            config_types,
            head_vals,
            weights,
            energy_weights,
            forces_weights,
            stress_weights,
            info_props,
            arrays_props,
        )
        atoms_list.append(a)

    return atoms_list


def _load_single_atom_v2(f: h5py.File, idx: int) -> ase.Atoms:
    """Load a single structure by index using hyperslab reads."""
    start, end = int(f["offsets"][idx]), int(f["offsets"][idx + 1])

    a = ase.Atoms(
        numbers=f["atomic_numbers"][start:end],
        positions=f["positions"][start:end],
        cell=f["cell"][idx],
        pbc=f["pbc"][idx],
    )

    # Config metadata
    ct_raw = f["config_metadata/config_type"][idx]
    ct = ct_raw.decode() if isinstance(ct_raw, bytes) else str(ct_raw)
    if ct:
        a.info["config_type"] = ct

    h_raw = f["config_metadata/head"][idx]
    h = h_raw.decode() if isinstance(h_raw, bytes) else str(h_raw)
    if h:
        a.info["head"] = h

    w = float(f["config_metadata/weight"][idx])
    if not np.isnan(w):
        a.info["config_weight"] = w

    for pw_name, info_key in [
        ("energy", "energy_weight"),
        ("forces", "forces_weight"),
        ("stress", "stress_weight"),
    ]:
        v = float(f[f"property_weights/{pw_name}"][idx])
        if not np.isnan(v):
            a.info[info_key] = v

    # Info properties (per-config)
    if "properties/info" in f:
        for key in f["properties/info"].keys():
            val = np.array(f[f"properties/info/{key}"][idx])
            if not np.all(np.isnan(np.atleast_1d(val).ravel())):
                if val.ndim == 0:
                    a.info[key] = float(val)
                else:
                    a.info[key] = val

    # Arrays properties (per-atom)
    if "properties/arrays" in f:
        for key in f["properties/arrays"].keys():
            val = np.array(
                f[f"properties/arrays/{key}"][start:end]
            )
            if not np.all(np.isnan(val.ravel())):
                a.arrays[key] = val

    return a


def _fill_atoms_metadata(
    a: ase.Atoms,
    i: int,
    start: int,
    end: int,
    config_types,
    head_vals,
    weights,
    energy_weights,
    forces_weights,
    stress_weights,
    info_props: Dict[str, np.ndarray],
    arrays_props: Dict[str, np.ndarray],
) -> None:
    """Fill metadata onto an already-constructed Atoms (batch-load path)."""
    ct_raw = config_types[i]
    ct = ct_raw.decode() if isinstance(ct_raw, bytes) else str(ct_raw)
    if ct:
        a.info["config_type"] = ct

    h_raw = head_vals[i]
    h = h_raw.decode() if isinstance(h_raw, bytes) else str(h_raw)
    if h:
        a.info["head"] = h

    w = float(weights[i])
    if not np.isnan(w):
        a.info["config_weight"] = w

    ew = float(energy_weights[i])
    if not np.isnan(ew):
        a.info["energy_weight"] = ew

    fw = float(forces_weights[i])
    if not np.isnan(fw):
        a.info["forces_weight"] = fw

    sw = float(stress_weights[i])
    if not np.isnan(sw):
        a.info["stress_weight"] = sw

    for key, vals in info_props.items():
        val = vals[i]
        if not np.all(np.isnan(np.atleast_1d(val).ravel())):
            if np.ndim(val) == 0:
                a.info[key] = float(val)
            else:
                a.info[key] = np.asarray(val)

    for key, vals in arrays_props.items():
        val = vals[start:end]
        if not np.all(np.isnan(val.ravel())):
            a.arrays[key] = val


# ============================================================================
# Raw HDF5 Statistics (for lazy loading)
# ============================================================================


def scan_raw_hdf5_statistics(
    hdf5_path: str,
    r_max: float,
    r_max_lr: float,
    keyspec: "KeySpecification",
    num_neighbor_samples: int = 1000,
    seed: int = 42,
) -> Tuple[Dict[int, float], int, float]:
    """Scan raw HDF5 to compute statistics needed before training.

    Computes E0s and num_elements by reading only atomic_numbers
    and energies (fast, no neighbor lists). Estimates
    avg_num_neighbors by fully preprocessing a random sample.

    Args:
        hdf5_path: Path to raw HDF5 file.
        r_max: Short-range cutoff.
        r_max_lr: Long-range cutoff.
        keyspec: KeySpecification for property mapping.
        num_neighbor_samples: Number of structures to sample for
            avg_num_neighbors estimate.
        seed: Random seed for sampling.

    Returns:
        (e0s_dict, num_elements, avg_num_neighbors) where e0s_dict
        maps atomic number (int) to E0 (float).
    """
    atoms_list = load_atoms_from_hdf5(hdf5_path, index=None)
    num_configs = len(atoms_list)

    # --- E0s and num_elements (fast: no neighbor lists) ---
    configs = []
    all_atomic_numbers = set()
    for atoms in atoms_list:
        config = config_from_atoms(
            atoms, key_specification=keyspec
        )
        configs.append(config)
        all_atomic_numbers.update(config.atomic_numbers.tolist())

    num_elements = len(all_atomic_numbers)

    z_table = AtomicNumberTable(
        [int(z) for z in range(1, 119)]
    )
    present_zs = sorted(all_atomic_numbers)
    present_z_table = AtomicNumberTable(present_zs)
    present_e0s = compute_average_E0s(configs, present_z_table)
    e0s = {z: present_e0s.get(z, 0.0) for z in range(1, 119)}

    logging.info(
        f"Scanned {num_configs} structures: "
        f"{num_elements} elements, "
        f"E0s computed for {len(present_e0s)} present elements"
    )

    # --- avg_num_neighbors (sample + preprocess) ---
    n_sample = min(num_neighbor_samples, num_configs)
    rng = random.Random(seed)
    sample_indices = rng.sample(range(num_configs), n_sample)

    total_neighbors = 0
    total_atoms = 0
    for i in sample_indices:
        data = AtomicData.from_config(
            configs[i],
            z_table=z_table,
            cutoff=float(r_max),
            cutoff_lr=r_max_lr,
        )
        total_neighbors += data.edge_index.shape[1]
        total_atoms += data.num_nodes

    avg_num_neighbors = total_neighbors / max(total_atoms, 1)
    logging.info(
        f"Estimated avg_num_neighbors={avg_num_neighbors:.2f} "
        f"from {n_sample} sampled structures"
    )

    return e0s, num_elements, avg_num_neighbors


# ============================================================================
# Raw HDF5 — Configuration helpers
# ============================================================================


def configs_from_hdf5(
    hdf5_path: str,
    key_specification: Optional[KeySpecification] = None,
    head_name: str = "Default",
) -> List[Configuration]:
    """Load Configuration objects directly from raw HDF5.

    Args:
        hdf5_path: Path to HDF5 file
        key_specification: Key specification for properties
        head_name: Head name for configurations

    Returns:
        List of Configuration objects
    """
    if key_specification is None:
        # Try to read stored keyspec from HDF5 metadata
        with h5py.File(hdf5_path, "r") as f:
            if "keyspec_info" in f.attrs:
                info_keys = json.loads(f.attrs["keyspec_info"])
                arrays_keys = json.loads(f.attrs["keyspec_arrays"])
                key_specification = KeySpecification(
                    info_keys=info_keys,
                    arrays_keys=arrays_keys,
                )
            else:
                key_specification = KeySpecification.from_defaults()

    atoms_list = load_atoms_from_hdf5(hdf5_path, index=None)

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
    atomic_energy_shifts: Optional[Dict[int, float]] = None,
) -> None:
    """Save preprocessed AtomicData list to HDF5.

    Args:
        data_list: List of AtomicData objects
        output_path: Path to output HDF5 file
        r_max: Short-range cutoff used for neighbor lists
        r_max_lr: Long-range cutoff (optional)
        z_table: Atomic number table
        description: Optional dataset description
        atomic_energy_shifts: Optional dict mapping atomic number to E0
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

        # Store atomic energy shifts if provided
        if atomic_energy_shifts is not None:
            # Convert dict to JSON-serializable format (int keys → str)
            e0s_serializable = {
                str(z): float(e0) for z, e0 in atomic_energy_shifts.items()
            }
            f.attrs["atomic_energy_shifts"] = json.dumps(e0s_serializable)
            logging.info(
                f"Stored atomic energy shifts for "
                f"{len(atomic_energy_shifts)} elements"
            )

        # Compute average number of neighbors
        total_neighbors = sum(data.edge_index.shape[1] for data in data_list)
        total_atoms = sum(data.num_nodes for data in data_list)
        avg_num_neighbors = total_neighbors / max(total_atoms, 1)
        f.attrs["avg_num_neighbors"] = avg_num_neighbors

        # Compute number of unique elements
        unique_elements = set()
        for data in data_list:
            # Get atomic numbers from node_attrs one-hot encoding
            elements = data.node_attrs.argmax(dim=-1).flatten().tolist()
            unique_elements.update(elements)
        num_elements = len(unique_elements)
        f.attrs["num_elements"] = num_elements

        logging.info(
            f"Computed statistics: avg_num_neighbors={avg_num_neighbors:.2f}, "
            f"num_elements={num_elements}"
        )

        # Write each configuration
        for i, data in enumerate(data_list):
            group = f.create_group(f"config_{i}")
            _write_atomic_data_to_hdf5_group(group, data)

    logging.info(
        f"Saved {len(data_list)} preprocessed configurations to HDF5: "
        f"{output_path}"
    )


def compute_and_format_e0s(
    configs: "Configurations",
    z_table: "AtomicNumberTable",
) -> Dict[int, float]:
    """Compute E0s from configurations and format for full z_table.

    Returns dict mapping atomic number (1-118) to E0 value.
    Missing elements get 0.0.

    Args:
        configs: List of Configuration objects
        z_table: Atomic number table (should be full 1-118)

    Returns:
        Dictionary mapping int(Z) -> float(E0) for all elements 1-118
    """
    # Find elements actually present
    present_zs = set()
    for config in configs:
        present_zs.update(config.atomic_numbers)

    if not present_zs:
        logging.warning("No atoms found in configs, returning zero E0s")
        return {z: 0.0 for z in range(1, 119)}

    present_z_table = AtomicNumberTable(sorted(list(present_zs)))

    # Compute E0s for present elements only
    present_e0s = compute_average_E0s(
        collections_train=configs,
        z_table=present_z_table,
    )

    # Expand to full 118 elements (fill missing with 0)
    full_e0s = {z: present_e0s.get(z, 0.0) for z in range(1, 119)}

    logging.info(
        f"Computed E0s for {len(present_e0s)} elements present in data"
    )

    return full_e0s


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
    if data.edge_index_lr is not None and data.edge_index_lr.numel() > 0:
        edges_lr_grp = group.create_group("edges_lr")
        edges_lr_grp["edge_index_lr"] = data.edge_index_lr.cpu().numpy()
        edges_lr_grp["shifts_lr"] = data.shifts_lr.cpu().numpy()
        edges_lr_grp["unit_shifts_lr"] = data.unit_shifts_lr.cpu().numpy()

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
        props_grp["hirshfeld_ratios"] = data.hirshfeld_ratios.cpu().numpy()
    if data.total_charge is not None:
        props_grp["total_charge"] = data.total_charge.cpu().numpy()
    if data.total_spin is not None:
        props_grp["total_spin"] = data.total_spin.cpu().numpy()
    if data.elec_temp is not None:
        props_grp["elec_temp"] = data.elec_temp.cpu().numpy()

    # Property weights
    weights_grp = group.create_group("weights")
    if data.energy_weight is not None:
        weights_grp["energy_weight"] = data.energy_weight.cpu().numpy()
    if data.forces_weight is not None:
        weights_grp["forces_weight"] = data.forces_weight.cpu().numpy()
    if data.stress_weight is not None:
        weights_grp["stress_weight"] = data.stress_weight.cpu().numpy()
    if data.virials_weight is not None:
        weights_grp["virials_weight"] = data.virials_weight.cpu().numpy()
    if data.dipole_weight is not None:
        weights_grp["dipole_weight"] = data.dipole_weight.cpu().numpy()
    if data.charges_weight is not None:
        weights_grp["charges_weight"] = data.charges_weight.cpu().numpy()
    if data.hirshfeld_ratios_weight is not None:
        weights_grp["hirshfeld_ratios_weight"] = (
            data.hirshfeld_ratios_weight.cpu().numpy()
        )


def _read_atomic_data_from_hdf5_group(
    group: h5py.Group,
) -> AtomicData:
    """Read single AtomicData from HDF5 group."""
    # Basic structure
    num_nodes = int(group["num_nodes"][()])
    positions = torch.from_numpy(np.array(group["positions"]))
    atomic_numbers = torch.from_numpy(np.array(group["atomic_numbers"]))
    node_attrs = torch.from_numpy(np.array(group["node_attrs"]))
    cell = (
        torch.from_numpy(np.array(group["cell"])) if "cell" in group else None
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
    total_charge = (
        torch.from_numpy(np.array(props_grp["total_charge"]))
        if "total_charge" in props_grp
        else torch.tensor(0.0)
    )
    total_spin = (
        torch.from_numpy(np.array(props_grp["total_spin"]))
        if "total_spin" in props_grp
        else torch.tensor(0.0)
    )
    elec_temp = (
        torch.from_numpy(np.array(props_grp["elec_temp"]))
        if "elec_temp" in props_grp
        else torch.tensor(0.0)
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
        torch.from_numpy(np.array(weights_grp["hirshfeld_ratios_weight"]))
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
        total_charge=total_charge,
        total_spin=total_spin,
        elec_temp=elec_temp,
        edge_index_lr=edge_index_lr,
        shifts_lr=shifts_lr,
        unit_shifts_lr=unit_shifts_lr,
    )


class PreprocessedHDF5Dataset(torch.utils.data.Dataset):
    """PyTorch Dataset for lazy-loading preprocessed HDF5.

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

            # Read E0s if present
            if "atomic_energy_shifts" in f.attrs:
                try:
                    e0s_json = f.attrs["atomic_energy_shifts"]
                    if isinstance(e0s_json, bytes):
                        e0s_json = e0s_json.decode()
                    e0s_serialized = json.loads(e0s_json)
                    # Convert string keys back to int
                    self.atomic_energy_shifts = {
                        int(z): float(e0)
                        for z, e0 in e0s_serialized.items()
                    }
                    logging.info(
                        f"Loaded atomic energy shifts (E0s) for "
                        f"{len(self.atomic_energy_shifts)} elements from "
                        f"HDF5"
                    )
                except (json.JSONDecodeError, KeyError, ValueError) as e:
                    logging.warning(
                        f"Failed to load atomic energy shifts from HDF5: "
                        f"{e}. E0s will be computed during training."
                    )
                    self.atomic_energy_shifts = None
            else:
                self.atomic_energy_shifts = None
                logging.info(
                    "No atomic energy shifts found in HDF5 "
                    "(will be computed during training)"
                )

            # Store in metadata dict for easy access
            self.metadata["atomic_energy_shifts"] = self.atomic_energy_shifts

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
            return _read_atomic_data_from_hdf5_group(group)


# ============================================================================
# Format Detection and Validation
# ============================================================================


def detect_file_format(path: str) -> str:
    """Detect format: 'xyz', 'hdf5_raw', or 'hdf5_preprocessed'.

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
            logging.warning(f"Could not read HDF5 metadata from {path}: {e}")
            return "hdf5_raw"

    # Unknown format
    raise ValueError(f"Unsupported file format: {path}")


def validate_preprocessed_hdf5(
    hdf5_path: str,
    expected_r_max: Optional[float] = None,
    expected_r_max_lr: Optional[float] = None,
) -> None:
    """Validate that HDF5 file contains preprocessed data.

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
            raise ValueError(f"{hdf5_path} is missing format_type metadata")

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
        f"Validated preprocessed HDF5: {hdf5_path} " f"(r_max={r_max})"
    )


# ============================================================================
# HDF5 Merge Utilities
# ============================================================================


def iter_atoms_from_hdf5(
    hdf5_path: str,
    batch_size: int = 100_000,
) -> Iterable[ase.Atoms]:
    """Stream ASE Atoms objects from a raw HDF5 file.

    Yields atoms in batches to keep peak RAM proportional to
    ``batch_size``, not total dataset size.

    Args:
        hdf5_path: Path to raw HDF5 v2.0 file.
        batch_size: Number of structures loaded per read batch.

    Yields:
        Individual ASE Atoms objects.
    """
    with h5py.File(hdf5_path, "r") as f:
        num_configs = int(f.attrs["num_configs"])
    for start in range(0, num_configs, batch_size):
        end = min(start + batch_size, num_configs)
        batch = load_atoms_from_hdf5(
            hdf5_path, index=slice(start, end)
        )
        yield from batch


def merge_raw_hdf5_files(
    input_paths: List[str],
    output_path: str,
    batch_size: int = 100_000,
    description: Optional[str] = None,
) -> None:
    """Merge multiple raw HDF5 (v2.0) files into one.

    Operates directly on HDF5 datasets without deserializing to ASE
    Atoms objects, making it dramatically faster than a round-trip
    through Python objects.  Peak RAM is proportional to
    ``batch_size``, not total combined dataset size.

    Property schemas are reconciled automatically: properties absent
    in some files are filled with NaN.

    Args:
        input_paths: Ordered list of raw HDF5 files to merge.
        output_path: Path for the merged output file.
        batch_size: Structures processed per write batch.
        description: Optional description for the merged file.

    Raises:
        ValueError: If any input is not a raw HDF5 file.
    """
    # --- Pass 1: validate formats, collect sizes & property schemas ---
    file_meta: List[dict] = []
    all_info_keys: Dict[str, tuple] = {}  # key -> (shape, dtype)
    all_arrays_keys: Dict[str, tuple] = {}  # key -> (shape, dtype)
    key_specification = KeySpecification()

    for idx, p in enumerate(input_paths):
        fmt = detect_file_format(p)
        if fmt != "hdf5_raw":
            raise ValueError(
                f"{p} is not a raw HDF5 file (detected format: {fmt})"
            )
        with h5py.File(p, "r") as f:
            nc = int(f.attrs["num_configs"])
            na = int(f.attrs.get("num_atoms", 0))
            if na == 0 and nc > 0:
                na = int(f["n_atoms"][:].sum())
            file_meta.append(
                {"path": p, "num_configs": nc, "num_atoms": na}
            )
            # Recover keyspec from first file
            if idx == 0 and "keyspec_info" in f.attrs:
                key_specification = KeySpecification(
                    info_keys=json.loads(f.attrs["keyspec_info"]),
                    arrays_keys=json.loads(
                        f.attrs["keyspec_arrays"]
                    ),
                )
            # Collect property dataset schemas
            if "properties/info" in f:
                for k in f["properties/info"]:
                    ds = f[f"properties/info/{k}"]
                    all_info_keys.setdefault(
                        k, (ds.shape[1:], ds.dtype)
                    )
            if "properties/arrays" in f:
                for k in f["properties/arrays"]:
                    ds = f[f"properties/arrays/{k}"]
                    all_arrays_keys.setdefault(
                        k, (ds.shape[1:], ds.dtype)
                    )

    total_configs = sum(m["num_configs"] for m in file_meta)
    total_atoms = sum(m["num_atoms"] for m in file_meta)
    c = _HDF5_CHUNK

    logging.info(
        f"Merging {len(input_paths)} raw HDF5 files "
        f"({total_configs} total structures) → {output_path}"
    )

    # --- Pass 2: create output and copy data directly ---
    with h5py.File(output_path, "w") as out:
        # File-level attributes
        out.attrs["format_version"] = RAW_HDF5_FORMAT_VERSION
        out.attrs["format_type"] = "raw"
        out.attrs["timestamp"] = datetime.now().isoformat()
        out.attrs["num_configs"] = total_configs
        out.attrs["num_atoms"] = total_atoms
        if description:
            out.attrs["description"] = description
        out.attrs["keyspec_info"] = json.dumps(
            key_specification.info_keys
        )
        out.attrs["keyspec_arrays"] = json.dumps(
            key_specification.arrays_keys
        )

        str_dt = h5py.string_dtype(encoding="utf-8")

        # Pre-allocate all datasets at final size
        out.create_dataset(
            "n_atoms", shape=(total_configs,),
            dtype=np.int32, chunks=(min(c, total_configs),),
        )
        out.create_dataset(
            "atomic_numbers", shape=(total_atoms,),
            dtype=np.int32, chunks=(min(c, total_atoms),),
        )
        out.create_dataset(
            "positions", shape=(total_atoms, 3),
            dtype=np.float64, chunks=(min(c, total_atoms), 3),
        )
        out.create_dataset(
            "cell", shape=(total_configs, 3, 3),
            dtype=np.float64,
            chunks=(min(c, total_configs), 3, 3),
        )
        out.create_dataset(
            "pbc", shape=(total_configs, 3),
            dtype=bool, chunks=(min(c, total_configs), 3),
        )

        meta_grp = out.require_group("config_metadata")
        meta_grp.create_dataset(
            "config_type", shape=(total_configs,),
            dtype=str_dt, chunks=(min(c, total_configs),),
        )
        meta_grp.create_dataset(
            "head", shape=(total_configs,),
            dtype=str_dt, chunks=(min(c, total_configs),),
        )
        meta_grp.create_dataset(
            "weight", shape=(total_configs,),
            dtype=np.float64, chunks=(min(c, total_configs),),
        )

        pw_grp = out.require_group("property_weights")
        for pw_name in ("energy", "forces", "stress"):
            pw_grp.create_dataset(
                pw_name, shape=(total_configs,),
                dtype=np.float64,
                chunks=(min(c, total_configs),),
            )

        if all_info_keys:
            info_grp = out.require_group("properties/info")
            for k, (ishape, dt) in all_info_keys.items():
                ds_shape = (total_configs,) + ishape
                ds_chunks = (min(c, total_configs),) + ishape
                ds = info_grp.create_dataset(
                    k, shape=ds_shape, dtype=dt,
                    chunks=ds_chunks,
                )
                ds[:] = np.nan

        if all_arrays_keys:
            arr_grp = out.require_group("properties/arrays")
            for k, (ishape, dt) in all_arrays_keys.items():
                ds_shape = (total_atoms,) + ishape
                ds_chunks = (min(c, total_atoms),) + ishape
                ds = arr_grp.create_dataset(
                    k, shape=ds_shape, dtype=dt,
                    chunks=ds_chunks,
                )
                ds[:] = np.nan

        # Stream data from each input file
        cfg_off = 0
        atm_off = 0
        start_time = time.time()

        for fm in file_meta:
            p = fm["path"]
            nc = fm["num_configs"]
            if nc == 0:
                continue

            with h5py.File(p, "r") as inp:
                # Copy in batches to limit RAM
                for b_start in range(0, nc, batch_size):
                    b_end = min(b_start + batch_size, nc)
                    b_len = b_end - b_start

                    # Per-config datasets
                    n_atoms_b = inp["n_atoms"][b_start:b_end]
                    b_atoms = int(n_atoms_b.sum())

                    # Atom range in source
                    if b_start == 0:
                        a_start = 0
                    else:
                        a_start = int(
                            inp["n_atoms"][:b_start].sum()
                        )
                    a_end = a_start + b_atoms

                    out["n_atoms"][
                        cfg_off:cfg_off + b_len
                    ] = n_atoms_b
                    out["atomic_numbers"][
                        atm_off:atm_off + b_atoms
                    ] = inp["atomic_numbers"][a_start:a_end]
                    out["positions"][
                        atm_off:atm_off + b_atoms
                    ] = inp["positions"][a_start:a_end]
                    out["cell"][
                        cfg_off:cfg_off + b_len
                    ] = inp["cell"][b_start:b_end]
                    out["pbc"][
                        cfg_off:cfg_off + b_len
                    ] = inp["pbc"][b_start:b_end]

                    # Config metadata
                    out["config_metadata/config_type"][
                        cfg_off:cfg_off + b_len
                    ] = inp["config_metadata/config_type"][
                        b_start:b_end
                    ]
                    out["config_metadata/head"][
                        cfg_off:cfg_off + b_len
                    ] = inp["config_metadata/head"][b_start:b_end]
                    out["config_metadata/weight"][
                        cfg_off:cfg_off + b_len
                    ] = inp["config_metadata/weight"][
                        b_start:b_end
                    ]

                    # Property weights
                    for pw_name in ("energy", "forces", "stress"):
                        out[f"property_weights/{pw_name}"][
                            cfg_off:cfg_off + b_len
                        ] = inp[f"property_weights/{pw_name}"][
                            b_start:b_end
                        ]

                    # Info properties (per-config)
                    for k in all_info_keys:
                        src = f"properties/info/{k}"
                        if src in inp:
                            out[src][
                                cfg_off:cfg_off + b_len
                            ] = inp[src][b_start:b_end]

                    # Arrays properties (per-atom)
                    for k in all_arrays_keys:
                        src = f"properties/arrays/{k}"
                        if src in inp:
                            out[src][
                                atm_off:atm_off + b_atoms
                            ] = inp[src][a_start:a_end]

                    cfg_off += b_len
                    atm_off += b_atoms

                    elapsed = time.time() - start_time
                    rate = (
                        cfg_off / elapsed if elapsed > 0 else 0
                    )
                    logging.info(
                        f"Progress: {cfg_off}/{total_configs} "
                        f"configs ({rate:.0f} configs/s)"
                    )

        # Build CSR offset index
        n_atoms_all = out["n_atoms"][:]
        offsets = np.concatenate(
            [[0], np.cumsum(n_atoms_all).astype(np.int64)]
        )
        out.create_dataset("offsets", data=offsets)

    logging.info(f"Merge complete: {output_path}")


def merge_preprocessed_hdf5_files(
    input_paths: List[str],
    output_path: str,
) -> None:
    """Merge multiple preprocessed HDF5 files into one.

    Copies ``config_N`` groups via h5py with renumbering so that the
    merged file contains groups ``config_0 … config_{total-1}``.
    All input files must have identical ``r_max`` and ``r_max_lr``.

    Args:
        input_paths: Ordered list of preprocessed HDF5 files to merge.
        output_path: Path for the merged output file.

    Raises:
        ValueError: If any input is not a preprocessed HDF5 file, or
                    if ``r_max`` / ``r_max_lr`` differ across inputs.
    """
    # Validate formats and r_max / r_max_lr in a single pass
    r_max_ref: Optional[float] = None
    r_max_lr_ref: Optional[float] = None
    num_configs_per_file: List[int] = []
    avg_nn_per_file: List[float] = []

    for p in input_paths:
        fmt = detect_file_format(p)
        if fmt != "hdf5_preprocessed":
            raise ValueError(
                f"{p} is not a preprocessed HDF5 file "
                f"(detected format: {fmt})"
            )
        with h5py.File(p, "r") as f:
            r_max_val = float(f.attrs["r_max"])
            r_max_lr_val = (
                float(f.attrs["r_max_lr"]) if "r_max_lr" in f.attrs else None
            )
            if r_max_ref is None:
                r_max_ref = r_max_val
                r_max_lr_ref = r_max_lr_val
            else:
                if abs(r_max_val - r_max_ref) > 1e-6:
                    raise ValueError(
                        f"r_max mismatch: {input_paths[0]} has "
                        f"r_max={r_max_ref}, but {p} has r_max={r_max_val}"
                    )
                if r_max_lr_ref != r_max_lr_val:
                    # Allow (None, None) match; reject any other mismatch
                    if not (
                        r_max_lr_ref is None and r_max_lr_val is None
                    ):
                        raise ValueError(
                            f"r_max_lr mismatch: {input_paths[0]} has "
                            f"r_max_lr={r_max_lr_ref}, but {p} has "
                            f"r_max_lr={r_max_lr_val}"
                        )
            num_configs_per_file.append(int(f.attrs["num_configs"]))
            avg_nn_per_file.append(float(f.attrs.get("avg_num_neighbors", 0.0)))

    total_configs = sum(num_configs_per_file)
    # Weighted average of avg_num_neighbors by num_configs
    total_weight = sum(num_configs_per_file)
    avg_num_neighbors = (
        sum(a * n for a, n in zip(avg_nn_per_file, num_configs_per_file))
        / max(total_weight, 1)
    )

    logging.info(
        f"Merging {len(input_paths)} preprocessed HDF5 files "
        f"({total_configs} total configs) → {output_path}"
    )

    with h5py.File(output_path, "w") as out_f:
        # Copy root attributes from first file and update
        with h5py.File(input_paths[0], "r") as first_f:
            for k, v in first_f.attrs.items():
                out_f.attrs[k] = v
        out_f.attrs["num_configs"] = total_configs
        out_f.attrs["avg_num_neighbors"] = avg_num_neighbors
        out_f.attrs["timestamp"] = datetime.now().isoformat()

        # Copy config groups with renumbering
        config_offset = 0
        for p, n in zip(input_paths, num_configs_per_file):
            with h5py.File(p, "r") as in_f:
                for i in range(n):
                    in_f.copy(
                        f"config_{i}",
                        out_f,
                        name=f"config_{config_offset + i}",
                    )
            config_offset += n
            logging.info(
                f"Copied {n} configs from {p} "
                f"(offset {config_offset - n})"
            )

    logging.info(f"Merge complete: {output_path}")
