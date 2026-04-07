"""Utility to load/save per-atom descriptors from/to an HDF5 file."""

from __future__ import annotations

from typing import Optional, Union
import numpy as np
import h5py


def save_descriptors_hdf5(
    filename: str,
    inv: Optional[list[np.ndarray]] = None,
    eqv: Optional[list[np.ndarray]] = None,
    mean_inv: Optional[np.ndarray] = None,
    mean_eqv: Optional[np.ndarray] = None,
) -> None:
    """Append descriptor datasets to an HDF5 file.

    Opens in append mode (``"a"``), so the file is created if it does
    not exist, or extended if it does.

    Parameters
    ----------
    filename:
        Path to the HDF5 file.
    inv:
        Per-atom invariant descriptors, one array of shape
        ``(num_atoms_i, num_features)`` per structure.
    eqv:
        Per-atom equivariant descriptors, one array of shape
        ``(num_atoms_i, sh_dim)`` per structure.
    mean_inv:
        Mean invariant descriptors, shape
        ``(num_structures, num_features)``.
    mean_eqv:
        Mean equivariant descriptors, shape
        ``(num_structures, sh_dim)``.
    """
    with h5py.File(filename, "a") as f:
        for key, arrays in (
            ("inv_descriptors", inv),
            ("eqv_descriptors", eqv),
        ):
            if arrays is not None:
                grp = f.create_group(key)
                data = np.concatenate(arrays, axis=0)
                ptr = np.concatenate(
                    [[0], np.cumsum([a.shape[0] for a in arrays])]
                )
                grp.create_dataset(
                    "data",
                    data=data,
                    chunks=(min(len(data), 65536), data.shape[1]),
                )
                grp.create_dataset("ptr", data=ptr)
        for key, arr in (
            ("mean_inv_descriptors", mean_inv),
            ("mean_eqv_descriptors", mean_eqv),
        ):
            if arr is not None:
                f.create_dataset(
                    key,
                    data=arr,
                    chunks=(min(len(arr), 65536), arr.shape[1]),
                )


def load_descriptors(
    filename: str,
) -> dict[str, Optional[Union[list[np.ndarray], np.ndarray]]]:
    """Load per-atom descriptors from an HDF5 file produced by
    ``torchkrates-eval``.

    Parameters
    ----------
    filename:
        Path to the HDF5 file written by :func:`save_descriptors_hdf5`.

    Returns
    -------
    dict with keys:

    ``"inv_descriptors"``
        ``list`` of ``(num_atoms_i, num_features)`` arrays, one per
        structure; or ``None`` if not present.
    ``"eqv_descriptors"``
        ``list`` of ``(num_atoms_i, sh_dim)`` arrays; or ``None``.
    ``"mean_inv_descriptors"``
        ``np.ndarray`` of shape ``(num_structures, num_features)``; or
        ``None``.
    ``"mean_eqv_descriptors"``
        ``np.ndarray`` of shape ``(num_structures, sh_dim)``; or
        ``None``.
    """
    result: dict[str, Optional[Union[list[np.ndarray], np.ndarray]]] = {
        "inv_descriptors": None,
        "eqv_descriptors": None,
        "mean_inv_descriptors": None,
        "mean_eqv_descriptors": None,
    }

    with h5py.File(filename, "r") as f:
        for key in ("inv_descriptors", "eqv_descriptors"):
            if key not in f:
                continue
            item = f[key]
            if isinstance(item, h5py.Dataset):
                # None sentinel stored by old save_results_hdf5 format
                continue
            data = item["data"][:]
            ptr = item["ptr"][:]
            result[key] = [
                data[ptr[i] : ptr[i + 1]] for i in range(len(ptr) - 1)
            ]
        for key in ("mean_inv_descriptors", "mean_eqv_descriptors"):
            if key in f:
                result[key] = f[key][:]

    return result


def save_mean_descriptors_npz(
    filename: str,
    mean_inv: Optional[np.ndarray] = None,
    mean_eqv: Optional[np.ndarray] = None,
) -> None:
    """Save mean descriptors as an uncompressed numpy archive (.npz).

    Uses :func:`numpy.savez` (not compressed) for maximum write speed.
    The resulting file can be loaded with
    :func:`load_mean_descriptors_npz`.

    Parameters
    ----------
    filename:
        Output path. A ``.npz`` extension is appended by numpy if not
        already present.
    mean_inv:
        Mean invariant descriptors, shape
        ``(num_structures, num_features)``.
    mean_eqv:
        Mean equivariant descriptors, shape
        ``(num_structures, sh_dim)``.
    """
    arrays: dict[str, np.ndarray] = {}
    if mean_inv is not None:
        arrays["mean_inv_descriptors"] = mean_inv
    if mean_eqv is not None:
        arrays["mean_eqv_descriptors"] = mean_eqv
    if arrays:
        np.savez(filename, **arrays)


def load_mean_descriptors_npz(
    filename: str,
) -> dict[str, Optional[np.ndarray]]:
    """Load mean descriptors from a .npz file written by
    :func:`save_mean_descriptors_npz`.

    Parameters
    ----------
    filename:
        Path to the ``.npz`` file.

    Returns
    -------
    dict with keys ``"mean_inv_descriptors"`` and
    ``"mean_eqv_descriptors"``, each an ``np.ndarray`` or ``None``.
    """
    result: dict[str, Optional[np.ndarray]] = {
        "mean_inv_descriptors": None,
        "mean_eqv_descriptors": None,
    }
    data = np.load(filename)
    for key in result:
        if key in data:
            result[key] = data[key]
    return result
