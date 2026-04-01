"""Utility to load per-atom descriptors from an HDF5 results file."""

from __future__ import annotations

from typing import Optional
import numpy as np
import h5py


def load_descriptors(
    filename: str,
) -> dict[str, Optional[list[np.ndarray]]]:
    """Load invariant and/or equivariant per-atom descriptors from an HDF5
    file produced by ``torchkrates-eval``.

    Parameters
    ----------
    filename:
        Path to the HDF5 file written by ``save_results_hdf5``.

    Returns
    -------
    dict with keys:
        - ``"inv_descriptors"``: list of ``(num_atoms, num_features)``
          arrays, one per structure; or ``None`` if not present.
        - ``"eqv_descriptors"``: list of ``(num_atoms, sh_dim)``
          arrays, one per structure; or ``None`` if not present.
    """
    result: dict[str, Optional[list[np.ndarray]]] = {
        "inv_descriptors": None,
        "eqv_descriptors": None,
    }

    with h5py.File(filename, "r") as f:
        for key in ("inv_descriptors", "eqv_descriptors"):
            if key not in f:
                continue
            item = f[key]
            # None values are stored as empty datasets with is_none attr
            if isinstance(item, h5py.Dataset):
                continue
            # Groups are named item_000000, item_000001, …
            sorted_keys = sorted(item.keys())
            result[key] = [item[k][:] for k in sorted_keys]

    return result
