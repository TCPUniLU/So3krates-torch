"""Per-atom dispersion ratio labels (a0/Hirshfeld ratios, C6 ratios).

Mirrors the JAX so3lr data path: label key fallbacks
(`dataloader_sparse_ase.py`) and NaN for missing labels so the loss
can mask them out.
"""

import h5py
import numpy as np
import pytest
import torch
from ase.build import molecule

from so3krates_torch.data.atomic_data import AtomicData
from so3krates_torch.data.hdf5_utils import (
    PreprocessedHDF5Dataset,
    save_atoms_to_hdf5,
    save_preprocessed_hdf5,
    scan_raw_hdf5_property_presence,
)
from so3krates_torch.data.utils import (
    KeySpecification,
    config_from_atoms,
)
from so3krates_torch.tools.utils import AtomicNumberTable

Z_TABLE = AtomicNumberTable([1, 8])
R_MAX = 5.0


def _water():
    atoms = molecule("H2O")
    atoms.info["REF_energy"] = -10.0
    atoms.arrays["REF_forces"] = np.zeros((3, 3))
    return atoms


def _data(atoms, keyspec=None):
    keyspec = keyspec or KeySpecification.from_defaults()
    config = config_from_atoms(atoms, keyspec)
    return AtomicData.from_config(config, z_table=Z_TABLE, cutoff=R_MAX)


def test_default_keyspec_has_c6_ratios():
    keyspec = KeySpecification.from_defaults()
    assert keyspec.arrays_keys["c6_ratios"] == "REF_c6_ratios"


def test_config_from_atoms_reads_c6_default_key():
    atoms = _water()
    atoms.arrays["REF_c6_ratios"] = np.array([0.9, 0.8, 0.7])
    data = _data(atoms)
    assert torch.allclose(
        data.c6_ratios,
        torch.tensor([0.9, 0.8, 0.7], dtype=data.c6_ratios.dtype),
    )
    assert float(data.c6_ratios_weight) == 1.0


def test_config_from_atoms_c6_jax_fallback_key():
    atoms = _water()
    atoms.arrays["c6_ratios"] = np.array([0.9, 0.8, 0.7])
    config = config_from_atoms(atoms, KeySpecification.from_defaults())
    assert np.allclose(config.properties["c6_ratios"], [0.9, 0.8, 0.7])
    assert config.property_weights["c6_ratios"] == 1.0


def test_config_from_atoms_hirshfeld_fallback_order():
    atoms = _water()
    atoms.arrays["a0_ratios"] = np.array([3.0, 3.0, 3.0])
    atoms.arrays["hirsh_ratios"] = np.array([2.0, 2.0, 2.0])
    config = config_from_atoms(atoms, KeySpecification.from_defaults())
    assert np.allclose(config.properties["hirshfeld_ratios"], 2.0)
    # configured key wins over every fallback
    atoms.arrays["REF_hirsh_ratios"] = np.array([1.0, 1.0, 1.0])
    config = config_from_atoms(atoms, KeySpecification.from_defaults())
    assert np.allclose(config.properties["hirshfeld_ratios"], 1.0)


def test_atomic_data_missing_ratio_labels_are_nan():
    data = _data(_water())
    assert torch.isnan(data.hirshfeld_ratios).all()
    assert torch.isnan(data.c6_ratios).all()
    assert float(data.c6_ratios_weight) == 0.0
    assert float(data.hirshfeld_ratios_weight) == 0.0
    # non-ratio labels keep zero-fill
    assert not torch.isnan(data.charges).any()
    assert not torch.isnan(data.forces).any()


def _roundtrip(data, tmp_path):
    path = str(tmp_path / "pre.h5")
    save_preprocessed_hdf5(
        [data], path, r_max=R_MAX, r_max_lr=None, z_table=Z_TABLE
    )
    return PreprocessedHDF5Dataset(path)[0], path


def test_hdf5_roundtrip_c6_ratios(tmp_path):
    atoms = _water()
    atoms.arrays["REF_c6_ratios"] = np.array([0.9, 0.8, 0.7])
    loaded, _ = _roundtrip(_data(atoms), tmp_path)
    assert torch.allclose(
        loaded.c6_ratios.double(), torch.tensor([0.9, 0.8, 0.7]).double()
    )
    assert float(loaded.c6_ratios_weight) == 1.0


def test_hdf5_roundtrip_missing_c6_is_nan(tmp_path):
    loaded, path = _roundtrip(_data(_water()), tmp_path)
    assert torch.isnan(loaded.c6_ratios).all()
    assert float(loaded.c6_ratios_weight) == 0.0
    # a file written before c6 support (no c6 datasets) also reads as NaN
    with h5py.File(path, "a") as f:
        del f["config_0/properties/c6_ratios"]
        del f["config_0/weights/c6_ratios_weight"]
    old = PreprocessedHDF5Dataset(path)[0]
    assert torch.isnan(old.c6_ratios).all()
    assert old.c6_ratios.shape == (3,)
    assert float(old.c6_ratios_weight) == 0.0


def test_raw_hdf5_presence_sees_fallback_key(tmp_path):
    atoms = _water()
    atoms.arrays["c6_ratios"] = np.array([0.9, 0.8, 0.7])
    path = str(tmp_path / "raw.h5")
    save_atoms_to_hdf5(
        [atoms], path, key_specification=KeySpecification.from_defaults()
    )
    presence = scan_raw_hdf5_property_presence(
        path, KeySpecification.from_defaults(), ["c6_ratios"]
    )
    assert presence["c6_ratios"] is True


def test_preprocessed_presence_accepts_partial_ratio_labels():
    from so3krates_torch.data.hdf5_utils import (
        raise_if_preprocessed_properties_missing,
    )

    # partially labelled ratio targets are masked in the loss -> allowed
    raise_if_preprocessed_properties_missing(
        {"c6_ratios": (2, 4), "hirshfeld_ratios": (1, 4)}, "x.h5"
    )
    # absent everywhere is still an error
    with pytest.raises(ValueError, match="c6_ratios"):
        raise_if_preprocessed_properties_missing({"c6_ratios": (0, 4)}, "x.h5")
    # other properties keep the strict consistency check
    with pytest.raises(ValueError, match="charges"):
        raise_if_preprocessed_properties_missing({"charges": (2, 4)}, "x.h5")


def _labelled_water(i, key=None):
    atoms = _water()
    if key is not None:
        atoms.arrays[key] = np.full(3, 0.5 + 0.1 * i)
    return atoms


def test_raw_hdf5_keeps_labels_first_seen_after_first_batch(tmp_path):
    from so3krates_torch.data.hdf5_utils import load_atoms_from_hdf5

    atoms_list = [_labelled_water(i) for i in range(3)] + [
        _labelled_water(i, "REF_c6_ratios") for i in range(3, 6)
    ]
    path = str(tmp_path / "raw.h5")
    save_atoms_to_hdf5(
        atoms_list,
        path,
        key_specification=KeySpecification.from_defaults(),
        batch_size=2,
    )
    loaded = load_atoms_from_hdf5(path)
    assert all("REF_c6_ratios" not in a.arrays for a in loaded[:3])
    for i, a in enumerate(loaded[3:], start=3):
        assert np.allclose(a.arrays["REF_c6_ratios"], 0.5 + 0.1 * i)


def test_raw_hdf5_mixed_fallback_keys_all_survive(tmp_path):
    from so3krates_torch.data.hdf5_utils import load_atoms_from_hdf5

    atoms_list = [
        _labelled_water(0, "hirsh_ratios"),
        _labelled_water(1, "a0_ratios"),
        _labelled_water(2),
    ]
    keyspec = KeySpecification.from_defaults()
    path = str(tmp_path / "raw.h5")
    save_atoms_to_hdf5(atoms_list, path, key_specification=keyspec)
    configs = [
        config_from_atoms(a, keyspec) for a in load_atoms_from_hdf5(path)
    ]
    assert np.allclose(configs[0].properties["hirshfeld_ratios"], 0.5)
    assert np.allclose(configs[1].properties["hirshfeld_ratios"], 0.6)
    assert configs[2].properties["hirshfeld_ratios"] is None
