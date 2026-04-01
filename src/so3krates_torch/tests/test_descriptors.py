"""Tests for descriptor extraction."""

import pytest
import torch
import numpy as np
from ase.build import molecule
from so3krates_torch.modules.models import So3krates


class TestGetDescriptorsUnpacking:
    """Verify that get_descriptors() returns correctly labelled features."""

    def test_invariant_features_are_per_node(
        self, default_model_config, make_batch, h2o_atoms, device
    ):
        """inv_features must have shape (num_atoms, num_features)."""
        model = So3krates(**default_model_config)
        model.eval()
        batch = make_batch(h2o_atoms, r_max=5.0)
        out = model(
            batch.to_dict(),
            return_descriptors=True,
        )
        inv = out["inv_features"]
        assert inv is not None
        # H2O has 3 atoms
        assert inv.shape[0] == 3
        assert inv.ndim == 2

    def test_equivariant_features_are_per_node(
        self, default_model_config, make_batch, h2o_atoms, device
    ):
        """ev_features must have shape (num_atoms, sh_dim)."""
        model = So3krates(**default_model_config)
        model.eval()
        batch = make_batch(h2o_atoms, r_max=5.0)
        out = model(
            batch.to_dict(),
            return_eqv_descriptors=True,
        )
        ev = out["ev_features"]
        assert ev is not None
        assert ev.shape[0] == 3
        assert ev.ndim == 2

    def test_inv_and_eqv_independent(
        self, default_model_config, make_batch, h2o_atoms, device
    ):
        """Invariant and equivariant features must differ from each other."""
        model = So3krates(**default_model_config)
        model.eval()
        batch = make_batch(h2o_atoms, r_max=5.0)
        out = model(
            batch.to_dict(),
            return_descriptors=True,
            return_eqv_descriptors=True,
        )
        inv = out["inv_features"].detach().cpu().numpy()
        ev = out["ev_features"].detach().cpu().numpy()
        # They should not be numerically identical (may differ in shape too)
        assert inv.shape != ev.shape or not np.allclose(inv, ev)


class TestEvaluateModelDescriptors:
    """Test descriptor extraction via evaluate_model()."""

    def test_inv_descriptors_shape(self, default_model_config, device):
        """inv_descriptors should be a list of (num_atoms, num_features)
        arrays, one per structure."""
        from ase.build import molecule
        from so3krates_torch.modules.models import So3krates
        from so3krates_torch.tools.eval import evaluate_model

        model = So3krates(**default_model_config)
        model.eval()
        atoms_list = [molecule("H2O"), molecule("NH3"), molecule("CH4")]
        # num atoms: H2O=3, NH3=4, CH4=5

        result = evaluate_model(
            atoms_list=atoms_list,
            model=model,
            batch_size=3,
            device=str(device),
            model_type="so3krates",
            multi_species=True,
            return_inv_descriptors=True,
        )
        inv = result["inv_descriptors"]
        assert inv is not None
        assert len(inv) == 3
        assert inv[0].shape == (3, default_model_config["num_features"])
        assert inv[1].shape == (4, default_model_config["num_features"])
        assert inv[2].shape == (5, default_model_config["num_features"])

    def test_eqv_descriptors_shape(self, default_model_config, device):
        """eqv_descriptors should be a list of (num_atoms, sh_dim) arrays."""
        from ase.build import molecule
        from so3krates_torch.modules.models import So3krates
        from so3krates_torch.tools.eval import evaluate_model

        model = So3krates(**default_model_config)
        model.eval()
        atoms_list = [molecule("H2O"), molecule("NH3")]

        result = evaluate_model(
            atoms_list=atoms_list,
            model=model,
            batch_size=2,
            device=str(device),
            model_type="so3krates",
            multi_species=True,
            return_eqv_descriptors=True,
        )
        eqv = result["eqv_descriptors"]
        assert eqv is not None
        assert len(eqv) == 2
        assert eqv[0].shape[0] == 3  # H2O has 3 atoms
        assert eqv[0].ndim == 2

    def test_none_when_not_requested(self, default_model_config, device):
        """Keys should be None when the corresponding flag is False."""
        from ase.build import molecule
        from so3krates_torch.modules.models import So3krates
        from so3krates_torch.tools.eval import evaluate_model

        model = So3krates(**default_model_config)
        model.eval()
        result = evaluate_model(
            atoms_list=[molecule("H2O")],
            model=model,
            batch_size=1,
            device=str(device),
            model_type="so3krates",
        )
        assert result["inv_descriptors"] is None
        assert result["eqv_descriptors"] is None


class TestLoadDescriptors:
    """Test HDF5 round-trip for descriptors."""

    def test_inv_descriptors_roundtrip(
        self, default_model_config, device, tmp_path
    ):
        """Save inv_descriptors to HDF5, load back, check equality."""
        from ase.build import molecule
        from so3krates_torch.modules.models import So3krates
        from so3krates_torch.tools.eval import evaluate_model
        from so3krates_torch.tools.load_descriptors import (
            load_descriptors,
            save_descriptors_hdf5,
        )

        model = So3krates(**default_model_config)
        model.eval()
        atoms_list = [molecule("H2O"), molecule("NH3"), molecule("CH4")]

        result = evaluate_model(
            atoms_list=atoms_list,
            model=model,
            batch_size=3,
            device=str(device),
            model_type="so3krates",
            multi_species=True,
            return_inv_descriptors=True,
        )

        out_path = str(tmp_path / "descriptors.h5")
        save_descriptors_hdf5(out_path, inv=result["inv_descriptors"])

        loaded = load_descriptors(out_path)
        assert loaded["inv_descriptors"] is not None
        assert len(loaded["inv_descriptors"]) == 3
        for orig, back in zip(
            result["inv_descriptors"], loaded["inv_descriptors"]
        ):
            np.testing.assert_array_equal(orig, back)

    def test_eqv_descriptors_roundtrip(
        self, default_model_config, device, tmp_path
    ):
        """Save eqv_descriptors to HDF5, load back, check equality."""
        from ase.build import molecule
        from so3krates_torch.modules.models import So3krates
        from so3krates_torch.tools.eval import evaluate_model
        from so3krates_torch.tools.load_descriptors import (
            load_descriptors,
            save_descriptors_hdf5,
        )

        model = So3krates(**default_model_config)
        model.eval()
        atoms_list = [molecule("H2O"), molecule("NH3")]

        result = evaluate_model(
            atoms_list=atoms_list,
            model=model,
            batch_size=2,
            device=str(device),
            model_type="so3krates",
            multi_species=True,
            return_eqv_descriptors=True,
        )

        out_path = str(tmp_path / "descriptors_eqv.h5")
        save_descriptors_hdf5(out_path, eqv=result["eqv_descriptors"])

        loaded = load_descriptors(out_path)
        assert loaded["eqv_descriptors"] is not None
        assert len(loaded["eqv_descriptors"]) == 2
        for orig, back in zip(
            result["eqv_descriptors"], loaded["eqv_descriptors"]
        ):
            np.testing.assert_array_equal(orig, back)

    def test_both_descriptors_roundtrip(
        self, default_model_config, device, tmp_path
    ):
        """Both descriptor types round-trip correctly together."""
        from ase.build import molecule
        from so3krates_torch.modules.models import So3krates
        from so3krates_torch.tools.eval import evaluate_model
        from so3krates_torch.tools.load_descriptors import (
            load_descriptors,
            save_descriptors_hdf5,
        )

        model = So3krates(**default_model_config)
        model.eval()
        atoms_list = [molecule("H2O")]

        result = evaluate_model(
            atoms_list=atoms_list,
            model=model,
            batch_size=1,
            device=str(device),
            model_type="so3krates",
            return_inv_descriptors=True,
            return_eqv_descriptors=True,
        )

        out_path = str(tmp_path / "both.h5")
        save_descriptors_hdf5(
            out_path,
            inv=result["inv_descriptors"],
            eqv=result["eqv_descriptors"],
        )

        loaded = load_descriptors(out_path)
        assert loaded["inv_descriptors"] is not None
        assert loaded["eqv_descriptors"] is not None

    def test_missing_descriptor_key_returns_none(self, tmp_path):
        """load_descriptors does not fail when keys are absent."""
        import h5py
        from so3krates_torch.tools.load_descriptors import load_descriptors

        out_path = str(tmp_path / "no_desc.h5")
        with h5py.File(out_path, "w") as f:
            grp = f.create_group("energies")
            grp.create_dataset("item_000000", data=np.array([-10.0]))

        loaded = load_descriptors(out_path)
        assert loaded.get("inv_descriptors") is None
        assert loaded.get("eqv_descriptors") is None
        assert loaded.get("mean_inv_descriptors") is None
        assert loaded.get("mean_eqv_descriptors") is None

    def test_concatenated_hdf5_structure(self, tmp_path):
        """save_descriptors_hdf5 writes data+ptr groups, not item_* datasets."""
        import h5py
        from so3krates_torch.tools.load_descriptors import (
            save_descriptors_hdf5,
        )

        inv = [
            np.random.randn(3, 8).astype(np.float64),
            np.random.randn(4, 8).astype(np.float64),
        ]
        out_path = str(tmp_path / "concat.h5")
        save_descriptors_hdf5(out_path, inv=inv)

        with h5py.File(out_path, "r") as f:
            assert isinstance(f["inv_descriptors"], h5py.Group)
            assert "data" in f["inv_descriptors"]
            assert "ptr" in f["inv_descriptors"]
            assert "item_000000" not in f["inv_descriptors"]
            # ptr should be [0, 3, 7]
            ptr = f["inv_descriptors"]["ptr"][:]
            np.testing.assert_array_equal(ptr, [0, 3, 7])
