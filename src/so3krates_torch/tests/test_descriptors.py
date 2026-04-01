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
