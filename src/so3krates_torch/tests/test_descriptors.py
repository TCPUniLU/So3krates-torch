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
