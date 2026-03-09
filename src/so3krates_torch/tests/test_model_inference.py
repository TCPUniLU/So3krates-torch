"""Tests for model inference (forward pass)."""

import pytest
import torch
import numpy as np
from ase.build import molecule, bulk
from so3krates_torch.modules.models import (
    So3krates,
    SO3LR,
    MultiHeadSO3LR,
)


def _random_rotation():
    """Random rotation matrix via QR decomposition."""
    Q, _ = torch.linalg.qr(torch.randn(3, 3, dtype=torch.float64))
    if torch.linalg.det(Q) < 0:
        Q[:, 0] *= -1  # ensure det = +1
    return Q


class TestSo3kratesInference:
    """Test So3krates base model inference."""

    def test_forward_returns_energy_and_forces(
        self, default_model_config, make_batch, h2o_atoms
    ):
        model = So3krates(**default_model_config)
        model.eval()
        batch = make_batch(h2o_atoms, r_max=5.0)
        out = model(batch.to_dict(), compute_stress=False)
        assert out["energy"].shape == (1,)
        assert torch.isfinite(out["energy"]).all()
        assert out["forces"].shape == (3, 3)
        assert torch.isfinite(out["forces"]).all()

    def test_forward_batched(self, default_model_config, make_batch_list):
        model = So3krates(**default_model_config)
        model.eval()
        atoms_list = [
            molecule("H2O"),
            molecule("NH3"),
            molecule("CH4"),
        ]
        batch = make_batch_list(atoms_list, r_max=5.0)
        out = model(batch.to_dict(), compute_stress=False)
        assert out["energy"].shape == (3,)
        total_atoms = 3 + 4 + 5
        assert out["forces"].shape == (total_atoms, 3)

    def test_forward_deterministic(
        self, default_model_config, make_batch, h2o_atoms
    ):
        model = So3krates(**default_model_config)
        model.eval()
        batch = make_batch(h2o_atoms, r_max=5.0)
        out1 = model(batch.to_dict(), compute_stress=False)
        e1 = out1["energy"].detach().clone()
        f1 = out1["forces"].detach().clone()
        out2 = model(batch.to_dict(), compute_stress=False)
        assert torch.allclose(e1, out2["energy"])
        assert torch.allclose(f1, out2["forces"])

    def test_forward_different_degrees(
        self, default_model_config, make_batch, h2o_atoms
    ):
        for degrees in [[1], [1, 2], [1, 2, 3, 4]]:
            config = {
                **default_model_config,
                "degrees": degrees,
            }
            model = So3krates(**config)
            model.eval()
            batch = make_batch(h2o_atoms, r_max=5.0)
            out = model(batch.to_dict(), compute_stress=False)
            assert out["energy"].shape == (1,)
            assert out["forces"].shape == (3, 3)
            assert torch.isfinite(out["energy"]).all()
            assert torch.isfinite(out["forces"]).all()

    def test_forward_stress_periodic(
        self, default_model_config, make_batch, si_bulk
    ):
        model = So3krates(**default_model_config)
        model.eval()
        batch = make_batch(si_bulk, r_max=5.0)
        out = model(batch.to_dict(), compute_stress=True)
        assert out["stress"].shape == (1, 3, 3)
        assert torch.isfinite(out["stress"]).all()


class TestSO3LRInference:
    """Test SO3LR model inference."""

    def test_forward_returns_lr_properties(
        self, so3lr_model_config, make_batch, h2o_atoms
    ):
        model = SO3LR(**so3lr_model_config)
        model.eval()
        batch = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
        out = model(batch.to_dict(), compute_stress=False)
        assert out["partial_charges"].shape == (3,)
        assert out["dipole"].shape == (1, 3)
        assert out["hirshfeld_ratios"].shape == (3,)
        assert out["zbl_repulsion"].shape == (3, 1)
        assert torch.isfinite(out["energy"]).all()
        assert torch.isfinite(out["zbl_repulsion"]).all()

    def test_forward_with_lr_disabled(
        self, default_model_config, make_batch, h2o_atoms
    ):
        config = {
            **default_model_config,
            "r_max_lr": 10.0,
            "zbl_repulsion_bool": False,
            "electrostatic_energy_bool": False,
            "dispersion_energy_bool": False,
        }
        model = SO3LR(**config)
        model.eval()
        batch = make_batch(h2o_atoms, r_max=5.0)
        out = model(batch.to_dict(), compute_stress=False)
        assert out["partial_charges"] is None
        assert out["dipole"] is None
        assert out["hirshfeld_ratios"] is None
        assert torch.isfinite(out["energy"]).all()
        assert torch.isfinite(out["forces"]).all()

    def test_forward_periodic(self, so3lr_model_config, make_batch, si_bulk):
        model = SO3LR(**so3lr_model_config)
        model.eval()
        batch = make_batch(si_bulk, r_max=5.0, cutoff_lr=10.0)
        out = model(batch.to_dict(), compute_stress=False)
        assert out["energy"].shape == (1,)
        num_atoms = len(si_bulk)
        assert out["forces"].shape == (num_atoms, 3)
        assert torch.isfinite(out["energy"]).all()
        assert torch.isfinite(out["forces"]).all()


class TestMultiHeadSO3LRInference:
    """Test MultiHeadSO3LR model inference."""

    def test_multihead_output_shapes(
        self, multihead_model_config, make_batch, h2o_atoms
    ):
        model = MultiHeadSO3LR(**multihead_model_config)
        model.eval()
        batch = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
        out = model(batch.to_dict(), compute_stress=False)
        assert out["energy"].shape == (4, 1)
        assert out["forces"].shape == (4, 3, 3)

    def test_multihead_head_selection(
        self, multihead_model_config, make_batch, h2o_atoms
    ):
        model = MultiHeadSO3LR(**multihead_model_config)
        model.eval()
        model.select_heads = True
        batch = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
        out = model(batch.to_dict(), compute_stress=False)
        assert out["energy"].shape == (1,)
        assert out["forces"].shape == (3, 3)


class TestSO3LRReferenceValues:
    """Test SO3LR model against deterministic reference values.

    Uses a freshly initialized SO3LR model (not pretrained)
    with fixed seed and config to verify reproducibility of
    the forward pass.
    """

    @pytest.fixture
    def so3lr_model(self, so3lr_model_config):
        model = SO3LR(**so3lr_model_config)
        model.eval()
        return model

    def test_h2o_energy_and_forces(
        self,
        so3lr_model,
        so3lr_model_config,
        make_batch,
        h2o_atoms,
    ):
        batch = make_batch(
            h2o_atoms,
            r_max=so3lr_model_config["r_max"],
            cutoff_lr=so3lr_model_config["r_max_lr"],
        )
        out = so3lr_model(batch.to_dict(), compute_stress=False)
        ref_energy = np.array([-0.14982067])
        ref_forces = np.array(
            [
                [-0.0, -0.0, 11.37196199],
                [-0.0, 7.32691155, -5.68598099],
                [-0.0, -7.32691155, -5.68598099],
            ]
        )
        np.testing.assert_allclose(
            out["energy"].detach().numpy(),
            ref_energy,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            out["forces"].detach().numpy(),
            ref_forces,
            rtol=1e-4,
        )

    def test_nh3_energy_and_forces(
        self,
        so3lr_model,
        so3lr_model_config,
        make_batch,
        nh3_atoms,
    ):
        batch = make_batch(
            nh3_atoms,
            r_max=so3lr_model_config["r_max"],
            cutoff_lr=so3lr_model_config["r_max_lr"],
        )
        out = so3lr_model(batch.to_dict(), compute_stress=False)
        ref_energy = np.array([0.38790223])
        ref_forces = np.array(
            [
                [-0.0, 1.69453622e-06, 6.91859831],
                [-0.0, 5.59683490, -2.30619734],
                [4.84700830, -2.79841830, -2.30620049],
                [-4.84700830, -2.79841830, -2.30620049],
            ]
        )
        np.testing.assert_allclose(
            out["energy"].detach().numpy(),
            ref_energy,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            out["forces"].detach().numpy(),
            ref_forces,
            rtol=1e-4,
        )

    def test_ch4_energy_and_forces(
        self,
        so3lr_model,
        so3lr_model_config,
        make_batch,
        ch4_atoms,
    ):
        batch = make_batch(
            ch4_atoms,
            r_max=so3lr_model_config["r_max"],
            cutoff_lr=so3lr_model_config["r_max_lr"],
        )
        out = so3lr_model(batch.to_dict(), compute_stress=False)
        ref_energy = np.array([-0.16540115])
        ref_forces = np.array(
            [
                [-0.0, -0.0, -0.0],
                [1.75721934, 1.75721934, 1.75721934],
                [-1.75721934, -1.75721934, 1.75721934],
                [1.75721934, -1.75721934, -1.75721934],
                [-1.75721934, 1.75721934, -1.75721934],
            ]
        )
        np.testing.assert_allclose(
            out["energy"].detach().numpy(),
            ref_energy,
            rtol=1e-4,
        )
        np.testing.assert_allclose(
            out["forces"].detach().numpy(),
            ref_forces,
            rtol=1e-4,
        )

    def test_ethanol_energy_and_forces(
        self,
        so3lr_model,
        so3lr_model_config,
        make_batch,
        ethanol_atoms,
    ):
        batch = make_batch(
            ethanol_atoms,
            r_max=so3lr_model_config["r_max"],
            cutoff_lr=so3lr_model_config["r_max_lr"],
        )
        out = so3lr_model(batch.to_dict(), compute_stress=False)
        ref_energy = np.array([-1.74773691])
        np.testing.assert_allclose(
            out["energy"].detach().numpy(),
            ref_energy,
            rtol=1e-4,
        )
        assert torch.isfinite(out["forces"]).all()
        assert out["forces"].shape == (9, 3)

    def test_partial_charges_neutrality(
        self,
        so3lr_model,
        so3lr_model_config,
        make_batch,
    ):
        for atoms in [
            molecule("H2O"),
            molecule("NH3"),
            molecule("CH4"),
        ]:
            batch = make_batch(
                atoms,
                r_max=so3lr_model_config["r_max"],
                cutoff_lr=so3lr_model_config["r_max_lr"],
            )
            out = so3lr_model(batch.to_dict(), compute_stress=False)
            charge_sum = out["partial_charges"].detach().sum().item()
            assert abs(charge_sum) < 1e-5, (
                f"{atoms.get_chemical_formula()}: "
                f"charge sum = {charge_sum}"
            )

    def test_h2o_partial_charges_values(
        self,
        so3lr_model,
        so3lr_model_config,
        make_batch,
        h2o_atoms,
    ):
        batch = make_batch(
            h2o_atoms,
            r_max=so3lr_model_config["r_max"],
            cutoff_lr=so3lr_model_config["r_max_lr"],
        )
        out = so3lr_model(batch.to_dict(), compute_stress=False)
        ref_charges = np.array([0.91145137, -0.45572569, -0.45572569])
        np.testing.assert_allclose(
            out["partial_charges"].detach().numpy(),
            ref_charges,
            rtol=1e-4,
        )

    def test_hirshfeld_ratios_positive(
        self,
        so3lr_model,
        so3lr_model_config,
        make_batch,
    ):
        for atoms in [
            molecule("H2O"),
            molecule("NH3"),
            molecule("CH4"),
        ]:
            batch = make_batch(
                atoms,
                r_max=so3lr_model_config["r_max"],
                cutoff_lr=so3lr_model_config["r_max_lr"],
            )
            out = so3lr_model(batch.to_dict(), compute_stress=False)
            hr = out["hirshfeld_ratios"].detach().numpy()
            assert (hr > 0).all(), (
                f"{atoms.get_chemical_formula()}: "
                f"negative hirshfeld ratio found"
            )

    def test_h2o_hirshfeld_reference(
        self,
        so3lr_model,
        so3lr_model_config,
        make_batch,
        h2o_atoms,
    ):
        batch = make_batch(
            h2o_atoms,
            r_max=so3lr_model_config["r_max"],
            cutoff_lr=so3lr_model_config["r_max_lr"],
        )
        out = so3lr_model(batch.to_dict(), compute_stress=False)
        ref_hr = np.array([0.55005068, 0.30834503, 0.30834503])
        np.testing.assert_allclose(
            out["hirshfeld_ratios"].detach().numpy(),
            ref_hr,
            rtol=1e-4,
        )

    def test_h2o_dipole_reference(
        self,
        so3lr_model,
        so3lr_model_config,
        make_batch,
        h2o_atoms,
    ):
        batch = make_batch(
            h2o_atoms,
            r_max=so3lr_model_config["r_max"],
            cutoff_lr=so3lr_model_config["r_max_lr"],
        )
        out = so3lr_model(batch.to_dict(), compute_stress=False)
        ref_dipole = np.array([[0.0, 0.0, 0.54350666]])
        np.testing.assert_allclose(
            out["dipole"].detach().numpy(),
            ref_dipole,
            atol=1e-5,
        )
        dipole_magnitude = np.linalg.norm(out["dipole"].detach().numpy())
        assert dipole_magnitude > 0


class TestEquivariance:
    """Test SO(3) equivariance and translation invariance."""

    def test_energy_rotation_invariant(
        self, default_model_config, make_batch, h2o_atoms
    ):
        """E(R*x) == E(x) for a random rotation R."""
        model = So3krates(**default_model_config)
        model.eval()
        R = _random_rotation()

        batch_orig = make_batch(h2o_atoms, r_max=5.0)
        E_orig = model(
            batch_orig.to_dict(), compute_stress=False
        )["energy"].detach()

        h2o_rot = h2o_atoms.copy()
        pos = torch.from_numpy(h2o_atoms.get_positions())
        h2o_rot.set_positions((R @ pos.T).T.numpy())
        batch_rot = make_batch(h2o_rot, r_max=5.0)
        E_rot = model(
            batch_rot.to_dict(), compute_stress=False
        )["energy"].detach()

        assert torch.allclose(E_orig, E_rot, atol=1e-5)

    def test_forces_rotation_equivariant(
        self, default_model_config, make_batch, h2o_atoms
    ):
        """F(R*x) == R*F(x) for a random rotation R."""
        model = So3krates(**default_model_config)
        model.eval()
        R = _random_rotation()

        batch_orig = make_batch(h2o_atoms, r_max=5.0)
        F_orig = model(
            batch_orig.to_dict(), compute_stress=False
        )["forces"].detach()

        h2o_rot = h2o_atoms.copy()
        pos = torch.from_numpy(h2o_atoms.get_positions())
        h2o_rot.set_positions((R @ pos.T).T.numpy())
        batch_rot = make_batch(h2o_rot, r_max=5.0)
        F_rot = model(
            batch_rot.to_dict(), compute_stress=False
        )["forces"].detach()

        # F(R*x) must equal R*F(x)
        F_expected = (R @ F_orig.T).T
        assert torch.allclose(F_rot, F_expected, atol=1e-5)

    def test_energy_translation_invariant(
        self, default_model_config, make_batch, h2o_atoms
    ):
        """E(x + t) == E(x) for any constant shift t."""
        model = So3krates(**default_model_config)
        model.eval()

        batch_orig = make_batch(h2o_atoms, r_max=5.0)
        E_orig = model(
            batch_orig.to_dict(), compute_stress=False
        )["energy"].detach()

        h2o_shifted = h2o_atoms.copy()
        h2o_shifted.set_positions(
            h2o_atoms.get_positions() + [3.7, -1.2, 5.5]
        )
        batch_shift = make_batch(h2o_shifted, r_max=5.0)
        E_shift = model(
            batch_shift.to_dict(), compute_stress=False
        )["energy"].detach()

        assert torch.allclose(E_orig, E_shift, atol=1e-6)


class TestForcesFiniteDifferences:
    """Test autograd forces against finite difference reference."""

    def test_forces_match_finite_differences(
        self, default_model_config, make_batch, h2o_atoms
    ):
        """Autograd forces match central finite differences."""
        model = So3krates(**default_model_config)
        model.eval()
        eps = 1e-4
        positions = h2o_atoms.get_positions().copy()
        N = len(h2o_atoms)

        batch_ref = make_batch(h2o_atoms, r_max=5.0)
        F_autograd = model(
            batch_ref.to_dict(), compute_stress=False
        )["forces"].detach()

        F_fd = torch.zeros(N, 3, dtype=torch.float64)
        for i in range(N):
            for j in range(3):
                for sign, delta in [(+1, eps), (-1, -eps)]:
                    pos = positions.copy()
                    pos[i, j] += delta
                    atoms_pert = h2o_atoms.copy()
                    atoms_pert.set_positions(pos)
                    batch_pert = make_batch(atoms_pert, r_max=5.0)
                    E_pert = model(
                        batch_pert.to_dict(),
                        compute_stress=False,
                    )["energy"].detach().item()
                    F_fd[i, j] -= sign * E_pert / (2 * eps)

        assert torch.allclose(F_autograd, F_fd, atol=1e-3)


class TestStressTensor:
    """Test stress tensor properties."""

    def test_stress_tensor_symmetric(
        self, default_model_config, make_batch, si_bulk
    ):
        """Stress tensor from model forward pass must be symmetric."""
        model = So3krates(**default_model_config)
        model.eval()
        batch = make_batch(si_bulk, r_max=5.0)
        stress = model(
            batch.to_dict(), compute_stress=True
        )["stress"]  # (num_graphs, 3, 3)

        for g in range(stress.shape[0]):
            s = stress[g]
            assert torch.allclose(s, s.T, atol=1e-6), (
                f"Stress tensor not symmetric:\n{s}"
            )
