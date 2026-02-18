"""Tests for LAMMPS MLIAP integration.

All tests use mocks for LAMMPS-specific imports (lammps.mliap) since
LAMMPS is not available on the development machine. Full integration
testing with actual LAMMPS should be done on HPC.
"""

import os
from unittest.mock import Mock

import pytest
import torch
from ase.build import bulk, molecule
from ase.data import atomic_numbers as ase_atomic_numbers

from so3krates_torch.calculator.lammps_mliap_so3 import (
    LAMMPS_MLIAP_SO3,
    So3EdgeForcesWrapper,
    So3LammpsConfig,
)
from so3krates_torch.modules.models import SO3LR, MultiHeadSO3LR


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def so3lr_short_range_config(default_model_config):
    """SO3LR config with only short-range interactions (ZBL only)."""
    return {
        **default_model_config,
        "zbl_repulsion_bool": True,
        "electrostatic_energy_bool": False,
        "dispersion_energy_bool": False,
    }


@pytest.fixture
def so3lr_short_range_model(so3lr_short_range_config):
    """SO3LR model with ZBL but no electrostatics/dispersion."""
    model = SO3LR(**so3lr_short_range_config)
    model.eval()
    return model


@pytest.fixture
def so3lr_short_range_no_zbl_config(default_model_config):
    """SO3LR config with no physical potentials at all."""
    return {
        **default_model_config,
        "zbl_repulsion_bool": False,
        "electrostatic_energy_bool": False,
        "dispersion_energy_bool": False,
    }


@pytest.fixture
def so3lr_short_range_no_zbl_model(so3lr_short_range_no_zbl_config):
    """SO3LR model with no ZBL, electrostatics, or dispersion."""
    model = SO3LR(**so3lr_short_range_no_zbl_config)
    model.eval()
    return model


@pytest.fixture
def so3lr_electrostatic_config(default_model_config):
    """SO3LR config with electrostatics enabled (should fail validation)."""
    return {
        **default_model_config,
        "r_max_lr": 10.0,
        "zbl_repulsion_bool": True,
        "electrostatic_energy_bool": True,
        "dispersion_energy_bool": False,
    }


@pytest.fixture
def so3lr_electrostatic_model(so3lr_electrostatic_config):
    """SO3LR model with electrostatics (for validation rejection tests)."""
    model = SO3LR(**so3lr_electrostatic_config)
    model.eval()
    return model


@pytest.fixture
def so3lr_dispersion_config(default_model_config):
    """SO3LR config with dispersion enabled (should fail validation)."""
    return {
        **default_model_config,
        "r_max_lr": 10.0,
        "zbl_repulsion_bool": True,
        "electrostatic_energy_bool": False,
        "dispersion_energy_bool": True,
        "dispersion_energy_cutoff_lr_damping": 2.0,
    }


@pytest.fixture
def so3lr_dispersion_model(so3lr_dispersion_config):
    """SO3LR model with dispersion (for validation rejection tests)."""
    model = SO3LR(**so3lr_dispersion_config)
    model.eval()
    return model


@pytest.fixture
def multihead_short_range_config(default_model_config):
    """MultiHead config with only short-range interactions."""
    return {
        **default_model_config,
        "num_output_heads": 3,
        "energy_regression_dim": default_model_config["num_features"],
        "zbl_repulsion_bool": True,
        "electrostatic_energy_bool": False,
        "dispersion_energy_bool": False,
    }


@pytest.fixture
def multihead_short_range_model(multihead_short_range_config):
    """MultiHeadSO3LR model with short-range only."""
    model = MultiHeadSO3LR(**multihead_short_range_config)
    model.eval()
    return model


@pytest.fixture
def water_atomic_numbers():
    """Atomic numbers for H2O system."""
    return [1, 8]  # H, O


@pytest.fixture
def si_atomic_numbers():
    """Atomic numbers for Si system."""
    return [14]


def _make_mock_lammps_data(
    natoms, nghosts, npairs, atomic_numbers_map, dtype=torch.float64
):
    """Create mock LAMMPS data structure for testing.

    Args:
        natoms: Number of real atoms.
        nghosts: Number of ghost atoms.
        npairs: Number of atom pairs (edges).
        atomic_numbers_map: List of atomic numbers (e.g. [1, 8] for H, O).
        dtype: Data type for floating point tensors.
    """
    ntotal = natoms + nghosts
    num_species = len(atomic_numbers_map)
    mock = Mock()
    mock.nlocal = natoms
    mock.ntotal = ntotal
    mock.npairs = npairs

    # Element type indices (0-based, maps to atomic_numbers_map)
    mock.elems = torch.randint(0, num_species, (ntotal,))
    # Random pair vectors
    mock.rij = torch.randn(npairs, 3, dtype=dtype)
    # Pair indices (within ntotal range)
    mock.pair_i = torch.randint(0, ntotal, (npairs,))
    mock.pair_j = torch.randint(0, ntotal, (npairs,))
    # Energy storage
    mock.eatoms = torch.zeros(natoms, dtype=dtype)
    mock.energy = torch.tensor(0.0, dtype=dtype)
    # Force update mock
    mock.update_pair_forces_gpu = Mock()
    # Module for device detection (not kokkos)
    mock.__class__ = type(
        "RegularModule", (), {"__module__": "regular_module"}
    )
    return mock


# ---------------------------------------------------------------------------
# TestSo3LammpsConfig
# ---------------------------------------------------------------------------


class TestSo3LammpsConfig:
    """Test configuration parsing from environment variables."""

    def test_default_values(self):
        config = So3LammpsConfig()
        assert config.debug_time is False
        assert config.debug_profile is False
        assert config.allow_cpu is False
        assert config.force_cpu is False
        assert config.profile_start_step == 5
        assert config.profile_end_step == 10

    def test_env_var_parsing(self, monkeypatch):
        monkeypatch.setenv("SO3_TIME", "true")
        monkeypatch.setenv("SO3_PROFILE", "1")
        monkeypatch.setenv("SO3_PROFILE_START", "10")
        monkeypatch.setenv("SO3_PROFILE_END", "20")
        monkeypatch.setenv("SO3_ALLOW_CPU", "yes")
        monkeypatch.setenv("SO3_FORCE_CPU", "t")

        config = So3LammpsConfig()
        assert config.debug_time is True
        assert config.debug_profile is True
        assert config.profile_start_step == 10
        assert config.profile_end_step == 20
        assert config.allow_cpu is True
        assert config.force_cpu is True

    def test_env_var_false_values(self, monkeypatch):
        monkeypatch.setenv("SO3_TIME", "false")
        monkeypatch.setenv("SO3_PROFILE", "0")
        monkeypatch.setenv("SO3_ALLOW_CPU", "no")

        config = So3LammpsConfig()
        assert config.debug_time is False
        assert config.debug_profile is False
        assert config.allow_cpu is False


# ---------------------------------------------------------------------------
# TestSo3EdgeForcesWrapper
# ---------------------------------------------------------------------------


class TestSo3EdgeForcesWrapper:
    """Test model wrapper validation and construction."""

    def test_validation_rejects_electrostatics(
        self, so3lr_electrostatic_model, water_atomic_numbers
    ):
        with pytest.raises(ValueError, match="electrostatic"):
            So3EdgeForcesWrapper(
                so3lr_electrostatic_model,
                atomic_numbers=water_atomic_numbers,
            )

    def test_validation_rejects_dispersion(
        self, so3lr_dispersion_model, water_atomic_numbers
    ):
        with pytest.raises(ValueError, match="dispersion"):
            So3EdgeForcesWrapper(
                so3lr_dispersion_model,
                atomic_numbers=water_atomic_numbers,
            )

    def test_accepts_short_range_with_zbl(
        self, so3lr_short_range_model, water_atomic_numbers
    ):
        wrapper = So3EdgeForcesWrapper(
            so3lr_short_range_model,
            atomic_numbers=water_atomic_numbers,
        )
        assert wrapper.num_elements == 118
        assert torch.equal(
            wrapper.atomic_numbers_map,
            torch.tensor([1, 8], dtype=torch.long),
        )

    def test_accepts_short_range_no_zbl(
        self, so3lr_short_range_no_zbl_model, water_atomic_numbers
    ):
        wrapper = So3EdgeForcesWrapper(
            so3lr_short_range_no_zbl_model,
            atomic_numbers=water_atomic_numbers,
        )
        assert wrapper.num_elements == 118

    def test_parameters_frozen(
        self, so3lr_short_range_model, water_atomic_numbers
    ):
        wrapper = So3EdgeForcesWrapper(
            so3lr_short_range_model,
            atomic_numbers=water_atomic_numbers,
        )
        for p in wrapper.model.parameters():
            assert p.requires_grad is False

    def test_r_max_buffer(self, so3lr_short_range_model, water_atomic_numbers):
        wrapper = So3EdgeForcesWrapper(
            so3lr_short_range_model,
            atomic_numbers=water_atomic_numbers,
        )
        assert float(wrapper.r_max) == so3lr_short_range_model.r_max

    def test_head_default_selection(
        self, so3lr_short_range_model, water_atomic_numbers
    ):
        wrapper = So3EdgeForcesWrapper(
            so3lr_short_range_model,
            atomic_numbers=water_atomic_numbers,
        )
        # Default head should be the last one
        assert wrapper.head.item() == 0  # Only one head "Default"

    def test_multihead_head_selection(
        self, multihead_short_range_model, water_atomic_numbers
    ):
        # MultiHeadSO3LR doesn't have heads attribute by default
        multihead_short_range_model.heads = ["head_a", "head_b", "head_c"]
        wrapper = So3EdgeForcesWrapper(
            multihead_short_range_model,
            atomic_numbers=water_atomic_numbers,
            head="head_b",
        )
        assert wrapper.head.item() == 1


# ---------------------------------------------------------------------------
# TestLAMMPS_MLIAP_SO3
# ---------------------------------------------------------------------------


class TestLAMMPS_MLIAP_SO3:
    """Test LAMMPS MLIAP interface construction and attributes."""

    def test_initialization(
        self, so3lr_short_range_model, water_atomic_numbers
    ):
        calc = LAMMPS_MLIAP_SO3(
            so3lr_short_range_model,
            atomic_numbers=water_atomic_numbers,
        )
        assert calc.num_species == 2
        assert calc.element_types == ["H", "O"]
        assert calc.rcutfac == 0.5 * float(so3lr_short_range_model.r_max)
        assert calc.ndescriptors == 1
        assert calc.nparams == 1
        assert calc.initialized is False
        assert calc.step == 0

    def test_initialization_single_element(
        self, so3lr_short_range_model, si_atomic_numbers
    ):
        calc = LAMMPS_MLIAP_SO3(
            so3lr_short_range_model,
            atomic_numbers=si_atomic_numbers,
        )
        assert calc.num_species == 1
        assert calc.element_types == ["Si"]

    def test_device_initialization_cpu(
        self, so3lr_short_range_model, water_atomic_numbers
    ):
        calc = LAMMPS_MLIAP_SO3(
            so3lr_short_range_model,
            atomic_numbers=water_atomic_numbers,
        )
        mock_data = Mock()
        mock_data.__class__ = type(
            "RegularModule", (), {"__module__": "regular_module"}
        )

        calc._initialize_device(mock_data)
        assert calc.device == torch.device("cpu")
        assert calc.initialized is True

    def test_prepare_batch_keys(
        self, so3lr_short_range_model, water_atomic_numbers
    ):
        calc = LAMMPS_MLIAP_SO3(
            so3lr_short_range_model,
            atomic_numbers=water_atomic_numbers,
        )
        calc.device = torch.device("cpu")

        mock_data = _make_mock_lammps_data(
            natoms=3,
            nghosts=2,
            npairs=10,
            atomic_numbers_map=water_atomic_numbers,
        )
        species = torch.as_tensor(mock_data.elems, dtype=torch.int64)

        batch = calc._prepare_batch(mock_data, 3, 2, species)

        expected_keys = {
            "vectors",
            "node_attrs",
            "edge_index",
            "atomic_numbers",
            "batch",
            "ptr",
            "total_charge",
            "total_spin",
            "lammps_class",
            "natoms",
        }
        assert set(batch.keys()) == expected_keys

    def test_prepare_batch_shapes(
        self, so3lr_short_range_model, water_atomic_numbers
    ):
        calc = LAMMPS_MLIAP_SO3(
            so3lr_short_range_model,
            atomic_numbers=water_atomic_numbers,
        )
        calc.device = torch.device("cpu")

        natoms, nghosts, npairs = 3, 2, 10
        mock_data = _make_mock_lammps_data(
            natoms=natoms,
            nghosts=nghosts,
            npairs=npairs,
            atomic_numbers_map=water_atomic_numbers,
        )
        species = torch.as_tensor(mock_data.elems, dtype=torch.int64)

        batch = calc._prepare_batch(mock_data, natoms, nghosts, species)

        assert batch["vectors"].shape == (npairs, 3)
        assert batch["node_attrs"].shape == (natoms + nghosts, 118)
        assert batch["edge_index"].shape == (2, npairs)
        assert batch["atomic_numbers"].shape == (natoms + nghosts,)
        assert batch["batch"].shape == (natoms + nghosts,)
        assert batch["ptr"].shape == (2,)
        assert batch["natoms"] == (natoms, natoms + nghosts)

    def test_prepare_batch_one_hot_encoding(
        self, so3lr_short_range_model, water_atomic_numbers
    ):
        """Verify one-hot encoding maps LAMMPS type indices to Z-1 positions."""
        calc = LAMMPS_MLIAP_SO3(
            so3lr_short_range_model,
            atomic_numbers=water_atomic_numbers,
        )
        calc.device = torch.device("cpu")

        mock_data = _make_mock_lammps_data(
            natoms=3,
            nghosts=0,
            npairs=5,
            atomic_numbers_map=water_atomic_numbers,
        )
        # Force specific element assignment: H, O, H
        mock_data.elems = torch.tensor(
            [0, 1, 0]
        )  # type 0=H(Z=1), type 1=O(Z=8)
        species = torch.as_tensor(mock_data.elems, dtype=torch.int64)

        batch = calc._prepare_batch(mock_data, 3, 0, species)

        # Check atomic numbers
        assert batch["atomic_numbers"][0].item() == 1  # H
        assert batch["atomic_numbers"][1].item() == 8  # O
        assert batch["atomic_numbers"][2].item() == 1  # H

        # Check one-hot: H should be hot at index 0 (Z-1=0), O at index 7 (Z-1=7)
        assert batch["node_attrs"][0, 0].item() == 1.0  # H at index 0
        assert batch["node_attrs"][1, 7].item() == 1.0  # O at index 7
        assert batch["node_attrs"][2, 0].item() == 1.0  # H at index 0

        # Check rest is zero
        assert batch["node_attrs"][0].sum().item() == 1.0
        assert batch["node_attrs"][1].sum().item() == 1.0

    def test_update_lammps_data_energy_only_real_atoms(
        self, so3lr_short_range_model, water_atomic_numbers
    ):
        """Energy should only sum over real atoms, not ghosts."""
        calc = LAMMPS_MLIAP_SO3(
            so3lr_short_range_model,
            atomic_numbers=water_atomic_numbers,
        )

        natoms = 3
        nghosts = 2
        ntotal = natoms + nghosts
        atom_energies = torch.tensor(
            [1.0, 2.0, 3.0, 100.0, 200.0], dtype=torch.float64
        )
        pair_forces = torch.randn(10, 3, dtype=torch.float64)

        mock_data = _make_mock_lammps_data(
            natoms=natoms,
            nghosts=nghosts,
            npairs=10,
            atomic_numbers_map=water_atomic_numbers,
        )

        calc._update_lammps_data(mock_data, atom_energies, pair_forces, natoms)

        # Total energy should only sum real atoms: 1+2+3=6, not including ghosts
        assert mock_data.energy.item() == pytest.approx(6.0)
        # eatoms should have first 3 values
        eatoms = torch.as_tensor(mock_data.eatoms)
        assert eatoms[0].item() == pytest.approx(1.0)
        assert eatoms[1].item() == pytest.approx(2.0)
        assert eatoms[2].item() == pytest.approx(3.0)

    def test_update_lammps_data_dtype_conversion(
        self, so3lr_short_range_model, water_atomic_numbers
    ):
        """Float32 pair forces should be converted to float64 for LAMMPS."""
        calc = LAMMPS_MLIAP_SO3(
            so3lr_short_range_model,
            atomic_numbers=water_atomic_numbers,
        )
        calc.dtype = torch.float32

        natoms = 3
        atom_energies = torch.tensor([1.0, 2.0, 3.0], dtype=torch.float32)
        pair_forces = torch.randn(10, 3, dtype=torch.float32)

        mock_data = _make_mock_lammps_data(
            natoms=natoms,
            nghosts=0,
            npairs=10,
            atomic_numbers_map=water_atomic_numbers,
            dtype=torch.float32,
        )

        calc._update_lammps_data(mock_data, atom_energies, pair_forces, natoms)

        # Check that update_pair_forces_gpu was called with float64
        called_forces = mock_data.update_pair_forces_gpu.call_args[0][0]
        assert called_forces.dtype == torch.float64

    def test_compute_forces_early_return_empty(
        self, so3lr_short_range_model, water_atomic_numbers
    ):
        """Should return early when natoms=0."""
        calc = LAMMPS_MLIAP_SO3(
            so3lr_short_range_model,
            atomic_numbers=water_atomic_numbers,
        )
        calc.initialized = True
        calc.device = torch.device("cpu")

        mock_data = Mock()
        mock_data.nlocal = 0
        mock_data.ntotal = 0
        mock_data.npairs = 0
        mock_data.elems = torch.tensor([], dtype=torch.int64)
        mock_data.__class__ = type(
            "RegularModule", (), {"__module__": "regular_module"}
        )

        # Should not raise
        calc.compute_forces(mock_data)
        assert calc.step == 1

    def test_update_lammps_data_squeeze_multidim(
        self, so3lr_short_range_model, water_atomic_numbers
    ):
        """Node energies with trailing dim should be squeezed."""
        calc = LAMMPS_MLIAP_SO3(
            so3lr_short_range_model,
            atomic_numbers=water_atomic_numbers,
        )

        natoms = 3
        # Shape (3, 1) instead of (3,)
        atom_energies = torch.tensor(
            [[1.0], [2.0], [3.0]], dtype=torch.float64
        )
        pair_forces = torch.randn(10, 3, dtype=torch.float64)

        mock_data = _make_mock_lammps_data(
            natoms=natoms,
            nghosts=0,
            npairs=10,
            atomic_numbers_map=water_atomic_numbers,
        )

        calc._update_lammps_data(mock_data, atom_energies, pair_forces, natoms)
        assert mock_data.energy.item() == pytest.approx(6.0)


# ---------------------------------------------------------------------------
# TestModelOutputNodeEnergy
# ---------------------------------------------------------------------------


class TestModelOutputNodeEnergy:
    """Test that SO3LR forward pass now includes node_energy in output."""

    def test_so3lr_output_has_node_energy(
        self, so3lr_short_range_model, make_batch, h2o_atoms
    ):
        model = so3lr_short_range_model
        batch = make_batch(h2o_atoms, r_max=model.r_max)

        output = model(
            batch,
            training=False,
            compute_force=True,
            compute_edge_forces=False,
        )

        assert "node_energy" in output
        assert output["node_energy"] is not None
        # node_energy shape should be (num_atoms, 1) or (num_atoms,)
        assert output["node_energy"].shape[0] == len(h2o_atoms)

    def test_so3lr_node_energy_sums_to_total(
        self, so3lr_short_range_model, make_batch, h2o_atoms
    ):
        model = so3lr_short_range_model
        batch = make_batch(h2o_atoms, r_max=model.r_max)

        output = model(
            batch,
            training=False,
            compute_force=True,
        )

        node_energy = output["node_energy"]
        if node_energy.dim() > 1:
            node_energy = node_energy.squeeze(-1)
        total_from_nodes = node_energy.sum()
        total_energy = output["energy"].squeeze()

        assert torch.allclose(total_from_nodes, total_energy, atol=1e-6)

    def test_so3lr_no_zbl_node_energy(
        self, so3lr_short_range_no_zbl_model, make_batch, h2o_atoms
    ):
        model = so3lr_short_range_no_zbl_model
        batch = make_batch(h2o_atoms, r_max=model.r_max)

        output = model(
            batch,
            training=False,
            compute_force=True,
        )

        assert "node_energy" in output
        node_energy = output["node_energy"]
        if node_energy.dim() > 1:
            node_energy = node_energy.squeeze(-1)
        total_from_nodes = node_energy.sum()
        total_energy = output["energy"].squeeze()

        assert torch.allclose(total_from_nodes, total_energy, atol=1e-6)


# ---------------------------------------------------------------------------
# TestCLIValidation
# ---------------------------------------------------------------------------


class TestCLIValidation:
    """Test CLI helper functions."""

    def test_validate_elements_valid(self):
        from so3krates_torch.cli.create_lammps_model import validate_elements

        z_list = validate_elements(["Si", "O"])
        assert z_list == [14, 8]

    def test_validate_elements_single(self):
        from so3krates_torch.cli.create_lammps_model import validate_elements

        z_list = validate_elements(["H"])
        assert z_list == [1]

    def test_validate_elements_invalid(self):
        from so3krates_torch.cli.create_lammps_model import validate_elements

        with pytest.raises(ValueError, match="Unknown element symbol"):
            validate_elements(["Si", "Xx"])

    def test_validate_model_rejects_electrostatics(
        self, so3lr_electrostatic_model
    ):
        from so3krates_torch.cli.create_lammps_model import validate_model

        with pytest.raises(ValueError, match="electrostatic"):
            validate_model(so3lr_electrostatic_model)

    def test_validate_model_rejects_dispersion(self, so3lr_dispersion_model):
        from so3krates_torch.cli.create_lammps_model import validate_model

        with pytest.raises(ValueError, match="dispersion"):
            validate_model(so3lr_dispersion_model)

    def test_validate_model_accepts_short_range(self, so3lr_short_range_model):
        from so3krates_torch.cli.create_lammps_model import validate_model

        # Should not raise
        validate_model(so3lr_short_range_model)


# ---------------------------------------------------------------------------
# TestEndToEndForward
# ---------------------------------------------------------------------------


class TestEndToEndForward:
    """End-to-end tests running the wrapper forward pass with real models.

    These tests use the actual So3krates model but with mock LAMMPS data
    structures (no actual LAMMPS needed).
    """

    def _atoms_to_mock_lammps_data(
        self, atoms, model, atomic_numbers_list, dtype=torch.float64
    ):
        """Convert ASE Atoms to mock LAMMPS data by computing neighbor list.

        This mimics what LAMMPS would provide via MLIAP.
        """
        from ase.neighborlist import neighbor_list

        r_max = model.r_max
        i_indices, j_indices, d_vectors = neighbor_list(
            "ijD", atoms, cutoff=r_max, self_interaction=False
        )

        natoms = len(atoms)
        npairs = len(i_indices)

        # Map actual atomic numbers to LAMMPS type indices
        z_to_type = {z: idx for idx, z in enumerate(atomic_numbers_list)}

        mock = Mock()
        mock.nlocal = natoms
        mock.ntotal = natoms  # No ghosts for non-periodic / single domain
        mock.npairs = npairs
        mock.elems = torch.tensor(
            [z_to_type[z] for z in atoms.get_atomic_numbers()],
            dtype=torch.int64,
        )
        mock.rij = torch.tensor(d_vectors, dtype=dtype)
        mock.pair_i = torch.tensor(i_indices, dtype=torch.int64)
        mock.pair_j = torch.tensor(j_indices, dtype=torch.int64)
        mock.eatoms = torch.zeros(natoms, dtype=dtype)
        mock.energy = torch.tensor(0.0, dtype=dtype)
        mock.update_pair_forces_gpu = Mock()
        mock.__class__ = type(
            "RegularModule", (), {"__module__": "regular_module"}
        )
        return mock

    def test_wrapper_forward_h2o(
        self, so3lr_short_range_model, water_atomic_numbers
    ):
        """Run wrapper forward pass on H2O and check output shapes."""
        atoms = molecule("H2O")
        model = so3lr_short_range_model
        model.double()

        wrapper = So3EdgeForcesWrapper(
            model, atomic_numbers=water_atomic_numbers
        )
        wrapper.eval()

        mock_data = self._atoms_to_mock_lammps_data(
            atoms, model, water_atomic_numbers
        )

        # Build batch dict like LAMMPS_MLIAP_SO3._prepare_batch would
        natoms = mock_data.nlocal
        ntotal = mock_data.ntotal
        species = mock_data.elems

        atomic_numbers_map = wrapper.atomic_numbers_map
        actual_z = atomic_numbers_map[species]

        node_attrs = torch.zeros(ntotal, 118, dtype=torch.float64)
        node_attrs.scatter_(1, (actual_z - 1).unsqueeze(1), 1.0)

        batch_dict = {
            "vectors": mock_data.rij.to(torch.float64),
            "node_attrs": node_attrs,
            "edge_index": torch.stack(
                [
                    mock_data.pair_i.to(torch.int64),
                    mock_data.pair_j.to(torch.int64),
                ],
                dim=0,
            ),
            "atomic_numbers": actual_z,
            "batch": torch.zeros(natoms, dtype=torch.int64),
            "ptr": torch.tensor([0, natoms], dtype=torch.long),
            "total_charge": torch.zeros(1, dtype=torch.float64),
            "total_spin": torch.zeros(1, dtype=torch.float64),
            "lammps_class": mock_data,
            "natoms": (natoms, ntotal),
        }

        total_energy, node_energy, pair_forces = wrapper(batch_dict)

        assert total_energy.shape == ()
        assert node_energy.shape[0] == natoms
        assert pair_forces.shape == (mock_data.npairs, 3)

    def test_lammps_calc_h2o(
        self, so3lr_short_range_model, water_atomic_numbers
    ):
        """Full LAMMPS_MLIAP_SO3.compute_forces on H2O."""
        atoms = molecule("H2O")
        model = so3lr_short_range_model
        model.double()

        calc = LAMMPS_MLIAP_SO3(model, atomic_numbers=water_atomic_numbers)
        calc.device = torch.device("cpu")
        calc.initialized = True
        calc.model = calc.model.to("cpu")

        mock_data = self._atoms_to_mock_lammps_data(
            atoms, model, water_atomic_numbers
        )

        calc.compute_forces(mock_data)

        # Energy should be set
        assert mock_data.energy.item() != 0.0
        # Pair forces should have been passed to LAMMPS
        mock_data.update_pair_forces_gpu.assert_called_once()
        called_forces = mock_data.update_pair_forces_gpu.call_args[0][0]
        assert called_forces.shape == (mock_data.npairs, 3)

    def test_energy_matches_ase_calculator(
        self,
        so3lr_short_range_no_zbl_model,
        make_batch,
        water_atomic_numbers,
    ):
        """Compare LAMMPS mock path vs ASE calculator path.

        This is the gold standard validation: energies from the LAMMPS
        MLIAP code path must match the standard ASE calculator code path.
        """
        atoms = molecule("H2O")
        model = so3lr_short_range_no_zbl_model
        model.double()

        # --- ASE path ---
        batch = make_batch(atoms, r_max=model.r_max)
        ase_output = model(
            batch,
            training=False,
            compute_force=False,
        )
        ase_energy = ase_output["energy"].squeeze().item()

        # --- LAMMPS mock path ---
        calc = LAMMPS_MLIAP_SO3(model, atomic_numbers=water_atomic_numbers)
        calc.device = torch.device("cpu")
        calc.initialized = True
        calc.model = calc.model.to("cpu")

        mock_data = self._atoms_to_mock_lammps_data(
            atoms, model, water_atomic_numbers
        )
        calc.compute_forces(mock_data)
        lammps_energy = mock_data.energy.item()

        # Energies should match within numerical precision
        assert lammps_energy == pytest.approx(
            ase_energy, rel=1e-5
        ), f"Energy mismatch: LAMMPS={lammps_energy}, ASE={ase_energy}"

    def test_forces_match_ase_calculator(
        self,
        so3lr_short_range_no_zbl_model,
        make_batch,
        water_atomic_numbers,
    ):
        """Compare atomic forces from LAMMPS edge forces vs ASE forces.

        Converts LAMMPS pair forces to atomic forces:
            f_i = sum_{j: pair_i=i} pair_forces_ij - sum_{j: pair_j=i} pair_forces_ji
        and checks they match ASE-path forces.
        """
        atoms = molecule("H2O")
        model = so3lr_short_range_no_zbl_model
        model.double()

        # --- ASE path: compute forces via positions gradient ---
        batch = make_batch(atoms, r_max=model.r_max)
        ase_output = model(
            batch,
            training=False,
            compute_force=True,
        )
        ase_forces = ase_output["forces"].detach()

        # --- LAMMPS mock path: compute edge forces ---
        calc = LAMMPS_MLIAP_SO3(model, atomic_numbers=water_atomic_numbers)
        calc.device = torch.device("cpu")
        calc.initialized = True
        calc.model = calc.model.to("cpu")

        mock_data = self._atoms_to_mock_lammps_data(
            atoms, model, water_atomic_numbers
        )
        calc.compute_forces(mock_data)

        # Get the pair forces that were passed to LAMMPS
        pair_forces = mock_data.update_pair_forces_gpu.call_args[0][0]
        pair_i = mock_data.pair_i
        pair_j = mock_data.pair_j
        natoms = mock_data.nlocal

        # Convert pair forces to atomic forces (LAMMPS convention):
        # f[i] += pair_forces[n] for all n where pair_i[n] == i
        # f[j] -= pair_forces[n] for all n where pair_j[n] == j
        lammps_atomic_forces = torch.zeros(natoms, 3, dtype=torch.float64)
        for n in range(len(pair_i)):
            i = pair_i[n].item()
            j = pair_j[n].item()
            lammps_atomic_forces[i] += pair_forces[n]
            if j < natoms:  # Only real atoms
                lammps_atomic_forces[j] -= pair_forces[n]

        # Forces should match
        print(f"\nASE forces:\n{ase_forces}")
        print(
            f"\nLAMMPS atomic forces (from edge forces):\n{lammps_atomic_forces}"
        )
        print(f"\nDifference:\n{ase_forces - lammps_atomic_forces}")
        print(
            f"\nMax abs diff: {(ase_forces - lammps_atomic_forces).abs().max().item()}"
        )

        torch.testing.assert_close(
            lammps_atomic_forces,
            ase_forces,
            atol=1e-5,
            rtol=1e-5,
            msg=f"Force mismatch!\nASE: {ase_forces}\nLAMMPS: {lammps_atomic_forces}",
        )

    def _atoms_to_mock_lammps_data_periodic(
        self, atoms, model, atomic_numbers_list, dtype=torch.float64
    ):
        """Convert periodic ASE Atoms to mock LAMMPS data WITH ghost atoms.

        For periodic systems, LAMMPS creates ghost atoms for periodic images.
        This mock reproduces that behavior by creating ghost atoms for
        neighbors that cross periodic boundaries.

        The mock also provides ``forward_exchange`` / ``reverse_exchange``
        callables that mimic the LAMMPS Kokkos ghost-communication routines
        (used by ``LAMMPS_MP``).
        """
        import numpy as np
        from ase.neighborlist import neighbor_list

        r_max = model.r_max
        i_indices, j_indices, d_vectors, shifts = neighbor_list(
            "ijDS", atoms, cutoff=r_max, self_interaction=False
        )
        natoms = len(atoms)

        # Create ghost atoms for periodic images.
        # Each unique (j_atom, shift) with shift != (0,0,0) needs a ghost.
        ghost_map = {}  # (j_atom, shift_tuple) -> ghost_index
        ghost_types = []  # LAMMPS type index for each ghost
        next_ghost_idx = natoms

        z_to_type = {z: idx for idx, z in enumerate(atomic_numbers_list)}
        atomic_numbers = atoms.get_atomic_numbers()

        new_pair_i = []
        new_pair_j = []
        new_rij = []

        for n in range(len(i_indices)):
            i = i_indices[n]
            j = j_indices[n]
            shift = tuple(shifts[n])
            rij = d_vectors[n]

            new_pair_i.append(i)
            new_rij.append(rij)

            if shift == (0, 0, 0):
                # Same cell - j is a real atom
                new_pair_j.append(j)
            else:
                # Periodic image - need a ghost atom
                key = (j, shift)
                if key not in ghost_map:
                    ghost_map[key] = next_ghost_idx
                    ghost_types.append(z_to_type[atomic_numbers[j]])
                    next_ghost_idx += 1
                new_pair_j.append(ghost_map[key])

        nghosts = len(ghost_map)
        ntotal = natoms + nghosts

        # Build ghost→real parent index mapping (for forward/reverse exchange)
        ghost_to_real = torch.zeros(nghosts, dtype=torch.long)
        for (real_atom, _shift), ghost_idx in ghost_map.items():
            ghost_to_real[ghost_idx - natoms] = real_atom

        # Build element types: real atoms + ghost atoms
        real_types = [z_to_type[z] for z in atomic_numbers]
        all_types = real_types + ghost_types

        mock = Mock()
        mock.nlocal = natoms
        mock.ntotal = ntotal
        mock.npairs = len(new_pair_i)
        mock.elems = torch.tensor(all_types, dtype=torch.int64)
        mock.rij = torch.tensor(np.array(new_rij), dtype=dtype)
        mock.pair_i = torch.tensor(new_pair_i, dtype=torch.int64)
        mock.pair_j = torch.tensor(new_pair_j, dtype=torch.int64)
        mock.eatoms = torch.zeros(natoms, dtype=dtype)
        mock.energy = torch.tensor(0.0, dtype=dtype)
        mock.update_pair_forces_gpu = Mock()
        mock.__class__ = type(
            "RegularModule", (), {"__module__": "regular_module"}
        )

        # -- Mock LAMMPS ghost communication (forward/reverse exchange) ------
        # In real LAMMPS these are Kokkos-level MPI calls. Here we just
        # copy features from real parents to their ghost copies (forward)
        # and accumulate ghost gradients back onto parents (reverse).
        _n_real = natoms
        _g2r = ghost_to_real  # captured by closures

        def _forward_exchange(feats, out, vec_len):
            """Copy real atom features → out; copy parent features → ghost."""
            out.copy_(feats)
            if nghosts > 0:
                out[_n_real:] = feats[_g2r]

        def _reverse_exchange(grad, gout, vec_len):
            """Accumulate ghost gradients onto parents; zero ghosts."""
            gout.copy_(grad)
            if nghosts > 0:
                gout[:_n_real].scatter_add_(
                    0,
                    _g2r.unsqueeze(1).expand_as(grad[_n_real:]),
                    grad[_n_real:],
                )
                gout[_n_real:] = 0.0

        mock.forward_exchange = _forward_exchange
        mock.reverse_exchange = _reverse_exchange
        mock.ghost_to_real = ghost_to_real  # for test force accumulation

        return mock

    def test_forces_match_ase_periodic(
        self,
        so3lr_short_range_no_zbl_model,
        make_batch,
        si_atomic_numbers,
    ):
        """Compare forces for PERIODIC system with ghost atoms.

        This is the critical test: periodic systems require ghost atoms
        in LAMMPS, and the edge force → atomic force conversion must
        correctly handle ghost indices.
        """
        atoms = bulk("Si", "diamond", a=5.43)
        model = so3lr_short_range_no_zbl_model
        model.double()

        # --- ASE path ---
        batch = make_batch(atoms, r_max=model.r_max)
        ase_output = model(
            batch,
            training=False,
            compute_force=True,
        )
        ase_forces = ase_output["forces"].detach()
        ase_energy = ase_output["energy"].squeeze().item()

        # --- LAMMPS mock path (with ghost atoms) ---
        calc = LAMMPS_MLIAP_SO3(model, atomic_numbers=si_atomic_numbers)
        calc.device = torch.device("cpu")
        calc.initialized = True
        calc.model = calc.model.to("cpu")

        mock_data = self._atoms_to_mock_lammps_data_periodic(
            atoms, model, si_atomic_numbers
        )
        calc.compute_forces(mock_data)
        lammps_energy = mock_data.energy.item()

        print(
            f"\nPeriodic Si: natoms={mock_data.nlocal}, "
            f"nghosts={mock_data.ntotal - mock_data.nlocal}, "
            f"npairs={mock_data.npairs}"
        )
        print(f"ASE energy: {ase_energy}")
        print(f"LAMMPS energy: {lammps_energy}")

        # Energy should match
        assert lammps_energy == pytest.approx(
            ase_energy, rel=1e-5
        ), f"Energy mismatch: LAMMPS={lammps_energy}, ASE={ase_energy}"

        # Convert edge forces to atomic forces (Newton's 3rd law).
        # Ghost atom forces must be mapped back to their real parent atoms,
        # mirroring what LAMMPS does via reverse communication.
        pair_forces = mock_data.update_pair_forces_gpu.call_args[0][0]
        pair_i = mock_data.pair_i
        pair_j = mock_data.pair_j
        natoms = mock_data.nlocal
        g2r = mock_data.ghost_to_real

        lammps_atomic_forces = torch.zeros(natoms, 3, dtype=torch.float64)
        for n in range(len(pair_i)):
            i = pair_i[n].item()
            j = pair_j[n].item()
            lammps_atomic_forces[i] += pair_forces[n]
            if j < natoms:
                lammps_atomic_forces[j] -= pair_forces[n]
            else:
                # Ghost j → map force to real parent
                real_parent = g2r[j - natoms].item()
                lammps_atomic_forces[real_parent] -= pair_forces[n]

        print(f"\nASE forces:\n{ase_forces}")
        print(f"\nLAMMPS atomic forces:\n{lammps_atomic_forces}")
        print(f"\nDifference:\n{ase_forces - lammps_atomic_forces}")
        print(
            f"\nMax abs diff: {(ase_forces - lammps_atomic_forces).abs().max().item()}"
        )

        torch.testing.assert_close(
            lammps_atomic_forces,
            ase_forces,
            atol=1e-10,
            rtol=1e-10,
            msg=f"Force mismatch (periodic)!\nASE: {ase_forces}\nLAMMPS: {lammps_atomic_forces}",
        )
