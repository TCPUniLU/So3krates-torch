"""Shared pytest fixtures for So3krates-torch tests."""
import os
import pytest
import torch
import numpy as np
from ase.build import molecule, bulk


@pytest.fixture(autouse=True)
def set_deterministic():
    """Ensure deterministic behavior for all tests."""
    seed = 42
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    torch.use_deterministic_algorithms(True, warn_only=True)
    yield
    torch.use_deterministic_algorithms(False)


@pytest.fixture
def device():
    """Return available device (CUDA if available, else CPU)."""
    return torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )


@pytest.fixture
def h2o_atoms():
    """H2O molecule for testing."""
    return molecule("H2O")


@pytest.fixture
def ethanol_atoms():
    """Ethanol molecule for testing (larger system)."""
    return molecule("CH3CH2OH")


@pytest.fixture
def si_bulk():
    """Silicon bulk structure for periodic boundary testing."""
    return bulk("Si", "diamond", a=5.43)


@pytest.fixture
def si_supercell(si_bulk):
    """2x2x2 silicon supercell for larger periodic tests."""
    return si_bulk * (2, 2, 2)


@pytest.fixture
def random_batch_atoms():
    """Create a batch of 3 random small molecules."""
    return [molecule(name) for name in ["H2O", "NH3", "CH4"]]


@pytest.fixture
def default_model_config():
    """Minimal model configuration for fast tests."""
    return {
        "r_max": 5.0,
        "num_radial_basis_fn": 8,
        "degrees": [1, 2],
        "num_features": 16,
        "num_heads": 2,
        "num_layers": 1,
        "num_elements": 118,
        "avg_num_neighbors": 10.0,
        "final_mlp_layers": 1,
        "dtype": torch.float64,
        "seed": 42,
    }


@pytest.fixture
def so3lr_model_config(default_model_config):
    """SO3LR model configuration with long-range components."""
    return {
        **default_model_config,
        "r_max_lr": 10.0,
        "zbl_repulsion_bool": True,
        "electrostatic_energy_bool": True,
        "dispersion_energy_bool": True,
    }


@pytest.fixture
def multihead_model_config(so3lr_model_config):
    """MultiHead SO3LR model configuration."""
    return {
        **so3lr_model_config,
        "num_output_heads": 4,
        "energy_regression_dim": so3lr_model_config[
            "num_features"
        ],
    }
