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
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


@pytest.fixture
def h2o_atoms():
    """H2O molecule for testing."""
    return molecule("H2O")


@pytest.fixture
def nh3_atoms():
    """NH3 molecule for testing."""
    return molecule("NH3")


@pytest.fixture
def ch4_atoms():
    """CH4 molecule for testing."""
    return molecule("CH4")


@pytest.fixture
def ethanol_atoms():
    """Ethanol molecule for testing (larger system)."""
    return molecule("CH3CH2OH")


@pytest.fixture
def si_bulk():
    """Silicon bulk structure for periodic boundary testing."""
    return bulk("Si", "diamond", a=5.43)


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
        "dispersion_energy_cutoff_lr_damping": 2.0,
    }


@pytest.fixture
def multihead_model_config(so3lr_model_config):
    """MultiHead SO3LR model configuration."""
    return {
        **so3lr_model_config,
        "num_output_heads": 4,
        "energy_regression_dim": so3lr_model_config["num_features"],
    }


@pytest.fixture
def mock_batch_for_loss():
    """Batch-like object with known values for hand-computing loss."""
    from so3krates_torch.tools.torch_geometric import Batch

    # 2 graphs: 3 atoms (graph 0) + 4 atoms (graph 1)
    # Known energy/forces/weights for manual loss calculation
    batch = Batch(
        ptr=torch.tensor([0, 3, 7]),  # Graph boundaries
        batch=torch.tensor([0, 0, 0, 1, 1, 1, 1]),  # Atom mapping
        energy=torch.tensor([10.0, 20.0]),
        forces=torch.tensor(
            [
                [1.0, 0.0, 0.0],  # Graph 0
                [0.0, 1.0, 0.0],
                [0.0, 0.0, 1.0],
                [2.0, 0.0, 0.0],  # Graph 1
                [0.0, 2.0, 0.0],
                [0.0, 0.0, 2.0],
                [1.0, 1.0, 1.0],
            ]
        ),
        dipole=torch.tensor([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]),
        hirshfeld_ratios=torch.ones(7),
        weight=torch.tensor([1.0, 2.0]),  # Per-graph
        energy_weight=torch.tensor([1.0, 1.5]),
        forces_weight=torch.tensor([10.0, 20.0]),
        dipole_weight=torch.tensor([[1.0, 1.0, 1.0], [2.0, 2.0, 2.0]]),
        hirshfeld_ratios_weight=torch.tensor([1.0, 2.0]),
    )

    return batch


@pytest.fixture
def make_batch(device):
    """Factory fixture to convert ASE Atoms to model-ready batch."""

    def _make(atoms, r_max, cutoff_lr=None, dtype=torch.float64):
        from so3krates_torch.data.utils import (
            KeySpecification,
            config_from_atoms,
        )
        from so3krates_torch.data.atomic_data import AtomicData
        from so3krates_torch.tools import torch_geometric
        from so3krates_torch.tools.utils import (
            AtomicNumberTable,
        )

        torch.set_default_dtype(dtype)
        z_table = AtomicNumberTable([int(z) for z in range(1, 119)])
        config = config_from_atoms(atoms, key_specification=KeySpecification())
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                AtomicData.from_config(
                    config,
                    z_table=z_table,
                    cutoff=r_max,
                    cutoff_lr=cutoff_lr,
                )
            ],
            batch_size=1,
            shuffle=False,
            drop_last=False,
        )
        return next(iter(data_loader)).to(device)

    return _make


@pytest.fixture
def make_batch_list(device):
    """Factory fixture to batch multiple ASE Atoms."""

    def _make(atoms_list, r_max, cutoff_lr=None, dtype=torch.float64):
        from so3krates_torch.data.utils import (
            KeySpecification,
            config_from_atoms,
        )
        from so3krates_torch.data.atomic_data import AtomicData
        from so3krates_torch.tools import torch_geometric
        from so3krates_torch.tools.utils import (
            AtomicNumberTable,
        )

        torch.set_default_dtype(dtype)
        z_table = AtomicNumberTable([int(z) for z in range(1, 119)])
        dataset = [
            AtomicData.from_config(
                config_from_atoms(
                    a,
                    key_specification=KeySpecification(),
                ),
                z_table=z_table,
                cutoff=r_max,
                cutoff_lr=cutoff_lr,
            )
            for a in atoms_list
        ]
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=dataset,
            batch_size=len(atoms_list),
            shuffle=False,
            drop_last=False,
        )
        return next(iter(data_loader)).to(device)

    return _make


@pytest.fixture
def example_xyz_with_data(tmp_path):
    """Create XYZ file with energy/forces for testing."""
    from ase.build import molecule
    import ase.io

    # Create a few molecules with properties
    atoms_list = []
    for mol_name in ["H2O", "NH3", "CH4"]:
        atoms = molecule(mol_name)
        # Add fake energy and forces
        atoms.info["REF_energy"] = -10.0 * len(atoms)
        atoms.arrays["REF_forces"] = np.random.randn(len(atoms), 3) * 0.1
        atoms_list.append(atoms)

    # Write to XYZ
    xyz_path = tmp_path / "test_data.xyz"
    ase.io.write(xyz_path, atoms_list)
    return str(xyz_path)


@pytest.fixture
def example_raw_hdf5(tmp_path):
    """Create raw HDF5 file for testing."""
    from ase.build import molecule
    from so3krates_torch.data.hdf5_utils import save_atoms_to_hdf5
    from so3krates_torch.data.utils import KeySpecification

    # Create molecules with properties
    atoms_list = []
    for mol_name in ["H2O", "NH3", "CH4"]:
        atoms = molecule(mol_name)
        atoms.info["REF_energy"] = -10.0 * len(atoms)
        atoms.arrays["REF_forces"] = np.random.randn(len(atoms), 3) * 0.1
        atoms_list.append(atoms)

    # Save to raw HDF5
    hdf5_path = tmp_path / "test_raw.h5"
    keyspec = KeySpecification(
        info_keys={"energy": "REF_energy"},
        arrays_keys={"forces": "REF_forces"},
    )
    save_atoms_to_hdf5(atoms_list, str(hdf5_path), keyspec)
    return str(hdf5_path)


@pytest.fixture
def example_preprocessed_hdf5(tmp_path):
    """Create preprocessed HDF5 file for testing."""
    from ase.build import molecule
    from so3krates_torch.data.hdf5_utils import (
        save_preprocessed_hdf5,
    )
    from so3krates_torch.data.utils import (
        KeySpecification,
        config_from_atoms,
    )
    from so3krates_torch.data.atomic_data import AtomicData
    from so3krates_torch.tools.utils import AtomicNumberTable

    # Create molecules with properties
    atoms_list = []
    for mol_name in ["H2O", "NH3", "CH4"]:
        atoms = molecule(mol_name)
        atoms.info["REF_energy"] = -10.0 * len(atoms)
        atoms.arrays["REF_forces"] = np.random.randn(len(atoms), 3) * 0.1
        atoms_list.append(atoms)

    # Convert to AtomicData
    keyspec = KeySpecification(
        info_keys={"energy": "REF_energy"},
        arrays_keys={"forces": "REF_forces"},
    )
    configs = [config_from_atoms(atoms, keyspec) for atoms in atoms_list]

    # Create z_table
    all_zs = set()
    for config in configs:
        all_zs.update(config.atomic_numbers)
    z_table = AtomicNumberTable(sorted(list(all_zs)))

    # Preprocess
    r_max = 5.0
    r_max_lr = None
    data_list = [
        AtomicData.from_config(
            config, z_table=z_table, cutoff=r_max, cutoff_lr=r_max_lr
        )
        for config in configs
    ]

    # Save to preprocessed HDF5
    hdf5_path = tmp_path / "test_preprocessed.h5"
    save_preprocessed_hdf5(
        data_list,
        str(hdf5_path),
        r_max=r_max,
        r_max_lr=r_max_lr,
        z_table=z_table,
    )
    return str(hdf5_path)
