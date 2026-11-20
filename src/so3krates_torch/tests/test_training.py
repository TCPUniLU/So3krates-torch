"""Tests for training functionality."""

import json
from pathlib import Path

import pytest
import torch
from ase.io import read
from pydantic import ValidationError

from so3krates_torch.tools.model_setup import create_model
from so3krates_torch.config import ArchitectureConfig
from so3krates_torch.data.utils import KeySpecification
from so3krates_torch.modules.loss import (
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesDipoleHirshfeldLoss,
)
from so3krates_torch.modules.models import So3krates, SO3LR
from so3krates_torch.tools.utils import (
    AtomicNumberTable,
    create_dataloader_from_list,
)


_DATA_DIR = Path(__file__).parent / "data"
_REF_PATH = _DATA_DIR / "training_references.json"


def _load_references():
    if _REF_PATH.exists():
        with open(_REF_PATH) as f:
            return json.load(f)
    return {}


def _save_references(refs):
    with open(_REF_PATH, "w") as f:
        json.dump(refs, f, indent=2)


@pytest.mark.parametrize(
    "model_class,loss_class,use_lr,reference_key",
    [
        (
            So3krates,
            WeightedEnergyForcesLoss,
            False,
            "so3krates_energy_forces",
        ),
        (
            SO3LR,
            WeightedEnergyForcesDipoleHirshfeldLoss,
            True,
            "so3lr_full_properties",
        ),
    ],
    ids=["so3krates_base", "so3lr_full"],
)
def test_single_epoch_training_deterministic(
    model_class,
    loss_class,
    use_lr,
    reference_key,
    device,
):
    """Train for single epoch on real data, compare to ref."""
    dtype = torch.float64
    torch.set_default_dtype(dtype)
    r_max = 5.0
    r_max_lr = 10.0 if use_lr else None

    # --- Load real data ---
    data_path = _DATA_DIR / "aqm_small.xyz"
    atoms_list = read(str(data_path), index=":")[:5]

    keyspec = KeySpecification(
        info_keys={
            "energy": "REF_energy",
            "dipole": "REF_dipole",
        },
        arrays_keys={
            "hirshfeld_ratios": "REF_hirsh_ratios",
            "forces": "REF_forces",
        },
    )

    z_table = AtomicNumberTable([int(z) for z in range(1, 119)])

    train_loader = create_dataloader_from_list(
        atoms_list=atoms_list,
        batch_size=2,
        r_max=r_max,
        r_max_lr=r_max_lr,
        key_specification=keyspec,
        shuffle=False,
        z_table=z_table,
    )

    # --- Create model ---
    model_config = {
        "r_max": r_max,
        "num_radial_basis_fn": 8,
        "degrees": [1, 2],
        "num_features": 16,
        "num_heads": 2,
        "num_layers": 1,
        "num_elements": 118,
        "avg_num_neighbors": 10.0,
        "final_mlp_layers": 1,
        "dtype": dtype,
        "seed": 42,
    }

    if use_lr:
        model_config.update(
            {
                "r_max_lr": r_max_lr,
                "zbl_repulsion_bool": True,
                "electrostatic_energy_bool": True,
                "dispersion_energy_bool": True,
                "dispersion_energy_cutoff_lr_damping": 2.0,
            }
        )

    model = model_class(**model_config).to(device)
    model.train()

    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    if loss_class == WeightedEnergyForcesLoss:
        loss_fn = loss_class(energy_weight=1.0, forces_weight=100.0)
    else:
        loss_fn = loss_class(
            energy_weight=1.0,
            forces_weight=100.0,
            dipole_weight=1.0,
            hirshfeld_weight=1.0,
        )

    # --- Train one epoch ---
    total_loss = 0.0
    num_batches = 0

    for batch in train_loader:
        batch = batch.to(device)
        batch.positions.requires_grad_(True)
        batch_dict = batch.to_dict()

        optimizer.zero_grad()

        output = model(batch_dict, training=True, compute_force=True)

        loss = loss_fn(pred=output, ref=batch)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        num_batches += 1

    avg_loss = total_loss / num_batches

    # --- Compute metrics ---
    model.eval()
    first_batch = next(iter(train_loader)).to(device)
    first_batch.positions.requires_grad_(True)
    first_output = model(
        first_batch.to_dict(),
        training=False,
        compute_force=True,
    )

    total_param_norm = sum(
        p.norm().item() for p in model.parameters() if p.requires_grad
    )

    metrics = {
        "avg_loss": avg_loss,
        "total_param_norm": total_param_norm,
        "first_batch_energy_0": (first_output["energy"][0].item()),
        "first_batch_forces_norm": (first_output["forces"].norm().item()),
    }

    if "dipole" in first_output:
        metrics["first_batch_dipole_norm"] = (
            first_output["dipole"].norm().item()
        )
    if "hirshfeld_ratios" in first_output:
        metrics["first_batch_hirshfeld_norm"] = (
            first_output["hirshfeld_ratios"].norm().item()
        )

    # --- Compare to reference ---
    references = _load_references()

    if reference_key not in references:
        references[reference_key] = metrics
        _save_references(references)
        pytest.skip(f"Generated reference for '{reference_key}'")

    ref = references[reference_key]
    rtol = 1e-6
    atol = 1e-8

    for key, actual in metrics.items():
        assert key in ref, f"Missing reference metric: {key}"
        expected = ref[key]

        if abs(expected) > atol:
            rel_diff = abs(actual - expected) / abs(expected)
            assert rel_diff < rtol, (
                f"{key}: {actual} != {expected} "
                f"(rel_diff={rel_diff:.2e}, rtol={rtol:.2e})"
            )
        else:
            abs_diff = abs(actual - expected)
            assert abs_diff < atol, (
                f"{key}: {actual} != {expected} "
                f"(abs_diff={abs_diff:.2e}, atol={atol:.2e})"
            )


def test_all_parameters_receive_gradients(
    default_model_config, make_batch, h2o_atoms
):
    """After a backward pass, every requires_grad parameter has .grad."""
    from so3krates_torch.modules.loss import WeightedEnergyForcesLoss

    model = So3krates(**default_model_config)
    model.train()

    batch = make_batch(h2o_atoms, r_max=5.0)
    batch_dict = batch.to_dict()
    batch_dict["positions"].requires_grad_(True)

    output = model(batch_dict, training=True, compute_force=True)

    # Create a simple loss from energy and forces
    loss = output["energy"].sum() + output["forces"].sum()
    loss.backward()

    no_grad = [
        name
        for name, p in model.named_parameters()
        if p.requires_grad and p.grad is None
    ]
    assert (
        no_grad == []
    ), f"Parameters without gradients after backward: {no_grad}"


def test_arch_config_validates_lr_cutoff_missing():
    """Test that ArchitectureConfig raises error when r_max_lr is
    None but long-range physics is enabled."""
    with pytest.raises(
        ValidationError, match="Long-range cutoff.*must be specified"
    ):
        ArchitectureConfig(
            degrees=[1, 2],
            r_max=5.0,
            r_max_lr=None,
            electrostatic_energy_bool=True,
            dispersion_energy_bool=False,
        )


def test_arch_config_validates_lr_cutoff_with_dispersion():
    """Test that ArchitectureConfig raises error when r_max_lr is
    None but dispersion is enabled."""
    with pytest.raises(
        ValidationError, match="Long-range cutoff.*must be specified"
    ):
        ArchitectureConfig(
            degrees=[1, 2],
            r_max=5.0,
            r_max_lr=None,
            electrostatic_energy_bool=False,
            dispersion_energy_bool=True,
        )


def test_arch_config_accepts_unused_lr_cutoff():
    """Test that ArchitectureConfig accepts r_max_lr when both
    long-range physics features are disabled."""
    cfg = ArchitectureConfig(
        degrees=[1, 2],
        r_max=5.0,
        r_max_lr=10.0,
        electrostatic_energy_bool=False,
        dispersion_energy_bool=False,
    )
    assert cfg.r_max_lr == 10.0


def test_create_model_accepts_valid_lr_config(device):
    """Test that create_model accepts valid long-range configuration."""
    config = {
        "GENERAL": {
            "name_exp": "test",
            "seed": 42,
            "default_dtype": "float32",
        },
        "ARCHITECTURE": {
            "degrees": [1, 2],
            "r_max": 5.0,
            "r_max_lr": 10.0,
            "electrostatic_energy_bool": True,
            "dispersion_energy_bool": True,
        },
    }

    model = create_model(config, device)
    assert model is not None
    assert isinstance(model, SO3LR)
    assert model.r_max_lr == 10.0


def test_create_model_honors_final_layer_bias_false(device):
    """Regression: ``final_layer_bias`` in ARCHITECTURE must reach the
    model. Several bundled v2 checkpoints (e.g. so3lr-s/-m/-l) save
    ``final_layer_bias: false`` -- if ``create_model`` silently dropped
    this key (defaulting to ``True``), a model built from such a
    checkpoint's own settings for ``pretrained_weights`` warm-starting
    would have an extra ``atomic_energy_output_block.final_layer.bias``
    parameter absent from the checkpoint's state_dict, which is a real
    (non-PME) missing key rather than an intentional PME buffer."""
    config = {
        "GENERAL": {"seed": 42, "default_dtype": "float32"},
        "ARCHITECTURE": {
            "degrees": [1, 2],
            "r_max": 5.0,
            "r_max_lr": 10.0,
            "final_layer_bias": False,
        },
    }
    model = create_model(config, device)
    assert model.atomic_energy_output_block.final_layer.bias is None


def test_setup_optimizer_and_scheduler_adam(default_model_config):
    from so3krates_torch.tools.training_setup import (
        setup_optimizer_and_scheduler,
    )
    from so3krates_torch.modules.models import So3krates

    model = So3krates(**default_model_config)
    config = {
        "TRAINING": {
            "optimizer": "adam",
            "lr": 1e-3,
            "weight_decay": 1e-4,
            "amsgrad": False,
            "scheduler": "exponential_decay",
            "lr_scheduler_gamma": 0.99,
        }
    }
    optimizer, scheduler = setup_optimizer_and_scheduler(model, config)
    assert isinstance(optimizer, torch.optim.Adam)
    assert abs(optimizer.param_groups[0]["lr"] - 1e-3) < 1e-9
    assert isinstance(scheduler, torch.optim.lr_scheduler.ExponentialLR)
    assert abs(scheduler.gamma - 0.99) < 1e-9


def test_setup_optimizer_and_scheduler_plateau(default_model_config):
    from so3krates_torch.tools.training_setup import (
        setup_optimizer_and_scheduler,
    )
    from so3krates_torch.modules.models import So3krates

    model = So3krates(**default_model_config)
    config = {
        "TRAINING": {
            "lr": 5e-4,
            "scheduler": "reduce_on_plateau",
            "scheduler_patience": 10,
            "lr_factor": 0.5,
        }
    }
    _, scheduler = setup_optimizer_and_scheduler(model, config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau)
    assert scheduler.patience == 10
    assert abs(scheduler.factor - 0.5) < 1e-9


def test_setup_optimizer_invalid_name_raises(default_model_config):
    from so3krates_torch.tools.training_setup import (
        setup_optimizer_and_scheduler,
    )
    from so3krates_torch.modules.models import So3krates

    model = So3krates(**default_model_config)
    config = {"TRAINING": {"lr": 1e-3, "optimizer": "sgd"}}
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        setup_optimizer_and_scheduler(model, config)


def test_process_config_atomic_energies_int_keys():
    from so3krates_torch.tools.model_setup import (
        process_config_atomic_energies,
    )

    shifts = process_config_atomic_energies({1: -13.6, 6: -1027.0, 8: -2042.0})
    assert shifts[1] == -13.6
    assert shifts[6] == -1027.0
    assert shifts[8] == -2042.0
    # Missing elements default to 0.0
    assert shifts[7] == 0.0
    # Full range 1-118
    assert len(shifts) == 118


def test_process_config_atomic_energies_str_keys():
    from so3krates_torch.tools.model_setup import (
        process_config_atomic_energies,
    )

    shifts = process_config_atomic_energies({"1": -13.6, "6": -1027.0})
    assert shifts[1] == -13.6
    assert shifts[6] == -1027.0


def test_process_config_atomic_energies_with_default_shifts():
    from so3krates_torch.tools.model_setup import (
        process_config_atomic_energies,
    )

    default_shifts = torch.arange(1, 119, dtype=torch.float64) * -0.01
    shifts = process_config_atomic_energies(
        {6: -1027.0}, default_shifts=default_shifts
    )
    # overridden element uses the provided value
    assert shifts[6] == -1027.0
    # unspecified elements fall back to default_shifts, not 0.0
    assert abs(shifts[1] - default_shifts[0].item()) < 1e-12
    assert abs(shifts[7] - default_shifts[6].item()) < 1e-12


def test_set_avg_num_neighbors_in_model(default_model_config):
    from so3krates_torch.tools.model_setup import (
        set_avg_num_neighbors_in_model,
    )
    from so3krates_torch.modules.models import So3krates

    model = So3krates(**default_model_config)
    set_avg_num_neighbors_in_model(model, 15.5)
    assert model.avg_num_neighbors == 15.5
    for layer in model.euclidean_transformers:
        assert layer.euclidean_attention_block.att_norm_inv == 15.5
        assert layer.euclidean_attention_block.att_norm_ev == 15.5


def test_set_atomic_energy_shifts_in_model(default_model_config, device):
    from so3krates_torch.tools.model_setup import (
        set_atomic_energy_shifts_in_model,
    )
    from so3krates_torch.modules.models import So3krates

    model = So3krates(**default_model_config).to(device)
    shifts = {z: float(z) * -0.1 for z in range(1, 119)}
    set_atomic_energy_shifts_in_model(model, shifts)
    stored = model.atomic_energy_output_block.energy_shifts
    assert stored is not None
    # shifts dict is sorted by key (z), so index 0 -> z=1 (H), index 5 -> z=6 (C)
    assert abs(stored[0].item() - (-0.1)) < 1e-9
    assert abs(stored[5].item() - (-0.6)) < 1e-9


def test_resolve_atomic_energy_shifts_warm_start_merge(
    default_model_config, device
):
    from so3krates_torch.tools.model_setup import (
        resolve_atomic_energy_shifts,
        set_atomic_energy_shifts_in_model,
    )

    model = So3krates(**default_model_config).to(device)
    pretrained_shifts = {z: float(z) * -0.1 for z in range(1, 119)}
    set_atomic_energy_shifts_in_model(model, pretrained_shifts)

    config = {
        "ARCHITECTURE": {"atomic_energy_shifts": {6: -999.0}},
        "TRAINING": {},
    }
    average_atomic_energy_shifts = torch.zeros(118, dtype=torch.float64)

    result = resolve_atomic_energy_shifts(
        config,
        model,
        warm_start=True,
        average_atomic_energy_shifts=average_atomic_energy_shifts,
    )

    # overridden element uses the provided value
    assert result[6] == -999.0
    # unspecified elements retain the pretrained checkpoint's own shifts
    assert abs(result[1] - pretrained_shifts[1]) < 1e-9
    assert abs(result[7] - pretrained_shifts[7]) < 1e-9


def test_resolve_atomic_energy_shifts_warm_start_mutually_exclusive_raises(
    default_model_config, device
):
    from so3krates_torch.tools.model_setup import (
        resolve_atomic_energy_shifts,
    )

    model = So3krates(**default_model_config).to(device)
    config = {
        "ARCHITECTURE": {"atomic_energy_shifts": {6: -999.0}},
        "TRAINING": {"force_use_average_shifts": True},
    }
    average_atomic_energy_shifts = torch.zeros(118, dtype=torch.float64)

    with pytest.raises(ValueError, match="mutually exclusive|cannot both"):
        resolve_atomic_energy_shifts(
            config,
            model,
            warm_start=True,
            average_atomic_energy_shifts=average_atomic_energy_shifts,
        )


def test_resolve_atomic_energy_shifts_cold_start_unchanged(
    default_model_config, device
):
    from so3krates_torch.tools.model_setup import (
        resolve_atomic_energy_shifts,
    )

    model = So3krates(**default_model_config).to(device)
    config = {
        "ARCHITECTURE": {"atomic_energy_shifts": {6: -1027.0}},
        "TRAINING": {},
    }
    average_atomic_energy_shifts = torch.zeros(118, dtype=torch.float64)

    result = resolve_atomic_energy_shifts(
        config,
        model,
        warm_start=False,
        average_atomic_energy_shifts=average_atomic_energy_shifts,
    )

    assert result[6] == -1027.0
    # cold start still zero-fills unspecified elements
    assert result[1] == 0.0


def test_set_dtype_model_float32(default_model_config):
    from so3krates_torch.tools.model_setup import set_dtype_model
    from so3krates_torch.modules.models import So3krates

    model = So3krates(**default_model_config)
    set_dtype_model(model, "float32")
    for param in model.parameters():
        assert param.dtype == torch.float32


def test_set_dtype_model_float64(default_model_config):
    from so3krates_torch.tools.model_setup import set_dtype_model
    from so3krates_torch.modules.models import So3krates

    model = So3krates(**default_model_config)
    set_dtype_model(model, "float64")
    for param in model.parameters():
        assert param.dtype == torch.float64


def test_set_dtype_model_invalid_raises(default_model_config):
    from so3krates_torch.tools.model_setup import set_dtype_model
    from so3krates_torch.modules.models import So3krates

    model = So3krates(**default_model_config)
    with pytest.raises(ValueError, match="Unsupported dtype"):
        set_dtype_model(model, "int32")


def test_setup_loss_function_auto_energy_forces():
    from so3krates_torch.tools.training_setup import setup_loss_function
    from so3krates_torch.modules.loss import WeightedPropertiesLoss

    config = {
        "TRAINING": {
            "energy_weight": 2.0,
            "forces_weight": 500.0,
        }
    }
    loss = setup_loss_function(config)
    assert isinstance(loss, WeightedPropertiesLoss)
    assert loss.properties == ["energy", "forces"]
    assert abs(loss.energy_weight.item() - 2.0) < 1e-9
    assert abs(loss.forces_weight.item() - 500.0) < 1e-9


def test_setup_loss_function_invalid_type_raises():
    from so3krates_torch.tools.training_setup import setup_loss_function

    config = {"TRAINING": {"loss_type": "nonexistent"}}
    with pytest.raises(ValueError, match="Unknown loss_type"):
        setup_loss_function(config)


def test_get_loss_property_weights_defaults():
    from so3krates_torch.tools.training_setup import (
        get_loss_property_weights,
    )

    weights = get_loss_property_weights(
        {"TRAINING": {"hirshfeld_weight": 3.0}}
    )
    assert weights == {
        "energy": 1.0,
        "forces": 1000.0,
        "dipole": 0.0,
        "hirshfeld_ratios": 3.0,
        "charges": 0.0,
        "c6_ratios": 0.0,
    }


def _singlehead_config(train_path, **training_overrides):
    config = {
        "GENERAL": {},
        "ARCHITECTURE": {"r_max": 5.0, "r_max_lr": None},
        "TRAINING": {
            "path_to_train_data": str(train_path),
            "batch_size": 2,
            "valid_batch_size": 2,
        },
    }
    config["TRAINING"].update(training_overrides)
    return config


def test_missing_loss_property_raises_for_raw_xyz(example_xyz_with_data):
    """A loss weight configured for a property absent from the training
    file must stop with a clear error instead of silently zero-filling."""
    from so3krates_torch.tools.data_setup import (
        _setup_singlehead_data_loaders,
    )

    config = _singlehead_config(example_xyz_with_data, hirshfeld_weight=1.0)
    with pytest.raises(ValueError, match="hirshfeld_ratios"):
        _setup_singlehead_data_loaders(config)


def test_missing_loss_property_raises_for_lazy_raw_hdf5(example_raw_hdf5):
    from so3krates_torch.tools.data_setup import (
        _setup_singlehead_data_loaders,
    )

    config = _singlehead_config(
        example_raw_hdf5,
        hirshfeld_weight=1.0,
        lazy_loading=True,
    )
    with pytest.raises(ValueError, match="hirshfeld_ratios"):
        _setup_singlehead_data_loaders(config)


def test_missing_loss_property_raises_for_preprocessed_hdf5(
    example_preprocessed_hdf5_full_keyspec,
):
    from so3krates_torch.tools.data_setup import (
        _setup_singlehead_data_loaders,
    )

    config = _singlehead_config(
        example_preprocessed_hdf5_full_keyspec,
        hirshfeld_weight=1.0,
        data_preprocessed=True,
    )
    with pytest.raises(ValueError, match="hirshfeld_ratios"):
        _setup_singlehead_data_loaders(config)


def test_missing_loss_property_raises_for_multihead(tmp_path):
    """Aggregate multi-head check: a loss-required property absent from
    every head's data must still raise."""
    import numpy as np
    from ase.build import molecule
    import ase.io
    from so3krates_torch.tools.data_setup import (
        _setup_multihead_data_loaders,
    )

    def make_xyz(path):
        atoms_list = []
        for mol_name in ["H2O", "NH3", "CH4"]:
            atoms = molecule(mol_name)
            atoms.info["REF_energy"] = -10.0 * len(atoms)
            atoms.arrays["REF_forces"] = np.random.randn(len(atoms), 3) * 0.1
            atoms_list.append(atoms)
        ase.io.write(path, atoms_list)

    path_a = tmp_path / "a.xyz"
    path_b = tmp_path / "b.xyz"
    make_xyz(path_a)
    make_xyz(path_b)

    config = {
        "GENERAL": {},
        "ARCHITECTURE": {"r_max": 5.0, "r_max_lr": None},
        "TRAINING": {
            "batch_size": 2,
            "valid_batch_size": 2,
            "hirshfeld_weight": 1.0,
            "heads": {
                "head_a": {"path_to_train_data": str(path_a)},
                "head_b": {"path_to_train_data": str(path_b)},
            },
        },
    }
    with pytest.raises(ValueError, match="hirshfeld_ratios"):
        _setup_multihead_data_loaders(config)


def test_heterogeneous_multihead_property_raises_by_default(tmp_path):
    """A property present in only one of several heads must raise by
    default: presence is checked per head, not in aggregate, so a
    property missing from an entire head is no longer silently
    tolerated just because another head happens to have it."""
    import numpy as np
    from ase.build import molecule
    import ase.io
    from so3krates_torch.tools.data_setup import (
        _setup_multihead_data_loaders,
    )

    def make_xyz(path, with_hirshfeld):
        atoms_list = []
        for mol_name in ["H2O", "NH3", "CH4"]:
            atoms = molecule(mol_name)
            atoms.info["REF_energy"] = -10.0 * len(atoms)
            atoms.arrays["REF_forces"] = np.random.randn(len(atoms), 3) * 0.1
            if with_hirshfeld:
                atoms.arrays["REF_hirsh_ratios"] = np.ones(len(atoms))
            atoms_list.append(atoms)
        ase.io.write(path, atoms_list)

    path_a = tmp_path / "a.xyz"
    path_b = tmp_path / "b.xyz"
    make_xyz(path_a, with_hirshfeld=True)
    make_xyz(path_b, with_hirshfeld=False)

    config = {
        "GENERAL": {},
        "ARCHITECTURE": {"r_max": 5.0, "r_max_lr": None},
        "TRAINING": {
            "batch_size": 2,
            "valid_batch_size": 2,
            "hirshfeld_weight": 1.0,
            "heads": {
                "head_a": {
                    "path_to_train_data": str(path_a),
                    "num_valid": 1,
                },
                "head_b": {
                    "path_to_train_data": str(path_b),
                    "num_valid": 1,
                },
            },
        },
    }
    with pytest.raises(ValueError, match="hirshfeld_ratios"):
        _setup_multihead_data_loaders(config)


def test_heterogeneous_multihead_property_excluded_does_not_raise(tmp_path):
    """A head can opt out of a globally-required property via
    `exclude_loss_properties`, for legitimately heterogeneous
    multi-head setups (e.g. a head that's an energy/forces-only
    ensemble member)."""
    import numpy as np
    from ase.build import molecule
    import ase.io
    from so3krates_torch.tools.data_setup import (
        _setup_multihead_data_loaders,
    )

    def make_xyz(path, with_hirshfeld):
        atoms_list = []
        for mol_name in ["H2O", "NH3", "CH4"]:
            atoms = molecule(mol_name)
            atoms.info["REF_energy"] = -10.0 * len(atoms)
            atoms.arrays["REF_forces"] = np.random.randn(len(atoms), 3) * 0.1
            if with_hirshfeld:
                atoms.arrays["REF_hirsh_ratios"] = np.ones(len(atoms))
            atoms_list.append(atoms)
        ase.io.write(path, atoms_list)

    path_a = tmp_path / "a.xyz"
    path_b = tmp_path / "b.xyz"
    make_xyz(path_a, with_hirshfeld=True)
    make_xyz(path_b, with_hirshfeld=False)

    config = {
        "GENERAL": {},
        "ARCHITECTURE": {"r_max": 5.0, "r_max_lr": None},
        "TRAINING": {
            "batch_size": 2,
            "valid_batch_size": 2,
            "hirshfeld_weight": 1.0,
            "heads": {
                "head_a": {
                    "path_to_train_data": str(path_a),
                    "valid_ratio": 0.4,
                    "num_valid": 1,
                },
                "head_b": {
                    "path_to_train_data": str(path_b),
                    "valid_ratio": 0.4,
                    "num_valid": 1,
                    "exclude_loss_properties": ["hirshfeld_ratios"],
                },
            },
        },
    }
    # Should not raise: head_b explicitly opted out.
    _setup_multihead_data_loaders(config)


def test_missing_loss_property_raises_with_empty_validation_split(
    example_xyz_with_data,
):
    """An empty validation split (e.g. too few structures for the
    default valid_ratio) must not cause every property to look
    spuriously 'missing' -- the train-set check should still fire
    correctly for a genuinely absent property."""
    from so3krates_torch.tools.data_setup import (
        _setup_singlehead_data_loaders,
    )

    config = _singlehead_config(
        example_xyz_with_data,
        hirshfeld_weight=1.0,
        valid_ratio=0.0,
    )
    with pytest.raises(ValueError, match="hirshfeld_ratios"):
        _setup_singlehead_data_loaders(config)


def test_present_loss_property_does_not_raise(example_xyz_with_data):
    """Sanity check: the new validation must not fire for properties
    that are actually present (e.g. energy/forces, always required)."""
    from so3krates_torch.tools.data_setup import (
        _setup_singlehead_data_loaders,
    )

    config = _singlehead_config(example_xyz_with_data)
    # Should not raise.
    _setup_singlehead_data_loaders(config)


def test_present_property_with_explicit_zero_weight_does_not_raise(
    tmp_path,
):
    """A deliberate per-structure `config_<name>_weight=0` override on
    data that IS present must not be mistaken for the property being
    absent (false-positive regression test, end-to-end)."""
    import numpy as np
    from ase.build import molecule
    import ase.io
    from so3krates_torch.tools.data_setup import (
        _setup_singlehead_data_loaders,
    )

    atoms_list = []
    for mol_name in ["H2O", "NH3", "CH4"]:
        atoms = molecule(mol_name)
        atoms.info["REF_energy"] = -10.0 * len(atoms)
        atoms.arrays["REF_forces"] = np.random.randn(len(atoms), 3) * 0.1
        atoms.arrays["REF_hirsh_ratios"] = np.ones(len(atoms))
        # Data is present, but every structure explicitly downweights
        # this loss term to zero.
        atoms.info["config_hirshfeld_ratios_weight"] = 0.0
        atoms_list.append(atoms)
    train_path = tmp_path / "zero_weight_override.xyz"
    ase.io.write(train_path, atoms_list)

    config = _singlehead_config(train_path, hirshfeld_weight=1.0)
    # Should not raise: the data is present, only its weight is zero.
    _setup_singlehead_data_loaders(config)


def test_missing_loss_property_raises_for_lazy_raw_hdf5_validation(
    tmp_path, example_raw_hdf5
):
    """The lazy_loading validation-data path must fail fast when a
    required property is missing from the validation file, and must
    do so without eagerly loading the whole file into ase.Atoms
    (regression test for the lazy-loading validation-loader fix)."""
    from unittest import mock

    import numpy as np
    from ase.build import molecule
    import ase.io
    from so3krates_torch.tools.data_setup import (
        _setup_singlehead_data_loaders,
    )

    # Training data DOES have hirshfeld_ratios, so only the
    # validation path (example_raw_hdf5, which lacks it) is what
    # triggers the error here.
    atoms_list = []
    for mol_name in ["H2O", "NH3", "CH4"]:
        atoms = molecule(mol_name)
        atoms.info["REF_energy"] = -10.0 * len(atoms)
        atoms.arrays["REF_forces"] = np.random.randn(len(atoms), 3) * 0.1
        atoms.arrays["REF_hirsh_ratios"] = np.ones(len(atoms))
        atoms_list.append(atoms)
    train_path = tmp_path / "train_with_hirshfeld.xyz"
    ase.io.write(train_path, atoms_list)

    config = _singlehead_config(
        train_path,
        hirshfeld_weight=1.0,
        path_to_val_data=example_raw_hdf5,
        lazy_loading=True,
    )

    with mock.patch(
        "so3krates_torch.tools.data_setup.load_atoms_from_hdf5"
    ) as mock_load:
        with pytest.raises(ValueError, match="hirshfeld_ratios"):
            _setup_singlehead_data_loaders(config)
    mock_load.assert_not_called()


def test_replay_dataset_missing_property_raises_raw(tmp_path):
    """Fine-tuning with a raw/XYZ replay dataset missing a required
    property must fail fast, the same as the main train/val data."""
    import numpy as np
    from ase.build import molecule
    import ase.io
    from so3krates_torch.tools.data_setup import (
        _setup_singlehead_data_loaders,
    )

    def make_xyz(path, with_hirshfeld):
        atoms_list = []
        for mol_name in ["H2O", "NH3", "CH4"]:
            atoms = molecule(mol_name)
            atoms.info["REF_energy"] = -10.0 * len(atoms)
            atoms.arrays["REF_forces"] = np.random.randn(len(atoms), 3) * 0.1
            if with_hirshfeld:
                atoms.arrays["REF_hirsh_ratios"] = np.ones(len(atoms))
            atoms_list.append(atoms)
        ase.io.write(path, atoms_list)

    ft_path = tmp_path / "finetune.xyz"
    replay_path = tmp_path / "replay.xyz"
    make_xyz(ft_path, with_hirshfeld=True)
    make_xyz(replay_path, with_hirshfeld=False)

    config = _singlehead_config(
        ft_path,
        hirshfeld_weight=1.0,
        replay_datasets=[str(replay_path)],
        replay_fractions=[1.0],
        replay_total=2,
    )
    with pytest.raises(ValueError, match="hirshfeld_ratios"):
        _setup_singlehead_data_loaders(config)


def test_replay_dataset_missing_property_raises_preprocessed(
    tmp_path, example_preprocessed_hdf5_full_keyspec
):
    """Fine-tuning with a preprocessed-HDF5 replay dataset missing a
    required property must fail fast, the same as the main data."""
    import numpy as np
    from ase.build import molecule
    import ase.io
    from so3krates_torch.tools.data_setup import (
        _setup_singlehead_data_loaders,
    )

    atoms_list = []
    for mol_name in ["H2O", "NH3", "CH4"]:
        atoms = molecule(mol_name)
        atoms.info["REF_energy"] = -10.0 * len(atoms)
        atoms.arrays["REF_forces"] = np.random.randn(len(atoms), 3) * 0.1
        atoms.arrays["REF_hirsh_ratios"] = np.ones(len(atoms))
        atoms_list.append(atoms)
    ft_path = tmp_path / "finetune_with_hirshfeld.xyz"
    ase.io.write(ft_path, atoms_list)

    config = _singlehead_config(
        ft_path,
        hirshfeld_weight=1.0,
        replay_datasets=[str(example_preprocessed_hdf5_full_keyspec)],
        replay_fractions=[1.0],
        replay_total=2,
    )
    with pytest.raises(ValueError, match="hirshfeld_ratios"):
        _setup_singlehead_data_loaders(config)


def test_replay_dataset_missing_property_raises_on_resample(tmp_path):
    """A replay-resample config (replay_resample_per_epoch=True) must
    be protected too, not just the plain replay path — the initial
    setup call already validates (since every _load_replay_data call
    checks required_properties), and the per-epoch builder closure
    must carry the same required_properties through as well so a
    later resample can't silently reintroduce the bug."""
    import numpy as np
    from ase.build import molecule
    import ase.io
    from so3krates_torch.tools.data_setup import (
        _setup_singlehead_data_loaders,
    )

    def make_xyz(path, with_hirshfeld):
        atoms_list = []
        for mol_name in ["H2O", "NH3", "CH4"]:
            atoms = molecule(mol_name)
            atoms.info["REF_energy"] = -10.0 * len(atoms)
            atoms.arrays["REF_forces"] = np.random.randn(len(atoms), 3) * 0.1
            if with_hirshfeld:
                atoms.arrays["REF_hirsh_ratios"] = np.ones(len(atoms))
            atoms_list.append(atoms)
        ase.io.write(path, atoms_list)

    ft_path = tmp_path / "finetune.xyz"
    replay_path = tmp_path / "replay.xyz"
    make_xyz(ft_path, with_hirshfeld=True)
    make_xyz(replay_path, with_hirshfeld=False)

    config = _singlehead_config(
        ft_path,
        hirshfeld_weight=1.0,
        replay_datasets=[str(replay_path)],
        replay_fractions=[1.0],
        replay_total=2,
        replay_resample_per_epoch=True,
    )
    with pytest.raises(ValueError, match="hirshfeld_ratios"):
        _setup_singlehead_data_loaders(config)


def test_select_valid_subset_split_ratio():
    from so3krates_torch.tools.data_setup import select_valid_subset

    data = list(range(100))
    train, val = select_valid_subset(data, valid_ratio=0.2)
    assert len(train) == 80
    assert len(val) == 20
    assert set(train) | set(val) == set(range(100))
    assert set(train) & set(val) == set()


def test_select_valid_subset_num_train_limit():
    from so3krates_torch.tools.data_setup import select_valid_subset

    data = list(range(100))
    train, val = select_valid_subset(data, valid_ratio=0.1, num_train=30)
    assert len(train) == 30
    assert len(val) == 10


def test_select_valid_subset_num_train_with_val_path(tmp_path, monkeypatch):
    """num_train must be applied when path_to_val_data is set (raw path)."""
    import random as _random
    from so3krates_torch.tools import data_setup as rt

    # Patch heavy helpers so _load_training_dataset runs the raw branch
    data = list(range(100))
    monkeypatch.setattr(rt, "detect_file_format", lambda p: "xyz")
    monkeypatch.setattr(rt, "read", lambda path, index: data)
    import types

    monkeypatch.setattr(
        rt,
        "create_configs_from_list",
        lambda **kw: [
            types.SimpleNamespace(weight=1.0) for _ in kw["atoms_list"]
        ],
    )
    monkeypatch.setattr(
        rt, "create_data_from_configs", lambda cfgs, **kw: cfgs
    )

    val_path = str(tmp_path / "val.xyz")
    config = {
        "TRAINING": {
            "path_to_train_data": str(tmp_path / "train.xyz"),
            "path_to_val_data": val_path,
            "num_train": 20,
        }
    }
    train_data, _, _, _, val_split = rt._load_training_dataset(
        config, r_max=5.0, r_max_lr=None, keyspec=None
    )
    assert len(train_data) == 20
    assert val_split is None


def test_preprocessed_lazy_split_applies_num_train_num_valid():
    """num_train / num_valid are honoured in the Preprocessed/Lazy split."""
    from so3krates_torch.tools.data_setup import select_valid_subset

    # Mirror the split logic: n_train = total - n_valid, then capped
    data = list(range(200))
    train, val = select_valid_subset(
        data, valid_ratio=0.1, num_train=50, num_valid=5
    )
    assert len(train) == 50
    assert len(val) == 5


def test_determine_num_elements(example_xyz_with_data):
    from so3krates_torch.tools.model_setup import determine_num_elements
    from so3krates_torch.tools.utils import create_dataloader_from_list
    from ase.io import read

    atoms_list = read(example_xyz_with_data, index=":")
    loader = create_dataloader_from_list(
        atoms_list, batch_size=8, r_max=5.0, r_max_lr=None, shuffle=False
    )
    n = determine_num_elements(loader)
    # H2O -> H,O; NH3 -> N,H; CH4 -> C,H  ->  {H, O, N, C} = 4 elements
    assert n == 4


# ── End-to-end: dispersion ratio labels (a0 + C6), mixed dataset ─


def _ratio_label_config(tmp_path):
    import ase.io
    import numpy as np
    import yaml
    from ase.build import molecule

    rng = np.random.default_rng(0)
    atoms_list = []
    for i, name in enumerate(["H2O", "NH3", "CH4", "H2O", "NH3", "CH4"]):
        atoms = molecule(name)
        atoms.rattle(0.05, seed=i)
        atoms.info["REF_energy"] = -10.0 * len(atoms)
        atoms.arrays["REF_forces"] = rng.normal(size=(len(atoms), 3)) * 0.1
        if i % 2 == 0:  # half the structures carry dispersion labels
            atoms.arrays["REF_hirsh_ratios"] = rng.uniform(
                0.6, 1.0, len(atoms)
            )
            atoms.arrays["REF_c6_ratios"] = rng.uniform(0.4, 1.0, len(atoms))
        atoms_list.append(atoms)
    data_path = tmp_path / "train.xyz"
    ase.io.write(data_path, atoms_list)

    example = (
        Path(__file__).resolve().parents[3]
        / "examples"
        / "training"
        / "train_settings_example.yaml"
    )
    raw = yaml.safe_load(example.read_text())
    raw["GENERAL"].update(
        checkpoints_dir=str(tmp_path / "ckpt"),
        model_dir=str(tmp_path / "model"),
        log_dir=str(tmp_path / "logs"),
        name_exp="ratio_labels",
    )
    raw["ARCHITECTURE"].update(
        degrees=[1],
        num_features=8,
        num_heads=2,
        num_layers=1,
        num_radial_basis_fn=4,
        energy_regression_dim=8,
        r_max=4.0,
        r_max_lr=6.0,
        electrostatic_energy_bool=False,
        dispersion_energy_bool=True,
        c6_ratios_bool=True,
        use_simple_hirshfeld=True,
        zbl_repulsion_bool=False,
    )
    raw["TRAINING"].update(
        path_to_train_data=str(data_path),
        # validate on all six so labelled structures are always present
        path_to_val_data=str(data_path),
        batch_size=2,
        valid_batch_size=2,
        num_epochs=1,
        energy_weight=0.01,
        forces_weight=1.0,
        a0_ratios_weight=0.1,
        c6_ratios_weight=0.1,
    )
    raw["MISC"].update(device="cpu", error_table="PerAtomMAE")
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.dump(raw))
    return config_path


def test_train_with_c6_and_a0_labels_mixed_dataset(tmp_path):
    from so3krates_torch.cli.run_train import setup_config_from_yaml
    from so3krates_torch.modules.loss import WeightedPropertiesLoss
    from so3krates_torch.tools.data_setup import setup_data_loaders
    from so3krates_torch.tools.model_setup import create_model
    from so3krates_torch.tools.training_setup import setup_loss_function

    torch.set_default_dtype(torch.float64)
    try:
        config = setup_config_from_yaml(str(_ratio_label_config(tmp_path)))
        assert config["TRAINING"]["hirshfeld_weight"] == 0.1
        model = create_model(config, torch.device("cpu"))
        train_loader, *_ = setup_data_loaders(config)
        loss_fn = setup_loss_function(config)
        assert isinstance(loss_fn, WeightedPropertiesLoss)
        assert "c6_ratios" in loss_fn.properties

        saw_missing = saw_labelled = False
        for batch in train_loader:
            saw_missing |= bool(torch.isnan(batch.c6_ratios).any())
            saw_labelled |= bool(torch.isfinite(batch.c6_ratios).any())
            model.zero_grad()
            out = model(batch.to_dict(), training=True, compute_force=True)
            loss = loss_fn(ref=batch, pred=out, ddp=False)
            assert torch.isfinite(loss)
            loss.backward()
        assert saw_missing and saw_labelled

        c6_grads = [
            p.grad
            for p in model.c6_ratios_output_block.parameters()
            if p.grad is not None
        ]
        assert c6_grads and any(bool((g != 0).any()) for g in c6_grads)
        assert all(bool(torch.isfinite(g).all()) for g in c6_grads)
    finally:
        torch.set_default_dtype(torch.float32)


def test_run_training_logs_c6_metrics(tmp_path, monkeypatch):
    from so3krates_torch.cli.run_train import (
        run_training,
        setup_config_from_yaml,
    )

    config = setup_config_from_yaml(str(_ratio_label_config(tmp_path)))
    # run_training saves '<name_exp>.model' relative to the cwd
    monkeypatch.chdir(tmp_path)
    try:
        run_training(config)
    finally:
        torch.set_default_dtype(torch.float32)
    log_text = "".join(
        p.read_text() for p in (tmp_path / "logs").rglob("*.log")
    )
    assert "MAE_c6_ratios=" in log_text
    assert "loss=nan" not in log_text.lower()
    assert (tmp_path / "ratio_labels.model").exists()


# ── Adaptive robust loss: optimizer, DDP, logging ──────────────


def _adaptive_loss():
    from so3krates_torch.modules.loss import WeightedPropertiesLoss

    return WeightedPropertiesLoss(
        {"energy": 1.0, "forces": 1.0}, robust="adaptive"
    )


def test_optimizer_includes_loss_params():
    from so3krates_torch.tools.training_setup import (
        setup_optimizer_and_scheduler,
    )

    model = torch.nn.Linear(3, 2)
    loss_fn = _adaptive_loss()
    for opt in ("adam", "adamw"):
        config = {"TRAINING": {"lr": 1e-3, "optimizer": opt}}
        optimizer, _ = setup_optimizer_and_scheduler(model, config, loss_fn)
        ids = {id(p) for g in optimizer.param_groups for p in g["params"]}
        assert all(id(p) in ids for p in loss_fn.parameters())
        assert all(id(p) in ids for p in model.parameters())
    # no loss params -> unchanged single group
    optimizer, _ = setup_optimizer_and_scheduler(
        model, {"TRAINING": {"lr": 1e-3}}, None
    )
    assert len(optimizer.param_groups) == 1


def test_loss_param_grads_averaged_across_ranks(monkeypatch):
    import torch.distributed as dist
    from so3krates_torch.tools.train import allreduce_loss_param_grads_

    loss_fn = _adaptive_loss()
    for p in loss_fn.parameters():
        p.grad = torch.ones_like(p)

    def all_reduce(t, op=None):
        t += 3.0  # other rank's gradient

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    allreduce_loss_param_grads_(loss_fn)
    for p in loss_fn.parameters():
        assert torch.allclose(p.grad, torch.full_like(p, 2.0))


def test_valid_err_log_reports_robust_params(caplog):
    import logging
    from so3krates_torch.modules.loss import robust_param_metrics
    from so3krates_torch.tools.train import valid_err_log

    class _Logger:
        def log(self, _):
            pass

    metrics = {"mae_e_per_atom": 0.001, "mae_f": 0.01}
    metrics.update(robust_param_metrics(_adaptive_loss()))
    assert metrics["robust_alpha_forces"] == pytest.approx(1.0, abs=1e-6)
    with caplog.at_level(logging.INFO):
        valid_err_log(0.1, metrics, _Logger(), "PerAtomMAE", epoch=1)
    assert "robust: energy alpha=1.000 scale=1.000" in caplog.text


def test_run_training_adaptive_per_graph_smoke(tmp_path, monkeypatch):
    import yaml
    from so3krates_torch.cli.run_train import (
        run_training,
        setup_config_from_yaml,
    )

    config_path = _ratio_label_config(tmp_path)
    raw = yaml.safe_load(config_path.read_text())
    raw["TRAINING"].update(
        loss_reduction="per_graph",
        use_adaptive_robust_loss=True,
        num_epochs=2,
    )
    config_path.write_text(yaml.dump(raw))
    config = setup_config_from_yaml(str(config_path))
    monkeypatch.chdir(tmp_path)
    try:
        run_training(config)
    finally:
        torch.set_default_dtype(torch.float32)
    log_text = "".join(
        p.read_text() for p in (tmp_path / "logs").rglob("*.log")
    )
    assert "robust: " in log_text
    for prop in ("energy", "forces", "hirshfeld_ratios", "c6_ratios"):
        assert f"{prop} alpha=" in log_text
    assert "loss=nan" not in log_text.lower()
    ckpts = list((tmp_path / "ckpt").glob("*.pt"))
    assert ckpts and "loss_fn" in torch.load(ckpts[0], weights_only=False)


def test_clip_gradients_includes_loss_params():
    """JAX clips the global norm over the full params tree, including
    the adaptive robust-loss parameters."""
    from so3krates_torch.tools.train import clip_gradients_

    model = torch.nn.Linear(1, 1, bias=False)
    model.weight.grad = torch.ones_like(model.weight)
    loss_fn = _adaptive_loss()
    for p in loss_fn.parameters():
        p.grad = torch.full_like(p, 100.0)
    norm = clip_gradients_(model, loss_fn, max_norm=1.0)
    assert norm.item() == pytest.approx((1 + 4 * 100.0**2) ** 0.5)
    assert model.weight.grad.item() < 0.01  # scaled by the global norm
    total = torch.sqrt(
        model.weight.grad.square().sum()
        + sum(p.grad.square().sum() for p in loss_fn.parameters())
    )
    assert total.item() == pytest.approx(1.0, rel=1e-4)


# ── setup_loss_functions: validation robust + two-stage ────────


def _loss_cfg(**training):
    return {
        "TRAINING": {"energy_weight": 0.001, "forces_weight": 1.0, **training}
    }


def test_setup_loss_functions_defaults_share_train_loss():
    from so3krates_torch.tools.training_setup import setup_loss_functions

    fns = setup_loss_functions(_loss_cfg())
    assert fns.valid is fns.train
    assert fns.stage_two is None


def test_validation_robust_overrides():
    from so3krates_torch.tools.training_setup import setup_loss_functions

    fns = setup_loss_functions(
        _loss_cfg(use_robust_loss=True, use_robust_loss_validation=False)
    )
    assert fns.train.robust == "fixed" and fns.valid.robust == "none"
    fns = setup_loss_functions(
        _loss_cfg(
            use_robust_loss_validation=True, robust_loss_alpha_validation=0.5
        )
    )
    assert fns.train.robust == "none"
    assert fns.valid.robust == "fixed" and fns.valid.robust_alpha == 0.5


def test_adaptive_validation_uses_training_loss():
    from so3krates_torch.tools.training_setup import setup_loss_functions

    fns = setup_loss_functions(
        _loss_cfg(
            use_adaptive_robust_loss=True, use_robust_loss_validation=False
        )
    )
    assert fns.valid is fns.train


def test_stage_two_weights_inherit_and_override():
    from so3krates_torch.tools.training_setup import setup_loss_functions

    fns = setup_loss_functions(
        _loss_cfg(
            stage_two_start_epoch=4,
            stage_two_loss_weights={"energy_weight": 0.01},
        )
    )
    s2 = fns.stage_two
    assert s2.start_epoch == 4
    assert s2.loss_fn.energy_weight.item() == pytest.approx(0.01)
    assert s2.loss_fn.forces_weight.item() == pytest.approx(1.0)
    assert fns.train.energy_weight.item() == pytest.approx(0.001)
    assert s2.valid_loss_fn is s2.loss_fn


def test_stage_two_shares_adaptive_params():
    from so3krates_torch.tools.training_setup import setup_loss_functions

    fns = setup_loss_functions(
        _loss_cfg(
            use_adaptive_robust_loss=True,
            stage_two_start_epoch=2,
            stage_two_loss_weights={"dipole_weight": 0.1},
        )
    )
    assert fns.stage_two.loss_fn.adaptive is fns.train.adaptive
    assert set(fns.train.adaptive.alpha_raw.keys()) == {
        "energy",
        "forces",
        "dipole",
    }
    names = {n for n, _ in fns.train.named_parameters()}
    assert names == {n for n, _ in fns.stage_two.loss_fn.named_parameters()}


def test_legacy_loss_type_rejects_stage_two():
    from so3krates_torch.tools.training_setup import setup_loss_functions

    with pytest.raises(ValueError, match="stage_two"):
        setup_loss_functions(
            _loss_cfg(
                loss_type="energy_forces",
                stage_two_start_epoch=2,
                stage_two_loss_weights={"energy_weight": 0.01},
            )
        )


def test_required_loss_properties_include_stage_two():
    from so3krates_torch.tools.training_setup import (
        get_required_loss_properties,
    )

    required = get_required_loss_properties(
        _loss_cfg(
            stage_two_start_epoch=2,
            stage_two_loss_weights={"c6_ratios_weight": 0.1},
        )
    )
    assert set(required) == {"energy", "forces", "c6_ratios"}


def test_reset_plateau_scheduler_keeps_lr():
    from so3krates_torch.tools.train import reset_plateau_scheduler_

    model = torch.nn.Linear(1, 1)
    opt = torch.optim.Adam(model.parameters(), lr=1.0)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, patience=0, factor=0.5
    )
    sched.step(1.0)
    sched.step(2.0)  # no improvement -> lr halved
    assert opt.param_groups[0]["lr"] == 0.5
    sched.step(3.0)
    reset_plateau_scheduler_(sched)
    assert sched.best == sched.mode_worse
    assert sched.num_bad_epochs == 0 and sched.cooldown_counter == 0
    assert opt.param_groups[0]["lr"] == 0.25  # reduction kept
    reset_plateau_scheduler_(
        torch.optim.lr_scheduler.ExponentialLR(opt, 0.9)
    )  # no-op for other schedulers


def test_run_training_two_stage_switch(tmp_path, monkeypatch):
    import yaml
    from so3krates_torch.cli.run_train import (
        run_training,
        setup_config_from_yaml,
    )

    config_path = _ratio_label_config(tmp_path)
    raw = yaml.safe_load(config_path.read_text())
    raw["TRAINING"].update(
        num_epochs=3,
        stage_two_start_epoch=1,
        stage_two_loss_weights={"energy_weight": 1.0},
    )
    config_path.write_text(yaml.dump(raw))
    config = setup_config_from_yaml(str(config_path))
    monkeypatch.chdir(tmp_path)
    try:
        run_training(config)
    finally:
        torch.set_default_dtype(torch.float32)
    log_text = (tmp_path / "logs" / "ratio_labels.log").read_text()
    debug_text = (tmp_path / "logs" / "ratio_labels_debug.log").read_text()
    assert log_text.count("switching to the stage-two loss at epoch 1") == 1
    assert "loss=nan" not in log_text.lower()
    # best-loss tracker reset: the first stage-two epoch is checkpointed
    # even though its (re-weighted) loss is higher than stage one's
    assert "ratio_labels_epoch-1.pt" in debug_text


# ── Two-stage: restarts, stale plateau losses, startup output check ─


def _two_stage_run(tmp_path, monkeypatch, **training):
    import yaml
    from so3krates_torch.cli.run_train import (
        run_training,
        setup_config_from_yaml,
    )

    config_path = _ratio_label_config(tmp_path)
    raw = yaml.safe_load(config_path.read_text())
    raw["TRAINING"].pop("stage_two_start_epoch", None)
    raw["TRAINING"].pop("stage_two_loss_weights", None)
    raw["TRAINING"].update(scheduler="reduce_on_plateau", **training)
    raw["MISC"]["keep_checkpoints"] = True  # checkpoint every epoch
    config_path.write_text(yaml.dump(raw))
    monkeypatch.chdir(tmp_path)
    try:
        run_training(setup_config_from_yaml(str(config_path)))
    finally:
        torch.set_default_dtype(torch.float32)
    return (tmp_path / "logs" / "ratio_labels.log").read_text()


def test_resume_from_stage_one_checkpoint_runs_full_switch(
    tmp_path, monkeypatch
):
    _two_stage_run(tmp_path, monkeypatch, num_epochs=2)
    log = _two_stage_run(
        tmp_path,
        monkeypatch,
        num_epochs=3,
        stage_two_start_epoch=2,
        stage_two_loss_weights={"energy_weight": 50.0},
    )
    # checkpoint was taken in stage one -> full switch incl. tracker reset
    assert "switching to the stage-two loss at epoch 2" in log
    assert "resuming in stage two" not in log


def test_resume_stage_two_checkpoint_without_stage_two_raises(
    tmp_path, monkeypatch
):
    _two_stage_run(
        tmp_path,
        monkeypatch,
        num_epochs=2,
        stage_two_start_epoch=1,
        stage_two_loss_weights={"energy_weight": 1.0},
    )
    with pytest.raises(ValueError, match="stage two"):
        _two_stage_run(tmp_path, monkeypatch, num_epochs=3)


def test_checkpoint_stores_stage_two_flag(tmp_path):
    from so3krates_torch.tools.checkpoint import (
        CheckpointBuilder,
        CheckpointState,
    )

    model = torch.nn.Linear(2, 1)
    opt = torch.optim.Adam(model.parameters())
    sched = torch.optim.lr_scheduler.ExponentialLR(opt, 0.9)
    ckpt = CheckpointBuilder.create_checkpoint(
        CheckpointState(model, opt, sched, stage_two_active=True)
    )
    assert ckpt["stage_two_active"] is True
    state = CheckpointState(model, opt, sched)
    CheckpointBuilder.load_checkpoint(state, ckpt, strict=True)
    assert state.stage_two_active is True


def test_stale_stage_one_loss_skips_plateau_step(tmp_path, monkeypatch):
    log = _two_stage_run(
        tmp_path,
        monkeypatch,
        num_epochs=4,
        eval_interval=2,
        stage_two_start_epoch=1,
        stage_two_loss_weights={"energy_weight": 50.0},
    )
    assert "skipping plateau scheduler step" in log


def test_startup_check_catches_stage_two_only_output(
    so3lr_model_config, make_batch, h2o_atoms
):
    from so3krates_torch.modules.models import SO3LR
    from so3krates_torch.tools.training_setup import (
        check_loss_outputs,
        setup_loss_functions,
    )

    config = {**so3lr_model_config, "electrostatic_energy_bool": False}
    model = SO3LR(**config).to(torch.float64)
    batch = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
    fns = setup_loss_functions(
        _loss_cfg(
            stage_two_start_epoch=5,
            stage_two_loss_weights={"dipole_weight": 0.1},
        )
    )
    with pytest.raises(KeyError, match="dipole"):
        check_loss_outputs(model, fns, batch)
    ok = setup_loss_functions(_loss_cfg())
    check_loss_outputs(model, ok, batch)  # no error


# ── Loss config warnings ───────────────────────────────────────

import logging  # noqa: E402

from so3krates_torch.tools.training_setup import (  # noqa: E402
    setup_loss_functions as _setup_loss_functions,
)


def test_adaptive_ignores_validation_settings_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _setup_loss_functions(
            _loss_cfg(
                use_adaptive_robust_loss=True,
                use_robust_loss_validation=False,
                robust_loss_alpha_validation=0.5,
            )
        )
    assert caplog.text.count("ignored with use_adaptive_robust_loss") == 1


def test_alpha_validation_without_robust_validation_warns(caplog):
    with caplog.at_level(logging.WARNING):
        _setup_loss_functions(_loss_cfg(robust_loss_alpha_validation=0.5))
    assert "robust_loss_alpha_validation has no effect" in caplog.text


def test_legacy_loss_type_names_validation_key():
    with pytest.raises(ValueError, match="use_robust_loss_validation"):
        _setup_loss_functions(
            _loss_cfg(
                loss_type="energy_forces", use_robust_loss_validation=True
            )
        )


def test_no_warning_by_default(caplog):
    with caplog.at_level(logging.WARNING):
        _setup_loss_functions(
            _loss_cfg(
                use_robust_loss_validation=None,
                robust_loss_alpha_validation=None,
            )
        )
    assert caplog.text == ""


def test_no_warning_when_validation_matches_training(caplog):
    with caplog.at_level(logging.WARNING):
        _setup_loss_functions(
            _loss_cfg(
                use_robust_loss=True,
                use_robust_loss_validation=True,
                robust_loss_alpha_validation=1.0,
            )
        )
    assert caplog.text == ""


def test_ignored_validation_warning_logged_once_with_stage_two(caplog):
    with caplog.at_level(logging.WARNING):
        _setup_loss_functions(
            _loss_cfg(
                use_adaptive_robust_loss=True,
                use_robust_loss_validation=True,
                stage_two_start_epoch=2,
                stage_two_loss_weights={"energy_weight": 0.01},
            )
        )
    assert caplog.text.count("ignored with use_adaptive_robust_loss") == 1


@pytest.mark.parametrize("start", [5, 7])
def test_stage_two_never_starts_warns(caplog, start):
    with caplog.at_level(logging.WARNING):
        _setup_loss_functions(
            _loss_cfg(
                num_epochs=5,
                stage_two_start_epoch=start,
                stage_two_loss_weights={"energy_weight": 0.01},
            )
        )
    assert "stage two never starts" in caplog.text
    assert f"stage_two_start_epoch={start}" in caplog.text
    assert "num_epochs=5" in caplog.text


def test_stage_two_reachable_no_warning(caplog):
    with caplog.at_level(logging.WARNING):
        _setup_loss_functions(
            _loss_cfg(
                num_epochs=5,
                stage_two_start_epoch=4,
                stage_two_loss_weights={"energy_weight": 0.01},
            )
        )
    assert caplog.text == ""


def test_stage_two_check_without_num_epochs(caplog):
    with caplog.at_level(logging.WARNING):
        _setup_loss_functions(
            _loss_cfg(
                stage_two_start_epoch=4,
                stage_two_loss_weights={"energy_weight": 0.01},
            )
        )
    assert caplog.text == ""


@pytest.mark.parametrize(
    "robust_key", ["use_robust_loss", "use_adaptive_robust_loss"]
)
def test_legacy_error_names_training_key_when_training_is_robust(robust_key):
    with pytest.raises(ValueError) as err:
        _setup_loss_functions(
            _loss_cfg(
                loss_type="energy_forces",
                use_robust_loss_validation=True,
                **{robust_key: True},
            )
        )
    assert "use_robust_loss / use_adaptive_robust_loss" in str(err.value)


def test_alpha_warning_no_suggestion_for_legacy_loss_type(caplog):
    with caplog.at_level(logging.WARNING):
        _setup_loss_functions(
            _loss_cfg(
                loss_type="energy_forces", robust_loss_alpha_validation=0.5
            )
        )
    assert "robust_loss_alpha_validation has no effect" in caplog.text
    assert "Set use_robust_loss_validation" not in caplog.text


def test_train_module_imports_without_pandas():
    # pandas belongs to the optional [vis] extra; training must not need it
    import subprocess
    import sys

    code = (
        "import sys; sys.modules['pandas'] = None; "
        "import so3krates_torch.tools.train"
    )
    result = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True
    )
    assert result.returncode == 0, result.stderr
