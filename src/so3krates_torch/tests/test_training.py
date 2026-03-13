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


def test_setup_optimizer_and_scheduler_adam(default_model_config):
    from so3krates_torch.tools.training_setup import setup_optimizer_and_scheduler
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
    from so3krates_torch.tools.training_setup import setup_optimizer_and_scheduler
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
    from so3krates_torch.tools.training_setup import setup_optimizer_and_scheduler
    from so3krates_torch.modules.models import So3krates

    model = So3krates(**default_model_config)
    config = {"TRAINING": {"lr": 1e-3, "optimizer": "sgd"}}
    with pytest.raises(ValueError, match="Unsupported optimizer"):
        setup_optimizer_and_scheduler(model, config)


def test_process_config_atomic_energies_int_keys():
    from so3krates_torch.tools.model_setup import process_config_atomic_energies

    shifts = process_config_atomic_energies({1: -13.6, 6: -1027.0, 8: -2042.0})
    assert shifts[1] == -13.6
    assert shifts[6] == -1027.0
    assert shifts[8] == -2042.0
    # Missing elements default to 0.0
    assert shifts[7] == 0.0
    # Full range 1-118
    assert len(shifts) == 118


def test_process_config_atomic_energies_str_keys():
    from so3krates_torch.tools.model_setup import process_config_atomic_energies

    shifts = process_config_atomic_energies({"1": -13.6, "6": -1027.0})
    assert shifts[1] == -13.6
    assert shifts[6] == -1027.0


def test_set_avg_num_neighbors_in_model(default_model_config):
    from so3krates_torch.tools.model_setup import set_avg_num_neighbors_in_model
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
    from so3krates_torch.modules.loss import WeightedEnergyForcesLoss

    config = {
        "TRAINING": {
            "energy_weight": 2.0,
            "forces_weight": 500.0,
        }
    }
    loss = setup_loss_function(config)
    assert isinstance(loss, WeightedEnergyForcesLoss)
    assert abs(loss.energy_weight.item() - 2.0) < 1e-9
    assert abs(loss.forces_weight.item() - 500.0) < 1e-9


def test_setup_loss_function_invalid_type_raises():
    from so3krates_torch.tools.training_setup import setup_loss_function

    config = {"TRAINING": {"loss_type": "nonexistent"}}
    with pytest.raises(ValueError, match="Unknown loss_type"):
        setup_loss_function(config)


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
    monkeypatch.setattr(
        rt, "create_configs_from_list", lambda **kw: kw["atoms_list"]
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
