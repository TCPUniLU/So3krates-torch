"""Tests for training functionality."""
import json
from pathlib import Path

import pytest
import torch
from ase.io import read

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

    z_table = AtomicNumberTable(
        [int(z) for z in range(1, 119)]
    )

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
        model_config.update({
            "r_max_lr": r_max_lr,
            "zbl_repulsion_bool": True,
            "electrostatic_energy_bool": True,
            "dispersion_energy_bool": True,
            "dispersion_energy_cutoff_lr_damping": 2.0,
        })

    model = model_class(**model_config).to(device)
    model.train()

    optimizer = torch.optim.Adam(
        model.parameters(), lr=0.001
    )

    if loss_class == WeightedEnergyForcesLoss:
        loss_fn = loss_class(
            energy_weight=1.0, forces_weight=100.0
        )
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

        output = model(
            batch_dict, training=True, compute_force=True
        )

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
        p.norm().item()
        for p in model.parameters()
        if p.requires_grad
    )

    metrics = {
        "avg_loss": avg_loss,
        "total_param_norm": total_param_norm,
        "first_batch_energy_0": (
            first_output["energy"][0].item()
        ),
        "first_batch_forces_norm": (
            first_output["forces"].norm().item()
        ),
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
        pytest.skip(
            f"Generated reference for '{reference_key}'"
        )

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
