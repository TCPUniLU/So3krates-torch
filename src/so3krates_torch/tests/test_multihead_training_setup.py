"""Multi-head conversion inside the ``torchkrates-train`` setup sequence:
``_setup_model_for_training`` (load) -> ``prepare_model_for_training``
(convert, shifts, fine-tuning, dtype)."""

import copy
from pathlib import Path

import pytest
import torch

from so3krates_torch.cli.run_train import setup_config_from_yaml
from so3krates_torch.modules.models import MultiHeadSO3LR, SO3LR
from so3krates_torch.tests.test_multihead_conversion import (
    _energy_forces,
    _trained_like,
)
from so3krates_torch.tools.model_setup import (
    _setup_model_for_training,
    create_model,
    prepare_model_for_training,
)

EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "examples/training/train_settings_example.yaml"
)
NUM_ELEMENTS = 118


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    before = torch.get_default_dtype()
    yield
    torch.set_default_dtype(before)


@pytest.fixture
def config():
    """Validated example config, shrunk to a small float64 model."""
    cfg = setup_config_from_yaml(str(EXAMPLE))
    cfg["ARCHITECTURE"].update(
        num_features=16,
        num_heads=2,
        num_layers=1,
        degrees=[1, 2],
        energy_regression_dim=16,
    )
    cfg["GENERAL"]["default_dtype"] = "float64"
    return cfg


def _with(cfg, **sections):
    cfg = copy.deepcopy(cfg)
    for section, values in sections.items():
        cfg[section].update(values)
    return cfg


def _prepare(cfg, model, warm_start=True):
    return prepare_model_for_training(
        cfg,
        model,
        warm_start=warm_start,
        avg_num_neighbors=10.0,
        average_atomic_energy_shifts=torch.zeros(NUM_ELEMENTS),
        num_elements=NUM_ELEMENTS,
        device_name="cpu",
    )


def test_theory_level_model_converts_in_training_setup(
    config, so3lr_model_config, make_batch, h2o_atoms
):
    sh = _trained_like(SO3LR(**so3lr_model_config, num_theory_levels=3))
    cfg = _with(
        config,
        ARCHITECTURE=dict(
            convert_to_multihead=True,
            num_output_heads=2,
            convert_theory_level=1,
        ),
        TRAINING=dict(finetune_choice="naive"),
    )
    model = _prepare(cfg, sh)
    head = model.atomic_energy_output_block
    assert type(model) is MultiHeadSO3LR
    assert torch.equal(
        head.energy_shifts,
        sh.atomic_energy_output_block.energy_shifts[:, 1],
    )
    energy, _ = _energy_forces(
        model, make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
    )
    assert energy.shape == (2, 1)


def test_pretrained_model_converts_without_finetune_choice(
    config, so3lr_model_config
):
    sh = SO3LR(**so3lr_model_config)
    cfg = _with(
        config,
        ARCHITECTURE=dict(convert_to_multihead=True, num_output_heads=2),
    )
    assert type(_prepare(cfg, sh)) is MultiHeadSO3LR


def test_pretrained_weights_single_head_keep_scales(config, tmp_path):
    pretrained = _trained_like(create_model(config, "cpu"))
    path = tmp_path / "weights.pt"
    torch.save(pretrained.state_dict(), path)
    cfg = _with(
        config,
        ARCHITECTURE=dict(convert_to_multihead=True, num_output_heads=3),
        TRAINING=dict(pretrained_weights=str(path), finetune_choice="naive"),
    )

    loaded, warm_start = _setup_model_for_training(cfg, "cpu")
    assert warm_start and type(loaded) is SO3LR
    model = _prepare(cfg, loaded)

    src = pretrained.atomic_energy_output_block
    head = model.atomic_energy_output_block
    assert type(model) is MultiHeadSO3LR and model.num_output_heads == 3
    assert torch.equal(head.energy_scales, src.energy_scales.weight.T)
    assert torch.equal(head.energy_shifts, src.energy_shifts)


def test_pretrained_weights_multihead_loaded_as_multihead(config, tmp_path):
    cfg = _with(
        config,
        ARCHITECTURE=dict(convert_to_multihead=True, num_output_heads=3),
    )
    pretrained = create_model(cfg, "cpu")
    assert type(pretrained) is MultiHeadSO3LR
    path = tmp_path / "weights.pt"
    torch.save(pretrained.state_dict(), path)
    cfg = _with(
        cfg,
        TRAINING=dict(pretrained_weights=str(path), finetune_choice="naive"),
    )

    loaded, _ = _setup_model_for_training(cfg, "cpu")
    model = _prepare(cfg, loaded)
    assert type(model) is MultiHeadSO3LR
    assert torch.equal(
        model.atomic_energy_output_block.final_layer_weights,
        pretrained.atomic_energy_output_block.final_layer_weights,
    )


def test_from_scratch_multihead_unchanged(config):
    cfg = _with(
        config,
        ARCHITECTURE=dict(convert_to_multihead=True, num_output_heads=2),
    )
    model = _prepare(cfg, create_model(cfg, "cpu"), warm_start=False)
    assert type(model) is MultiHeadSO3LR and model.num_output_heads == 2


def test_model_eval_multihead_batches_of_different_sizes(
    so3lr_model_config, make_batch
):
    """Validation of a converted multi-head model (no per-structure head
    selection) must work when batches differ in size; energy/force
    metrics use the mean over heads, as MultiHeadSO3LRCalculator does."""
    from ase.build import molecule

    from so3krates_torch.tools.eval import ModelEval
    from so3krates_torch.tools.multihead_utils import to_multihead_model

    mh = to_multihead_model(SO3LR(**so3lr_model_config), 3, seed=0).eval()
    ev = ModelEval()
    expected_delta_f = []
    for name in ("H2O", "CH4"):  # 3 and 5 atoms
        atoms = molecule(name)
        atoms.info["REF_energy"] = -1.0
        batch = make_batch(atoms, r_max=5.0, cutoff_lr=10.0)
        batch.forces = torch.zeros_like(batch.positions)
        out = mh(batch.to_dict(), training=False)
        assert out["forces"].dim() == 3  # (heads, atoms, 3)
        expected_delta_f.append(-out["forces"].mean(0).detach())
        ev.update(batch, out)
    aux = ev.compute()  # no loss_fn: only the metrics
    expected = torch.cat(expected_delta_f).abs().mean().item()
    assert aux["mae_f"] == pytest.approx(expected, rel=1e-10)
