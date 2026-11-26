"""Tests for long-range anchoring (``modules/anchoring.py``): a frozen copy
of the pretrained model labels every batch with its partial charges,
a0/C6 ratios and dipole, which the fine-tuned model is pulled towards."""

import logging
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml
from ase.build import molecule

from so3krates_torch.config import TrainConfig
from so3krates_torch.data.atomic_data import AtomicData
from so3krates_torch.data.utils import (
    KeySpecification,
    config_from_atoms,
    update_keyspec_from_kwargs,
)
from so3krates_torch.modules.anchoring import (
    AnchoredLoss,
    make_anchor_teacher,
    normalize_anchor_weights,
)
from so3krates_torch.modules.loss import (
    AdaptiveRobustParams,
    WeightedPropertiesLoss,
)
from so3krates_torch.modules.models import SO3LR
from so3krates_torch.tools import torch_geometric
from so3krates_torch.tools.default_keys import DefaultKeys
from so3krates_torch.tools.utils import AtomicNumberTable

EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "examples/training/train_settings_example.yaml"
)


ALL_ANCHORS = {
    "charges": 1.0,
    "hirshfeld_ratios": 1.0,
    "c6_ratios": 1.0,
    "dipole": 1.0,
}


@pytest.fixture(autouse=True)
def _float64():
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(before)


def _batch(atoms):
    """Batch built with the training key specification (DefaultKeys), so
    absent labels get per-structure weight 0 as in torchkrates-train."""
    keyspec = update_keyspec_from_kwargs(
        KeySpecification(), DefaultKeys.keydict()
    )
    config = config_from_atoms(atoms, key_specification=keyspec)
    data = AtomicData.from_config(
        config,
        z_table=AtomicNumberTable(list(range(1, 119))),
        cutoff=5.0,
        cutoff_lr=10.0,
    )
    loader = torch_geometric.dataloader.DataLoader([data], batch_size=1)
    return next(iter(loader))


@pytest.fixture
def lr_model(so3lr_model_config):
    """SO3LR predicting charges, dipole, a0 and C6 ratios."""
    return SO3LR(
        **so3lr_model_config, c6_ratios_bool=True, output_partial_charges=True
    )


@pytest.fixture
def batch():
    """Energies/forces-free water: no long-range labels at all."""
    return _batch(molecule("H2O"))


@pytest.fixture
def labelled_batch():
    atoms = molecule("H2O")
    atoms.arrays["REF_charges"] = np.array([-0.8, 0.4, 0.4])
    atoms.arrays["REF_hirsh_ratios"] = np.array([0.9, 0.7, 0.7])
    atoms.info["dipole"] = np.array([0.0, 0.0, 0.5])
    return _batch(atoms)


def _pred(model, batch):
    return model(batch.to_dict(), training=True)


def _drift(head):
    with torch.no_grad():
        for p in head.parameters():
            p.add_(0.1)


def _raw_config(**training):
    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["TRAINING"].update(training)
    return raw


# ---------------------------------------------------------------------------
# Config keys
# ---------------------------------------------------------------------------


def test_normalize_maps_jax_and_torch_names():
    weights = normalize_anchor_weights(
        {
            "partial_charges": 1.0,
            "a0_ratios": 0.5,
            "c6_ratios": 2.0,
            "dipole_vec": 0.1,
        }
    )
    assert weights == {
        "charges": 1.0,
        "hirshfeld_ratios": 0.5,
        "c6_ratios": 2.0,
        "dipole": 0.1,
    }
    assert list(weights) == [
        "charges",
        "hirshfeld_ratios",
        "c6_ratios",
        "dipole",
    ]
    assert normalize_anchor_weights({"charges": 1.0, "dipole": 1.0}) == {
        "charges": 1.0,
        "dipole": 1.0,
    }
    assert normalize_anchor_weights({"hirshfeld_ratios": 1.0}) == {
        "hirshfeld_ratios": 1.0
    }


def test_normalize_drops_zero_and_none():
    assert normalize_anchor_weights(None) == {}
    assert normalize_anchor_weights({}) == {}
    assert normalize_anchor_weights({"partial_charges": 0.0}) == {}


@pytest.mark.parametrize(
    "raw, match",
    [
        ({"energy": 1.0}, "nknown"),
        ({"a0_ratios": 1.0, "hirshfeld_ratios": 1.0}, "same quantity"),
        ({"partial_charges": 1.0, "charges": 1.0}, "same quantity"),
        ({"c6_ratios": -1.0}, "negative"),
    ],
)
def test_normalize_rejects(raw, match):
    with pytest.raises(ValueError, match=match):
        normalize_anchor_weights(raw)


def test_config_requires_pretrained_model():
    raw = _raw_config(long_range_anchor_weights={"partial_charges": 1.0})
    with pytest.raises(ValueError, match="pretrained"):
        TrainConfig.model_validate(raw)


def test_config_rejects_legacy_loss_type():
    raw = _raw_config(
        long_range_anchor_weights={"partial_charges": 1.0},
        pretrained_model="x.model",
        loss_type="energy_forces",
    )
    with pytest.raises(ValueError, match="loss_type"):
        TrainConfig.model_validate(raw)


def test_config_rejects_unknown_anchor_key():
    raw = _raw_config(
        long_range_anchor_weights={"energy": 1.0},
        pretrained_model="x.model",
    )
    with pytest.raises(ValueError, match="nknown"):
        TrainConfig.model_validate(raw)


def test_config_accepts_anchor_weights():
    anchors = {"partial_charges": 1.0, "a0_ratios": 1.0, "c6_ratios": 1.0}
    raw = _raw_config(
        long_range_anchor_weights=anchors, pretrained_model="x.model"
    )
    cfg = TrainConfig.model_validate(raw).model_dump()
    assert cfg["TRAINING"]["long_range_anchor_weights"] == anchors


def test_config_default_has_no_anchors():
    cfg = TrainConfig.model_validate(_raw_config()).model_dump()
    assert cfg["TRAINING"]["long_range_anchor_weights"] is None


# ---------------------------------------------------------------------------
# Frozen teacher and AnchoredLoss
# ---------------------------------------------------------------------------


def test_teacher_is_frozen_copy(lr_model):
    flags = [p.requires_grad for p in lr_model.parameters()]
    teacher = make_anchor_teacher(lr_model, "float64")
    assert teacher is not lr_model
    assert not teacher.training
    assert all(not p.requires_grad for p in teacher.parameters())
    assert [p.requires_grad for p in lr_model.parameters()] == flags


def test_teacher_cast_to_training_dtype(so3lr_model_config):
    model = SO3LR(**{**so3lr_model_config, "dtype": torch.float32})
    torch.set_default_dtype(torch.float64)
    teacher = make_anchor_teacher(model, "float64")
    assert all(p.dtype == torch.float64 for p in teacher.parameters())
    assert all(p.dtype == torch.float32 for p in model.parameters())


def test_anchor_loss_zero_for_identical_student(lr_model, batch):
    base = WeightedPropertiesLoss({"forces": 1.0})
    loss = AnchoredLoss(
        base, make_anchor_teacher(lr_model, "float64"), ALL_ANCHORS
    )
    pred = _pred(lr_model, batch)
    assert torch.allclose(
        loss(ref=batch, pred=pred), base(ref=batch, pred=pred), atol=1e-12
    )


def test_anchor_loss_positive_and_differentiable_after_drift(lr_model, batch):
    teacher = make_anchor_teacher(lr_model, "float64")
    _drift(lr_model.partial_charges_output_block)
    loss = AnchoredLoss(
        WeightedPropertiesLoss({"forces": 0.0}), teacher, {"charges": 1.0}
    )
    value = loss(ref=batch, pred=_pred(lr_model, batch))
    assert value > 0
    value.backward()
    assert all(p.grad is None for p in teacher.parameters())
    assert any(
        p.grad is not None
        for p in lr_model.partial_charges_output_block.parameters()
    )


def test_anchor_terms_ignore_label_weights(lr_model, batch):
    # `batch` has no charge labels (charges_weight == 0); the anchor term
    # must still count every atom.
    assert float(batch.charges_weight) == 0.0
    teacher = make_anchor_teacher(lr_model, "float64")
    _drift(lr_model.partial_charges_output_block)
    loss = AnchoredLoss(
        WeightedPropertiesLoss({"forces": 0.0}), teacher, {"charges": 1.0}
    )
    assert loss(ref=batch, pred=_pred(lr_model, batch)) > 0


def test_last_anchor_targets_are_teacher_predictions(lr_model, batch):
    teacher = make_anchor_teacher(lr_model, "float64")
    loss = AnchoredLoss(
        WeightedPropertiesLoss({"forces": 1.0}), teacher, ALL_ANCHORS
    )
    loss(ref=batch, pred=_pred(lr_model, batch))
    out = teacher(batch.to_dict(), training=False, compute_force=False)
    targets = loss.last_anchor_targets
    assert torch.allclose(targets["charges"], out["partial_charges"])
    for key in ("hirshfeld_ratios", "c6_ratios", "dipole"):
        assert torch.allclose(targets[key], out[key])


def test_teacher_hidden_from_loss_module(lr_model):
    loss = AnchoredLoss(
        WeightedPropertiesLoss({"forces": 1.0}),
        make_anchor_teacher(lr_model, "float64"),
        {"charges": 1.0},
    )
    assert sum(p.numel() for p in loss.parameters()) == 0
    assert not any("teacher" in k for k in loss.state_dict())
    assert not any(isinstance(m, SO3LR) for m in loss.modules())


def test_real_batch_not_mutated(lr_model, labelled_batch):
    before = {
        k: v.clone()
        for k, v in labelled_batch.to_dict().items()
        if torch.is_tensor(v)
    }
    loss = AnchoredLoss(
        WeightedPropertiesLoss({"charges": 1.0}),
        make_anchor_teacher(lr_model, "float64"),
        ALL_ANCHORS,
    )
    loss(ref=labelled_batch, pred=_pred(lr_model, labelled_batch))
    after = labelled_batch.to_dict()
    for key, value in before.items():  # absent C6 labels are NaN
        torch.testing.assert_close(
            after[key], value, rtol=0, atol=0, equal_nan=True
        )


@pytest.mark.parametrize("reduction", ["flat", "per_graph"])
def test_anchor_term_follows_base_reduction(lr_model, batch, reduction):
    teacher = make_anchor_teacher(lr_model, "float64")
    _drift(lr_model.hirshfeld_output_block)
    base = WeightedPropertiesLoss({"forces": 0.0}, reduction=reduction)
    loss = AnchoredLoss(base, teacher, {"hirshfeld_ratios": 1.0})
    assert loss(ref=batch, pred=_pred(lr_model, batch)) > 0


def test_adaptive_anchor_params_are_separate(lr_model):
    adaptive = AdaptiveRobustParams(["anchor_charges"], 1.0, 1.0)
    base = WeightedPropertiesLoss(
        {"forces": 1.0, "charges": 1.0}, robust="adaptive"
    )
    loss = AnchoredLoss(
        base,
        make_anchor_teacher(lr_model, "float64"),
        {"charges": 1.0},
        adaptive=adaptive,
    )
    names = [n for n, _ in loss.named_parameters()]
    assert any("anchor_charges" in n for n in names)
    assert any(n.endswith("charges") and "anchor" not in n for n in names)


def test_adaptive_anchor_term_uses_anchor_params(lr_model, batch):
    adaptive = AdaptiveRobustParams(["anchor_charges"], 1.0, 1.0)
    teacher = make_anchor_teacher(lr_model, "float64")
    _drift(lr_model.partial_charges_output_block)
    base = WeightedPropertiesLoss({"forces": 1.0}, robust="adaptive")
    loss = AnchoredLoss(base, teacher, {"charges": 1.0}, adaptive=adaptive)
    loss(ref=batch, pred=_pred(lr_model, batch)).backward()
    grads = [
        p.grad for n, p in adaptive.named_parameters() if "anchor_charges" in n
    ]
    assert grads and all(g is not None for g in grads)


def test_check_teacher_names_missing_quantity(so3lr_model_config, batch):
    no_c6 = SO3LR(**so3lr_model_config)
    loss = AnchoredLoss(
        WeightedPropertiesLoss({"forces": 1.0}),
        make_anchor_teacher(no_c6, "float64"),
        {"c6_ratios": 1.0},
    )
    with pytest.raises(ValueError, match="c6_ratios"):
        loss.check_teacher(batch)


# ---------------------------------------------------------------------------
# Warning when the data has hard labels for an anchored quantity
# ---------------------------------------------------------------------------


def _anchor_warnings(caplog, name):
    return [
        r
        for r in caplog.records
        if r.levelno == logging.WARNING
        and "long_range_anchor_weights" in r.getMessage()
        and f"'{name}'" in r.getMessage()
    ]


def test_warns_when_hard_labels_present(lr_model, labelled_batch, caplog):
    loss = AnchoredLoss(
        WeightedPropertiesLoss({"forces": 1.0, "charges": 1.0}),
        make_anchor_teacher(lr_model, "float64"),
        {"charges": 1.0},
    )
    with caplog.at_level(logging.WARNING):
        loss(ref=labelled_batch, pred=_pred(lr_model, labelled_batch))
    (record,) = _anchor_warnings(caplog, "charges")
    msg = record.getMessage()
    assert "hard labels" in msg
    assert "Both" in msg  # label weight > 0: both terms are trained


def test_warning_says_labels_ignored_when_label_weight_zero(
    lr_model, labelled_batch, caplog
):
    loss = AnchoredLoss(
        WeightedPropertiesLoss({"forces": 1.0}),
        make_anchor_teacher(lr_model, "float64"),
        {"charges": 1.0},
    )
    with caplog.at_level(logging.WARNING):
        loss(ref=labelled_batch, pred=_pred(lr_model, labelled_batch))
    (record,) = _anchor_warnings(caplog, "charges")
    assert "ignored" in record.getMessage()


@pytest.mark.parametrize("name", ["hirshfeld_ratios", "dipole"])
def test_warns_for_ratio_and_dipole_labels(
    lr_model, labelled_batch, caplog, name
):
    loss = AnchoredLoss(
        WeightedPropertiesLoss({"forces": 1.0}),
        make_anchor_teacher(lr_model, "float64"),
        {name: 1.0},
    )
    with caplog.at_level(logging.WARNING):
        loss(ref=labelled_batch, pred=_pred(lr_model, labelled_batch))
    assert len(_anchor_warnings(caplog, name)) == 1


def test_warns_once_per_quantity(lr_model, labelled_batch, caplog):
    teacher = make_anchor_teacher(lr_model, "float64")
    warned = set()
    losses = [
        AnchoredLoss(
            WeightedPropertiesLoss({"forces": 1.0}),
            teacher,
            {"charges": 1.0},
            warned=warned,
        )
        for _ in range(2)
    ]
    with caplog.at_level(logging.WARNING):
        for loss in losses + losses:
            loss(ref=labelled_batch, pred=_pred(lr_model, labelled_batch))
    assert len(_anchor_warnings(caplog, "charges")) == 1


def test_no_warning_without_labels(lr_model, batch, caplog):
    loss = AnchoredLoss(
        WeightedPropertiesLoss({"forces": 1.0}),
        make_anchor_teacher(lr_model, "float64"),
        ALL_ANCHORS,
    )
    with caplog.at_level(logging.WARNING):
        loss(ref=batch, pred=_pred(lr_model, batch))
    assert not [
        r for r in caplog.records if "long_range_anchor_weights" in r.msg
    ]


def test_zero_structure_weight_is_not_a_label(
    lr_model, labelled_batch, caplog
):
    labelled_batch.weight[:] = 0.0
    loss = AnchoredLoss(
        WeightedPropertiesLoss({"forces": 1.0}),
        make_anchor_teacher(lr_model, "float64"),
        ALL_ANCHORS,
    )
    with caplog.at_level(logging.WARNING):
        loss(ref=labelled_batch, pred=_pred(lr_model, labelled_batch))
    assert not [
        r for r in caplog.records if "long_range_anchor_weights" in r.msg
    ]


def test_nan_ratio_labels_are_not_labels(lr_model, batch, caplog):
    # Absent ratios are NaN; force a non-zero label weight to show that
    # NaN alone keeps the structure from counting as labelled.
    batch.hirshfeld_ratios_weight[:] = 1.0
    assert torch.isnan(batch.hirshfeld_ratios).all()
    loss = AnchoredLoss(
        WeightedPropertiesLoss({"forces": 1.0}),
        make_anchor_teacher(lr_model, "float64"),
        {"hirshfeld_ratios": 1.0},
    )
    with caplog.at_level(logging.WARNING):
        loss(ref=batch, pred=_pred(lr_model, batch))
    assert not _anchor_warnings(caplog, "hirshfeld_ratios")
