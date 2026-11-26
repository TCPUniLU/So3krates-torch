"""Tests for long-range anchoring (``modules/anchoring.py``): a frozen copy
of the pretrained model labels every batch with its partial charges,
a0/C6 ratios and dipole, which the fine-tuned model is pulled towards."""

from pathlib import Path

import pytest
import yaml

from so3krates_torch.config import TrainConfig
from so3krates_torch.modules.anchoring import normalize_anchor_weights

EXAMPLE = (
    Path(__file__).resolve().parents[3]
    / "examples/training/train_settings_example.yaml"
)


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
