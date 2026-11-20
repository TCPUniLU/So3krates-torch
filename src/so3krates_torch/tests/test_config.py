"""Tests for Pydantic configuration models."""

from pathlib import Path

import pytest
import yaml
from pydantic import ValidationError

from so3krates_torch.config import (
    ArchitectureConfig,
    CreateLammpsArgs,
    EvalArgs,
    GeneralConfig,
    Jax2TorchArgs,
    MergeArgs,
    MetricArgs,
    MiscConfig,
    PreprocessArgs,
    Torch2JaxArgs,
    TrainConfig,
    TrainingConfig,
)

_EXAMPLE_YAML = (
    Path(__file__).resolve().parents[3]
    / "examples"
    / "training"
    / "train_settings_example.yaml"
)


# ── 1. Valid YAML backward compatibility ────────────────────────


def test_example_yaml_loads_through_train_config():
    """Load the shipped example YAML through TrainConfig."""
    raw = yaml.safe_load(_EXAMPLE_YAML.read_text())
    cfg = TrainConfig.model_validate(raw)
    assert cfg.GENERAL.name_exp == "my_model"
    assert cfg.ARCHITECTURE.r_max == 6.0
    assert cfg.TRAINING.batch_size == 5


# ── 2. Missing required field ──────────────────────────────────


def test_missing_required_field_degrees():
    """ArchitectureConfig without 'degrees' must fail."""
    with pytest.raises(ValidationError, match="degrees"):
        ArchitectureConfig(
            r_max=5.0,
            electrostatic_energy_bool=False,
            dispersion_energy_bool=False,
        )


def test_missing_required_section():
    """TrainConfig without GENERAL section must fail."""
    with pytest.raises(ValidationError, match="GENERAL"):
        TrainConfig.model_validate(
            {
                "ARCHITECTURE": {
                    "degrees": [1, 2],
                    "r_max": 5.0,
                    "electrostatic_energy_bool": False,
                    "dispersion_energy_bool": False,
                },
                "TRAINING": {
                    "batch_size": 5,
                    "valid_batch_size": 5,
                    "lr": 0.001,
                    "num_epochs": 10,
                    "path_to_train_data": "./data",
                },
            }
        )


# ── 3. Typo in section name (extra="forbid") ──────────────────


def test_typo_in_section_name_rejected():
    """TrainConfig rejects unknown top-level keys like 'TRANING'."""
    raw = yaml.safe_load(_EXAMPLE_YAML.read_text())
    raw["TRANING"] = raw.pop("TRAINING")
    with pytest.raises(ValidationError, match="TRANING"):
        TrainConfig.model_validate(raw)


# ── 4. Conditional: r_max_lr required for long-range ───────────


def test_r_max_lr_required_for_electrostatics():
    with pytest.raises(ValidationError, match="Long-range cutoff"):
        ArchitectureConfig(
            degrees=[1, 2],
            r_max=5.0,
            r_max_lr=None,
            electrostatic_energy_bool=True,
            dispersion_energy_bool=False,
        )


def test_r_max_lr_required_for_dispersion():
    with pytest.raises(ValidationError, match="Long-range cutoff"):
        ArchitectureConfig(
            degrees=[1, 2],
            r_max=5.0,
            r_max_lr=None,
            electrostatic_energy_bool=False,
            dispersion_energy_bool=True,
        )


# ── 5. Conditional: preprocessed mode requires r_max ───────────


def test_preprocess_preprocessed_requires_r_max():
    with pytest.raises(ValidationError, match="r-max"):
        PreprocessArgs(
            input="in.xyz",
            output="out.h5",
            mode="preprocessed",
            r_max=None,
        )


def test_preprocess_raw_accepts_no_r_max():
    cfg = PreprocessArgs(input="in.xyz", output="out.h5", mode="raw")
    assert cfg.r_max is None


# ── 6. Mutual exclusion: pretrained_weights / pretrained_model ─


def test_pretrained_mutual_exclusion():
    with pytest.raises(ValidationError, match="Cannot specify"):
        TrainingConfig(
            batch_size=5,
            valid_batch_size=5,
            lr=0.001,
            num_epochs=10,
            path_to_train_data="./data",
            pretrained_weights="w.pt",
            pretrained_model="m.pt",
        )


# ── 7. Defaults populated correctly ───────────────────────────


def test_general_defaults():
    cfg = GeneralConfig(name_exp="test")
    assert cfg.checkpoints_dir == "./checkpoints"
    assert cfg.model_dir == "./model"
    assert cfg.log_dir == "./logs"
    assert cfg.default_dtype == "float64"
    assert cfg.seed == 100


def test_misc_defaults():
    cfg = MiscConfig()
    assert cfg.device == "cpu"
    assert cfg.distributed is False
    assert cfg.log_level == "INFO"
    assert cfg.restart_latest is True
    assert cfg.error_table == "PerAtomMAE"


def test_training_defaults():
    cfg = TrainingConfig(
        batch_size=5,
        valid_batch_size=5,
        lr=0.001,
        num_epochs=10,
        path_to_train_data="./data",
    )
    assert cfg.weight_decay == 0.0
    assert cfg.optimizer == "adam"
    assert cfg.forces_weight == 1000.0
    assert cfg.valid_ratio == 0.1
    assert cfg.clip_grad == 10.0
    assert cfg.patience == 50


# ── 8. CLI model round-trip ────────────────────────────────────


def test_eval_args_round_trip():
    """Simulate argparse vars → Pydantic → model_dump."""
    args_dict = {
        "model_path": "model.pt",
        "data_path": "data.xyz",
        "output_file": "out.h5",
        "ensemble_size": 1,
        "device": "cpu",
        "batch_size": 10,
        "model_type": "so3lr",
        "r_max_lr": None,
        "multispecies": False,
        "multihead_model": False,
        "compute_dipole": False,
        "compute_stress": False,
        "compute_hirshfeld": False,
        "compute_partial_charges": False,
        "dispersion_energy_cutoff_lr_damping": 2.0,
        "energy_key": "REF_energy",
        "forces_key": "REF_forces",
        "stress_key": "REF_stress",
        "virials_key": "REF_virials",
        "dipole_key": "REF_dipoles",
        "charges_key": "REF_charges",
        "total_charge_key": "charge",
        "total_spin_key": "total_spin",
        "hirshfeld_key": "REF_hirsh_ratios",
        "head_key": "head",
        "head": "head",
        "dtype": "float32",
        "return_att": False,
    }
    validated = EvalArgs.model_validate(args_dict)
    dumped = validated.model_dump()
    assert dumped["model_path"] == "model.pt"
    assert dumped["batch_size"] == 10


def test_merge_args_min_inputs():
    """MergeArgs requires at least 2 inputs."""
    with pytest.raises(ValidationError, match="2 input"):
        MergeArgs(inputs=["one.h5"], output="out.h5")


def test_merge_args_valid():
    cfg = MergeArgs(inputs=["a.h5", "b.h5"], output="out.h5")
    assert len(cfg.inputs) == 2


def test_jax2torch_requires_save_path():
    with pytest.raises(ValidationError, match="save"):
        Jax2TorchArgs(
            path_to_params="p.pkl",
            path_to_hyperparams="h.yaml",
        )


def test_torch2jax_requires_save_path():
    with pytest.raises(ValidationError, match="save"):
        Torch2JaxArgs(
            path_to_state_dict="s.pt",
            path_to_hyperparams="h.yaml",
        )


def test_preprocess_validate_alias():
    """The 'validate' alias maps to validate_output field."""
    cfg = PreprocessArgs.model_validate(
        {
            "input": "in.xyz",
            "output": "out.h5",
            "mode": "raw",
            "validate": True,
        }
    )
    assert cfg.validate_output is True


# ── Dispersion-label loss weights (c6_ratios / a0_ratios) ──────


def _example_raw():
    return yaml.safe_load(_EXAMPLE_YAML.read_text())


def _training(**kw):
    section = _example_raw()["TRAINING"]
    section.pop("hirshfeld_weight", None)
    section.update(kw)
    return TrainingConfig(**section)


def test_c6_weight_default_zero_and_in_property_weights():
    from so3krates_torch.tools.training_setup import (
        get_loss_property_weights,
    )

    assert get_loss_property_weights({"TRAINING": {}})["c6_ratios"] == 0.0
    weights = get_loss_property_weights(
        {"TRAINING": {"c6_ratios_weight": 0.1}}
    )
    assert weights["c6_ratios"] == 0.1
    dumped = _training().model_dump()
    assert dumped["c6_ratios_weight"] == 0.0


def test_a0_ratios_weight_alias():
    cfg = _training(a0_ratios_weight=0.1)
    assert cfg.hirshfeld_weight == 0.1
    assert "a0_ratios_weight" not in cfg.model_dump()
    # same value under both names is accepted
    assert (
        _training(a0_ratios_weight=0.1, hirshfeld_weight=0.1).hirshfeld_weight
        == 0.1
    )
    with pytest.raises(ValidationError, match="a0_ratios_weight"):
        _training(a0_ratios_weight=0.1, hirshfeld_weight=0.2)


def test_c6_weight_requires_c6_head():
    raw = _example_raw()
    raw["ARCHITECTURE"]["dispersion_energy_bool"] = True
    raw["ARCHITECTURE"]["c6_ratios_bool"] = False
    raw["TRAINING"]["c6_ratios_weight"] = 0.1
    with pytest.raises(ValidationError, match="c6_ratios_bool"):
        TrainConfig.model_validate(raw)
    raw["ARCHITECTURE"]["c6_ratios_bool"] = True
    cfg = TrainConfig.model_validate(raw)
    assert cfg.TRAINING.c6_ratios_weight == 0.1


def test_c6_weight_with_pretrained_model_skips_architecture_check():
    """With pretrained_model the architecture comes from the pickled
    model, so ARCHITECTURE.c6_ratios_bool must not be required."""
    raw = _example_raw()
    raw["ARCHITECTURE"]["c6_ratios_bool"] = False
    raw["TRAINING"]["c6_ratios_weight"] = 0.1
    raw["TRAINING"]["pretrained_model"] = "/some/tauri.model"
    cfg = TrainConfig.model_validate(raw)
    assert cfg.TRAINING.c6_ratios_weight == 0.1


def test_loss_reduction_config_default_flat():
    assert _training().loss_reduction == "flat"
    assert _training(loss_reduction="per_graph").loss_reduction == "per_graph"
    with pytest.raises(ValidationError):
        _training(loss_reduction="mean")


def test_robust_flags_mutually_exclusive():
    cfg = _training()
    assert cfg.use_robust_loss is False
    assert cfg.use_adaptive_robust_loss is False
    assert cfg.robust_loss_alpha == 1.0
    assert cfg.adaptive_robust_loss_init_alpha == 1.0
    assert cfg.adaptive_robust_loss_init_scale == 1.0
    with pytest.raises(ValidationError, match="use_adaptive_robust_loss"):
        _training(use_robust_loss=True, use_adaptive_robust_loss=True)


# ── Two-stage loss weights / validation robust settings ────────


def test_stage_two_config_both_or_neither():
    cfg = _training(
        stage_two_start_epoch=3, stage_two_loss_weights={"energy_weight": 0.01}
    )
    assert cfg.stage_two_start_epoch == 3
    with pytest.raises(ValidationError, match="stage_two"):
        _training(stage_two_start_epoch=3)
    with pytest.raises(ValidationError, match="stage_two"):
        _training(stage_two_loss_weights={"energy_weight": 0.01})


def test_stage_two_config_rejects_bad_values():
    with pytest.raises(ValidationError, match="stage_two_start_epoch"):
        _training(
            stage_two_start_epoch=0,
            stage_two_loss_weights={"energy_weight": 0.01},
        )
    with pytest.raises(ValidationError, match="dipole_vec"):
        _training(
            stage_two_start_epoch=2,
            stage_two_loss_weights={"dipole_vec": 0.1},
        )


def test_stage_two_a0_alias_normalized():
    cfg = _training(
        stage_two_start_epoch=2,
        stage_two_loss_weights={"a0_ratios_weight": 0.1},
    )
    assert cfg.stage_two_loss_weights == {"hirshfeld_weight": 0.1}


def test_stage_two_c6_requires_head():
    raw = _example_raw()
    raw["ARCHITECTURE"]["c6_ratios_bool"] = False
    raw["TRAINING"]["stage_two_start_epoch"] = 2
    raw["TRAINING"]["stage_two_loss_weights"] = {"c6_ratios_weight": 0.1}
    with pytest.raises(ValidationError, match="c6_ratios_bool"):
        TrainConfig.model_validate(raw)


def test_validation_robust_defaults():
    cfg = _training()
    assert cfg.use_robust_loss_validation is None
    assert cfg.robust_loss_alpha_validation is None
