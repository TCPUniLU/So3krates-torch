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
    assert cfg.replay_as_heads is False


def test_replay_as_heads_with_manual_heads_rejected():
    """replay_as_heads builds its own heads dict internally, so a
    manually-specified 'heads' dict at the same time is ambiguous."""
    with pytest.raises(ValidationError, match="not supported with"):
        TrainingConfig(
            batch_size=5,
            valid_batch_size=5,
            lr=0.001,
            num_epochs=10,
            path_to_train_data="./data",
            replay_as_heads=True,
            replay_datasets=["old.xyz"],
            replay_fractions=[1.0],
            replay_total=10,
            heads={"a": {"path_to_train_data": "a.xyz"}},
        )


def test_replay_as_heads_allowed_without_manual_heads():
    cfg = TrainingConfig(
        batch_size=5,
        valid_batch_size=5,
        lr=0.001,
        num_epochs=10,
        path_to_train_data="./data",
        replay_as_heads=True,
        replay_datasets=["old.xyz"],
        replay_fractions=[1.0],
        replay_total=10,
    )
    assert cfg.replay_as_heads is True
    assert cfg.heads is None


def test_replay_head_names_length_mismatch_rejected():
    with pytest.raises(ValidationError, match="replay_head_names"):
        TrainingConfig(
            batch_size=5,
            valid_batch_size=5,
            lr=0.001,
            num_epochs=10,
            path_to_train_data="./data",
            replay_as_heads=True,
            replay_datasets=["old_a.xyz", "old_b.xyz"],
            replay_fractions=[0.5, 0.5],
            replay_total=10,
            replay_head_names=["only_one"],
        )


def test_replay_head_names_reserved_finetune_name_rejected():
    with pytest.raises(ValidationError, match="reserved"):
        TrainingConfig(
            batch_size=5,
            valid_batch_size=5,
            lr=0.001,
            num_epochs=10,
            path_to_train_data="./data",
            replay_as_heads=True,
            replay_datasets=["old_a.xyz"],
            replay_fractions=[1.0],
            replay_total=10,
            replay_head_names=["finetune"],
        )


def test_replay_head_names_allows_grouping():
    cfg = TrainingConfig(
        batch_size=5,
        valid_batch_size=5,
        lr=0.001,
        num_epochs=10,
        path_to_train_data="./data",
        replay_as_heads=True,
        replay_datasets=["old_a.xyz", "old_b.xyz", "old_c.xyz"],
        replay_fractions=[0.3, 0.3, 0.4],
        replay_total=10,
        replay_head_names=["old_md", "old_md", "old_qm"],
    )
    assert cfg.replay_head_names == ["old_md", "old_md", "old_qm"]


def test_post_training_export_defaults_to_multihead():
    cfg = TrainingConfig(
        batch_size=5,
        valid_batch_size=5,
        lr=0.001,
        num_epochs=10,
        path_to_train_data="./data",
    )
    assert cfg.post_training_export == "multihead"


def test_post_training_export_rejects_unknown_value():
    with pytest.raises(ValidationError):
        TrainingConfig(
            batch_size=5,
            valid_batch_size=5,
            lr=0.001,
            num_epochs=10,
            path_to_train_data="./data",
            post_training_export="bogus",
        )


def test_post_training_export_singlehead_requires_multihead_run():
    """Exporting per-head models makes no sense for a plain
    single-head run (no 'heads' dict, no replay_as_heads)."""
    with pytest.raises(ValidationError, match="post_training_export"):
        TrainingConfig(
            batch_size=5,
            valid_batch_size=5,
            lr=0.001,
            num_epochs=10,
            path_to_train_data="./data",
            post_training_export="singlehead",
        )


def test_post_training_export_singlehead_allowed_with_heads():
    cfg = TrainingConfig(
        batch_size=5,
        valid_batch_size=5,
        lr=0.001,
        num_epochs=10,
        path_to_train_data="./data",
        post_training_export="singlehead",
        heads={"a": {"path_to_train_data": "a.xyz"}},
    )
    assert cfg.post_training_export == "singlehead"


def test_post_training_export_singlehead_allowed_with_replay_as_heads():
    cfg = TrainingConfig(
        batch_size=5,
        valid_batch_size=5,
        lr=0.001,
        num_epochs=10,
        path_to_train_data="./data",
        post_training_export="multihead+singlehead",
        replay_as_heads=True,
        replay_datasets=["old.xyz"],
        replay_fractions=[1.0],
        replay_total=10,
    )
    assert cfg.post_training_export == "multihead+singlehead"


def test_replay_as_heads_without_replay_triple_rejected():
    """replay_as_heads=True with no replay_datasets/fractions/total
    must be rejected at config time, not crash mid-setup with a bare
    TypeError from len(None)."""
    with pytest.raises(ValidationError, match="replay_as_heads"):
        TrainingConfig(
            batch_size=5,
            valid_batch_size=5,
            lr=0.001,
            num_epochs=10,
            path_to_train_data="./data",
            replay_as_heads=True,
        )


def test_replay_as_heads_rejects_replay_resample_per_epoch():
    """Per-epoch replay resampling isn't implemented for the
    replay_as_heads path (data_setup.py always returns replay_builder
    = None there); accept-then-silently-ignore is worse than
    rejecting."""
    with pytest.raises(ValidationError, match="replay_resample_per_epoch"):
        TrainingConfig(
            batch_size=5,
            valid_batch_size=5,
            lr=0.001,
            num_epochs=10,
            path_to_train_data="./data",
            replay_as_heads=True,
            replay_datasets=["old.xyz"],
            replay_fractions=[1.0],
            replay_total=10,
            replay_resample_per_epoch=True,
        )


def test_replay_as_heads_rejects_explicit_replay_oversample_finetune():
    """replay_oversample_finetune only affects the single-head replay
    path's fine-tune:replay ratio; it has no effect under
    replay_as_heads (each head is its own loader, no combining)."""
    with pytest.raises(ValidationError, match="replay_oversample_finetune"):
        TrainingConfig(
            batch_size=5,
            valid_batch_size=5,
            lr=0.001,
            num_epochs=10,
            path_to_train_data="./data",
            replay_as_heads=True,
            replay_datasets=["old.xyz"],
            replay_fractions=[1.0],
            replay_total=10,
            replay_oversample_finetune=False,
        )


def test_replay_oversample_finetune_defaults_to_none_and_is_allowed():
    """Leaving replay_oversample_finetune untouched (the common case)
    must not be treated as 'explicitly set' by the check above."""
    cfg = TrainingConfig(
        batch_size=5,
        valid_batch_size=5,
        lr=0.001,
        num_epochs=10,
        path_to_train_data="./data",
        replay_as_heads=True,
        replay_datasets=["old.xyz"],
        replay_fractions=[1.0],
        replay_total=10,
    )
    assert cfg.replay_oversample_finetune is None


def _minimal_architecture(**overrides):
    return {
        "degrees": [1, 2],
        "zbl_repulsion_bool": False,
        "electrostatic_energy_bool": False,
        "dispersion_energy_bool": False,
        **overrides,
    }


def _minimal_train_config(**training_overrides):
    return TrainConfig(
        GENERAL={"name_exp": "test"},
        ARCHITECTURE=training_overrides.pop(
            "ARCHITECTURE", _minimal_architecture()
        ),
        TRAINING={
            "batch_size": 5,
            "valid_batch_size": 5,
            "lr": 0.001,
            "num_epochs": 10,
            "path_to_train_data": "./data",
            **training_overrides,
        },
    )


def test_post_training_export_requires_convert_to_multihead():
    """A manual 'heads:' dict alone isn't enough: without
    convert_to_multihead/use_multihead the model never actually
    becomes multi-head, so 'singlehead'/'multihead+singlehead' export
    would train fine and then die at export time with a KeyError."""
    with pytest.raises(ValidationError, match="convert_to_multihead"):
        _minimal_train_config(
            heads={"a": {"path_to_train_data": "a.xyz"}},
            post_training_export="singlehead",
        )


def test_post_training_export_allowed_with_convert_to_multihead():
    cfg = _minimal_train_config(
        ARCHITECTURE=_minimal_architecture(
            convert_to_multihead=True,
            use_multihead=True,
            num_output_heads=1,
        ),
        heads={"a": {"path_to_train_data": "a.xyz"}},
        post_training_export="singlehead",
    )
    assert cfg.TRAINING.post_training_export == "singlehead"


def test_replay_as_heads_requires_use_multihead():
    """Without use_multihead, model.select_heads is never turned on
    (run_train.py gates that on ARCHITECTURE.use_multihead), so the
    model would train ignoring the multi-head structure entirely."""
    with pytest.raises(ValidationError, match="use_multihead"):
        _minimal_train_config(
            ARCHITECTURE=_minimal_architecture(
                convert_to_multihead=True, num_output_heads=2
            ),
            replay_as_heads=True,
            replay_datasets=["old.xyz"],
            replay_fractions=[1.0],
            replay_total=10,
        )


def test_replay_as_heads_requires_matching_num_output_heads():
    with pytest.raises(ValidationError, match="num_output_heads"):
        _minimal_train_config(
            ARCHITECTURE=_minimal_architecture(
                convert_to_multihead=True,
                use_multihead=True,
                num_output_heads=5,
            ),
            replay_as_heads=True,
            replay_datasets=["old.xyz"],
            replay_fractions=[1.0],
            replay_total=10,
        )


def test_replay_as_heads_valid_config_round_trips_through_model_dump():
    """The full, correctly-configured replay_as_heads path must
    survive a model_dump() -> model_validate() round trip (this is
    exactly what setup_config_from_yaml does, and what the
    replay_oversample_finetune None-sentinel check must tolerate)."""
    cfg = _minimal_train_config(
        ARCHITECTURE=_minimal_architecture(
            convert_to_multihead=True,
            use_multihead=True,
            num_output_heads=2,
        ),
        replay_as_heads=True,
        replay_datasets=["old.xyz"],
        replay_fractions=[1.0],
        replay_total=10,
    )
    dumped = cfg.model_dump()
    round_tripped = TrainConfig.model_validate(dumped)
    assert round_tripped.TRAINING.replay_as_heads is True


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
