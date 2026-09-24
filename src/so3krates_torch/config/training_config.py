from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator


class GeneralConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name_exp: str
    checkpoints_dir: str = "./checkpoints"
    model_dir: str = "./model"
    log_dir: str = "./logs"
    default_dtype: Literal[
        "float32", "float64", "float16", "bfloat16"
    ] = "float64"
    seed: int = 100
    # inference-only: stress is computed during evaluation but has no
    # loss weight and is never used as a training target
    compute_stress: bool = False


class ArchitectureConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    degrees: List[int]
    r_max: float = 4.5
    r_max_lr: Optional[float] = None
    num_features: int = 128
    num_heads: int = 4
    num_layers: int = 3
    num_radial_basis_fn: int = 32
    activation_fn: str = "silu"
    energy_activation_fn: str = "silu"
    cutoff_fn: str = "cosine"
    radial_basis_fn: str = "bernstein"
    message_normalization: str = "avg_num_neighbors"
    energy_regression_dim: int = 128
    energy_learn_atomic_type_shifts: bool = False
    energy_learn_atomic_type_scales: bool = False
    layer_normalization_1: bool = False
    layer_normalization_2: bool = False
    residual_mlp_1: bool = False
    residual_mlp_2: bool = False
    use_charge_embed: bool = False
    use_spin_embed: bool = False
    qk_non_linearity: str = "identity"
    input_convention: str = "positions"
    layers_behave_like_identity_fn_at_init: bool = False
    output_is_zero_at_init: bool = False
    # SO3LR-specific
    zbl_repulsion_bool: bool = True
    electrostatic_energy_bool: bool = True
    electrostatic_energy_scale: float = 4.0
    dispersion_energy_bool: bool = True
    dispersion_energy_scale: float = 1.2
    dispersion_energy_cutoff_lr_damping: Optional[float] = None
    neighborlist_format_lr: str = "sparse"
    # PME electrostatics
    use_pme: bool = False
    pme_smearing: Optional[float] = None
    pme_mesh_spacing: Optional[float] = None
    compute_avg_num_neighbors: bool = True
    # Multi-head
    convert_to_multihead: bool = False
    num_output_heads: Optional[int] = None
    use_multihead: bool = False

    @model_validator(mode="after")
    def validate_long_range(self):
        electrostatics_needs_lr = (
            self.electrostatic_energy_bool and not self.use_pme
        )
        dispersion_needs_lr = self.dispersion_energy_bool
        if (
            electrostatics_needs_lr or dispersion_needs_lr
        ) and self.r_max_lr is None:
            raise ValueError(
                "Long-range cutoff 'r_max_lr' must be specified "
                "when electrostatic_energy_bool or "
                "dispersion_energy_bool is True (and the "
                "corresponding PME flag is False). "
                f"Current: r_max_lr={self.r_max_lr}, "
                f"electrostatic_energy_bool="
                f"{self.electrostatic_energy_bool}, "
                f"use_pme={self.use_pme}, "
                f"dispersion_energy_bool="
                f"{self.dispersion_energy_bool}"
            )
        if (
            self.dispersion_energy_bool
            and self.dispersion_energy_cutoff_lr_damping is None
        ):
            raise ValueError(
                "dispersion_energy_cutoff_lr_damping must be "
                "specified when dispersion_energy_bool is True. "
                f"Got dispersion_energy_cutoff_lr_damping="
                f"{self.dispersion_energy_cutoff_lr_damping}"
            )
        if self.convert_to_multihead and (self.num_output_heads is None):
            raise ValueError(
                "num_output_heads must be specified when using "
                "convert_to_multihead"
            )
        return self


class TrainingConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    batch_size: int
    valid_batch_size: int
    lr: float
    num_epochs: int
    path_to_train_data: str
    weight_decay: float = 0.0
    optimizer: str = "adam"
    amsgrad: bool = False
    scheduler: str = "exponential_decay"
    lr_scheduler_gamma: float = 0.85
    warmup_steps: int = 0
    energy_weight: float = 1.0
    forces_weight: float = 1000.0
    dipole_weight: float = 0.0
    hirshfeld_weight: float = 0.0
    eval_interval: int = 1
    valid_ratio: float = 0.1
    clip_grad: float = 10.0
    patience: int = 50
    early_stopping_min_delta: float = 0.0
    early_stopping_warmup: int = 0
    loss_type: str = "auto"
    path_to_val_data: Optional[str] = None
    keys: Optional[Dict[str, str]] = None
    heads: Optional[Dict[str, Any]] = None
    pretrained_weights: Optional[str] = None
    pretrained_model: Optional[str] = None
    finetune_choice: Optional[str] = None
    lazy_loading: bool = False
    num_workers: int = 4
    prefetch_factor: int = 2
    num_neighbor_samples: int = 1000
    data_preprocessed: Optional[bool] = None
    ema: bool = False
    ema_decay: float = 0.99
    # Replay settings
    replay_datasets: Optional[List[str]] = None
    replay_fractions: Optional[List[float]] = None
    replay_total: Optional[int] = None
    # None means "use the single-head path's default of True";
    # kept as a tri-state (rather than bool = True) so validate_replay
    # can tell an explicit setting apart from the default even after a
    # model_dump() -> model_validate() round trip (which otherwise
    # marks every field as explicitly set, defeating
    # model_fields_set-based detection) — needed since this flag has
    # no effect at all under replay_as_heads and should be rejected
    # rather than silently ignored there.
    replay_oversample_finetune: Optional[bool] = None
    replay_resample_per_epoch: bool = False
    replay_as_heads: bool = False
    # Optional per-dataset head name for replay_as_heads. Datasets
    # sharing the same name are routed to the same head instead of
    # each getting its own. Defaults to one head per replay dataset.
    replay_head_names: Optional[List[str]] = None
    # Per-config-type loss weight multipliers
    config_type_weights: Optional[Dict[str, float]] = None
    # For a manual `heads:` multi-head run, which head's validation
    # loss drives checkpointing/early-stopping/LR scheduling (default:
    # the last-declared head, for backwards compatibility). Ignored
    # for replay_as_heads, which always uses the "finetune" head.
    primary_valid_head: Optional[str] = None
    # What to export at the end of training: the (possibly multi-head)
    # trained model, per-head single-head models extracted from it, or
    # both.
    post_training_export: Literal[
        "multihead", "multihead+singlehead", "singlehead"
    ] = "multihead"

    @model_validator(mode="after")
    def validate_pretrained(self):
        if (
            self.pretrained_weights is not None
            and self.pretrained_model is not None
        ):
            raise ValueError(
                "Cannot specify both 'pretrained_weights' and "
                "'pretrained_model'. Use one or the other."
            )
        return self

    @model_validator(mode="after")
    def validate_replay(self):
        replay_fields = [
            self.replay_datasets,
            self.replay_fractions,
            self.replay_total,
        ]
        any_set = any(f is not None for f in replay_fields)
        all_set = all(f is not None for f in replay_fields)
        if any_set and not all_set:
            raise ValueError(
                "'replay_datasets', 'replay_fractions', and "
                "'replay_total' must all be specified together."
            )
        if not any_set:
            if self.replay_as_heads:
                raise ValueError(
                    "replay_as_heads=True requires 'replay_datasets', "
                    "'replay_fractions', and 'replay_total' to also "
                    "be specified."
                )
            return self
        if len(self.replay_datasets) != len(self.replay_fractions):
            raise ValueError(
                f"replay_datasets has {len(self.replay_datasets)} "
                f"entries but replay_fractions has "
                f"{len(self.replay_fractions)}. They must match."
            )
        if any(f < 0 for f in self.replay_fractions):
            raise ValueError("All replay_fractions must be >= 0.")
        if abs(sum(self.replay_fractions) - 1.0) > 1e-6:
            raise ValueError(
                f"replay_fractions must sum to 1.0, "
                f"got {sum(self.replay_fractions):.6f}."
            )
        if self.replay_total <= 0:
            raise ValueError(
                f"replay_total must be > 0, got {self.replay_total}."
            )
        if self.heads is not None:
            raise ValueError(
                "Data replay is not supported with multi-head "
                "training. Remove 'heads' or 'replay_datasets'."
            )
        if self.replay_as_heads and self.replay_resample_per_epoch:
            raise ValueError(
                "replay_resample_per_epoch is not supported with "
                "replay_as_heads (each replay dataset is sampled "
                "once, at setup time). Remove 'replay_resample_per_epoch'."
            )
        if (
            self.replay_as_heads
            and self.replay_oversample_finetune is not None
        ):
            raise ValueError(
                "replay_oversample_finetune has no effect under "
                "replay_as_heads (there is no combined fine-tune + "
                "replay loader to balance; each head is trained via "
                "its own loader). Remove 'replay_oversample_finetune'."
            )
        if self.replay_head_names is not None:
            if len(self.replay_head_names) != len(self.replay_datasets):
                raise ValueError(
                    f"replay_head_names has "
                    f"{len(self.replay_head_names)} entries but "
                    f"replay_datasets has {len(self.replay_datasets)}. "
                    "They must match."
                )
            # Must match data_setup.FINETUNE_HEAD_NAME. Not imported
            # here to keep this leaf config module free of a
            # dependency on the tools layer.
            if "finetune" in self.replay_head_names:
                raise ValueError(
                    "'finetune' is reserved for the main training "
                    "dataset's head and cannot be used in "
                    "'replay_head_names'."
                )
        return self

    @model_validator(mode="after")
    def validate_post_training_export(self):
        is_multihead_run = self.heads is not None or self.replay_as_heads
        if self.post_training_export != "multihead" and not is_multihead_run:
            raise ValueError(
                f"post_training_export='{self.post_training_export}' "
                "requires a multi-head run ('heads' or "
                "'replay_as_heads')."
            )
        return self


class MiscConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    device: str = "cpu"
    distributed: bool = False
    launcher: Optional[str] = None
    log_level: str = "INFO"
    restart_latest: bool = True
    no_checkpoint: bool = False
    log_wandb: bool = False
    keep_checkpoints: bool = False
    error_table: str = "PerAtomMAE"
    deterministic_seed: bool = False


class TrainConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")
    GENERAL: GeneralConfig
    ARCHITECTURE: ArchitectureConfig
    TRAINING: TrainingConfig
    MISC: MiscConfig = MiscConfig()

    @model_validator(mode="after")
    def validate_multihead_cross_section(self):
        is_multihead_flagged = (
            self.ARCHITECTURE.convert_to_multihead
            or self.ARCHITECTURE.use_multihead
        )
        if (
            self.TRAINING.post_training_export != "multihead"
            and not is_multihead_flagged
        ):
            raise ValueError(
                f"TRAINING.post_training_export="
                f"'{self.TRAINING.post_training_export}' requires "
                "ARCHITECTURE.convert_to_multihead or use_multihead "
                "to be true (a plain single-head model has no heads "
                "to extract)."
            )
        if self.TRAINING.replay_as_heads:
            if not self.ARCHITECTURE.use_multihead:
                raise ValueError(
                    "TRAINING.replay_as_heads requires "
                    "ARCHITECTURE.use_multihead=true (otherwise "
                    "model.select_heads is never enabled during "
                    "training and per-sample head routing has no "
                    "effect)."
                )
            replay_head_names = self.TRAINING.replay_head_names
            if replay_head_names is None:
                num_replay_heads = len(self.TRAINING.replay_datasets or [])
            else:
                num_replay_heads = len(set(replay_head_names))
            expected_heads = num_replay_heads + 1
            if self.ARCHITECTURE.num_output_heads != expected_heads:
                raise ValueError(
                    "TRAINING.replay_as_heads requires "
                    "ARCHITECTURE.num_output_heads == number of "
                    f"distinct replay heads + 1 = {expected_heads} "
                    "(one fine-tune head plus one per distinct "
                    "replay_head_names entry), got "
                    f"num_output_heads="
                    f"{self.ARCHITECTURE.num_output_heads}."
                )
        return self
