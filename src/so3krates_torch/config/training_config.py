from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, model_validator


class GeneralConfig(BaseModel):
    model_config = ConfigDict(extra="allow")
    name_exp: str
    checkpoints_dir: str = "./checkpoints"
    model_dir: str = "./model"
    log_dir: str = "./logs"
    default_dtype: Literal["float32", "float64", "float16", "bfloat16"] = (
        "float64"
    )
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
    use_rms_norm: bool = False
    residual_mlp_1: bool = False
    residual_mlp_2: bool = False
    use_charge_embed: bool = False
    use_spin_embed: bool = False
    qk_non_linearity: str = "identity"
    qk_norm: bool = False
    use_residual_scalars: bool = False
    input_convention: str = "positions"
    layers_behave_like_identity_fn_at_init: bool = False
    output_is_zero_at_init: bool = False
    # SO3LR-specific
    zbl_repulsion_bool: bool = True
    nhl_repulsion_bool: bool = False
    c6_ratios_bool: bool = False
    use_simple_hirshfeld: bool = False
    num_theory_levels: int = 1
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
    # PME dispersion (k-space correction added on top of the real-space
    # C6 residual + C8/C10 terms — does not replace them, unlike use_pme
    # above for electrostatics)
    use_pme_dispersion: bool = False
    pme_dispersion_smearing: Optional[float] = None
    pme_dispersion_mesh_spacing: Optional[float] = None
    # Mesh-size rounding rule for both PME terms: "jax" matches so3lr_dev
    # (jax-pme), "torchpme" is the stock torch-pme rule.
    pme_mesh_rule: Literal["jax", "torchpme"] = "jax"
    legacy_dispersion_bool: bool = True
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
        if self.nhl_repulsion_bool and self.zbl_repulsion_bool:
            import logging

            logging.warning(
                "nhl_repulsion_bool and zbl_repulsion_bool are both "
                "True. NHL takes precedence; ZBL module is instantiated "
                "but never called."
            )
        if self.c6_ratios_bool and not self.dispersion_energy_bool:
            raise ValueError(
                "c6_ratios_bool=True requires dispersion_energy_bool=True."
            )
        if self.use_simple_hirshfeld and not self.dispersion_energy_bool:
            raise ValueError(
                "use_simple_hirshfeld=True requires "
                "dispersion_energy_bool=True."
            )
        if self.num_theory_levels < 1:
            raise ValueError("num_theory_levels must be >= 1.")
        return self


STAGE_TWO_WEIGHT_KEYS = (
    "energy_weight",
    "forces_weight",
    "dipole_weight",
    "hirshfeld_weight",
    "a0_ratios_weight",
    "c6_ratios_weight",
    "charges_weight",
)


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
    # Per-atom C6-ratio label (paper Eq. S13, JAX `c6_ratios`); needs
    # ARCHITECTURE.c6_ratios_bool=True. `hirshfeld_weight` may also be
    # given as `a0_ratios_weight` (the tauri/JAX name, same quantity).
    c6_ratios_weight: float = 0.0
    # "flat": mean over all elements in the batch (MACE-style, default);
    # "per_graph": JAX so3lr reduction (per-structure mean, then mean
    # over structures carrying the label), DDP-correct.
    loss_reduction: Literal["flat", "per_graph"] = "flat"
    # Robust loss (Barron 2019), as in JAX so3lr. Fixed: penalty
    # 2*rho(x, robust_loss_alpha, 1). Adaptive: per-property alpha/scale
    # learned through the NLL; JAX parity needs loss_reduction per_graph.
    use_robust_loss: bool = False
    robust_loss_alpha: float = 1.0
    use_adaptive_robust_loss: bool = False
    adaptive_robust_loss_init_alpha: float = 1.0
    adaptive_robust_loss_init_scale: float = 1.0
    # Validation robust loss (JAX use_robust_loss_validation /
    # robust_loss_alpha_validation); None = same as training. Ignored with
    # the adaptive loss, whose validation always uses the training loss.
    use_robust_loss_validation: Optional[bool] = None
    robust_loss_alpha_validation: Optional[float] = None
    # Two-stage loss weights (JAX stage_two_*): from this epoch on, train
    # and validate with stage_two_loss_weights (keys as the *_weight
    # options above; unlisted ones keep their stage-one value). Best-loss,
    # early-stopping and plateau trackers reset at the switch.
    stage_two_start_epoch: Optional[int] = None
    stage_two_loss_weights: Optional[Dict[str, float]] = None
    # Fine-tuning (finetune_choice != naive): keep dispersion heads frozen.
    # `freeze_hirshfeld` may also be given as `freeze_a0_ratios`.
    freeze_hirshfeld: bool = True
    freeze_c6_ratios: bool = True
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
    replay_oversample_finetune: bool = True
    replay_resample_per_epoch: bool = False
    # Per-config-type loss weight multipliers
    config_type_weights: Optional[Dict[str, float]] = None

    @model_validator(mode="before")
    @classmethod
    def resolve_a0_ratios_alias(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        data = dict(data)
        for alias, name in (
            ("a0_ratios_weight", "hirshfeld_weight"),
            ("freeze_a0_ratios", "freeze_hirshfeld"),
        ):
            if alias not in data:
                continue
            value = data.pop(alias)
            if name in data and data[name] != value:
                raise ValueError(
                    f"Both '{name}' and its alias '{alias}' are set with "
                    f"different values ({data[name]} vs {value}). Set only "
                    "one."
                )
            data[name] = value
        return data

    @model_validator(mode="after")
    def validate_stage_two(self):
        start = self.stage_two_start_epoch
        weights = self.stage_two_loss_weights
        if (start is None) != (weights is None):
            raise ValueError(
                "Set both 'stage_two_start_epoch' and "
                "'stage_two_loss_weights', or neither."
            )
        if start is None:
            return self
        if start < 1:
            raise ValueError("stage_two_start_epoch must be >= 1.")
        unknown = set(weights) - set(STAGE_TWO_WEIGHT_KEYS)
        if unknown:
            raise ValueError(
                f"Unknown stage_two_loss_weights keys {sorted(unknown)}; "
                f"use {sorted(STAGE_TWO_WEIGHT_KEYS)}."
            )
        if "a0_ratios_weight" in weights:
            weights = dict(weights)
            a0 = weights.pop("a0_ratios_weight")
            if weights.get("hirshfeld_weight", a0) != a0:
                raise ValueError(
                    "stage_two_loss_weights sets both 'hirshfeld_weight' "
                    "and its alias 'a0_ratios_weight' to different values."
                )
            weights["hirshfeld_weight"] = a0
            self.stage_two_loss_weights = weights
        return self

    @model_validator(mode="after")
    def validate_robust_loss(self):
        if self.use_robust_loss and self.use_adaptive_robust_loss:
            raise ValueError(
                "Cannot use both 'use_robust_loss' (fixed) and "
                "'use_adaptive_robust_loss' (learned). Enable only one."
            )
        return self

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
    def validate_c6_ratios_loss(self):
        # With pretrained_model the architecture is the pickled model's;
        # WeightedPropertiesLoss raises if that model has no C6 head.
        stage_two_c6 = (self.TRAINING.stage_two_loss_weights or {}).get(
            "c6_ratios_weight", 0.0
        )
        if (
            max(self.TRAINING.c6_ratios_weight, stage_two_c6) > 0
            and self.TRAINING.pretrained_model is None
            and not getattr(self.ARCHITECTURE, "c6_ratios_bool", False)
        ):
            raise ValueError(
                "TRAINING.c6_ratios_weight > 0 (either stage) requires "
                "ARCHITECTURE.c6_ratios_bool=True (the model needs a C6 "
                "ratio head to supervise)."
            )
        return self
