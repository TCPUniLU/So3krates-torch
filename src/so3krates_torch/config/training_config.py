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
    compute_avg_num_neighbors: bool = True
    # Multi-head
    convert_to_multihead: bool = False
    num_output_heads: Optional[int] = None
    use_multihead: bool = False

    @model_validator(mode="after")
    def validate_long_range(self):
        if (
            self.electrostatic_energy_bool or self.dispersion_energy_bool
        ) and self.r_max_lr is None:
            raise ValueError(
                "Long-range cutoff 'r_max_lr' must be specified "
                "when electrostatic_energy_bool or "
                "dispersion_energy_bool is True. "
                f"Current: r_max_lr={self.r_max_lr}, "
                f"electrostatic_energy_bool="
                f"{self.electrostatic_energy_bool}, "
                f"dispersion_energy_bool="
                f"{self.dispersion_energy_bool}"
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
    neighbors_lr_cutoff: float = 100.0
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
