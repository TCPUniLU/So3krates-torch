import logging
import dataclasses
from typing import Dict, List, Optional

import torch

from so3krates_torch.modules.loss import (
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesHirshfeldLoss,
    WeightedEnergyForcesDipoleHirshfeldLoss,
    WeightedEnergyForcesChargesLoss,
    WeightedEnergyForcesChargesHirshfeldLoss,
    WeightedPropertiesLoss,
    AdaptiveRobustParams,
)
from so3krates_torch.tools.checkpoint import (
    CheckpointHandler,
    CheckpointState,
    LossStateMismatchError,
)
from so3krates_torch.tools.utils import MetricsLogger
from torch_ema import ExponentialMovingAverage


def get_loss_property_weights(config: dict) -> Dict[str, float]:
    """Return the configured loss weight for each loss-relevant
    property, using the same defaults as setup_loss_function."""
    loss_config = config["TRAINING"]
    return {
        "energy": loss_config.get("energy_weight", 1.0),
        "forces": loss_config.get("forces_weight", 1000.0),
        "dipole": loss_config.get("dipole_weight", 0.0),
        "hirshfeld_ratios": loss_config.get(
            "hirshfeld_weight", loss_config.get("a0_ratios_weight", 0.0)
        ),
        "c6_ratios": loss_config.get("c6_ratios_weight", 0.0),
        "charges": loss_config.get("charges_weight", 0.0),
    }


def _stage_two_config(config: dict) -> Optional[dict]:
    """Config with the stage-two weights merged over the stage-one ones
    (unlisted keys keep their stage-one value); None without stage two."""
    loss_config = config["TRAINING"]
    overrides = loss_config.get("stage_two_loss_weights")
    if loss_config.get("stage_two_start_epoch") is None or overrides is None:
        return None
    overrides = dict(overrides)
    if "a0_ratios_weight" in overrides:
        overrides["hirshfeld_weight"] = overrides.pop("a0_ratios_weight")
    return {**config, "TRAINING": {**loss_config, **overrides}}


def get_required_loss_properties(config: dict) -> List[str]:
    """Properties with a non-zero loss weight in either training stage,
    i.e. the labels the training data must provide."""
    configs = [config]
    stage_two = _stage_two_config(config)
    if stage_two is not None:
        configs.append(stage_two)
    required = []
    for cfg in configs:
        for name, weight in get_loss_property_weights(cfg).items():
            if weight > 0 and name not in required:
                required.append(name)
    return required


@dataclasses.dataclass
class StageTwoLoss:
    start_epoch: int
    loss_fn: torch.nn.Module
    valid_loss_fn: torch.nn.Module


@dataclasses.dataclass
class LossFunctions:
    train: torch.nn.Module
    valid: torch.nn.Module
    stage_two: Optional[StageTwoLoss] = None

    def modules(self) -> List[torch.nn.Module]:
        mods = [self.train, self.valid]
        if self.stage_two is not None:
            mods += [self.stage_two.loss_fn, self.stage_two.valid_loss_fn]
        return mods


def _validation_loss(
    config: dict,
    train_loss: torch.nn.Module,
    adaptive: Optional[AdaptiveRobustParams],
) -> torch.nn.Module:
    """JAX make_val_fn: the adaptive loss validates with the training
    loss; otherwise use_robust_loss_validation / _alpha_validation
    (default: the training settings) pick the validation penalty."""
    loss_config = config["TRAINING"]
    if loss_config.get("use_adaptive_robust_loss", False):
        return train_loss
    use_fixed = loss_config.get("use_robust_loss", False)
    alpha = loss_config.get("robust_loss_alpha", 1.0)
    val_fixed = loss_config.get("use_robust_loss_validation")
    val_alpha = loss_config.get("robust_loss_alpha_validation")
    val_fixed = use_fixed if val_fixed is None else val_fixed
    val_alpha = alpha if val_alpha is None else val_alpha
    if val_fixed == use_fixed and (not val_fixed or val_alpha == alpha):
        return train_loss
    val_config = {
        **config,
        "TRAINING": {
            **loss_config,
            "use_robust_loss": val_fixed,
            "robust_loss_alpha": val_alpha,
        },
    }
    logging.info("Validation loss uses its own robust-loss settings:")
    return setup_loss_function(val_config, adaptive=adaptive)


def _check_validation_settings(loss_config: dict) -> None:
    """Warn about validation robust settings that have no effect, and
    reject them for legacy loss types with an error naming them."""
    val_flag = loss_config.get("use_robust_loss_validation")
    val_alpha = loss_config.get("robust_loss_alpha_validation")
    loss_type = loss_config.get("loss_type", "auto")
    legacy = loss_type not in ("auto", "properties")
    training_robust = loss_config.get(
        "use_robust_loss", False
    ) or loss_config.get("use_adaptive_robust_loss", False)
    # With robust training the training-side error (naming
    # use_robust_loss / use_adaptive_robust_loss) fires instead.
    if val_flag and legacy and not training_robust:
        raise ValueError(
            f"loss_type '{loss_type}' does not support "
            "use_robust_loss_validation. Use loss_type 'auto' or "
            "'properties'."
        )
    if loss_config.get("use_adaptive_robust_loss", False):
        if val_flag is not None or val_alpha is not None:
            logging.warning(
                "use_robust_loss_validation / robust_loss_alpha_validation "
                "are ignored with use_adaptive_robust_loss: validation uses "
                "the training loss (learned alpha/scale), as in JAX so3lr."
            )
        return
    effective = (
        loss_config.get("use_robust_loss", False)
        if val_flag is None
        else val_flag
    )
    if val_alpha is not None and not effective:
        hint = (
            ""
            if legacy
            else " Set use_robust_loss_validation: true to validate with "
            "the fixed robust loss."
        )
        logging.warning(
            "robust_loss_alpha_validation has no effect: the validation loss "
            "uses the squared error." + hint
        )


def _check_stage_two_reachable(loss_config: dict) -> None:
    """Warn when stage_two_start_epoch lies at or beyond num_epochs (epochs
    run 0 .. num_epochs-1). A warning, not an error: num_epochs may be
    raised on a later restart."""
    start = loss_config.get("stage_two_start_epoch")
    num_epochs = loss_config.get("num_epochs")
    if start is None or num_epochs is None or start < num_epochs:
        return
    logging.warning(
        f"stage two never starts: stage_two_start_epoch={start} but "
        f"num_epochs={num_epochs} (epochs 0..{num_epochs - 1}). Lower "
        "stage_two_start_epoch or raise num_epochs."
    )


def setup_loss_functions(config: dict) -> LossFunctions:
    """Training and validation losses for both training stages."""
    loss_config = config["TRAINING"]
    _check_validation_settings(loss_config)
    _check_stage_two_reachable(loss_config)
    stage_two_config = _stage_two_config(config)
    loss_type = loss_config.get("loss_type", "auto")
    if stage_two_config is not None and loss_type not in (
        "auto",
        "properties",
    ):
        raise ValueError(
            f"loss_type '{loss_type}' does not support stage_two_loss_weights"
            " / stage_two_start_epoch. Use loss_type 'auto' or 'properties'."
        )

    shared = None
    if (
        loss_config.get("use_adaptive_robust_loss", False)
        and stage_two_config is not None
    ):
        # One set of alpha/scale for both stages, covering every property
        # weighted in either (stored by name, unlike JAX's positional).
        shared = AdaptiveRobustParams(
            get_required_loss_properties(config),
            loss_config.get("adaptive_robust_loss_init_alpha", 1.0),
            loss_config.get("adaptive_robust_loss_init_scale", 1.0),
        )

    train = setup_loss_function(config, adaptive=shared)
    fns = LossFunctions(
        train=train, valid=_validation_loss(config, train, shared)
    )
    if stage_two_config is not None:
        logging.info(
            "Two-stage loss weights: stage two starts at epoch "
            f"{loss_config['stage_two_start_epoch']}:"
        )
        train_two = setup_loss_function(stage_two_config, adaptive=shared)
        fns.stage_two = StageTwoLoss(
            start_epoch=int(loss_config["stage_two_start_epoch"]),
            loss_fn=train_two,
            valid_loss_fn=_validation_loss(
                stage_two_config, train_two, shared
            ),
        )
    return fns


def check_loss_outputs(
    model: torch.nn.Module, loss_fns: LossFunctions, batch
) -> None:
    """Fail at startup, not at the stage-two switch: run one forward pass
    and check that every loss (both stages, train and validation) finds
    the model outputs it needs."""
    output = model(batch.to_dict(), training=False, compute_force=True)
    for module in loss_fns.modules():
        if hasattr(module, "check_outputs"):
            module.check_outputs(output)


def setup_loss_function(
    config: dict, adaptive: Optional[AdaptiveRobustParams] = None
) -> torch.nn.Module:
    """Setup loss function based on configuration."""
    loss_config = config["TRAINING"]

    # Get loss weights
    weights = get_loss_property_weights(config)
    energy_w = weights["energy"]
    forces_w = weights["forces"]
    dipole_w = weights["dipole"]
    hirshfeld_w = weights["hirshfeld_ratios"]
    charges_w = weights["charges"]

    # Check if explicit loss type is specified
    loss_type = loss_config.get("loss_type", "auto")
    reduction = loss_config.get("loss_reduction", "flat")
    use_fixed = loss_config.get("use_robust_loss", False)
    use_adaptive = loss_config.get("use_adaptive_robust_loss", False)
    if use_fixed and use_adaptive:
        raise ValueError(
            "Cannot use both 'use_robust_loss' and "
            "'use_adaptive_robust_loss'."
        )
    robust = "adaptive" if use_adaptive else "fixed" if use_fixed else "none"

    if loss_type in ("auto", "properties"):
        # One term per non-zero weight (see WeightedPropertiesLoss).
        loss_fn = WeightedPropertiesLoss(
            weights,
            reduction=reduction,
            robust=robust,
            robust_alpha=loss_config.get("robust_loss_alpha", 1.0),
            adaptive_init_alpha=loss_config.get(
                "adaptive_robust_loss_init_alpha", 1.0
            ),
            adaptive_init_scale=loss_config.get(
                "adaptive_robust_loss_init_scale", 1.0
            ),
            adaptive=adaptive,
        )
        logging.info(f"Selected {loss_fn}")
    else:
        # Explicit loss type specification
        if robust != "none":
            raise ValueError(
                f"loss_type '{loss_type}' does not support the robust loss "
                "(use_robust_loss / use_adaptive_robust_loss). Use loss_type "
                "'auto' or 'properties'."
            )
        if reduction != "flat":
            raise ValueError(
                f"loss_type '{loss_type}' only supports loss_reduction "
                f"'flat' (got '{reduction}'). Use loss_type 'auto' or "
                "'properties'."
            )
        if weights["c6_ratios"] > 0:
            raise ValueError(
                f"loss_type '{loss_type}' has no C6 ratio term but "
                "c6_ratios_weight > 0. Use loss_type 'auto' or "
                "'properties'."
            )
        if loss_type == "energy_forces":
            loss_fn = WeightedEnergyForcesLoss(
                energy_weight=energy_w, forces_weight=forces_w
            )
        elif loss_type == "energy_forces_dipole":
            loss_fn = WeightedEnergyForcesDipoleLoss(
                energy_weight=energy_w,
                forces_weight=forces_w,
                dipole_weight=dipole_w,
            )
        elif loss_type == "energy_forces_hirshfeld":
            loss_fn = WeightedEnergyForcesHirshfeldLoss(
                energy_weight=energy_w,
                forces_weight=forces_w,
                hirshfeld_weight=hirshfeld_w,
            )
        elif loss_type == "energy_forces_dipole_hirshfeld":
            loss_fn = WeightedEnergyForcesDipoleHirshfeldLoss(
                energy_weight=energy_w,
                forces_weight=forces_w,
                dipole_weight=dipole_w,
                hirshfeld_weight=hirshfeld_w,
            )
        elif loss_type == "energy_forces_charges":
            loss_fn = WeightedEnergyForcesChargesLoss(
                energy_weight=energy_w,
                forces_weight=forces_w,
                charges_weight=charges_w,
            )
        elif loss_type == "energy_forces_charges_hirshfeld":
            loss_fn = WeightedEnergyForcesChargesHirshfeldLoss(
                energy_weight=energy_w,
                forces_weight=forces_w,
                charges_weight=charges_w,
                hirshfeld_weight=hirshfeld_w,
            )
        else:
            supported_types = [
                "auto",
                "properties",
                "energy_forces",
                "energy_forces_dipole",
                "energy_forces_hirshfeld",
                "energy_forces_dipole_hirshfeld",
                "energy_forces_charges",
                "energy_forces_charges_hirshfeld",
            ]
            raise ValueError(
                f"Unknown loss_type: {loss_type}. "
                f"Supported types: {supported_types}"
            )

        logging.info(f"Explicit loss type: {loss_type}")

    logging.info(
        f"Loss weights: energy={energy_w}, forces={forces_w}, "
        f"dipole={dipole_w}, hirshfeld={hirshfeld_w}, charges={charges_w}, "
        f"c6_ratios={weights['c6_ratios']}"
    )

    return loss_fn


def setup_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: dict,
    loss_fn: Optional[torch.nn.Module] = None,
) -> tuple:
    """Setup optimizer and learning rate scheduler.

    Learned loss parameters (adaptive robust loss) get their own param
    group with the same lr and weight decay, like JAX so3lr's single
    optimizer over the full params tree. Without them the optimizer is
    unchanged (one group), so existing checkpoints stay loadable."""
    train_config = config["TRAINING"]
    loss_params = (
        [p for p in loss_fn.parameters() if p.requires_grad]
        if loss_fn is not None
        else []
    )

    def _with_loss_params(groups):
        if not loss_params:
            return groups
        if not isinstance(groups[0], dict):
            groups = [{"params": list(groups)}]
        return list(groups) + [{"params": loss_params}]

    optimizer_name = train_config.get("optimizer", "adam").lower()
    lr = train_config["lr"]
    weight_decay = train_config.get("weight_decay", 0.0)
    amsgrad = train_config.get("amsgrad", False)

    use_lora_plus = train_config.get("use_lora_plus", False)
    lora_B_lr = train_config.get("lora_B_lr", None)

    if optimizer_name == "adam":
        if use_lora_plus:
            assert (
                lora_B_lr is not None
            ), "lora_B_lr must be provided for LoRA+"
            # for LoRA+ adjust learning rate for A and B matrices
            optimizer = torch.optim.Adam(
                _with_loss_params(
                    [
                        {
                            "params": [
                                p
                                for n, p in model.named_parameters()
                                if "lora_A" in n
                            ],
                            "lr": lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in model.named_parameters()
                                if "lora_B" in n
                            ],
                            "lr": lora_B_lr,
                        },
                        {
                            "params": [
                                p
                                for n, p in model.named_parameters()
                                if "lora_A" not in n and "lora_B" not in n
                            ]
                        },
                    ]
                ),
                lr=lr,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )
        else:
            optimizer = torch.optim.Adam(
                _with_loss_params(list(model.parameters())),
                lr=lr,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            _with_loss_params(list(model.parameters())),
            lr=lr,
            weight_decay=weight_decay,
            betas=train_config.get("betas", (0.9, 0.999)),
            eps=train_config.get("eps", 1e-8),
        )
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Setup learning rate scheduler
    scheduler_name = train_config.get("scheduler", "exponential_decay")

    scheduler_args = train_config.get("scheduler_args", {})

    if scheduler_name == "exponential_decay":
        gamma = train_config.get(
            "lr_scheduler_gamma",
            scheduler_args.get("gamma", 0.9993),
        )
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer,
            gamma=gamma,
        )
    elif scheduler_name == "lambda":
        gamma = scheduler_args.get("gamma", 0.85)
        step_size = scheduler_args.get("step_size", 33)
        lr_lambda = lambda epoch: gamma ** (epoch / step_size)
        lr_scheduler = torch.optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=lr_lambda
        )
    elif scheduler_name == "step":
        lr_scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            **{
                "step_size": scheduler_args.get("step_size", 1000),
                "gamma": scheduler_args.get("gamma", 0.1),
            },
        )
    elif scheduler_name == "reduce_on_plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=train_config.get("scheduler_patience", 5),
            factor=train_config.get("lr_factor", 0.85),
            min_lr=1e-6,
        )
    elif scheduler_name == "cosine_annealing":
        T_max = scheduler_args.get(
            "T_max",
            train_config.get("num_epochs", 1000),
        )
        eta_min = scheduler_args.get("eta_min", 0.0)
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=T_max,
            eta_min=eta_min,
        )
    elif scheduler_name == "warmup_cosine":
        warmup_steps = train_config.get("warmup_steps", 0)
        T_max = scheduler_args.get(
            "T_max",
            train_config.get("num_epochs", 1000),
        )
        eta_min = scheduler_args.get("eta_min", 0.0)
        warmup_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer,
            start_factor=1e-3,
            end_factor=1.0,
            total_iters=warmup_steps,
        )
        cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer,
            T_max=max(T_max - warmup_steps, 1),
            eta_min=eta_min,
        )
        lr_scheduler = torch.optim.lr_scheduler.SequentialLR(
            optimizer,
            schedulers=[
                warmup_scheduler,
                cosine_scheduler,
            ],
            milestones=[warmup_steps],
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    logging.info(f"Optimizer: {optimizer_name}, Learning rate: {lr}")
    logging.info(f"Scheduler: {scheduler_name}")

    return optimizer, lr_scheduler


def setup_training_tools(config: dict, model: torch.nn.Module) -> tuple:
    """Setup training tools like EMA, checkpoint handler, and logger."""
    # Setup metrics logger
    train_logger = MetricsLogger(
        directory=config["GENERAL"]["log_dir"],
        tag=config["GENERAL"]["name_exp"] + "_train",
    )
    valid_logger = MetricsLogger(
        directory=config["GENERAL"]["log_dir"],
        tag=config["GENERAL"]["name_exp"] + "_valid",
    )
    logger = {"train": train_logger, "valid": valid_logger}

    # Setup checkpoint handler
    checkpoint_handler = CheckpointHandler(
        directory=config["GENERAL"]["checkpoints_dir"],
        tag=config["GENERAL"]["name_exp"],
    )

    # Setup EMA if enabled
    ema = None
    if config["TRAINING"].get("ema", False):
        ema_decay = config["TRAINING"].get("ema_decay", 0.99)
        ema = ExponentialMovingAverage(model.parameters(), decay=ema_decay)
        logging.info(f"Using EMA with decay: {ema_decay}")

    return logger, checkpoint_handler, ema


def load_checkpoint_if_exists(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    lr_scheduler: torch.optim.lr_scheduler._LRScheduler,
    checkpoint_handler: CheckpointHandler,
    ema: ExponentialMovingAverage,
    device: torch.device,
    config: dict,
    loss_fn: Optional[torch.nn.Module] = None,
    resume_state: Optional[dict] = None,
) -> int:
    """
    Load checkpoint if it exists and return the starting epoch.
    `resume_state`, if given, receives {"stage_two_active": bool | None}.
    Returns 0 if no checkpoint is found.
    """
    # Check if we should restart from latest checkpoint
    restart_latest = config["MISC"].get("restart_latest", True)

    if not restart_latest:
        logging.info("Checkpoint restart disabled, starting from epoch 0")
        return 0

    # Create checkpoint state
    checkpoint_state = CheckpointState(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        loss_fn=loss_fn,
    )

    # Try to load latest checkpoint
    try:
        start_epoch = checkpoint_handler.load_latest(
            state=checkpoint_state,
            swa=False,  # Don't load SWA checkpoint initially
            device=device,
            strict=False,
        )

        if start_epoch is not None:
            logging.info(f"Loaded checkpoint from epoch {start_epoch}")
            if resume_state is not None:
                resume_state["stage_two_active"] = (
                    checkpoint_state.stage_two_active
                )

            # Load EMA state if available and EMA is enabled
            if ema is not None:
                try:
                    # Try to load EMA state from checkpoint
                    io_handler = checkpoint_handler.io
                    latest_path = io_handler._get_latest_checkpoint_path(
                        swa=False
                    )
                    if latest_path:
                        checkpoint_data = torch.load(
                            latest_path, map_location=device
                        )
                        if "ema" in checkpoint_data:
                            ema.load_state_dict(checkpoint_data["ema"])
                            logging.info("Loaded EMA state from checkpoint")
                except Exception as e:
                    logging.warning(f"Could not load EMA state: {e}")

            return start_epoch + 1  # Start from next epoch
        else:
            logging.info("No checkpoint found, starting from epoch 0")
            return 0

    except LossStateMismatchError:
        raise
    except Exception as e:
        logging.warning(f"Error loading checkpoint: {e}")
        logging.info("Starting from epoch 0")
        return 0
