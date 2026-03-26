import logging
from typing import Optional

import torch

from so3krates_torch.modules.loss import (
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesHirshfeldLoss,
    WeightedEnergyForcesDipoleHirshfeldLoss,
)
from so3krates_torch.tools.checkpoint import (
    CheckpointHandler,
    CheckpointState,
)
from so3krates_torch.tools.utils import MetricsLogger
from torch_ema import ExponentialMovingAverage


def setup_loss_function(config: dict) -> torch.nn.Module:
    """Setup loss function based on configuration."""
    loss_config = config["TRAINING"]

    # Get loss weights
    energy_w = loss_config.get("energy_weight", 1.0)
    forces_w = loss_config.get("forces_weight", 1000.0)
    dipole_w = loss_config.get("dipole_weight", 0.0)
    hirshfeld_w = loss_config.get("hirshfeld_weight", 0.0)

    # Check if explicit loss type is specified
    loss_type = loss_config.get("loss_type", "auto")

    if loss_type == "auto":
        # Auto-determine loss function type based on weights
        has_dipole = dipole_w > 0
        has_hirshfeld = hirshfeld_w > 0

        if has_dipole and has_hirshfeld:
            loss_fn = WeightedEnergyForcesDipoleHirshfeldLoss(
                energy_weight=energy_w,
                forces_weight=forces_w,
                dipole_weight=dipole_w,
                hirshfeld_weight=hirshfeld_w,
            )
            logging.info(
                "Auto-selected " "WeightedEnergyForcesDipoleHirshfeldLoss"
            )
        elif has_dipole:
            loss_fn = WeightedEnergyForcesDipoleLoss(
                energy_weight=energy_w,
                forces_weight=forces_w,
                dipole_weight=dipole_w,
            )
            logging.info("Auto-selected WeightedEnergyForcesDipoleLoss")
        elif has_hirshfeld:
            loss_fn = WeightedEnergyForcesHirshfeldLoss(
                energy_weight=energy_w,
                forces_weight=forces_w,
                hirshfeld_weight=hirshfeld_w,
            )
            logging.info("Auto-selected WeightedEnergyForcesHirshfeldLoss")
        else:
            # Default to basic energy-forces loss from MACE
            loss_fn = WeightedEnergyForcesLoss(
                energy_weight=energy_w, forces_weight=forces_w
            )
            logging.info("Auto-selected WeightedEnergyForcesLoss")
    else:
        # Explicit loss type specification
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
        else:
            supported_types = [
                "auto",
                "energy_forces",
                "energy_forces_dipole",
                "energy_forces_hirshfeld",
                "energy_forces_dipole_hirshfeld",
            ]
            raise ValueError(
                f"Unknown loss_type: {loss_type}. "
                f"Supported types: {supported_types}"
            )

        logging.info(f"Explicit loss type: {loss_type}")

    logging.info(
        f"Loss weights: energy={energy_w}, forces={forces_w}, "
        f"dipole={dipole_w}, hirshfeld={hirshfeld_w}"
    )

    return loss_fn


def setup_optimizer_and_scheduler(
    model: torch.nn.Module,
    config: dict,
) -> tuple:
    """Setup optimizer and learning rate scheduler."""
    train_config = config["TRAINING"]

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
                ],
                lr=lr,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )
        else:
            optimizer = torch.optim.Adam(
                model.parameters(),
                lr=lr,
                weight_decay=weight_decay,
                amsgrad=amsgrad,
            )
    elif optimizer_name == "adamw":
        optimizer = torch.optim.AdamW(
            model.parameters(),
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
        warmup_steps = train_config.get("warmup_steps", 100)
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
) -> int:
    """
    Load checkpoint if it exists and return the starting epoch.
    Returns 0 if no checkpoint is found.
    """
    # Check if we should restart from latest checkpoint
    restart_latest = config["MISC"].get("restart_latest", True)

    if not restart_latest:
        logging.info("Checkpoint restart disabled, starting from epoch 0")
        return 0

    # Create checkpoint state
    checkpoint_state = CheckpointState(
        model=model, optimizer=optimizer, lr_scheduler=lr_scheduler
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

    except Exception as e:
        logging.warning(f"Error loading checkpoint: {e}")
        logging.info("Starting from epoch 0")
        return 0
