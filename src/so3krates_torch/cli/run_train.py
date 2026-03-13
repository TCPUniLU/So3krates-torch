import argparse
import logging

import yaml
import torch

from so3krates_torch.config import TrainConfig
from so3krates_torch.tools.distributed_tools import init_distributed
from so3krates_torch.tools.utils import (
    setup_logger,
)
from so3krates_torch.tools.train import train
from so3krates_torch.tools.finetune import fuse_lora_weights
from so3krates_torch.tools.torch_geometric import seed_everything
import os
from torch.nn.parallel import DistributedDataParallel as DDP

from so3krates_torch.tools.model_setup import (
    _setup_model_for_training,
    set_avg_num_neighbors_in_model,
    set_atomic_energy_shifts_in_model,
    process_config_atomic_energies,
    set_dtype_model,
    handle_finetuning,
)
from so3krates_torch.tools.data_setup import (
    setup_data_loaders,
)
from so3krates_torch.tools.training_setup import (
    setup_loss_function,
    setup_optimizer_and_scheduler,
    setup_training_tools,
    load_checkpoint_if_exists,
)


DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def setup_config_from_yaml(config_path: str) -> dict:
    """Load and parse configuration from YAML file."""
    with open(config_path, "r") as f:
        raw = yaml.safe_load(f)
    validated = TrainConfig.model_validate(raw)
    return validated.model_dump()


def setup_logging(config: dict) -> None:
    """Setup logging based on configuration."""
    log_level = getattr(logging, config["MISC"].get("log_level", "INFO"))
    setup_logger(
        level=log_level,
        tag=config["GENERAL"]["name_exp"],
        directory=config["GENERAL"]["log_dir"],
    )


def init_distributed_from_config(config: dict):
    """Initialise distributed training and return rank info.

    Returns (rank, local_rank, world_size, distributed_flag).
    When distributed is disabled the process group is *not* created
    and all ranks default to 0 / world_size 1.
    """
    distributed = config["MISC"].get("distributed", False)
    rank, local_rank, world_size = init_distributed(
        distributed=distributed,
        launcher=config["MISC"].get("launcher", None),
    )
    is_distributed = distributed and world_size > 1
    return rank, local_rank, world_size, is_distributed


def wrap_model_ddp(model, local_rank):
    """Wrap an already-device-placed model in DDP."""
    device = torch.device(f"cuda:{local_rank}")
    model = model.to(device)
    return DDP(
        model,
        device_ids=[local_rank],
        find_unused_parameters=True,
    )


def run_training(config: dict) -> None:
    """Execute the complete training pipeline."""
    # ---- distributed init (must come first) ----
    rank, local_rank, world_size, distributed = init_distributed_from_config(
        config
    )

    logging.info(
        f"Distributed setup: rank={rank}, local_rank={local_rank}, "
        f"world_size={world_size}, distributed={distributed}"
    )
    # Setup logging (only on rank 0 to avoid duplicate logs)
    if rank == 0:
        logging.getLogger().handlers.clear()
        logging.Logger.manager.loggerDict.clear()
        setup_logging(config)

    if distributed:
        logging.info(
            f"Distributed training: rank={rank}, "
            f"local_rank={local_rank}, world_size={world_size}"
        )

    if rank == 0:
        config_save_path = os.path.join(
            config["GENERAL"]["checkpoints_dir"],
            "config.yaml",
        )
        os.makedirs(
            config["GENERAL"]["checkpoints_dir"],
            exist_ok=True,
        )
        with open(config_save_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Config saved to {config_save_path}")

    dtype_str = config["GENERAL"].get("default_dtype", "float32")
    torch.set_default_dtype(DTYPE_MAP[dtype_str])

    seed = config["GENERAL"].get("seed", 42)
    seed_everything(
        seed,
        deterministic=config["MISC"].get("deterministic_seed", False),
    )
    logging.info(f"Global seed set to {seed}")

    no_checkpoint = config["MISC"].get("no_checkpoint", False)

    if no_checkpoint:
        config["MISC"]["restart_latest"] = False

    # ---- device ----
    if distributed:
        device = torch.device(f"cuda:{local_rank}")
        torch.cuda.set_device(device)
    else:
        device_name = config["MISC"].get("device", "cuda")
        device = torch.device(device_name)
    logging.info(f"Using device: {device}")

    # ---- model creation ----
    model, warm_start = _setup_model_for_training(config, device)

    # ---- data loaders (with DistributedSampler when needed) ----
    (
        train_loader,
        valid_loaders,
        train_sampler,
        avg_num_neighbors,
        num_elements,
        average_atomic_energy_shifts,
    ) = setup_data_loaders(
        config,
        distributed=distributed,
        rank=rank,
        world_size=world_size,
    )

    if warm_start:
        if config["TRAINING"].get("ft_update_avg_num_neighbors", False):
            logging.info(
                "Updating average number of neighbors from "
                "training data for fine-tuning."
            )
            model.avg_num_neighbors = avg_num_neighbors
        else:
            logging.info(
                "Retaining average number of neighbors from "
                "pretrained model for fine-tuning."
            )
            avg_num_neighbors = model.avg_num_neighbors
        atomic_energy_shifts = model.atomic_energy_output_block.energy_shifts
        if config["TRAINING"].get("force_use_average_shifts", False):
            atomic_energy_shifts = average_atomic_energy_shifts
            logging.info(
                "Forcing use of average atomic energy shifts "
                "computed from training data for training."
            )
    else:
        model.avg_num_neighbors = avg_num_neighbors
        atomic_shifts_config = config["ARCHITECTURE"].get(
            "atomic_energy_shifts", None
        )
        if atomic_shifts_config is not None:
            atomic_energy_shifts = process_config_atomic_energies(
                atomic_shifts_config
            )
            logging.info(
                "Using provided atomic energy shifts for " "training."
            )
        else:
            atomic_energy_shifts = average_atomic_energy_shifts
            logging.info(
                "Using average atomic energy shifts computed "
                "from training data for training."
            )

    # Setup finetuning if specified
    if config["TRAINING"].get("finetune_choice", None):
        model = handle_finetuning(config, model, num_elements, str(device))

    logging.info(f"Atomic energy shifts: {atomic_energy_shifts}")
    set_atomic_energy_shifts_in_model(model, atomic_energy_shifts)
    set_avg_num_neighbors_in_model(model, avg_num_neighbors)
    set_dtype_model(model, config["GENERAL"].get("default_dtype", "float32"))

    # ---- wrap model in DDP (after all weight mutations) ----
    ddp_model = None
    if distributed:
        ddp_model = wrap_model_ddp(model, local_rank)

    # Setup loss function
    loss_fn = setup_loss_function(config)

    # Setup optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, config)

    # Setup training tools
    logger, checkpoint_handler, ema = setup_training_tools(config, model)

    # Load checkpoint if exists
    start_epoch = 0
    if not warm_start:
        start_epoch = load_checkpoint_if_exists(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_handler=checkpoint_handler,
            ema=ema,
            device=device,
            config=config,
        )

    if rank == 0:
        _first_batch = next(iter(train_loader))
        for _key in ["energy", "forces"]:
            _val = getattr(_first_batch, _key, None)
            if _val is not None and not torch.isfinite(_val).all():
                raise RuntimeError(
                    f"Training data contains NaN/Inf"
                    f" in '{_key}'. Check your dataset."
                )

    logging.info("Model, data loaders, and training components initialized")
    if start_epoch > 0:
        logging.info(f"Resuming training from epoch {start_epoch}")
    else:
        logging.info("Starting fresh training.")

    # Training parameters
    max_num_epochs = config["TRAINING"]["num_epochs"]
    eval_interval = config["TRAINING"].get("eval_interval", 1)
    patience = config["TRAINING"].get("patience", 50)
    max_grad_norm = config["TRAINING"].get("clip_grad", 10.0)
    log_wandb = config["MISC"].get("log_wandb", False)
    save_all_checkpoints = config["MISC"].get("keep_checkpoints", False)

    output_args = {
        "forces": True,
        "virials": config["GENERAL"].get("compute_stress", False),
        "stress": config["GENERAL"].get("compute_stress", False),
    }

    log_errors = config["MISC"].get("error_table", "PerAtomMAE")

    if config["ARCHITECTURE"].get("use_multihead", False):
        logging.info(
            "Enabling head selection for multi-head model " "during training."
        )
        model.select_heads = True

    logging.info("Starting training loop...")
    train(
        model=model,
        loss_fn=loss_fn,
        train_loader=train_loader,
        valid_loaders=valid_loaders,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
        start_epoch=start_epoch,
        max_num_epochs=max_num_epochs,
        patience=patience,
        checkpoint_handler=checkpoint_handler,
        logger_train=logger["train"],
        logger_valid=logger["valid"],
        eval_interval=eval_interval,
        output_args=output_args,
        device=device,
        log_errors=log_errors,
        ema=ema,
        max_grad_norm=max_grad_norm,
        log_wandb=log_wandb,
        distributed=distributed,
        save_all_checkpoints=save_all_checkpoints,
        plotter=None,
        distributed_model=ddp_model,
        train_sampler=train_sampler,
        rank=rank,
        early_stopping_min_delta=config["TRAINING"].get(
            "early_stopping_min_delta", 0.0
        ),
        early_stopping_warmup=config["TRAINING"].get(
            "early_stopping_warmup", 0
        ),
        config=config,
    )
    logging.info("Training completed successfully!")

    if config["ARCHITECTURE"].get("use_multihead", False):
        logging.info(
            "Disabling head selection for multi-head model " "after training."
        )
        model.select_heads = False

    if config["TRAINING"].get("finetune_choice", None) in [
        "dora",
        "lora",
        "vera",
    ]:
        logging.info("Fusing LoRA weights into base model for saving...")
        model = fuse_lora_weights(model)
        logging.info("LoRA weights fused successfully.")

    # Only rank 0 saves the final model
    if rank == 0:
        final_model_path = f'{config["GENERAL"]["name_exp"]}.pth'
        torch.save(model.state_dict(), final_model_path)
        torch.save(model, final_model_path.replace(".pth", ".model"))


def run_dry_run(config: dict) -> None:
    """Validate config and run one forward pass, then exit."""
    logging.getLogger().handlers.clear()
    logging.Logger.manager.loggerDict.clear()
    setup_logging(config)
    logging.info(
        "Dry-run mode: validating config and running one forward pass"
    )

    dtype_str = config["GENERAL"].get("default_dtype", "float32")
    torch.set_default_dtype(DTYPE_MAP[dtype_str])

    device_name = config["MISC"].get("device", "cuda")
    device = torch.device(device_name)
    logging.info(f"Using device: {device}")

    model, _ = _setup_model_for_training(config, device)

    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )
    logging.info(f"Total parameters: {total_params}")
    logging.info(f"Trainable parameters: {trainable_params}")

    (
        train_loader,
        _valid_loaders,
        _train_sampler,
        avg_num_neighbors,
        _num_elements,
        average_atomic_energy_shifts,
    ) = setup_data_loaders(config, distributed=False, rank=0, world_size=1)

    set_atomic_energy_shifts_in_model(model, average_atomic_energy_shifts)
    set_avg_num_neighbors_in_model(model, avg_num_neighbors)
    set_dtype_model(model, dtype_str)

    batch = next(iter(train_loader))
    batch = batch.to(device)
    batch_dict = batch.to_dict()

    logging.info("Running one forward pass...")
    with torch.no_grad():
        output = model(
            batch_dict,
            training=False,
            compute_force=True,
            compute_virials=False,
            compute_stress=False,
        )
    logging.info(
        f"Forward pass successful. Output keys: {list(output.keys())}"
    )
    logging.info("Dry-run completed successfully. Exiting.")


def main():
    parser = argparse.ArgumentParser(description="Train SO3LR model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help=(
            "Validate config and run one forward pass,"
            " then exit without training"
        ),
    )
    args = parser.parse_args()

    # Load configuration
    config = setup_config_from_yaml(args.config)

    # Run the training pipeline
    if args.dry_run:
        run_dry_run(config)
    else:
        run_training(config)


if __name__ == "__main__":
    main()
