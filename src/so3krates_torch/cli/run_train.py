import argparse
import logging
import yaml
import torch
import torch.nn as nn
from ase.io import read
import random
from typing import Tuple, Union
from so3krates_torch.modules.models import SO3LR, MultiHeadSO3LR
from so3krates_torch.tools.utils import (
    create_dataloader_from_list,
    create_data_from_list,
    create_dataloader_from_data,
    create_configs_from_list,
    create_data_from_configs,
)
from so3krates_torch.modules.loss import (
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesHirshfeldLoss,
    WeightedEnergyForcesDipoleHirshfeldLoss,
)
from so3krates_torch.data.utils import KeySpecification, compute_average_E0s
from so3krates_torch.tools.utils import (
    AtomicNumberTable,
    MetricsLogger,
    compute_avg_num_neighbors,
    setup_logger,
)
from so3krates_torch.tools.checkpoint import (
    CheckpointHandler,
    CheckpointState,
)
from torch_ema import ExponentialMovingAverage
from so3krates_torch.tools.train import train
from so3krates_torch.tools.finetune import fuse_lora_weights, setup_finetuning
import os


DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


def setup_config_from_yaml(config_path: str) -> dict:
    """Load and parse configuration from YAML file."""
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    return config


def setup_logging(config: dict) -> None:
    """Setup logging based on configuration."""
    log_level = getattr(logging, config["MISC"].get("log_level", "INFO"))
    setup_logger(
        level=log_level,
        tag=config["GENERAL"]["name_exp"],
        directory=config["GENERAL"]["log_dir"],
    )


def create_model(config: dict, device: torch.device) -> SO3LR:
    """Create and initialize the SO3LR model."""
    arch_config = config["ARCHITECTURE"]

    # Map YAML parameters to model parameters
    model_params = {
        # Base So3krates parameters
        "r_max": arch_config.get("r_max", 4.5),  # cutoff -> r_max
        "num_radial_basis_fn": arch_config.get("num_radial_basis_fn", 32),
        "degrees": arch_config["degrees"],
        "num_features": arch_config.get("num_features", 128),
        "num_heads": arch_config.get("num_heads", 4),
        "num_layers": arch_config.get("num_layers", 3),
        "num_elements": 118,  # Default for periodic table
        "energy_regression_dim": arch_config.get("energy_regression_dim", 128),
        "message_normalization": arch_config.get(
            "message_normalization", "avg_num_neighbors"
        ),
        "radial_basis_fn": arch_config.get("radial_basis_fn", "bernstein"),
        "energy_learn_atomic_type_shifts": arch_config.get(
            "energy_learn_atomic_type_shifts", False
        ),
        "energy_learn_atomic_type_scales": arch_config.get(
            "energy_learn_atomic_type_scales", False
        ),
        "layer_normalization_1": arch_config.get(
            "layer_normalization_1", False
        ),
        "layer_normalization_2": arch_config.get(
            "layer_normalization_2", False
        ),
        "residual_mlp_1": arch_config.get("residual_mlp_1", False),
        "residual_mlp_2": arch_config.get("residual_mlp_2", False),
        "use_charge_embed": arch_config.get("use_charge_embed", False),
        "use_spin_embed": arch_config.get("use_spin_embed", False),
        "qk_non_linearity": arch_config.get("qk_non_linearity", "identity"),
        "cutoff_fn": arch_config.get("cutoff_fn", "cosine"),
        "activation_fn": arch_config.get("activation_fn", "silu"),
        "energy_activation_fn": arch_config.get(
            "energy_activation_fn", "silu"
        ),
        "seed": config["GENERAL"].get("seed", 42),
        "device": device,
        "dtype": config["GENERAL"].get("default_dtype", "float32"),
        "layers_behave_like_identity_fn_at_init": arch_config.get(
            "layers_behave_like_identity_fn_at_init", False
        ),
        "output_is_zero_at_init": arch_config.get(
            "output_is_zero_at_init", False
        ),
        "input_convention": arch_config.get("input_convention", "positions"),
        # SO3LR specific parameters
        "zbl_repulsion_bool": arch_config.get("zbl_repulsion_bool", True),
        "electrostatic_energy_bool": arch_config.get(
            "electrostatic_energy_bool", True
        ),
        "electrostatic_energy_scale": arch_config.get(
            "electrostatic_energy_scale", 4.0
        ),
        "dispersion_energy_bool": arch_config.get(
            "dispersion_energy_bool", True
        ),
        "dispersion_energy_scale": arch_config.get(
            "dispersion_energy_scale", 1.2
        ),
        "dispersion_energy_cutoff_lr_damping": arch_config.get(
            "dispersion_energy_cutoff_lr_damping", None
        ),
        "r_max_lr": arch_config.get("r_max_lr", None),
        "neighborlist_format_lr": arch_config.get(
            "neighborlist_format_lr", "sparse"
        ),
    }

    if arch_config.get("convert_to_multihead", False):
        model_params["num_output_heads"] = arch_config.get(
            "num_output_heads", None
        )
        assert (
            model_params["num_output_heads"] is not None
        ), "num_output_heads must be specified when using convert_to_multihead"
        logging.info(
            f"Creating Multi-Head SO3LR model with {model_params['num_output_heads']} heads"
        )
        model = MultiHeadSO3LR(**model_params)
    else:
        logging.info(f"Createing SO3LR model")
        model = SO3LR(**model_params)

    model = model.to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logging.info(f"The model has {total_params} parameters")
    return model


def select_valid_subset(
    data: list,
    valid_ratio: float,
    num_train: int = None,
    num_valid: int = None,
) -> Tuple[list, list]:
    n_valid = int(len(data) * valid_ratio)
    n_train = len(data) - n_valid
    if num_train is not None:
        n_train = min(n_train, num_train)
    if num_valid is not None:
        n_valid = min(n_valid, num_valid)
    random.shuffle(data)

    return data[:n_train], data[n_train : n_train + n_valid]


def setup_data_loaders(config: dict) -> tuple:
    """Setup training and validation data loaders."""
    # Key specification for data loading
    keyspec = KeySpecification(
        info_keys={
            "energy": "REF_energy",
            "dipole": "REF_dipole",
            "total_charge": "charge",
        },
        arrays_keys={
            "hirshfeld_ratios": "REF_hirsh_ratios",
            "forces": "REF_forces",
        },
    )
    # Create data loaders
    batch_size = config["TRAINING"]["batch_size"]
    valid_batch_size = config["TRAINING"]["valid_batch_size"]
    r_max = config["ARCHITECTURE"].get("r_max", None)
    r_max_lr = config["ARCHITECTURE"].get("r_max_lr", None)

    heads = config["TRAINING"].get("heads", None)
    if heads is not None:
        train_configs = []
        val_data = {}
        for head_name, head_config in heads.items():
            head_data = read(head_config["path_to_train_data"], index=":")
            head_valid_path = head_config.get("path_to_val_data", None)
            if head_valid_path:
                head_val_data = read(head_valid_path, index=":")
                head_train_data = head_data
            else:
                valid_ratio = head_config.get("valid_ratio", 0.1)
                num_train = head_config.get("num_train", None)
                num_valid = head_config.get("num_valid", None)
                head_train_data, head_val_data = select_valid_subset(
                    head_data, valid_ratio, num_train, num_valid
                )
            logging.info(
                f"Head {head_name} - Training set size: {len(head_train_data)}"
            )
            logging.info(
                f"Head {head_name} - Validation set size: {len(head_val_data)}"
            )

            head_config_list_train = create_configs_from_list(
                atoms_list=head_train_data,
                key_specification=keyspec,
                head_name=head_name,
            )
            train_configs.extend(head_config_list_train)

            head_config_list_val = create_data_from_list(
                head_val_data,
                r_max=r_max,
                r_max_lr=r_max_lr,
                key_specification=keyspec,
                head_name=head_name,
                all_heads=list(heads.keys()),
            )
            val_data[head_name] = create_dataloader_from_data(
                head_config_list_val,
                batch_size=valid_batch_size,
                shuffle=False,
            )

        average_atomic_energy_shifts = compute_average_E0s(
            collections_train=train_configs,
            z_table=AtomicNumberTable([int(z) for z in range(1, 119)]),
        )
        train_data = create_data_from_configs(
            train_configs,
            r_max=r_max,
            r_max_lr=r_max_lr,
            all_heads=list(heads.keys()),
        )

        train_loader = create_dataloader_from_data(
            config_list=train_data,
            batch_size=batch_size,
            shuffle=True,
        )
        num_elements = determine_num_elements(train_loader)
        avg_num_neighbors = compute_avg_num_neighbors(train_loader)
        logging.info(f"Training set size: {len(train_data)}")
        logging.info(f"Validation set size: {len(val_data)}")
        logging.info(
            f"Number of unique elements in training set: {num_elements}"
        )
        return (
            train_loader,
            val_data,
            avg_num_neighbors,
            num_elements,
            average_atomic_energy_shifts,
        )

    else:
        # Load data
        data_path = config["TRAINING"]["path_to_train_data"]
        logging.info(f"Loading data from {data_path}")
        data = read(data_path, index=":")

        # Split data
        val_data_path = config["TRAINING"].get("path_to_val_data")
        if val_data_path:
            val_data = read(val_data_path, index=":")
            train_data = data
            logging.info(
                f"Using separate validation data from {val_data_path}"
            )
        else:
            valid_ratio = config["TRAINING"].get("valid_ratio", 0.1)
            num_train = config["TRAINING"].get("num_train", None)
            num_valid = config["TRAINING"].get("num_valid", None)
            train_data, val_data = select_valid_subset(
                data, valid_ratio, num_train, num_valid
            )
            logging.info(f"Splitting data with validation ratio {valid_ratio}")

        train_configs = create_configs_from_list(
            atoms_list=train_data,
            key_specification=keyspec,
        )

        average_atomic_energy_shifts = compute_average_E0s(
            collections_train=train_configs,
            z_table=AtomicNumberTable([int(z) for z in range(1, 119)]),
        )
        train_atomic_data = create_data_from_configs(
            train_configs,
            r_max=r_max,
            r_max_lr=r_max_lr,
        )

        train_loader = create_dataloader_from_data(
            config_list=train_atomic_data,
            batch_size=batch_size,
            shuffle=True,
        )

        num_elements = determine_num_elements(train_loader)
        avg_num_neighbors = compute_avg_num_neighbors(train_loader)

        valid_loader = create_dataloader_from_list(
            val_data,
            batch_size=valid_batch_size,
            r_max=r_max,
            r_max_lr=r_max_lr,
            key_specification=keyspec,
            shuffle=False,
        )
        logging.info(f"Training set size: {len(train_data)}")
        logging.info(f"Validation set size: {len(val_data)}")
        logging.info(
            f"Number of unique elements in training set: {num_elements}"
        )
        return (
            train_loader,
            {"main": valid_loader},
            avg_num_neighbors,
            num_elements,
            average_atomic_energy_shifts,
        )


def set_avg_num_neighbors_in_model(
    model: Union[SO3LR, MultiHeadSO3LR],
    avg_num_neighbors: float,
) -> None:
    logging.info(f"Average number of neighbors: {avg_num_neighbors:.2f}")
    model.avg_num_neighbors = avg_num_neighbors
    for layer in model.euclidean_transformers:
        layer.euclidean_attention_block.att_norm_inv = avg_num_neighbors
        layer.euclidean_attention_block.att_norm_ev = avg_num_neighbors


def set_atomic_energy_shifts_in_model(
    model: Union[SO3LR, MultiHeadSO3LR],
    atomic_energy_shifts: Union[dict, torch.tensor, nn.Parameter],
) -> None:
    model.atomic_energy_output_block.set_defined_energy_shifts(
        atomic_energy_shifts
    )


def process_config_atomic_energies(
    atomic_shifts_config: dict,
):
    atomic_energy_shifts = {}
    # turn keys to str if they are int
    atomic_shifts_config = {str(k): v for k, v in atomic_shifts_config.items()}
    for z in range(1, 119):
        if str(z) in atomic_shifts_config:
            atomic_energy_shifts[z] = atomic_shifts_config[str(z)]
        else:
            atomic_energy_shifts[z] = 0.0
    return atomic_energy_shifts


def determine_num_elements(data_loader: torch.utils.data.DataLoader) -> int:
    """Determine the number of unique elements in the dataset."""
    unique_elements = set()
    for batch in data_loader:
        one_hot = batch["node_attrs"]
        elements = one_hot.argmax(dim=-1).flatten().tolist()
        unique_elements.update(elements)
    return len(unique_elements)


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
    use_lora_plus: bool = False,
    lora_B_lr: float = None,
) -> tuple:
    """Setup optimizer and learning rate scheduler."""
    train_config = config["TRAINING"]

    # Setup optimizer
    optimizer_name = train_config.get("optimizer", "adam").lower()
    lr = train_config["lr"]
    weight_decay = train_config.get("weight_decay", 0.0)
    amsgrad = train_config.get("amsgrad", False)

    if optimizer_name == "adam":
        if use_lora_plus:
            assert (
                lora_B_lr is not None
            ), "lora_A_lr must be provided for LoRA+"
            lr = lora_B_lr
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

    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

    # Setup learning rate scheduler
    scheduler_name = train_config.get("scheduler", "exponential_decay")

    if scheduler_name == "exponential_decay":
        lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=train_config.get("lr_scheduler_gamma", 0.9993)
        )
    elif scheduler_name == "reduce_on_plateau":
        lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=train_config.get("scheduler_patience", 5),
            factor=train_config.get("lr_factor", 0.85),
            min_lr=1e-6,
        )
    else:
        raise ValueError(f"Unsupported scheduler: {scheduler_name}")

    logging.info(f"Optimizer: {optimizer_name}, Learning rate: {lr}")
    logging.info(f"Scheduler: {scheduler_name}")

    return optimizer, lr_scheduler


def load_pretrained_model_direct(
    pretrained_path: str, device: torch.device
) -> torch.nn.Module:
    """Load a complete pretrained model directly."""
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pretrained model not found: {pretrained_path}"
        )

    logging.info(f"Loading complete pretrained model from: {pretrained_path}")

    # Load the pretrained model
    loaded_object = torch.load(pretrained_path, map_location=device)

    if isinstance(loaded_object, torch.nn.Module):
        # If it's a complete model object, return it directly
        model = loaded_object.to(device)

        # Verify it's the right type of model
        if not isinstance(model, SO3LR):
            logging.warning(
                f"Loaded model type: {type(model).__name__}, "
                f"expected SO3LR"
            )

        # Ensure model has required attributes for training
        required_attrs = ["r_max"]
        missing_attrs = [
            attr for attr in required_attrs if not hasattr(model, attr)
        ]
        if missing_attrs:
            raise AttributeError(
                f"Loaded model missing required attributes: "
                f"{missing_attrs}"
            )

        logging.info("Loaded complete model object")
        total_params = sum(p.numel() for p in model.parameters())
        logging.info(f"Loaded model has {total_params} parameters")

        # Log some key model attributes for debugging
        if hasattr(model, "r_max"):
            logging.info(f"Model r_max: {model.r_max}")
        if hasattr(model, "num_features"):
            logging.info(f"Model num_features: {model.num_features}")
        if hasattr(model, "num_layers"):
            logging.info(f"Model num_layers: {model.num_layers}")

        return model
    else:
        raise ValueError(
            f"Expected a complete model object, got: " f"{type(loaded_object)}"
        )


def load_pretrained_weights(
    model: torch.nn.Module, pretrained_path: str, device: torch.device
) -> None:
    """Load pretrained model weights into existing model."""
    if not os.path.exists(pretrained_path):
        raise FileNotFoundError(
            f"Pretrained model not found: {pretrained_path}"
        )

    logging.info(f"Loading pretrained weights from: {pretrained_path}")

    # Load the pretrained weights
    loaded_object = torch.load(pretrained_path, map_location=device)

    if isinstance(loaded_object, torch.nn.Module):
        # If it's a model object, extract the state dict
        state_dict = loaded_object.state_dict()
        logging.info("Loaded model object, extracting state dict")
    elif isinstance(loaded_object, dict):
        if "model" in loaded_object:
            # Checkpoint format
            state_dict = loaded_object["model"]
            logging.info("Loaded checkpoint format")
        else:
            # Direct state dict
            state_dict = loaded_object
            logging.info("Loaded state dict format")
    else:
        raise ValueError(
            f"Unsupported pretrained model format: " f"{type(loaded_object)}"
        )

    # Load the state dict into our model
    missing_keys, unexpected_keys = model.load_state_dict(
        state_dict, strict=False
    )

    if missing_keys:
        logging.warning(
            f"Missing {len(missing_keys)} keys when loading "
            f"pretrained weights"
        )
    if unexpected_keys:
        logging.warning(
            f"Unexpected {len(unexpected_keys)} keys when loading "
            f"pretrained weights"
        )

    logging.info("Successfully loaded pretrained weights")


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


def handle_finetuning(
    config: dict,
    model: torch.nn.Module,
    num_elements: int,
    device_name: str,
) -> None:

    return setup_finetuning(
        model=model,
        finetune_choice=config["TRAINING"].get("finetune_choice", None),
        device_name=device_name,
        num_elements=num_elements,
        freeze_embedding=config["TRAINING"].get("freeze_embedding", True),
        freeze_zbl=config["TRAINING"].get("freeze_zbl", True),
        freeze_hirshfeld=config["TRAINING"].get("freeze_hirshfeld", True),
        freeze_partial_charges=config["TRAINING"].get(
            "freeze_partial_charges", True
        ),
        freeze_shifts=config["TRAINING"].get("freeze_shifts", False),
        freeze_scales=config["TRAINING"].get("freeze_scales", False),
        lora_rank=config["TRAINING"].get("lora_rank", 4),
        lora_alpha=config["TRAINING"].get("lora_alpha", 8.0),
        lora_freeze_A=config["TRAINING"].get("lora_freeze_A", False),
        dora_scaling_to_one=config["TRAINING"].get(
            "dora_scaling_to_one", True
        ),
        convert_to_multihead=config["ARCHITECTURE"].get(
            "convert_to_multihead", False
        ),
        architecture_settings=config["ARCHITECTURE"],
        seed=config["GENERAL"].get("seed", 42),
        log=True,
    )


def set_dtype_model(model: torch.nn.Module, dtype_str: str) -> None:

    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    dtype = DTYPE_MAP[dtype_str]
    # set all model params to dtype
    for param in model.parameters():
        param.data = param.data.to(dtype)


def run_training(config: dict) -> None:
    """
    Execute the complete training pipeline.

    Args:
        config: Configuration dictionary containing all training parameters
    """
    # Setup logging
    # Clear all handlers from root
    logging.getLogger().handlers.clear()

    # Clear all existing loggers
    logging.Logger.manager.loggerDict.clear()
    setup_logging(config)

    dtype_str = config["GENERAL"].get("default_dtype", "float32")
    torch.set_default_dtype(DTYPE_MAP[dtype_str])

    # Get pretrained model settings from config
    pretrained_weights = config["TRAINING"].get("pretrained_weights", None)
    pretrained_model = config["TRAINING"].get("pretrained_model", None)
    no_checkpoint = config["MISC"].get("no_checkpoint", False)

    # Validate pretrained settings
    if pretrained_weights and pretrained_model:
        raise ValueError(
            "Cannot specify both 'pretrained_weights' and "
            "'pretrained_model' in config. Use "
            "'pretrained_weights' for weights only or "
            "'pretrained_model' for complete model."
        )

    # Override checkpoint loading if specified in config
    if no_checkpoint:
        config["MISC"]["restart_latest"] = False

    # Setup device
    device_name = config["MISC"].get("device", "cuda")
    device = torch.device(device_name)
    logging.info(f"Using device: {device}")

    # Handle model creation and pretrained loading
    warm_start = False
    if pretrained_model:
        # Load complete pretrained model (ignores config architecture)
        model = load_pretrained_model_direct(pretrained_model, device)
        logging.info("Using complete pretrained model.")
        warm_start = True
    else:
        # Create model from config
        model = create_model(config, device)

        # Load pretrained weights if specified
        if pretrained_weights:
            load_pretrained_weights(model, pretrained_weights, device)
            warm_start = True

    assert model.r_max == config["ARCHITECTURE"].get("r_max", 4.5), (
        f"Model r_max ({model.r_max}) does not match config "
        f"r_max ({config['ARCHITECTURE'].get('r_max', 4.5)})"
    )

    assert model.r_max_lr == config["ARCHITECTURE"].get("r_max_lr", None), (
        f"Model r_max_lr ({model.r_max_lr}) does not match config "
        f"r_max_lr ({config['ARCHITECTURE'].get('r_max_lr', None)})"
    )

    # Setup data loaders
    (
        train_loader,
        valid_loaders,
        avg_num_neighbors,
        num_elements,
        average_atomic_energy_shifts,
    ) = setup_data_loaders(config)

    if warm_start:
        if config["TRAINING"].get("ft_update_avg_num_neighbors", False):
            logging.info(
                "Updating average number of neighbors from training data "
                "for fine-tuning."
            )
            model.avg_num_neighbors = avg_num_neighbors
        else:
            logging.info(
                "Retaining average number of neighbors from pretrained model "
                "for fine-tuning."
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
        # sort the shifts by atomic number and add all remaining elements as 0
        if atomic_shifts_config is not None:
            atomic_energy_shifts = process_config_atomic_energies(
                atomic_shifts_config
            )
            logging.info("Using provided atomic energy shifts for training.")

        else:
            atomic_energy_shifts = average_atomic_energy_shifts
            logging.info(
                "Using average atomic energy shifts computed from training data for training."
            )

    # Setup finetuning if specified
    if config["TRAINING"].get("finetune_choice", None):
        model = handle_finetuning(config, model, num_elements, device_name)

    logging.info(f"Atomic energy shifts: {atomic_energy_shifts}")
    set_atomic_energy_shifts_in_model(model, atomic_energy_shifts)
    set_avg_num_neighbors_in_model(model, avg_num_neighbors)

    set_dtype_model(model, config["GENERAL"].get("default_dtype", "float32"))

    # Setup loss function
    loss_fn = setup_loss_function(config)

    # Setup optimizer and scheduler
    optimizer, lr_scheduler = setup_optimizer_and_scheduler(model, config)

    # Setup training tools
    logger, checkpoint_handler, ema = setup_training_tools(config, model)

    # Load checkpoint if exists (unless pretrained model was loaded)
    start_epoch = 0
    if not pretrained_weights and not pretrained_model:
        # Skip checkpoint if any pretrained model was loaded
        start_epoch = load_checkpoint_if_exists(
            model=model,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            checkpoint_handler=checkpoint_handler,
            ema=ema,
            device=device,
            config=config,
        )

    logging.info("Model, data loaders, and training components initialized")
    if start_epoch > 0:
        logging.info(f"Resuming training from epoch {start_epoch}")
    else:
        logging.info("Starting fresh training.")

    # Setup training parameters
    max_num_epochs = config["TRAINING"]["num_epochs"]
    eval_interval = config["TRAINING"].get("eval_interval", 1)
    patience = config["TRAINING"].get("patience", 50)
    max_grad_norm = config["TRAINING"].get("clip_grad", 10.0)
    log_wandb = config["MISC"].get("log_wandb", False)
    save_all_checkpoints = config["MISC"].get("keep_checkpoints", False)

    # Setup output arguments for model evaluation
    output_args = {
        "forces": True,  # Always compute forces for training
        "virials": config["GENERAL"].get("compute_stress", False),
        "stress": config["GENERAL"].get("compute_stress", False),
    }

    # Get error logging type
    log_errors = config["MISC"].get("error_table", "PerAtomMAE")

    if config["ARCHITECTURE"].get("use_multihead", False):
        logging.info(
            "Enabling head selection for multi-head model during training."
        )
        model.select_heads = True
    logging.info("Starting training loop...")
    # Start training
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
        distributed=False,  # Single GPU training for now
        save_all_checkpoints=save_all_checkpoints,
        plotter=None,  # No plotting for now
        distributed_model=None,
        train_sampler=None,
        rank=0,
    )
    logging.info("Training completed successfully!")

    if config["ARCHITECTURE"].get("use_multihead", False):
        logging.info(
            "Disabling head selection for multi-head model after training."
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
    # TODO: use EMA weights if enabled before saving

    # save the model in the working directory
    final_model_path = f'{config["GENERAL"]["name_exp"]}.pth'

    torch.save(model.state_dict(), final_model_path)
    torch.save(model, final_model_path.replace(".pth", ".model"))


def main():
    parser = argparse.ArgumentParser(description="Train SO3LR model")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration YAML file",
    )
    args = parser.parse_args()

    # Load configuration
    config = setup_config_from_yaml(args.config)

    # Run the training pipeline
    run_training(config)


if __name__ == "__main__":
    main()
