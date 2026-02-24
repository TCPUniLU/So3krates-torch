import argparse
import logging
import yaml
import torch
import torch.nn as nn
from ase.io import read
import random
from typing import Optional, Tuple, Union
from so3krates_torch.modules.models import SO3LR, MultiHeadSO3LR
from so3krates_torch.tools.utils import (
    create_dataloader_from_list,
    create_data_from_list,
    create_dataloader_from_data,
    create_configs_from_list,
    create_data_from_configs,
)
from so3krates_torch.tools.distributed_tools import init_distributed
from so3krates_torch.modules.loss import (
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesHirshfeldLoss,
    WeightedEnergyForcesDipoleHirshfeldLoss,
)
from so3krates_torch.data.utils import (
    KeySpecification,
    compute_average_E0s,
)
from so3krates_torch.data.hdf5_utils import (
    detect_file_format,
    PreprocessedHDF5Dataset,
    validate_preprocessed_hdf5,
)
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
from so3krates_torch.tools.torch_geometric import DataLoader
import os
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data.distributed import DistributedSampler


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

    # Validate long-range configuration
    r_max_lr = arch_config.get("r_max_lr", None)
    electrostatic_bool = arch_config.get("electrostatic_energy_bool", True)
    dispersion_bool = arch_config.get("dispersion_energy_bool", True)

    if (electrostatic_bool or dispersion_bool) and r_max_lr is None:
        raise ValueError(
            "Long-range cutoff 'r_max_lr' must be specified when "
            "electrostatic_energy_bool or dispersion_energy_bool is True. "
            f"Current: r_max_lr={r_max_lr}, "
            f"electrostatic_energy_bool={electrostatic_bool}, "
            f"dispersion_energy_bool={dispersion_bool}"
        )

    if r_max_lr is not None and not electrostatic_bool and not dispersion_bool:
        logging.warning(
            f"Long-range cutoff r_max_lr={r_max_lr} is set but both "
            "electrostatic_energy_bool and dispersion_energy_bool are False. "
            "Long-range neighbor lists will be computed but not used."
        )

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
        if model_params["num_output_heads"] is None:
            raise ValueError(
                "num_output_heads must be specified when using "
                "convert_to_multihead"
            )
        logging.info(
            f"Creating Multi-Head SO3LR model with {model_params['num_output_heads']} heads"
        )
        model = MultiHeadSO3LR(**model_params)
    else:
        logging.info("Creating SO3LR model")
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


def _load_training_dataset(
    config: dict,
    r_max: float,
    r_max_lr: float,
    keyspec,
) -> tuple:
    """Load training data.

    Returns (train_atomic_data, train_configs,
             avg_num_neighbors, num_elements, val_split).

    train_configs is None for preprocessed HDF5.
    avg_num_neighbors and num_elements are None when not in metadata.
    val_split is None if path_to_val_data is set.
    """
    train_path = config["TRAINING"]["path_to_train_data"]
    logging.info(f"Loading training data from {train_path}")

    is_preprocessed = config["TRAINING"].get("data_preprocessed", None)
    if is_preprocessed is None:
        file_format = detect_file_format(train_path)
        is_preprocessed = file_format == "hdf5_preprocessed"
        logging.info(f"Auto-detected train format: {file_format}")
    else:
        logging.info(
            f"Using config-specified data_preprocessed={is_preprocessed}"
        )
        if is_preprocessed:
            if not train_path.endswith((".h5", ".hdf5")):
                raise ValueError(
                    f"data_preprocessed=true but file is not HDF5: "
                    f"{train_path}"
                )
            validate_preprocessed_hdf5(
                train_path,
                expected_r_max=r_max,
                expected_r_max_lr=r_max_lr,
            )

    if is_preprocessed:
        logging.info(
            "Loading preprocessed HDF5 training data " "(with neighbor lists)"
        )
        dataset = PreprocessedHDF5Dataset(
            hdf5_path=train_path,
            validate_cutoffs=True,
            expected_r_max=r_max,
            expected_r_max_lr=r_max_lr,
        )
        avg_num_neighbors = dataset.metadata.get("avg_num_neighbors", None)
        num_elements = dataset.metadata.get("num_elements", None)
        return dataset, None, avg_num_neighbors, num_elements, None

    # Raw path (XYZ or raw HDF5)
    val_data_path = config["TRAINING"].get("path_to_val_data")
    if train_path.endswith(".xyz"):
        logging.info("Loading XYZ training data")
        data = read(train_path, index=":")
    elif train_path.endswith((".h5", ".hdf5")):
        logging.info("Loading raw HDF5 training data")
        from so3krates_torch.data.hdf5_utils import load_atoms_from_hdf5

        data = load_atoms_from_hdf5(train_path, index=None)
    else:
        raise ValueError(f"Unsupported training file format: {train_path}")

    if val_data_path:
        train_data = data
        val_split = None
    else:
        valid_ratio = config["TRAINING"].get("valid_ratio", 0.1)
        num_train = config["TRAINING"].get("num_train", None)
        num_valid = config["TRAINING"].get("num_valid", None)
        train_data, val_split = select_valid_subset(
            data, valid_ratio, num_train, num_valid
        )
        logging.info(
            f"Splitting training data with validation ratio " f"{valid_ratio}"
        )

    train_configs = create_configs_from_list(
        atoms_list=train_data, key_specification=keyspec
    )
    logging.info("Preprocessing training data (computing neighbor lists)")
    train_atomic_data = create_data_from_configs(
        train_configs, r_max=r_max, r_max_lr=r_max_lr
    )
    return train_atomic_data, train_configs, None, None, val_split


def _compute_e0s(
    train_configs,
    train_dataset=None,
    z_table=None,
) -> dict:
    """Compute average atomic energy shifts (E0s) for all 118 elements."""
    if train_configs is not None:
        present_zs = set()
        for cfg in train_configs:
            present_zs.update(cfg.atomic_numbers)
        present_z_table = AtomicNumberTable(sorted(list(present_zs)))
        present_e0s = compute_average_E0s(
            collections_train=train_configs,
            z_table=present_z_table,
        )
    elif (
        hasattr(train_dataset, "atomic_energy_shifts")
        and train_dataset.atomic_energy_shifts is not None
    ):
        logging.info(
            "Using atomic energy shifts (E0s) loaded from " "preprocessed HDF5"
        )
        present_e0s = train_dataset.atomic_energy_shifts
    else:
        from so3krates_torch.data.utils import (
            compute_average_E0s_from_dataset,
        )

        logging.info(
            "Computing E0s from preprocessed data " "(not found in HDF5)..."
        )
        present_e0s = compute_average_E0s_from_dataset(train_dataset, z_table)

    return {z: present_e0s.get(z, 0.0) for z in range(1, 119)}


def _load_validation_loader(
    config: dict,
    r_max: float,
    r_max_lr: float,
    keyspec,
    valid_batch_size: int,
    is_train_preprocessed: bool,
    val_split_from_train=None,
    valid_subset=None,
):
    """Return a DataLoader for validation data."""
    val_data_path = config["TRAINING"].get("path_to_val_data")

    if val_data_path:
        logging.info(f"Loading validation data from {val_data_path}")
        is_valid_preprocessed = config["TRAINING"].get(
            "valid_data_preprocessed", None
        )
        if is_valid_preprocessed is None:
            valid_format = detect_file_format(val_data_path)
            is_valid_preprocessed = valid_format == "hdf5_preprocessed"
            logging.info(f"Auto-detected validation format: {valid_format}")
        else:
            logging.info(
                f"Using config-specified validation "
                f"data_preprocessed={is_valid_preprocessed}"
            )
            if is_valid_preprocessed:
                validate_preprocessed_hdf5(
                    val_data_path,
                    expected_r_max=r_max,
                    expected_r_max_lr=r_max_lr,
                )

        if is_valid_preprocessed:
            valid_dataset = PreprocessedHDF5Dataset(
                hdf5_path=val_data_path,
                validate_cutoffs=True,
                expected_r_max=r_max,
                expected_r_max_lr=r_max_lr,
            )
            return create_dataloader_from_data(
                config_list=valid_dataset,
                batch_size=valid_batch_size,
                shuffle=False,
            )

        if val_data_path.endswith(".xyz"):
            val_data = read(val_data_path, index=":")
        elif val_data_path.endswith((".h5", ".hdf5")):
            from so3krates_torch.data.hdf5_utils import (
                load_atoms_from_hdf5,
            )

            val_data = load_atoms_from_hdf5(val_data_path, index=None)
        else:
            raise ValueError(
                f"Unsupported validation file format: {val_data_path}"
            )

        return create_dataloader_from_list(
            val_data,
            batch_size=valid_batch_size,
            r_max=r_max,
            r_max_lr=r_max_lr,
            key_specification=keyspec,
            shuffle=False,
        )

    # No separate val file — use split from training data
    if is_train_preprocessed:
        if valid_subset is None:
            raise ValueError(
                "valid_subset required when splitting preprocessed data"
            )
        return create_dataloader_from_data(
            config_list=valid_subset,
            batch_size=valid_batch_size,
            shuffle=False,
        )

    if val_split_from_train is None:
        raise ValueError(
            "val_split_from_train required when splitting raw data"
        )
    return create_dataloader_from_list(
        val_split_from_train,
        batch_size=valid_batch_size,
        r_max=r_max,
        r_max_lr=r_max_lr,
        key_specification=keyspec,
        shuffle=False,
    )


def _setup_multihead_data_loaders(
    config: dict,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple:
    """Setup data loaders for multi-head (multi-dataset) training.

    Returns (train_loader, valid_loaders, train_sampler,
             avg_num_neighbors, num_elements,
             average_atomic_energy_shifts).
    """
    from so3krates_torch.tools.default_keys import DefaultKeys
    from so3krates_torch.data.utils import update_keyspec_from_kwargs

    keydict = DefaultKeys.keydict()
    config_keys = config["TRAINING"].get("keys", {})
    keydict.update(config_keys)
    keyspec = update_keyspec_from_kwargs(KeySpecification(), keydict)
    batch_size = config["TRAINING"]["batch_size"]
    valid_batch_size = config["TRAINING"]["valid_batch_size"]
    r_max = config["ARCHITECTURE"].get("r_max", None)
    r_max_lr = config["ARCHITECTURE"].get("r_max_lr", None)

    heads = config["TRAINING"]["heads"]
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
            f"Head {head_name} - Training set size: " f"{len(head_train_data)}"
        )
        logging.info(
            f"Head {head_name} - Validation set size: " f"{len(head_val_data)}"
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

    # Find elements actually present in data
    present_zs = set()
    for cfg in train_configs:
        present_zs.update(cfg.atomic_numbers)
    present_z_table = AtomicNumberTable(sorted(list(present_zs)))

    # Compute E0s for present elements only
    present_e0s = compute_average_E0s(
        collections_train=train_configs,
        z_table=present_z_table,
    )

    # Expand to full 118 elements (fill missing with 0)
    average_atomic_energy_shifts = {
        z: present_e0s.get(z, 0.0) for z in range(1, 119)
    }
    train_data = create_data_from_configs(
        train_configs,
        r_max=r_max,
        r_max_lr=r_max_lr,
        all_heads=list(heads.keys()),
    )

    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    train_loader = create_dataloader_from_data(
        config_list=train_data,
        batch_size=batch_size,
        shuffle=True,
        sampler=train_sampler,
    )
    logging.info(
        "Computing dataset statistics (num_elements, avg_num_neighbors)..."
    )
    num_elements = determine_num_elements(train_loader)
    avg_num_neighbors = compute_avg_num_neighbors(train_loader)
    logging.info(
        f"Computed: num_elements={num_elements}, "
        f"avg_num_neighbors={avg_num_neighbors:.2f}"
    )
    logging.info(f"Training set size: {len(train_data)}")
    total_val = sum(len(loader.dataset) for loader in val_data.values())
    logging.info(
        f"Total validation set size: {total_val} "
        f"across {len(val_data)} heads"
    )
    logging.info(
        f"Number of unique elements in training set: " f"{num_elements}"
    )
    return (
        train_loader,
        val_data,
        train_sampler,
        avg_num_neighbors,
        num_elements,
        average_atomic_energy_shifts,
    )


def _setup_singlehead_data_loaders(
    config: dict,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple:
    """Setup data loaders for single-head training."""
    from so3krates_torch.tools.default_keys import DefaultKeys
    from so3krates_torch.data.utils import update_keyspec_from_kwargs

    keydict = DefaultKeys.keydict()
    keydict.update(config["TRAINING"].get("keys", {}))
    keyspec = update_keyspec_from_kwargs(KeySpecification(), keydict)

    r_max = config["ARCHITECTURE"].get("r_max", None)
    r_max_lr = config["ARCHITECTURE"].get("r_max_lr", None)
    batch_size = config["TRAINING"]["batch_size"]
    valid_batch_size = config["TRAINING"]["valid_batch_size"]
    valid_loader: Optional[DataLoader] = None

    (
        train_atomic_data,
        train_configs,
        avg_num_neighbors,
        num_elements,
        val_split,
    ) = _load_training_dataset(config, r_max, r_max_lr, keyspec)

    is_preprocessed = isinstance(train_atomic_data, PreprocessedHDF5Dataset)

    # Split preprocessed dataset if no separate val file
    valid_subset = None
    if is_preprocessed and not config["TRAINING"].get("path_to_val_data"):
        valid_ratio = config["TRAINING"].get("valid_ratio", 0.1)
        n_total = len(train_atomic_data)
        indices = list(range(n_total))
        random.shuffle(indices)
        n_valid = int(n_total * valid_ratio)
        n_train = n_total - n_valid
        from torch.utils.data import Subset

        valid_subset = Subset(train_atomic_data, indices[n_train:])
        train_atomic_data = Subset(train_atomic_data, indices[:n_train])
        logging.info(
            f"Split preprocessed data: {n_train} train, "
            f"{n_valid} valid (ratio={valid_ratio})"
        )

    # Train loader + optional DDP sampler
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_atomic_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )
    train_loader = create_dataloader_from_data(
        config_list=train_atomic_data,
        batch_size=batch_size,
        shuffle=(train_sampler is None),
        sampler=train_sampler,
    )

    # Compute dataset statistics (from metadata or iterate)
    if num_elements is None:
        logging.info(
            "num_elements not found in metadata, "
            "computing from data (slow)..."
        )
        num_elements = determine_num_elements(train_loader)
    else:
        logging.info(
            f"Loaded num_elements={num_elements} from " "preprocessed metadata"
        )

    if avg_num_neighbors is None:
        logging.info(
            "avg_num_neighbors not found in metadata, "
            "computing from data (slow)..."
        )
        avg_num_neighbors = compute_avg_num_neighbors(train_loader)
    else:
        logging.info(
            f"Loaded avg_num_neighbors={avg_num_neighbors:.2f} "
            "from preprocessed metadata"
        )

    # E0s
    if is_preprocessed:
        # For Subset, unwrap to get the underlying dataset
        raw_dataset = (
            train_atomic_data.dataset
            if hasattr(train_atomic_data, "dataset")
            else train_atomic_data
        )
        average_atomic_energy_shifts = _compute_e0s(
            train_configs=None,
            train_dataset=raw_dataset,
            z_table=AtomicNumberTable([int(z) for z in range(1, 119)]),
        )
    else:
        average_atomic_energy_shifts = _compute_e0s(
            train_configs=train_configs,
        )

    # Validation loader
    valid_loader = _load_validation_loader(
        config,
        r_max,
        r_max_lr,
        keyspec,
        valid_batch_size,
        is_train_preprocessed=is_preprocessed,
        val_split_from_train=val_split,
        valid_subset=valid_subset,
    )

    logging.info(f"Training set size: {len(train_atomic_data)}")
    if valid_loader is not None:
        logging.info(f"Validation set size: {len(valid_loader.dataset)}")
    logging.info(f"Number of unique elements in training set: {num_elements}")

    return (
        train_loader,
        {"main": valid_loader} if valid_loader else {},
        train_sampler,
        avg_num_neighbors,
        num_elements,
        average_atomic_energy_shifts,
    )


def setup_data_loaders(
    config: dict,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple:
    """Setup training and validation data loaders.

    Returns (train_loader, valid_loaders, train_sampler,
             avg_num_neighbors, num_elements,
             average_atomic_energy_shifts).
    """
    if config["TRAINING"].get("heads", None) is not None:
        return _setup_multihead_data_loaders(
            config, distributed, rank, world_size
        )
    return _setup_singlehead_data_loaders(
        config, distributed, rank, world_size
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
) -> tuple:
    """Setup optimizer and learning rate scheduler."""
    train_config = config["TRAINING"]

    optimizer_name = train_config.get("optimizer", "adam").lower()
    lr = train_config["lr"]
    weight_decay = train_config.get("weight_decay", 0.0)
    amsgrad = train_config.get("amsgrad", False)

    if optimizer_name == "adam":
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
    loaded_object = torch.load(
        pretrained_path, map_location=device, weights_only=False
    )

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


def _setup_model_for_training(
    config: dict,
    device: torch.device,
) -> tuple:
    """Create or load the model and return (model, warm_start).

    Validates that model attributes match the config, overriding
    r_max_lr and dispersion_energy_cutoff_lr_damping if they differ
    (with a warning).
    """
    pretrained_weights = config["TRAINING"].get("pretrained_weights", None)
    pretrained_model = config["TRAINING"].get("pretrained_model", None)

    if pretrained_weights and pretrained_model:
        raise ValueError(
            "Cannot specify both 'pretrained_weights' and "
            "'pretrained_model' in config. "
            "Use 'pretrained_weights' for weights only or "
            "'pretrained_model' for complete model."
        )

    warm_start = False
    if pretrained_model:
        model = load_pretrained_model_direct(pretrained_model, device)
        logging.info("Using complete pretrained model.")
        warm_start = True
    else:
        model = create_model(config, device)
        if pretrained_weights:
            load_pretrained_weights(model, pretrained_weights, device)
            warm_start = True

    config_r_max = config["ARCHITECTURE"].get("r_max", 4.5)
    if model.r_max != config_r_max:
        raise ValueError(
            f"Model r_max ({model.r_max}) does not match config "
            f"r_max ({config_r_max})"
        )

    config_r_max_lr = config["ARCHITECTURE"].get("r_max_lr", None)
    if model.r_max_lr != config_r_max_lr:
        logging.warning(
            f"Model r_max_lr ({model.r_max_lr}) does not match "
            f"config r_max_lr ({config_r_max_lr}). "
            f"Overriding model.r_max_lr with config value."
        )
        model.r_max_lr = config_r_max_lr

    config_cutoff_lr_damping = config["ARCHITECTURE"].get(
        "dispersion_energy_cutoff_lr_damping", None
    )
    if (
        hasattr(model, "dispersion_energy_cutoff_lr_damping")
        and model.dispersion_energy_cutoff_lr_damping
        != config_cutoff_lr_damping
    ):
        logging.warning(
            f"Model dispersion_energy_cutoff_lr_damping "
            f"({model.dispersion_energy_cutoff_lr_damping}) does not "
            f"match config dispersion_energy_cutoff_lr_damping "
            f"({config_cutoff_lr_damping}). "
            f"Overriding model value with config value."
        )
        model.dispersion_energy_cutoff_lr_damping = config_cutoff_lr_damping

    return model, warm_start


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

    dtype_str = config["GENERAL"].get("default_dtype", "float32")
    torch.set_default_dtype(DTYPE_MAP[dtype_str])

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
