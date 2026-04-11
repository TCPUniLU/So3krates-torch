import math
import random
import logging
import h5py
from typing import Callable, List, Optional, Tuple

import numpy as np
import torch
from torch.utils.data import Subset
from torch.utils.data.distributed import DistributedSampler
from ase.io import read
from so3krates_torch.tools.torch_geometric import DataLoader
from so3krates_torch.tools.torch_geometric.dataloader import Collater
from so3krates_torch.data.utils import (
    KeySpecification,
    compute_average_E0s,
    compute_average_E0s_from_dataset,
    update_keyspec_from_kwargs,
)
from so3krates_torch.data.hdf5_utils import (
    detect_file_format,
    load_atoms_from_hdf5,
    PreprocessedHDF5Dataset,
    scan_raw_hdf5_statistics,
    validate_preprocessed_hdf5,
)
from so3krates_torch.data.lazy_dataset import LazyAtomicDataset
from so3krates_torch.tools.default_keys import DefaultKeys
from so3krates_torch.tools.utils import (
    AtomicNumberTable,
    compute_avg_num_neighbors,
    create_dataloader_from_list,
    create_data_from_list,
    create_dataloader_from_data,
    create_configs_from_list,
    create_data_from_configs,
)
from so3krates_torch.tools.model_setup import determine_num_elements


def _worker_init_fn(worker_id: int) -> None:
    seed = torch.initial_seed() % (2**32)
    np.random.seed(seed)
    random.seed(seed)


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

    # Lazy loading path (raw HDF5 only)
    lazy_loading = config["TRAINING"].get("lazy_loading", False)
    if lazy_loading and train_path.endswith((".h5", ".hdf5")):
        logging.info("Using lazy loading for raw HDF5 training data")
        num_neighbor_samples = config["TRAINING"].get(
            "num_neighbor_samples", 1000
        )
        e0s, num_elements, avg_num_neighbors = scan_raw_hdf5_statistics(
            hdf5_path=train_path,
            r_max=r_max,
            r_max_lr=r_max_lr,
            keyspec=keyspec,
            num_neighbor_samples=num_neighbor_samples,
        )
        dataset = LazyAtomicDataset(
            hdf5_path=train_path,
            r_max=r_max,
            r_max_lr=r_max_lr,
            keyspec=keyspec,
        )
        # Store pre-computed E0s on the dataset for later use
        dataset.atomic_energy_shifts = e0s
        return dataset, None, avg_num_neighbors, num_elements, None

    # Raw path (XYZ or raw HDF5)
    val_data_path = config["TRAINING"].get("path_to_val_data")
    if train_path.endswith(".xyz"):
        logging.info("Loading XYZ training data")
        data = read(train_path, index=":")
    elif train_path.endswith((".h5", ".hdf5")):
        logging.info("Loading raw HDF5 training data")
        data = load_atoms_from_hdf5(train_path, index=None)
    else:
        raise ValueError(f"Unsupported training file format: {train_path}")

    if val_data_path:
        num_train = config["TRAINING"].get("num_train", None)
        if num_train is not None:
            random.shuffle(data)
            train_data = data[:num_train]
        else:
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

    config_type_weights = config["TRAINING"].get("config_type_weights", None)
    train_configs = create_configs_from_list(
        atoms_list=train_data,
        key_specification=keyspec,
        config_type_weights=config_type_weights,
    )
    n_weighted = sum(1 for c in train_configs if c.weight != 1.0)
    if n_weighted > 0:
        logging.info(
            f"Config weights found: {n_weighted}/{len(train_configs)} "
            f"training structures have non-default weights."
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
        logging.info(
            "Computing E0s from preprocessed data " "(not found in HDF5)..."
        )
        present_e0s = compute_average_E0s_from_dataset(train_dataset, z_table)

    return {z: present_e0s.get(z, 0.0) for z in range(1, 119)}


def _load_replay_data(
    config: dict,
    r_max: float,
    r_max_lr: float,
    keyspec,
) -> list:
    """Load and subsample replay datasets.

    Returns a flat list of AtomicData objects drawn from
    the configured replay datasets according to the specified
    fractions and total count.
    """
    replay_paths = config["TRAINING"]["replay_datasets"]
    fractions = config["TRAINING"]["replay_fractions"]
    total = config["TRAINING"]["replay_total"]

    # Compute per-dataset counts with remainder distribution
    counts = [int(math.floor(f * total)) for f in fractions]
    remainder = total - sum(counts)
    for i in range(remainder):
        counts[i] += 1

    all_replay = []
    for path, count in zip(replay_paths, counts):
        if count == 0:
            continue
        file_format = detect_file_format(path)
        logging.info(
            f"Loading replay dataset: {path} "
            f"(format={file_format}, n={count})"
        )

        if file_format == "hdf5_preprocessed":
            dataset = PreprocessedHDF5Dataset(
                hdf5_path=path,
                validate_cutoffs=True,
                expected_r_max=r_max,
                expected_r_max_lr=r_max_lr,
            )
            n_available = len(dataset)
            if count > n_available:
                logging.warning(
                    f"Replay dataset {path} has {n_available} "
                    f"structures but {count} requested. "
                    f"Sampling with replacement."
                )
                indices = random.choices(range(n_available), k=count)
            else:
                indices = random.sample(range(n_available), count)
            replay_data = [dataset[i] for i in indices]
        else:
            # Raw HDF5 or XYZ — load atoms
            if path.endswith((".h5", ".hdf5")):
                atoms_list = load_atoms_from_hdf5(path, index=None)
            elif path.endswith((".xyz", ".extxyz")):
                atoms_list = read(path, index=":")
            else:
                raise ValueError(f"Unsupported replay file format: {path}")

            n_available = len(atoms_list)
            if count > n_available:
                logging.warning(
                    f"Replay dataset {path} has {n_available} "
                    f"structures but {count} requested. "
                    f"Sampling with replacement."
                )
                indices = random.choices(range(n_available), k=count)
            else:
                indices = random.sample(range(n_available), count)
            sampled_atoms = [atoms_list[i] for i in indices]
            replay_data = create_data_from_list(
                sampled_atoms,
                r_max=r_max,
                r_max_lr=r_max_lr,
                key_specification=keyspec,
                config_type_weights=config["TRAINING"].get(
                    "config_type_weights", None
                ),
            )

        all_replay.extend(replay_data)

    logging.info(
        f"Loaded {len(all_replay)} total replay structures "
        f"from {len(replay_paths)} datasets"
    )
    return all_replay


def _materialize_dataset(dataset) -> list:
    """Convert a Subset or Dataset to a plain list."""
    if isinstance(dataset, list):
        return list(dataset)
    return [dataset[i] for i in range(len(dataset))]


def _build_combined_train_data(
    finetune_data: list,
    replay_data: list,
    oversample: bool,
) -> list:
    """Combine fine-tune and replay data with optional
    oversampling of fine-tune data for ~1:1 ratio."""
    n_ft = len(finetune_data)
    n_replay = len(replay_data)

    if oversample and n_ft < n_replay:
        repeats = math.ceil(n_replay / n_ft)
        finetune_data = (finetune_data * repeats)[:n_replay]
        logging.info(
            f"Oversampled fine-tune data: {n_ft} -> "
            f"{len(finetune_data)} to match "
            f"{n_replay} replay structures"
        )

    combined = finetune_data + replay_data
    random.shuffle(combined)
    logging.info(
        f"Combined training set: {len(combined)} "
        f"({len(finetune_data)} fine-tune + "
        f"{n_replay} replay)"
    )
    return combined


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
            val_data = load_atoms_from_hdf5(val_data_path, index=None)
        else:
            raise ValueError(
                f"Unsupported validation file format: {val_data_path}"
            )

        config_type_weights = config["TRAINING"].get(
            "config_type_weights", None
        )
        return create_dataloader_from_list(
            val_data,
            batch_size=valid_batch_size,
            r_max=r_max,
            r_max_lr=r_max_lr,
            key_specification=keyspec,
            shuffle=False,
            config_type_weights=config_type_weights,
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
    config_type_weights = config["TRAINING"].get("config_type_weights", None)
    return create_dataloader_from_list(
        val_split_from_train,
        batch_size=valid_batch_size,
        r_max=r_max,
        r_max_lr=r_max_lr,
        key_specification=keyspec,
        shuffle=False,
        config_type_weights=config_type_weights,
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

        head_ctw = head_config.get(
            "config_type_weights",
            config["TRAINING"].get("config_type_weights", None),
        )
        head_config_list_train = create_configs_from_list(
            atoms_list=head_train_data,
            key_specification=keyspec,
            head_name=head_name,
            config_type_weights=head_ctw,
        )
        train_configs.extend(head_config_list_train)

        head_config_list_val = create_data_from_list(
            head_val_data,
            r_max=r_max,
            r_max_lr=r_max_lr,
            key_specification=keyspec,
            head_name=head_name,
            all_heads=list(heads.keys()),
            config_type_weights=head_ctw,
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
        None,  # replay_builder (not supported for multi-head)
    )


def _setup_singlehead_data_loaders(
    config: dict,
    distributed: bool = False,
    rank: int = 0,
    world_size: int = 1,
) -> tuple:
    """Setup data loaders for single-head training."""
    keydict = DefaultKeys.keydict()
    keydict.update(config["TRAINING"].get("keys") or {})
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
    is_lazy = isinstance(train_atomic_data, LazyAtomicDataset)

    # Split preprocessed/lazy dataset if no separate val file
    valid_subset = None
    if (is_preprocessed or is_lazy) and not config["TRAINING"].get(
        "path_to_val_data"
    ):
        valid_ratio = config["TRAINING"].get("valid_ratio", 0.1)
        num_train = config["TRAINING"].get("num_train", None)
        num_valid = config["TRAINING"].get("num_valid", None)
        n_total = len(train_atomic_data)
        indices = list(range(n_total))
        random.shuffle(indices)
        n_valid = int(n_total * valid_ratio)
        n_train = n_total - n_valid
        if num_train is not None:
            n_train = min(n_train, num_train)
        if num_valid is not None:
            n_valid = min(n_valid, num_valid)
        valid_subset = Subset(
            train_atomic_data, indices[n_train : n_train + n_valid]
        )
        train_atomic_data = Subset(train_atomic_data, indices[:n_train])
        logging.info(
            f"Split data: {n_train} train, "
            f"{n_valid} valid (ratio={valid_ratio})"
        )
    elif (is_preprocessed or is_lazy) and config["TRAINING"].get("num_train"):
        num_train = config["TRAINING"]["num_train"]
        n = min(len(train_atomic_data), num_train)
        indices = list(range(len(train_atomic_data)))
        random.shuffle(indices)
        train_atomic_data = Subset(train_atomic_data, indices[:n])
        logging.info(f"Limiting training data to {n} samples")

    # Train loader + optional DDP sampler
    train_sampler = None
    if distributed:
        train_sampler = DistributedSampler(
            train_atomic_data,
            num_replicas=world_size,
            rank=rank,
            shuffle=True,
        )

    if is_lazy:
        num_workers = config["TRAINING"].get("num_workers", 4)
        prefetch_factor = config["TRAINING"].get("prefetch_factor", 2)
        train_loader = DataLoader(
            dataset=train_atomic_data,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            persistent_workers=(num_workers > 0),
            collate_fn=Collater([None], [None]),
            worker_init_fn=_worker_init_fn,
        )
        logging.info(
            f"Lazy DataLoader: num_workers={num_workers}, "
            f"prefetch_factor={prefetch_factor}"
        )
    else:
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
    if is_lazy:
        # For lazy datasets, E0s were pre-computed during
        # scan_raw_hdf5_statistics
        raw_dataset = (
            train_atomic_data.dataset
            if hasattr(train_atomic_data, "dataset")
            else train_atomic_data
        )
        average_atomic_energy_shifts = raw_dataset.atomic_energy_shifts
    elif is_preprocessed:
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

    # Replay data injection
    replay_builder = None
    replay_datasets_cfg = config["TRAINING"].get("replay_datasets", None)
    if replay_datasets_cfg:
        oversample = config["TRAINING"].get("replay_oversample_finetune", True)
        resample = config["TRAINING"].get("replay_resample_per_epoch", False)

        # Materialize fine-tune data once
        ft_data = _materialize_dataset(train_atomic_data)

        replay_data = _load_replay_data(config, r_max, r_max_lr, keyspec)
        combined = _build_combined_train_data(
            list(ft_data), replay_data, oversample
        )

        # Rebuild train_loader and train_sampler
        train_sampler = None
        if distributed:
            train_sampler = DistributedSampler(
                combined,
                num_replicas=world_size,
                rank=rank,
                shuffle=True,
            )
        train_loader = create_dataloader_from_data(
            config_list=combined,
            batch_size=batch_size,
            shuffle=(train_sampler is None),
            sampler=train_sampler,
        )

        if resample:

            def _make_replay_builder(
                _ft_data,
                _config,
                _r_max,
                _r_max_lr,
                _keyspec,
                _oversample,
                _batch_size,
                _distributed,
                _world_size,
                _rank,
            ):
                def builder():
                    replay = _load_replay_data(
                        _config, _r_max, _r_max_lr, _keyspec
                    )
                    combined = _build_combined_train_data(
                        list(_ft_data), replay, _oversample
                    )
                    sampler = None
                    if _distributed:
                        sampler = DistributedSampler(
                            combined,
                            num_replicas=_world_size,
                            rank=_rank,
                            shuffle=True,
                        )
                    loader = create_dataloader_from_data(
                        config_list=combined,
                        batch_size=_batch_size,
                        shuffle=(sampler is None),
                        sampler=sampler,
                    )
                    return loader, sampler

                return builder

            replay_builder = _make_replay_builder(
                ft_data,
                config,
                r_max,
                r_max_lr,
                keyspec,
                oversample,
                batch_size,
                distributed,
                world_size,
                rank,
            )

    # Validation loader
    valid_loader = _load_validation_loader(
        config,
        r_max,
        r_max_lr,
        keyspec,
        valid_batch_size,
        is_train_preprocessed=(is_preprocessed or is_lazy),
        val_split_from_train=val_split,
        valid_subset=valid_subset,
    )

    logging.info(f"Training set size: {len(train_loader.dataset)}")
    if valid_loader is not None:
        logging.info(f"Validation set size: " f"{len(valid_loader.dataset)}")
    logging.info(
        f"Number of unique elements in training set: " f"{num_elements}"
    )
    return (
        train_loader,
        {"main": valid_loader} if valid_loader else {},
        train_sampler,
        avg_num_neighbors,
        num_elements,
        average_atomic_energy_shifts,
        replay_builder,
    )


def build_ewc_fisher_loader(config: dict) -> DataLoader:
    """Build a DataLoader for EWC Fisher information estimation.

    Loads the dataset at ``config["TRAINING"]["ewc_fisher_data"]``
    (XYZ, raw HDF5, or preprocessed HDF5), shuffles it, and returns a
    DataLoader with the training ``batch_size``.  Loads at most
    ``config['TRAINING']['ewc_num_fisher_samples']`` structures (default
    1000) into the DataLoader. ``compute_fisher()`` will iterate over
    all of them.

    Parameters
    ----------
    config:
        Full training configuration dict.

    Returns
    -------
    DataLoader
        Iterates over structures from the pretraining dataset subset.
    """
    path = config["TRAINING"]["ewc_fisher_data"]
    r_max = config["ARCHITECTURE"].get("r_max", None)
    r_max_lr = config["ARCHITECTURE"].get("r_max_lr", None)
    batch_size = config["TRAINING"]["batch_size"]
    num_samples = config["TRAINING"].get("ewc_num_fisher_samples", 1000)

    keydict = DefaultKeys.keydict()
    keydict.update(config["TRAINING"].get("keys") or {})
    keyspec = update_keyspec_from_kwargs(KeySpecification(), keydict)

    file_format = detect_file_format(path)
    logging.info(
        f"EWC: loading Fisher dataset from {path} " f"(format={file_format})"
    )

    if file_format == "hdf5_preprocessed":
        dataset = PreprocessedHDF5Dataset(
            hdf5_path=path,
            validate_cutoffs=True,
            expected_r_max=r_max,
            expected_r_max_lr=r_max_lr,
        )
        indices = list(range(len(dataset)))
        random.shuffle(indices)
        indices = indices[:num_samples]
        data = [dataset[i] for i in indices]
    elif path.endswith((".h5", ".hdf5")):
        with h5py.File(path, "r") as _f:
            _n = int(_f.attrs["num_configs"])
        sampled_indices = random.sample(range(_n), min(num_samples, _n))
        atoms_list = load_atoms_from_hdf5(path, index=sampled_indices)
        data = create_data_from_list(
            atoms_list,
            r_max=r_max,
            r_max_lr=r_max_lr,
            key_specification=keyspec,
            config_type_weights=config["TRAINING"].get(
                "config_type_weights", None
            ),
        )
    elif path.endswith((".xyz", ".extxyz")):
        # XYZ has no random-access; full read is unavoidable.
        # Large pretraining datasets should use HDF5 instead.
        atoms_list = list(read(path, index=":"))
        random.shuffle(atoms_list)
        atoms_list = atoms_list[:num_samples]
        data = create_data_from_list(
            atoms_list,
            r_max=r_max,
            r_max_lr=r_max_lr,
            key_specification=keyspec,
            config_type_weights=config["TRAINING"].get(
                "config_type_weights", None
            ),
        )
    else:
        raise ValueError(f"Unsupported ewc_fisher_data file format: {path}")

    return create_dataloader_from_data(
        config_list=data,
        batch_size=batch_size,
        shuffle=False,  # already shuffled above
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
             average_atomic_energy_shifts, replay_builder).

    replay_builder is a callable that returns
    (DataLoader, Optional[DistributedSampler]) for per-epoch
    replay resampling, or None when replay is disabled or
    resample_per_epoch is False.
    """
    if config["TRAINING"].get("heads", None) is not None:
        return _setup_multihead_data_loaders(
            config, distributed, rank, world_size
        )
    return _setup_singlehead_data_loaders(
        config, distributed, rank, world_size
    )
