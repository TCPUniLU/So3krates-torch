import logging
import os
from typing import Union, Optional

import torch
import torch.nn as nn
from so3krates_torch.modules.models import SO3LR, MultiHeadSO3LR
from so3krates_torch.tools.finetune import fuse_lora_weights, setup_finetuning
from so3krates_torch.tools.utils import AtomicNumberTable

DTYPE_MAP = {
    "float32": torch.float32,
    "float64": torch.float64,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
}


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


def set_dtype_model(model: torch.nn.Module, dtype_str: str) -> None:
    if dtype_str not in DTYPE_MAP:
        raise ValueError(f"Unsupported dtype: {dtype_str}")
    dtype = DTYPE_MAP[dtype_str]
    # set all model params to dtype
    for param in model.parameters():
        param.data = param.data.to(dtype)


def determine_num_elements(data_loader: torch.utils.data.DataLoader) -> int:
    """Determine the number of unique elements in the dataset."""
    unique_elements = set()
    for batch in data_loader:
        one_hot = batch["node_attrs"]
        elements = one_hot.argmax(dim=-1).flatten().tolist()
        unique_elements.update(elements)
    return len(unique_elements)


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
