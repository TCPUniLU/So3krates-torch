from contextlib import contextmanager
from random import choice
import torch
from so3krates_torch.blocks.euclidean_transformer import (
    EuclideanAttentionBlockLORA,
    EuclideanAttentionBlockVeRA,
    EuclideanAttentionBlockDoRA,
    EuclideanAttentionBlock,
)
from so3krates_torch.tools.multihead_utils import pretrained_to_mh_model
from so3krates_torch.tools.utils import report_count_params
import math
import logging

POSSIBLE_FINETUNING_CHOICES = [
    "naive",
    "last_layer",
    "mlp",
    "qkv",
    "lora",
    "dora",
    "vera",
    "last_layer+mlp",
    "qkv+mlp",
    "lora+mlp",
]


@contextmanager
def preserve_grad_state(model):
    """
    Taken from
    https://github.com/ACEsuit/mace/pull/830/
    https://doi.org/10.1038/s41524-025-01727-x
    """
    # save the original requires_grad state for all parameters
    requires_grad_backup = {
        param: param.requires_grad for param in model.parameters()
    }
    try:
        # temporarily disable gradients for all parameters
        for param in model.parameters():
            param.requires_grad = False
        yield  # perform evaluation here
    finally:
        # restore the original requires_grad states
        for param, requires_grad in requires_grad_backup.items():
            param.requires_grad = requires_grad


def model_to_lora(
    model,
    rank: int = 4,
    alpha: int = 8,
    freeze_A: bool = False,
    use_dora: bool = False,
    scaling_to_one: bool = True,
    use_vera: bool = False,
    device: str = "cpu",
    seed: int = 42,
):
    """
    Convert a model to LoRA (Low-Rank Adaptation) format.
    This function modifies the model in place, replacing the normal euclidean attention blocks
    with their LoRA counterparts.
    """
    torch.manual_seed(seed)
    for i, transformer in enumerate(model.euclidean_transformers):
        if isinstance(
            transformer.euclidean_attention_block, EuclideanAttentionBlock
        ):
            degrees = transformer.euclidean_attention_block.degrees
            num_heads = transformer.euclidean_attention_block.num_heads
            num_features = transformer.euclidean_attention_block.num_features
            filter_net_inv = (
                transformer.euclidean_attention_block.filter_net_inv
            )
            filter_net_ev = transformer.euclidean_attention_block.filter_net_ev
            message_normalization = (
                transformer.euclidean_attention_block.message_normalization
            )
            qk_non_linearity = (
                transformer.euclidean_attention_block.qk_non_linearity
            )
            avg_num_neighbors = (
                transformer.euclidean_attention_block.avg_num_neighbors
            )
            inv_heads = transformer.euclidean_attention_block.inv_heads
            inv_head_dim = transformer.euclidean_attention_block.inv_head_dim
            ev_heads = transformer.euclidean_attention_block.ev_heads
            ev_head_dim = transformer.euclidean_attention_block.ev_head_dim

            # Create a new LoRA attention block with the same parameters
            if use_dora:
                lora_attention_block = EuclideanAttentionBlockDoRA(
                    degrees=degrees,
                    num_heads=num_heads,
                    num_features=num_features,
                    filter_net_inv=filter_net_inv,
                    filter_net_ev=filter_net_ev,
                    lora_rank=rank,
                    lora_alpha=alpha,
                    scaling_to_one=scaling_to_one,
                    message_normalization=message_normalization,
                    qk_non_linearity=qk_non_linearity,
                    avg_num_neighbors=avg_num_neighbors,
                    device=device,
                )

            elif use_vera:
                vera_A_matrix_inv = torch.nn.Parameter(
                    torch.empty((inv_heads, inv_head_dim, rank), device=device)
                )
                vera_B_matrix_inv = torch.nn.Parameter(
                    torch.empty((inv_heads, rank, inv_head_dim), device=device)
                )
                vera_A_matrix_ev = torch.nn.Parameter(
                    torch.empty((ev_heads, ev_head_dim, rank), device=device)
                )
                vera_B_matrix_ev = torch.nn.Parameter(
                    torch.empty((ev_heads, rank, ev_head_dim), device=device)
                )
                torch.nn.init.kaiming_uniform_(
                    vera_A_matrix_inv, a=math.sqrt(5)
                )
                torch.nn.init.kaiming_uniform_(
                    vera_B_matrix_inv, a=math.sqrt(5)
                )
                torch.nn.init.kaiming_uniform_(
                    vera_A_matrix_ev, a=math.sqrt(5)
                )
                torch.nn.init.kaiming_uniform_(
                    vera_B_matrix_ev, a=math.sqrt(5)
                )

                lora_attention_block = EuclideanAttentionBlockVeRA(
                    degrees=degrees,
                    num_heads=num_heads,
                    num_features=num_features,
                    vera_A_matrix_inv=vera_A_matrix_inv,
                    vera_B_matrix_inv=vera_B_matrix_inv,
                    vera_A_matrix_ev=vera_A_matrix_ev,
                    vera_B_matrix_ev=vera_B_matrix_ev,
                    filter_net_inv=filter_net_inv,
                    filter_net_ev=filter_net_ev,
                    lora_rank=rank,
                    lora_alpha=alpha,
                    message_normalization=message_normalization,
                    qk_non_linearity=qk_non_linearity,
                    avg_num_neighbors=avg_num_neighbors,
                    device=device,
                )
            else:
                lora_attention_block = EuclideanAttentionBlockLORA(
                    degrees=degrees,
                    num_heads=num_heads,
                    num_features=num_features,
                    filter_net_inv=filter_net_inv,
                    filter_net_ev=filter_net_ev,
                    lora_rank=rank,
                    lora_alpha=alpha,
                    freeze_A=freeze_A,
                    message_normalization=message_normalization,
                    qk_non_linearity=qk_non_linearity,
                    avg_num_neighbors=avg_num_neighbors,
                    device=device,
                )

            # Copy weights from the original attention block to the LoRA block
            lora_attention_block.load_state_dict(
                transformer.euclidean_attention_block.state_dict(),
                strict=False,  # Allow missing keys for LoRA-specific parameters
            )

            if use_dora:
                lora_attention_block.get_magnitude_vectors()

            # Replace the original attention block with the LoRA version
            transformer.euclidean_attention_block = lora_attention_block

    return model


def fuse_lora_weights(model):
    """
    Fuse the LoRA weights into the main model weights.
    This function modifies the model in place, combining the LoRA weights with the original weights.
    """
    for transformer in model.euclidean_transformers:
        if isinstance(
            transformer.euclidean_attention_block, EuclideanAttentionBlockLORA
        ):
            transformer.euclidean_attention_block.fuse_lora_weights()
    return model


def freeze_model_parameters(
    model,
    keep_trainable_choice: str,
    freeze_shifts: bool = False,
    freeze_scales: bool = False,
    freeze_partial_charges: bool = True,
    freeze_zbl: bool = True,
    freeze_hirshfeld: bool = True,
    freeze_embedding: bool = True,
    freeze_lora_A: bool = False,
):
    if keep_trainable_choice not in POSSIBLE_FINETUNING_CHOICES:
        raise ValueError(
            f"Invalid choice '{keep_trainable_choice}'. Must be one of {POSSIBLE_FINETUNING_CHOICES}."
        )

    keep_trainable = []

    # keep output heads trainable (energy, forces, etc.)
    if keep_trainable_choice in [
        "last_layer+mlp",
        "qkv+mlp",
        "mlp",
        "lora+mlp",
        "last_layer",
    ]:
        keep_trainable.append("atomic_energy_output_block.layers")
        keep_trainable.append("atomic_energy_output_block.final_layer")

    # Determine which transformer layers to keep trainable based on choice
    num_layers = len(model.euclidean_transformers)

    if (
        keep_trainable_choice == "last_layer"
        or keep_trainable_choice == "last_layer+mlp"
    ):
        # Keep only the last transformer layer trainable
        keep_trainable.append(f"euclidean_transformers.{num_layers-1}")

    elif keep_trainable_choice == "qkv" or keep_trainable_choice == "qkv+mlp":
        # Keep only the QKV projection layers trainable
        for i in range(num_layers):
            keep_trainable.append(
                f"euclidean_transformers.{i}.euclidean_attention_block"
            )

    elif (
        keep_trainable_choice == "lora" or keep_trainable_choice == "lora+mlp"
    ):
        # Keep only the LoRA parameters trainable
        for i in range(num_layers):
            if freeze_lora_A:
                keep_trainable.append(
                    f"euclidean_transformers.{i}.euclidean_attention_block.lora_B_"
                )
            else:
                keep_trainable.append(
                    f"euclidean_transformers.{i}.euclidean_attention_block.lora_"
                )

    elif keep_trainable_choice == "dora":
        # Keep only the DoRA parameters trainable
        for i in range(num_layers):
            keep_trainable.append(
                f"euclidean_transformers.{i}.euclidean_attention_block.lora_"
            )
            keep_trainable.append(
                f"euclidean_transformers.{i}.euclidean_attention_block.dora_"
            )
    elif keep_trainable_choice == "vera":
        # Keep only the VeRA parameters trainable
        for i in range(num_layers):
            keep_trainable.append(
                f"euclidean_transformers.{i}.euclidean_attention_block.d_k_"
            )
            keep_trainable.append(
                f"euclidean_transformers.{i}.euclidean_attention_block.d_q_"
            )
            keep_trainable.append(
                f"euclidean_transformers.{i}.euclidean_attention_block.d_v_"
            )
            keep_trainable.append(
                f"euclidean_transformers.{i}.euclidean_attention_block.b_k_"
            )
            keep_trainable.append(
                f"euclidean_transformers.{i}.euclidean_attention_block.b_q_"
            )
            keep_trainable.append(
                f"euclidean_transformers.{i}.euclidean_attention_block.b_v_"
            )

    # Handle optional components based on freeze flags
    if not freeze_shifts:
        keep_trainable.append("atomic_energy_output_block.energy_shifts")

    if not freeze_scales:
        keep_trainable.append("atomic_energy_output_block.energy_scales")

    if (
        not freeze_partial_charges
        and hasattr(model, "electrostatic_energy_bool")
        and model.electrostatic_energy_bool
    ):
        keep_trainable.append("partial_charges_output_block")

    if (
        not freeze_zbl
        and hasattr(model, "zbl_repulsion_bool")
        and model.zbl_repulsion_bool
    ):
        keep_trainable.append("zbl_repulsion")

    if (
        not freeze_hirshfeld
        and hasattr(model, "dispersion_energy_bool")
        and model.dispersion_energy_bool
    ):
        keep_trainable.append("hirshfeld_output_block")

    if not freeze_embedding:
        keep_trainable.append("embedding")

    # First, freeze all parameters
    for param in model.parameters():
        param.requires_grad = False

    # Then unfreeze only the specified ones
    for name, param in model.named_parameters():
        for keep_key in keep_trainable:
            if keep_key in name:
                param.requires_grad = True
                break  # Found match, no need to check other keys


def setup_finetuning(
    model: torch.nn.Module,
    finetune_choice: str,
    device_name: str,
    num_elements: int = None,
    freeze_embedding: bool = True,
    freeze_zbl: bool = True,
    freeze_hirshfeld: bool = True,
    freeze_partial_charges: bool = True,
    freeze_shifts: bool = False,
    freeze_scales: bool = False,
    convert_to_lora: bool = True,
    lora_rank: int = 4,
    lora_alpha: float = None,
    lora_freeze_A: bool = False,
    dora_scaling_to_one: bool = True,
    convert_to_multihead: bool = False,
    architecture_settings: dict = None,
    seed: int = 42,
    log: bool = False,
) -> torch.nn.Module:
    # TODO: docstring etc
    assert (
        finetune_choice in POSSIBLE_FINETUNING_CHOICES
    ), f"Invalid finetuning choice '{finetune_choice}'. Must be one of {POSSIBLE_FINETUNING_CHOICES}."

    # unfreeze all parameters in case loaded model has frozen params
    for param in model.parameters():
        param.requires_grad = True

    if log:
        logging.info(f"Setting up finetuning with choice: {finetune_choice}")

    if convert_to_multihead:
        assert (
            architecture_settings is not None
        ), "architecture_settings must be provided when convert_to_multihead is True"
        model = pretrained_to_mh_model(
            architecture_settings, model, device_name
        )

    if (
        finetune_choice == "lora"
        or finetune_choice == "lora+mlp"
        and convert_to_lora
    ):
        lora_alpha = 2.0 * lora_rank if lora_alpha is None else lora_alpha
        model = model_to_lora(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            freeze_A=lora_freeze_A,
            device=device_name,
            seed=seed,
        )
        if log:
            logging.info("Converted model to LoRA format")

    elif finetune_choice == "dora":
        model = model_to_lora(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            use_dora=True,
            device=device_name,
            scaling_to_one=dora_scaling_to_one,
        )
        logging.info("Converted model to DoRA format")

    elif finetune_choice == "vera":
        model = model_to_lora(
            model,
            rank=lora_rank,
            alpha=lora_alpha,
            use_vera=True,
            device=device_name,
            scaling_to_one=dora_scaling_to_one,
        )
        if log:
            logging.info("Converted model to VeRA format")

    if finetune_choice != "naive":
        freeze_model_parameters(
            model,
            finetune_choice,
            freeze_shifts=freeze_shifts,
            freeze_scales=freeze_scales,
            freeze_lora_A=lora_freeze_A,
            freeze_embedding=freeze_embedding,
            freeze_zbl=freeze_zbl,
            freeze_hirshfeld=freeze_hirshfeld,
            freeze_partial_charges=freeze_partial_charges,
        )

    if log:
        assert (
            num_elements is not None
        ), "num_elements must be provided for parameter reporting"
        use_electrostatics = model.electrostatic_energy_bool
        use_dispersion = model.dispersion_energy_bool
        report_count_params(
            model, num_elements, use_electrostatics, use_dispersion
        )
    return model
