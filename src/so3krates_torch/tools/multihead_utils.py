import inspect
import torch
from so3krates_torch.modules.models import So3krates, SO3LR, MultiHeadSO3LR
from so3krates_torch.blocks.output_block import MultiAtomicEnergyOutputHead
import logging


def _constructor_kwargs(*classes) -> set:
    """Union of __init__ keyword-parameter names across `classes`.

    Skips `self` and *args/**kwargs so that a subclass forwarding via
    `super().__init__(*args, **kwargs)` (SO3LR -> So3krates,
    MultiHeadSO3LR -> SO3LR) still surfaces the base class's real
    parameter names through the union.
    """
    names = set()
    for cls in classes:
        for name, param in inspect.signature(cls.__init__).parameters.items():
            if name == "self" or param.kind in (
                inspect.Parameter.VAR_POSITIONAL,
                inspect.Parameter.VAR_KEYWORD,
            ):
                continue
            names.add(name)
    return names


def _filter_architecture_settings(settings: dict, *classes) -> dict:
    """Keep only the keys `settings` that at least one of `classes`
    accepts as a constructor keyword argument.

    Used instead of an exclude-list so that unrelated
    ArchitectureConfig fields (the section allows extra keys, and a
    real config dict always carries every schema field once it has
    been through a pydantic model_dump()) never reach the model
    constructor and raise TypeError.
    """
    accepted = _constructor_kwargs(*classes)
    return {k: v for k, v in settings.items() if k in accepted}


def _broadcast_output_head_weights(sh_block, mh_block, num_heads: int):
    """Copy a single-head output block's weights into every head slice
    of a multi-head output block, so each head starts as an exact
    duplicate of the pretrained head.

    This is the inverse of the per-head extraction done by
    ``reduce_mh_model_to_sh``.
    """
    mh_block.energy_shifts.data = sh_block.energy_shifts.data.clone()
    # single-head: nn.Linear(num_elements, 1, bias=False) -> weight (1, E)
    # multi-head: plain (E, 1) Parameter shared across heads
    mh_block.energy_scales.data = sh_block.energy_scales.weight.data.T.clone()

    for i, sh_layer in enumerate(sh_block.layers):
        mh_block.layers_weights[i].data = sh_layer.weight.data.T.unsqueeze(
            0
        ).repeat(num_heads, 1, 1)
        mh_block.layers_bias[i].data = sh_layer.bias.data.unsqueeze(0).repeat(
            num_heads, 1
        )

    mh_block.final_layer_weights.data = (
        sh_block.final_layer.weight.data.T.unsqueeze(0).repeat(num_heads, 1, 1)
    )
    mh_block.final_layer_bias.data = sh_block.final_layer.bias.data.unsqueeze(
        0
    ).repeat(num_heads, 1)


def model_to_multihead(
    model: torch.nn.Module,
    num_output_heads: int,
    settings: dict,
    device: str = "cpu",
):
    source_block = model.atomic_energy_output_block
    source_is_multihead = isinstance(source_block, MultiAtomicEnergyOutputHead)
    if (
        source_is_multihead
        and source_block.num_output_heads != num_output_heads
    ):
        raise ValueError(
            "Cannot convert a model that is already multi-head with "
            f"{source_block.num_output_heads} heads to "
            f"num_output_heads={num_output_heads}: the head count "
            "must match when the source model is already multi-head "
            "(there is no single pretrained head left to duplicate)."
        )

    settings = _filter_architecture_settings(
        settings, MultiHeadSO3LR, SO3LR, So3krates
    )
    settings.pop("num_output_heads", None)  # passed explicitly below
    mh_model = MultiHeadSO3LR(num_output_heads=num_output_heads, **settings)
    mh_model.load_state_dict(
        model.state_dict(),
        strict=False,  # Allow missing keys for new output heads
    )
    if source_is_multihead:
        logging.info(
            "Source model is already multi-head; its per-head "
            "weights were already carried over by load_state_dict, "
            "skipping head duplication."
        )
    else:
        _broadcast_output_head_weights(
            source_block,
            mh_model.atomic_energy_output_block,
            num_output_heads,
        )
    model = mh_model.to(device)

    return model


def pretrained_to_mh_model(
    architecture_settings: dict,
    model: torch.nn.Module,
    device_name: str,
    log: bool = False,
) -> None:
    if log:
        logging.info(
            "Converting pretrained model to multi-head format. WARNING: Using provided settings for conversion."
        )

    num_output_heads = architecture_settings.get("num_output_heads", None)
    assert (
        num_output_heads is not None
    ), "num_output_heads must be specified for multi-head model"

    settings = dict(architecture_settings)
    settings["dtype"] = architecture_settings.get("default_dtype", "float32")
    settings["device"] = device_name
    model = model_to_multihead(
        model=model,
        settings=settings,
        num_output_heads=num_output_heads,
        device=device_name,
    )
    return model


def reduce_mh_model_to_sh(
    mh_model_state_dict,
    settings: dict,
    head_idx: int,
    model_choice: str = "so3lr",
    device: str = "cpu",
    dtype: str = "float32",
):
    num_layers = settings.get("final_mlp_layers", 2)
    settings = dict(settings)
    settings["device"] = device
    settings["dtype"] = dtype

    if model_choice == "so3lr":
        sh_model = SO3LR(
            **_filter_architecture_settings(settings, SO3LR, So3krates)
        )
    elif model_choice == "so3krates":
        sh_model = So3krates(
            **_filter_architecture_settings(settings, So3krates)
        )

    sh_model.load_state_dict(mh_model_state_dict, strict=False)

    # load energy scales
    mh_energy_scales = mh_model_state_dict[
        "atomic_energy_output_block.energy_scales"
    ].T
    sh_model.atomic_energy_output_block.energy_scales.weight.data = (
        mh_energy_scales.to(dtype=getattr(torch, dtype), device=device)
    )

    for layer in range(num_layers - 1):
        mh_weight = mh_model_state_dict[
            f"atomic_energy_output_block.layers_weights.{layer}"
        ]
        mh_bias = mh_model_state_dict[
            f"atomic_energy_output_block.layers_bias.{layer}"
        ]
        mh_weight = (
            mh_weight[head_idx].T
            if mh_weight[head_idx].ndim == 2
            else mh_weight[head_idx]
        )
        mh_bias = (
            mh_bias[head_idx].T
            if mh_bias[head_idx].ndim == 2
            else mh_bias[head_idx]
        )
        # Use .data to replace in-place
        sh_model.atomic_energy_output_block.layers[
            layer
        ].weight.data = mh_weight.to(
            dtype=getattr(torch, dtype), device=device
        )
        sh_model.atomic_energy_output_block.layers[
            layer
        ].bias.data = mh_bias.to(dtype=getattr(torch, dtype), device=device)
    # Final layer
    mh_weight = mh_model_state_dict[
        f"atomic_energy_output_block.final_layer_weights"
    ]
    mh_bias = mh_model_state_dict[
        f"atomic_energy_output_block.final_layer_bias"
    ]
    mh_weight = (
        mh_weight[head_idx].T
        if mh_weight[head_idx].ndim == 2
        else mh_weight[head_idx]
    )
    mh_bias = (
        mh_bias[head_idx].T
        if mh_bias[head_idx].ndim == 2
        else mh_bias[head_idx]
    )

    sh_model.atomic_energy_output_block.final_layer.weight.data = mh_weight.to(
        dtype=getattr(torch, dtype), device=device
    )
    sh_model.atomic_energy_output_block.final_layer.bias.data = mh_bias.to(
        dtype=getattr(torch, dtype), device=device
    )
    return sh_model
