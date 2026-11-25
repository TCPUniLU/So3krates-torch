import contextlib
import copy
import logging
from typing import Optional

import torch

from so3krates_torch.blocks.output_block import (
    AtomicEnergyOutputHead,
    MultiAtomicEnergyOutputHead,
)
from so3krates_torch.modules.models import So3krates, SO3LR, MultiHeadSO3LR


@contextlib.contextmanager
def _isolated_init(dtype: torch.dtype, device: torch.device, seed):
    """Build new modules in ``dtype`` without leaking global state.

    Restores ``torch.get_default_dtype()`` and the torch RNG state on
    exit; seeds the RNG inside the block when ``seed`` is given.
    """
    previous = torch.get_default_dtype()
    devices = [device] if device.type == "cuda" else []
    try:
        with torch.random.fork_rng(devices=devices):
            if seed is not None:
                torch.manual_seed(seed)
            torch.set_default_dtype(dtype)
            yield
    finally:
        torch.set_default_dtype(previous)


def _deepcopy_model(model: torch.nn.Module) -> torch.nn.Module:
    """``copy.deepcopy`` that tolerates tensors cached by a forward pass.

    The models keep geometry from their last forward (``vectors``,
    ``lengths``, ...) as attributes; those are attached to the autograd
    graph and cannot be deep-copied (also inside the cached ``ctx``
    tuple). They are copied detached instead;
    the source model is not modified.
    """
    memo = {}

    def collect(value):
        if torch.is_tensor(value):
            if value.grad_fn is not None:
                memo[id(value)] = value.detach().clone()
        elif isinstance(value, dict):
            for item in value.values():
                collect(item)
        elif isinstance(value, (list, tuple)):
            for item in value:
                collect(item)

    for module in model.modules():
        for name, value in vars(module).items():
            if name not in ("_parameters", "_buffers", "_modules"):
                collect(value)
    return copy.deepcopy(model, memo)


def _head_spec(head: AtomicEnergyOutputHead) -> dict:
    """Shapes and settings of a single-head energy MLP, read from its
    tensors rather than from model attributes or config."""
    first = head.layers[0] if len(head.layers) else head.final_layer
    weight = head.final_layer.weight
    return {
        "num_features": first.in_features,
        "energy_regression_dim": head.final_layer.in_features,
        "layers": len(head.layers) + 1,
        "non_linearity": type(head.non_linearity),
        "num_elements": head.energy_shifts.shape[0],
        "energy_learn_atomic_type_shifts": (
            head.energy_learn_atomic_type_shifts
        ),
        "energy_learn_atomic_type_scales": (
            head.energy_learn_atomic_type_scales
        ),
        "dtype": weight.dtype,
        "device": weight.device,
    }


def to_multihead_model(
    model: SO3LR,
    num_output_heads: int,
    seed: Optional[int] = None,
    theory_level: Optional[int] = None,
) -> MultiHeadSO3LR:
    """Return a ``MultiHeadSO3LR`` with ``model``'s backbone and
    ``num_output_heads`` freshly initialised, independent energy heads.

    The model is deep-copied and only ``atomic_energy_output_block`` is
    replaced, so backbone weights, ``r_max_lr``, dtype, device and all
    non-weight settings are kept. Per-element energy shifts and scales
    are copied into the new head. ``model`` is not modified.
    """
    if isinstance(model, MultiHeadSO3LR):
        raise ValueError(
            "Model is already a MultiHeadSO3LR; use to_single_head_model "
            "to extract a head first."
        )
    if not isinstance(model, SO3LR):
        raise ValueError(
            "to_multihead_model requires an SO3LR model, got "
            f"{type(model).__name__}."
        )
    if num_output_heads < 1:
        raise ValueError(
            f"num_output_heads must be >= 1, got {num_output_heads}."
        )
    src = model.atomic_energy_output_block
    if getattr(src, "num_theory_levels", 1) > 1:
        if theory_level is None:
            raise ValueError(
                f"Model has {src.num_theory_levels} theory levels; pass "
                "theory_level to choose the one to keep."
            )
        raise NotImplementedError("theory_level selection")
    if theory_level is not None:
        raise NotImplementedError("theory_level selection")

    spec = _head_spec(src)
    new = _deepcopy_model(model)
    with _isolated_init(spec["dtype"], spec["device"], seed):
        head = MultiAtomicEnergyOutputHead(
            num_output_heads=num_output_heads,
            num_features=spec["num_features"],
            energy_regression_dim=spec["energy_regression_dim"],
            final_output_features=1,
            layers=spec["layers"],
            bias=True,
            use_non_linearity=True,
            non_linearity=spec["non_linearity"],
            final_non_linearity=False,
            atomic_type_shifts=None,
            energy_learn_atomic_type_shifts=spec[
                "energy_learn_atomic_type_shifts"
            ],
            energy_learn_atomic_type_scales=spec[
                "energy_learn_atomic_type_scales"
            ],
            num_elements=spec["num_elements"],
            device=spec["device"],
        )
    head = head.to(device=spec["device"], dtype=spec["dtype"])
    with torch.no_grad():
        head.energy_shifts.copy_(src.energy_shifts)
        head.energy_scales.copy_(src.energy_scales.weight.T)

    new.atomic_energy_output_block = head
    new.__class__ = MultiHeadSO3LR
    new.num_output_heads = num_output_heads
    new.select_heads = False
    new.return_mean = False
    return new


def model_to_multihead(
    model: torch.nn.Module,
    num_output_heads: int,
    settings: dict,
    device: str = "cpu",
):
    mh_model = MultiHeadSO3LR(num_output_heads=num_output_heads, **settings)
    mh_model.load_state_dict(
        model.state_dict(),
        strict=False,  # Allow missing keys for new output heads
    )
    model = mh_model.to(device)

    return model


def to_single_head_model(model: MultiHeadSO3LR, head_idx: int) -> SO3LR:
    """Return an ``SO3LR`` that keeps head ``head_idx`` of a
    ``MultiHeadSO3LR`` (0-based; negative indices are rejected).

    The model is deep-copied and only ``atomic_energy_output_block`` is
    replaced, so the backbone and all settings are kept. Its energy
    equals the multi-head model's energy for that head. ``model`` is
    not modified.
    """
    if not isinstance(model, MultiHeadSO3LR):
        raise ValueError(
            "to_single_head_model requires a MultiHeadSO3LR model, got "
            f"{type(model).__name__}."
        )
    num_heads = model.num_output_heads
    if not 0 <= head_idx < num_heads:
        raise ValueError(
            f"head_idx must be in [0, {num_heads - 1}], got {head_idx}."
        )

    src = model.atomic_energy_output_block
    final_w = src.final_layer_weights  # (heads, in, 1)
    hidden = list(zip(src.layers_weights, src.layers_bias))
    num_features = hidden[0][0].shape[1] if hidden else final_w.shape[1]
    dtype, device = final_w.dtype, final_w.device

    new = _deepcopy_model(model)
    with _isolated_init(dtype, device, seed=None):
        head = AtomicEnergyOutputHead(
            num_features=num_features,
            energy_regression_dim=final_w.shape[1],
            final_output_features=1,
            layers=len(hidden) + 1,
            bias=True,
            final_layer_bias=True,
            use_non_linearity=True,
            non_linearity=type(src.non_linearity),
            final_non_linearity=False,
            atomic_type_shifts=None,
            energy_learn_atomic_type_shifts=(
                src.energy_learn_atomic_type_shifts
            ),
            energy_learn_atomic_type_scales=(
                src.energy_learn_atomic_type_scales
            ),
            num_elements=src.energy_shifts.shape[0],
            num_theory_levels=1,
            device=device,
        )
    head = head.to(device=device, dtype=dtype)
    with torch.no_grad():
        for layer, (weight, bias) in zip(head.layers, hidden):
            layer.weight.copy_(weight[head_idx].T)
            layer.bias.copy_(bias[head_idx])
        head.final_layer.weight.copy_(final_w[head_idx].T)
        head.final_layer.bias.copy_(src.final_layer_bias[head_idx])
        head.energy_shifts.copy_(src.energy_shifts)
        head.energy_scales.weight.copy_(src.energy_scales.T)

    new.atomic_energy_output_block = head
    new.__class__ = SO3LR
    for attr in ("num_output_heads", "select_heads", "return_mean"):
        if attr in new.__dict__:
            delattr(new, attr)
    return new


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

    exclude = [
        "convert_to_multihead",
        "use_multihead",
        "num_output_heads",
        "r_max_lr",
    ]
    settings = {
        k: v for k, v in architecture_settings.items() if k not in exclude
    }
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
    exclude = [
        "convert_to_multihead",
        "use_multihead",
        "num_output_heads",
        "r_max_lr",
    ]
    settings = {k: v for k, v in settings.items() if k not in exclude}
    settings["device"] = device
    settings["dtype"] = dtype

    if model_choice == "so3lr":
        sh_model = SO3LR(**settings)
    elif model_choice == "so3krates":
        sh_model = So3krates(**settings)

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
        sh_model.atomic_energy_output_block.layers[layer].weight.data = (
            mh_weight.to(dtype=getattr(torch, dtype), device=device)
        )
        sh_model.atomic_energy_output_block.layers[layer].bias.data = (
            mh_bias.to(dtype=getattr(torch, dtype), device=device)
        )
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
