"""Tests for fine-tuning utilities in tools/finetune.py.

Covers model_to_lora, fuse_lora_weights, freeze_model_parameters,
setup_finetuning, and preserve_grad_state. Uses a small So3krates
model (2 layers) for speed; fine-tuning logic is model-agnostic.
"""

import pytest
import torch

from so3krates_torch.blocks.euclidean_transformer import (
    EuclideanAttentionBlockDoRA,
    EuclideanAttentionBlockLORA,
    EuclideanAttentionBlockVeRA,
)
from so3krates_torch.modules.models import So3krates
from so3krates_torch.tools.finetune import (
    freeze_model_parameters,
    fuse_lora_weights,
    model_to_lora,
    preserve_grad_state,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def base_model(default_model_config):
    """So3krates model with 2 layers; required for last-layer tests."""
    config = {**default_model_config, "num_layers": 2}
    return So3krates(**config)


# ---------------------------------------------------------------------------
# model_to_lora
# ---------------------------------------------------------------------------


def test_model_to_lora_replaces_blocks(base_model):
    """LoRA conversion replaces every attention block with a LoRA variant."""
    model_to_lora(base_model, rank=4, alpha=8.0, device="cpu")

    for transformer in base_model.euclidean_transformers:
        assert isinstance(
            transformer.euclidean_attention_block,
            EuclideanAttentionBlockLORA,
        )


def test_model_to_lora_preserves_non_lora_weights(base_model):
    """LoRA conversion does not modify output-head parameters."""
    before = {
        name: param.detach().clone()
        for name, param in base_model.named_parameters()
        if "atomic_energy_output_block" in name
    }
    assert len(before) > 0, "No output-block params found; fixture broken"

    model_to_lora(base_model, rank=4, alpha=8.0, device="cpu")

    for name, orig in before.items():
        current = dict(base_model.named_parameters())[name]
        assert torch.allclose(current, orig), (
            f"Non-LoRA parameter {name} changed after model_to_lora"
        )


def test_model_to_dora(base_model):
    """DoRA conversion replaces every attention block with the DoRA variant."""
    model_to_lora(
        base_model, rank=4, alpha=8.0, use_dora=True, device="cpu"
    )

    for transformer in base_model.euclidean_transformers:
        assert isinstance(
            transformer.euclidean_attention_block,
            EuclideanAttentionBlockDoRA,
        )


def test_model_to_vera(base_model):
    """VeRA conversion replaces every attention block with the VeRA variant."""
    model_to_lora(
        base_model, rank=4, alpha=8.0, use_vera=True, device="cpu"
    )

    for transformer in base_model.euclidean_transformers:
        assert isinstance(
            transformer.euclidean_attention_block,
            EuclideanAttentionBlockVeRA,
        )


# ---------------------------------------------------------------------------
# fuse_lora_weights
# ---------------------------------------------------------------------------


def test_fuse_lora_weights(base_model):
    """Fusion removes LoRA parameters and marks block as fused."""
    model_to_lora(base_model, rank=4, alpha=8.0, device="cpu")

    first_block = (
        base_model.euclidean_transformers[0].euclidean_attention_block
    )
    assert hasattr(first_block, "lora_A_q_inv"), (
        "lora_A_q_inv missing before fusion; fixture broken"
    )

    fuse_lora_weights(base_model)

    assert not hasattr(first_block, "lora_A_q_inv"), (
        "lora_A_q_inv still present after fusion"
    )
    assert first_block.weights_fused


# ---------------------------------------------------------------------------
# freeze_model_parameters
# ---------------------------------------------------------------------------


def test_freeze_last_layer(base_model):
    """Only the last transformer layer and output head are trainable."""
    num_layers = len(base_model.euclidean_transformers)
    freeze_model_parameters(base_model, "last_layer")

    # Last transformer: all params trainable
    last_params = list(
        base_model.euclidean_transformers[num_layers - 1].parameters()
    )
    assert all(p.requires_grad for p in last_params)

    # First transformer: all params frozen
    first_params = list(
        base_model.euclidean_transformers[0].parameters()
    )
    assert all(not p.requires_grad for p in first_params)

    # Sanity: trainable is a strict subset of all params
    total = sum(p.numel() for p in base_model.parameters())
    trainable = sum(
        p.numel() for p in base_model.parameters() if p.requires_grad
    )
    assert trainable < total


def test_freeze_lora_only(base_model):
    """After LoRA conversion and freezing, only lora_ params are trainable."""
    model_to_lora(base_model, rank=4, alpha=8.0, device="cpu")
    freeze_model_parameters(
        base_model,
        "lora",
        freeze_shifts=True,
        freeze_scales=True,
    )

    for name, param in base_model.named_parameters():
        if "lora_" in name:
            assert param.requires_grad, f"{name} should be trainable"
        else:
            assert not param.requires_grad, f"{name} should be frozen"


# ---------------------------------------------------------------------------
# preserve_grad_state
# ---------------------------------------------------------------------------


def test_preserve_grad_state_restores(base_model):
    """Context manager restores requires_grad state on exit."""
    params = list(base_model.parameters())

    # Assign alternating grad state so the backup is non-trivial
    for i, param in enumerate(params):
        param.requires_grad = i % 2 == 0
    original_state = {id(p): p.requires_grad for p in params}

    with preserve_grad_state(base_model):
        # Inside: all gradients are disabled
        for param in base_model.parameters():
            assert not param.requires_grad

    # After exit: original state fully restored
    for param in base_model.parameters():
        assert param.requires_grad == original_state[id(param)]
