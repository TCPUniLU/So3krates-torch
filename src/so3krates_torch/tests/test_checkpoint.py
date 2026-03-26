"""Tests for checkpointing utilities in tools/checkpoint.py.

Covers CheckpointBuilder, CheckpointIO, and CheckpointHandler.
Uses minimal nn.Linear models for speed; checkpoint logic is
model-agnostic.
"""

import pytest
import torch
import torch.nn as nn

from so3krates_torch.tools.checkpoint import (
    CheckpointBuilder,
    CheckpointHandler,
    CheckpointIO,
    CheckpointState,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def minimal_state():
    """Minimal CheckpointState with a small linear model."""
    model = nn.Linear(10, 5)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
    lr_scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    return CheckpointState(
        model=model,
        optimizer=optimizer,
        lr_scheduler=lr_scheduler,
    )


# ---------------------------------------------------------------------------
# CheckpointBuilder
# ---------------------------------------------------------------------------


def test_checkpoint_builder_create_keys(minimal_state):
    """Checkpoint dict contains model, optimizer, and lr_scheduler keys."""
    checkpoint = CheckpointBuilder.create_checkpoint(minimal_state)

    assert "model" in checkpoint
    assert "optimizer" in checkpoint
    assert "lr_scheduler" in checkpoint


def test_checkpoint_builder_roundtrip(minimal_state):
    """Loading a checkpoint into a fresh state restores model parameters."""
    checkpoint = CheckpointBuilder.create_checkpoint(minimal_state)

    new_model = nn.Linear(10, 5)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    new_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        new_optimizer, gamma=0.9
    )
    new_state = CheckpointState(
        model=new_model,
        optimizer=new_optimizer,
        lr_scheduler=new_scheduler,
    )
    CheckpointBuilder.load_checkpoint(new_state, checkpoint, strict=True)

    for p_orig, p_new in zip(
        minimal_state.model.parameters(), new_state.model.parameters()
    ):
        assert torch.allclose(p_orig, p_new)


# ---------------------------------------------------------------------------
# CheckpointIO
# ---------------------------------------------------------------------------


def test_checkpoint_io_save_creates_file(minimal_state, tmp_path):
    """Saving creates a file with the expected name."""
    checkpoint = CheckpointBuilder.create_checkpoint(minimal_state)
    io = CheckpointIO(directory=str(tmp_path), tag="test")
    io.save(checkpoint, epochs=10)

    assert (tmp_path / "test_epoch-10.pt").exists()


def test_checkpoint_io_delete_old(minimal_state, tmp_path):
    """With keep=False, saving a new epoch deletes the previous file."""
    checkpoint = CheckpointBuilder.create_checkpoint(minimal_state)
    io = CheckpointIO(directory=str(tmp_path), tag="test", keep=False)
    io.save(checkpoint, epochs=5)
    io.save(checkpoint, epochs=10)

    assert not (tmp_path / "test_epoch-5.pt").exists()
    assert (tmp_path / "test_epoch-10.pt").exists()


def test_checkpoint_io_keep_old(minimal_state, tmp_path):
    """With keep=True, all saved checkpoint files are retained."""
    checkpoint = CheckpointBuilder.create_checkpoint(minimal_state)
    io = CheckpointIO(directory=str(tmp_path), tag="test", keep=True)
    io.save(checkpoint, epochs=5)
    io.save(checkpoint, epochs=10)

    assert (tmp_path / "test_epoch-5.pt").exists()
    assert (tmp_path / "test_epoch-10.pt").exists()


def test_checkpoint_io_swa_filename(minimal_state, tmp_path):
    """Checkpoint saved after swa_start has _swa in the filename."""
    checkpoint = CheckpointBuilder.create_checkpoint(minimal_state)
    io = CheckpointIO(directory=str(tmp_path), tag="test", swa_start=50)
    io.save(checkpoint, epochs=60)

    assert (tmp_path / "test_epoch-60_swa.pt").exists()


def test_checkpoint_io_load_latest(minimal_state, tmp_path):
    """load_latest returns the checkpoint saved at the highest epoch."""
    io = CheckpointIO(directory=str(tmp_path), tag="test", keep=True)
    for epoch in [5, 10, 15]:
        checkpoint = CheckpointBuilder.create_checkpoint(minimal_state)
        io.save(checkpoint, epochs=epoch)

    result = io.load_latest()
    assert result is not None
    _, returned_epoch = result
    assert returned_epoch == 15


# ---------------------------------------------------------------------------
# CheckpointHandler (integration)
# ---------------------------------------------------------------------------


def test_checkpoint_handler_full_cycle(minimal_state, tmp_path):
    """Handler save then load_latest restores model weights end-to-end."""
    handler = CheckpointHandler(directory=str(tmp_path), tag="test")
    handler.save(minimal_state, epochs=10)

    new_model = nn.Linear(10, 5)
    new_optimizer = torch.optim.Adam(new_model.parameters(), lr=0.001)
    new_scheduler = torch.optim.lr_scheduler.ExponentialLR(
        new_optimizer, gamma=0.9
    )
    new_state = CheckpointState(
        model=new_model,
        optimizer=new_optimizer,
        lr_scheduler=new_scheduler,
    )

    returned_epoch = handler.load_latest(new_state, strict=True)

    assert returned_epoch == 10
    for p_orig, p_new in zip(
        minimal_state.model.parameters(), new_state.model.parameters()
    ):
        assert torch.allclose(p_orig, p_new)
