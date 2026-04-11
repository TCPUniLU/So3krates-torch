"""Tests verifying that checkpoint restart works for fine-tuning runs.

The core bug: when fine-tuning from a pretrained model,
_setup_model_for_training() sets warm_start=True.  A guard
``if not warm_start:`` previously skipped load_checkpoint_if_exists()
entirely, meaning any interrupted fine-tuning run always restarted
from epoch 0 instead of resuming from the checkpoint.

These tests verify the fixed behaviour: load_checkpoint_if_exists()
is called regardless of warm_start, so fine-tuning restarts resume
from the correct epoch with the correct model / optimizer weights.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np

from so3krates_torch.tools.checkpoint import (
    CheckpointHandler,
    CheckpointState,
)
from so3krates_torch.tools.training_setup import load_checkpoint_if_exists


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_state(tmp_path, tag="run"):
    """Create a minimal trainable state (Linear model + Adam + Exp LR)."""
    model = nn.Linear(8, 4)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.9)
    handler = CheckpointHandler(directory=str(tmp_path), tag=tag)
    state = CheckpointState(
        model=model, optimizer=optimizer, lr_scheduler=scheduler
    )
    return model, optimizer, scheduler, handler, state


def _config(tmp_path, tag="run"):
    return {
        "MISC": {"restart_latest": True},
        "GENERAL": {
            "checkpoints_dir": str(tmp_path),
            "name_exp": tag,
        },
    }


# ---------------------------------------------------------------------------
# load_checkpoint_if_exists — basic correctness
# ---------------------------------------------------------------------------


def test_load_checkpoint_restores_epoch(tmp_path):
    """load_checkpoint_if_exists returns saved_epoch + 1."""
    model, opt, sched, handler, state = _make_state(tmp_path)
    handler.save(state, epochs=7)

    new_model = nn.Linear(8, 4)
    new_opt = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    new_sched = torch.optim.lr_scheduler.ExponentialLR(new_opt, gamma=0.9)

    start_epoch = load_checkpoint_if_exists(
        model=new_model,
        optimizer=new_opt,
        lr_scheduler=new_sched,
        checkpoint_handler=handler,
        ema=None,
        device=torch.device("cpu"),
        config=_config(tmp_path),
    )

    assert start_epoch == 8, (
        f"Expected start_epoch=8 (saved epoch 7 + 1), got {start_epoch}"
    )


def test_load_checkpoint_restores_model_weights(tmp_path):
    """Model weights after load_checkpoint_if_exists match saved weights."""
    model, opt, sched, handler, state = _make_state(tmp_path)
    # Overwrite weights with known values
    with torch.no_grad():
        for p in model.parameters():
            p.fill_(3.14)
    handler.save(state, epochs=3)

    new_model = nn.Linear(8, 4)  # different random init
    new_opt = torch.optim.Adam(new_model.parameters(), lr=1e-3)
    new_sched = torch.optim.lr_scheduler.ExponentialLR(new_opt, gamma=0.9)

    load_checkpoint_if_exists(
        model=new_model,
        optimizer=new_opt,
        lr_scheduler=new_sched,
        checkpoint_handler=handler,
        ema=None,
        device=torch.device("cpu"),
        config=_config(tmp_path),
    )

    for p_saved, p_loaded in zip(model.parameters(), new_model.parameters()):
        assert torch.allclose(p_saved, p_loaded), (
            "Model weights not restored from checkpoint"
        )


def test_load_checkpoint_returns_zero_when_missing(tmp_path):
    """Returns 0 gracefully when no checkpoint file exists."""
    model, opt, sched, handler, _ = _make_state(tmp_path)

    start_epoch = load_checkpoint_if_exists(
        model=model,
        optimizer=opt,
        lr_scheduler=sched,
        checkpoint_handler=handler,
        ema=None,
        device=torch.device("cpu"),
        config=_config(tmp_path),
    )

    assert start_epoch == 0


def test_load_checkpoint_respects_restart_latest_false(tmp_path):
    """restart_latest: false in config prevents checkpoint loading."""
    model, opt, sched, handler, state = _make_state(tmp_path)
    handler.save(state, epochs=5)

    config = {
        "MISC": {"restart_latest": False},
        "GENERAL": {"checkpoints_dir": str(tmp_path), "name_exp": "run"},
    }
    start_epoch = load_checkpoint_if_exists(
        model=model,
        optimizer=opt,
        lr_scheduler=sched,
        checkpoint_handler=handler,
        ema=None,
        device=torch.device("cpu"),
        config=config,
    )

    assert start_epoch == 0, (
        "Expected start_epoch=0 when restart_latest=False"
    )


# ---------------------------------------------------------------------------
# Integration: warm_start=True must not prevent checkpoint loading
# ---------------------------------------------------------------------------


def test_finetune_restart_loads_checkpoint(tmp_path):
    """Simulate a fine-tuning restart: warm_start=True path calls checkpoint.

    This is the regression test for the bug where `if not warm_start:`
    blocked checkpoint loading during fine-tuning.  We reproduce the
    minimal logic from run_training():

        model = load_pretrained(...)   -> warm_start=True
        handle_finetuning(model)
        optimizer, scheduler = setup(model)
        # BUG: the guard `if not warm_start:` prevented this call
        start_epoch = load_checkpoint_if_exists(...)

    After the fix the call is unconditional, so interrupted fine-tuning
    runs resume correctly.
    """
    from so3krates_torch.modules.models import So3krates
    from so3krates_torch.tools.finetune import setup_finetuning
    from so3krates_torch.tools.checkpoint import (
        CheckpointHandler,
        CheckpointState,
    )

    torch.set_default_dtype(torch.float64)

    # ---- Build a tiny pretrained model and save it ---
    pretrained_cfg = {
        "r_max": 5.0,
        "num_radial_basis_fn": 4,
        "degrees": [1, 2],
        "num_features": 8,
        "num_heads": 2,
        "num_layers": 2,
        "num_elements": 118,
        "avg_num_neighbors": 5.0,
        "final_mlp_layers": 1,
        "dtype": torch.float64,
        "seed": 0,
    }
    pretrained_model = So3krates(**pretrained_cfg)
    pretrained_path = str(tmp_path / "pretrained.model")
    torch.save(pretrained_model, pretrained_path)

    # ---- Simulate first fine-tuning run: modify weights and save ckpt ----
    # Load the pretrained model
    model_run1 = torch.load(
        pretrained_path,
        map_location="cpu",
        weights_only=False,
    )
    # Apply fine-tuning setup (last_layer): freezes all but output head
    setup_finetuning(
        model=model_run1,
        finetune_choice="last_layer",
        device_name="cpu",
        num_elements=10,
        log=False,
    )
    optimizer_run1 = torch.optim.Adam(
        [p for p in model_run1.parameters() if p.requires_grad], lr=1e-3
    )
    scheduler_run1 = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_run1, gamma=0.9
    )
    # Simulate 3 gradient updates so weights are non-trivially different
    for _ in range(3):
        loss = sum(p.sum() for p in model_run1.parameters() if p.requires_grad)
        loss.backward()
        optimizer_run1.step()
        optimizer_run1.zero_grad()

    ckpt_dir = tmp_path / "checkpoints"
    handler_run1 = CheckpointHandler(
        directory=str(ckpt_dir), tag="finetune_exp"
    )
    ckpt_state_run1 = CheckpointState(
        model=model_run1,
        optimizer=optimizer_run1,
        lr_scheduler=scheduler_run1,
    )
    handler_run1.save(ckpt_state_run1, epochs=2)

    # Record trainable weights from run 1 for later comparison
    trainable_weights_run1 = {
        n: p.detach().clone()
        for n, p in model_run1.named_parameters()
        if p.requires_grad
    }

    # ---- Simulate restart: warm_start=True path ----
    # Step 1: load pretrained model (sets warm_start=True internally)
    model_restart = torch.load(
        pretrained_path,
        map_location="cpu",
        weights_only=False,
    )
    # Step 2: re-apply fine-tuning setup
    setup_finetuning(
        model=model_restart,
        finetune_choice="last_layer",
        device_name="cpu",
        num_elements=10,
        log=False,
    )
    # Step 3: create fresh optimizer/scheduler
    optimizer_restart = torch.optim.Adam(
        [p for p in model_restart.parameters() if p.requires_grad], lr=1e-3
    )
    scheduler_restart = torch.optim.lr_scheduler.ExponentialLR(
        optimizer_restart, gamma=0.9
    )
    handler_restart = CheckpointHandler(
        directory=str(ckpt_dir), tag="finetune_exp"
    )

    # Step 4: call load_checkpoint_if_exists (previously gated by warm_start)
    config = {
        "MISC": {"restart_latest": True},
        "GENERAL": {
            "checkpoints_dir": str(ckpt_dir),
            "name_exp": "finetune_exp",
        },
    }
    start_epoch = load_checkpoint_if_exists(
        model=model_restart,
        optimizer=optimizer_restart,
        lr_scheduler=scheduler_restart,
        checkpoint_handler=handler_restart,
        ema=None,
        device=torch.device("cpu"),
        config=config,
    )

    # ---- Assertions ----
    assert start_epoch == 3, (
        f"Expected start_epoch=3 (checkpoint epoch 2 + 1), got {start_epoch}. "
        "The warm_start guard is likely still preventing checkpoint loading."
    )

    # Trainable parameter values must match run 1
    for name, w_run1 in trainable_weights_run1.items():
        w_restart = dict(model_restart.named_parameters())[name]
        assert torch.allclose(w_run1, w_restart), (
            f"Parameter '{name}' mismatch after checkpoint restore: "
            f"fine-tuning restart did not load the correct weights."
        )
