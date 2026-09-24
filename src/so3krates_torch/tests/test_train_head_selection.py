"""Tests for _select_valid_loss in tools/train.py.

train() previously kept only the *last* head's validation loss for
checkpointing, early stopping, and the LR scheduler (a leftover
`for`-loop variable), which meant a replay head decided model
selection instead of the fine-tune head it was inserted after. This
extracts that decision into a pure, unit-testable function.
"""

import numpy as np
import pytest

from so3krates_torch.tools.train import _select_valid_loss


def test_uses_primary_head_when_present():
    head_losses = {"finetune": 1.0, "replay_0": 5.0}
    assert _select_valid_loss(head_losses, "finetune") == 1.0


def test_falls_back_to_last_head_when_primary_not_given():
    """Preserves existing behavior for manual multi-head configs that
    don't set a primary head."""
    head_losses = {"head_a": 1.0, "head_b": 2.0, "head_c": 3.0}
    assert _select_valid_loss(head_losses, None) == 3.0


def test_falls_back_to_last_head_when_primary_absent_from_losses():
    head_losses = {"head_a": 1.0, "head_b": 2.0}
    assert _select_valid_loss(head_losses, "not_a_real_head") == 2.0


def test_returns_inf_for_empty_losses():
    """An empty valid_loaders dict (the single-head path can return
    {}) must not raise NameError, as it did before."""
    assert _select_valid_loss({}, "finetune") == np.inf
    assert _select_valid_loss({}, None) == np.inf


def test_single_head_dict_returns_its_only_value():
    assert _select_valid_loss({"main": 3.5}, None) == 3.5
