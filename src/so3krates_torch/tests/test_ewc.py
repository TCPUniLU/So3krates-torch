"""Tests for EWC (Elastic Weight Consolidation) implementation.

Covers:
  - EWC.compute_fisher(): value correctness, mode/grad cleanup,
    exception safety (try/finally), early stopping, frozen params
  - EWC.penalty(): value correctness, quadratic scaling, lambda
    scaling, DDP prefix stripping, missing-param warning
  - TrainingConfig validation: ewc_fisher_data required, adapter
    fine-tuning rejection, full fine-tuning acceptance
"""

import logging

import numpy as np
import pytest
import torch
from ase.build import molecule
from pydantic import ValidationError

from so3krates_torch.data.utils import KeySpecification
from so3krates_torch.modules.loss import WeightedEnergyForcesLoss
from so3krates_torch.modules.models import So3krates
from so3krates_torch.tools.ewc import EWC
from so3krates_torch.tools.utils import create_dataloader_from_list


# ---------------------------------------------------------------------------
# Module-local fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def small_model(default_model_config):
    """1-layer So3krates; fast enough for repeated backward passes."""
    return So3krates(**default_model_config)


@pytest.fixture
def ewc_loss_fn():
    return WeightedEnergyForcesLoss(
        energy_weight=1.0, forces_weight=100.0
    )


@pytest.fixture
def fisher_loader():
    """DataLoader over H2O/NH3/CH4 with energy+forces; batch_size=2."""
    rng = np.random.default_rng(0)
    atoms_list = [molecule(n) for n in ["H2O", "NH3", "CH4"]]
    for atoms in atoms_list:
        atoms.info["REF_energy"] = -10.0 * len(atoms)
        atoms.arrays["REF_forces"] = rng.standard_normal(
            (len(atoms), 3)
        ) * 0.1
    keyspec = KeySpecification(
        info_keys={"energy": "REF_energy"},
        arrays_keys={"forces": "REF_forces"},
    )
    return create_dataloader_from_list(
        atoms_list,
        batch_size=2,
        r_max=5.0,
        r_max_lr=None,
        key_specification=keyspec,
        shuffle=False,
    )


@pytest.fixture
def computed_ewc(small_model, ewc_loss_fn, fisher_loader, device):
    """EWC with Fisher already computed; model weights are at baseline."""
    small_model.to(device)
    torch.set_default_dtype(torch.float64)
    ewc = EWC(ewc_lambda=1.0)
    ewc.compute_fisher(
        small_model,
        ewc_loss_fn,
        fisher_loader,
        output_args={
            "forces": True,
            "virials": False,
            "stress": False,
        },
        device=device,
    )
    return ewc, small_model


# ---------------------------------------------------------------------------
# Tests: compute_fisher()
# ---------------------------------------------------------------------------


def test_fisher_values_nonnegative(computed_ewc):
    """Every Fisher entry must be >= 0 (accumulated squared gradients)."""
    ewc, _ = computed_ewc
    assert ewc._fisher, "Fisher dict is empty"
    for name, val in ewc._fisher.items():
        assert (val >= 0).all(), (
            f"Fisher entry '{name}' contains negative values"
        )


def test_fisher_baseline_matches_initial_params(
    computed_ewc,
):
    """Baseline must snapshot the exact parameter values at call time."""
    ewc, model = computed_ewc
    state = model.state_dict()
    for name, baseline in ewc._baseline.items():
        assert name in state, f"Baseline key '{name}' not in state_dict"
        assert torch.allclose(baseline, state[name].to(baseline.dtype)), (
            f"Baseline mismatch for '{name}'"
        )


def test_fisher_excludes_frozen_params(
    small_model, ewc_loss_fn, fisher_loader, device
):
    """Parameters with requires_grad=False must be absent from Fisher."""
    small_model.to(device)
    torch.set_default_dtype(torch.float64)
    # Freeze the output block
    for name, param in small_model.named_parameters():
        if "atomic_energy_output_block" in name:
            param.requires_grad = False

    frozen = {
        n
        for n, p in small_model.named_parameters()
        if not p.requires_grad
    }

    ewc = EWC(ewc_lambda=1.0)
    ewc.compute_fisher(
        small_model,
        ewc_loss_fn,
        fisher_loader,
        output_args={"forces": True, "virials": False, "stress": False},
        device=device,
    )

    for name in frozen:
        assert name not in ewc._fisher, (
            f"Frozen param '{name}' should not appear in Fisher"
        )
        assert name not in ewc._baseline, (
            f"Frozen param '{name}' should not appear in baseline"
        )


def test_fisher_training_mode_restored_after_success(
    small_model, ewc_loss_fn, fisher_loader, device
):
    """Model must return to eval mode if it was in eval before compute."""
    small_model.to(device)
    torch.set_default_dtype(torch.float64)
    small_model.eval()

    ewc = EWC(ewc_lambda=1.0)
    ewc.compute_fisher(
        small_model,
        ewc_loss_fn,
        fisher_loader,
        output_args={"forces": True, "virials": False, "stress": False},
        device=device,
    )

    assert not small_model.training, (
        "Model should be back in eval mode after compute_fisher"
    )


def test_fisher_training_mode_restored_after_exception(
    small_model, fisher_loader, device
):
    """try/finally: mode and grads must be clean even when loss raises."""
    small_model.to(device)
    torch.set_default_dtype(torch.float64)
    small_model.eval()

    class BrokenLoss(torch.nn.Module):
        def forward(self, pred, ref, **kwargs):
            raise RuntimeError("injected failure")

    ewc = EWC(ewc_lambda=1.0)
    with pytest.raises(RuntimeError, match="injected failure"):
        ewc.compute_fisher(
            small_model,
            BrokenLoss(),
            fisher_loader,
            output_args={
                "forces": True,
                "virials": False,
                "stress": False,
            },
            device=device,
        )

    # Mode must be restored
    assert not small_model.training, (
        "Model training mode not restored after exception"
    )
    # Gradients must be cleared
    for name, param in small_model.named_parameters():
        assert param.grad is None or (param.grad == 0).all(), (
            f"Stale gradient on '{name}' after exception"
        )


def test_fisher_grads_zeroed_after_success(computed_ewc):
    """All parameter gradients must be None after a successful compute."""
    ewc, model = computed_ewc
    for name, param in model.named_parameters():
        assert param.grad is None or (param.grad == 0).all(), (
            f"Parameter '{name}' has a non-zero gradient after "
            f"compute_fisher"
        )


def test_fisher_empty_loader_raises(
    small_model, ewc_loss_fn, device
):
    """An empty DataLoader must raise RuntimeError mentioning no batches."""
    small_model.to(device)
    torch.set_default_dtype(torch.float64)
    empty_loader = create_dataloader_from_list(
        [],
        batch_size=2,
        r_max=5.0,
        r_max_lr=None,
        shuffle=False,
    )
    ewc = EWC(ewc_lambda=1.0)
    with pytest.raises(RuntimeError, match="no batches"):
        ewc.compute_fisher(
            small_model,
            ewc_loss_fn,
            empty_loader,
            output_args={"forces": True, "virials": False, "stress": False},
            device=device,
        )


def test_fisher_num_samples_respected(
    small_model, ewc_loss_fn, device
):
    """compute_fisher stops processing once num_samples structures seen."""
    torch.set_default_dtype(torch.float64)
    rng = np.random.default_rng(1)
    # 10 structures; batch_size=2 → 5 batches
    atoms_list = [molecule(n) for n in ["H2O"] * 10]
    for atoms in atoms_list:
        atoms.info["REF_energy"] = -10.0 * len(atoms)
        atoms.arrays["REF_forces"] = rng.standard_normal(
            (len(atoms), 3)
        ) * 0.1
    keyspec = KeySpecification(
        info_keys={"energy": "REF_energy"},
        arrays_keys={"forces": "REF_forces"},
    )
    loader = create_dataloader_from_list(
        atoms_list,
        batch_size=2,
        r_max=5.0,
        r_max_lr=None,
        key_specification=keyspec,
        shuffle=False,
    )

    # Count how many loss.forward() calls are made
    call_count = [0]
    inner_loss = ewc_loss_fn

    class CountingLoss(torch.nn.Module):
        def forward(self, pred, ref, **kwargs):
            call_count[0] += 1
            return inner_loss(pred=pred, ref=ref)

    small_model.to(device)
    ewc = EWC(ewc_lambda=1.0)
    ewc.compute_fisher(
        small_model,
        CountingLoss(),
        loader,
        output_args={"forces": True, "virials": False, "stress": False},
        device=device,
        num_samples=4,  # should stop after 2 batches (4 structures)
    )

    # With num_samples=4 and batch_size=2, the break fires before the
    # third batch (total_structures reaches 4 after batch 2).
    assert call_count[0] <= 2, (
        f"Expected at most 2 loss calls for num_samples=4 with "
        f"batch_size=2, got {call_count[0]}"
    )
    assert call_count[0] >= 1, "At least 1 batch should be processed"


# ---------------------------------------------------------------------------
# Tests: penalty()
# ---------------------------------------------------------------------------


def test_penalty_zero_at_baseline(computed_ewc, device):
    """Penalty must be exactly 0 immediately after compute_fisher."""
    ewc, model = computed_ewc
    p = ewc.penalty(model)
    assert p.item() == pytest.approx(0.0, abs=1e-12), (
        f"Expected penalty=0 at baseline, got {p.item()}"
    )


def test_penalty_positive_after_drift(
    computed_ewc, ewc_loss_fn, fisher_loader, device
):
    """After one gradient step, penalty must be strictly positive."""
    ewc, model = computed_ewc
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-2)
    batch = next(iter(fisher_loader)).to(device)
    optimizer.zero_grad()
    output = model(batch.to_dict(), training=True, compute_force=True)
    loss = ewc_loss_fn(pred=output, ref=batch)
    loss.backward()
    optimizer.step()

    p = ewc.penalty(model)
    assert p.item() > 0.0, (
        f"Expected positive penalty after gradient step, got {p.item()}"
    )


def test_penalty_scales_quadratically_with_drift(computed_ewc, device):
    """Penalty must scale as delta^2: doubling drift -> 4x penalty."""
    ewc, model = computed_ewc

    # Pick the first trainable parameter
    param_name = next(iter(ewc._fisher))
    param = dict(model.named_parameters())[param_name]
    baseline = ewc._baseline[param_name]

    delta = torch.ones_like(param) * 1e-3

    with torch.no_grad():
        param.copy_(baseline + delta)
    p1 = ewc.penalty(model).item()

    with torch.no_grad():
        param.copy_(baseline + 2 * delta)
    p2 = ewc.penalty(model).item()

    # Restore
    with torch.no_grad():
        param.copy_(baseline)

    assert p1 > 0, "Penalty should be positive with delta drift"
    assert p2 / p1 == pytest.approx(4.0, rel=1e-5), (
        f"Expected 4x scaling, got {p2 / p1:.6f}"
    )


def test_penalty_scales_linearly_with_lambda(computed_ewc, device):
    """Penalty must scale linearly with ewc_lambda."""
    ewc, model = computed_ewc

    # Introduce a small drift
    param_name = next(iter(ewc._fisher))
    param = dict(model.named_parameters())[param_name]
    baseline = ewc._baseline[param_name]
    with torch.no_grad():
        param.copy_(baseline + 1e-3)

    ewc.ewc_lambda = 1.0
    p1 = ewc.penalty(model).item()

    ewc.ewc_lambda = 5.0
    p5 = ewc.penalty(model).item()

    # Restore
    with torch.no_grad():
        param.copy_(baseline)
    ewc.ewc_lambda = 1.0

    assert p1 > 0
    assert p5 / p1 == pytest.approx(5.0, rel=1e-9), (
        f"Expected 5x scaling with lambda, got {p5 / p1:.9f}"
    )


def test_penalty_raises_before_fisher(small_model):
    """Calling penalty() before compute_fisher() must raise RuntimeError."""
    ewc = EWC(ewc_lambda=1.0)
    with pytest.raises(RuntimeError, match="compute_fisher"):
        ewc.penalty(small_model)


def test_penalty_ddp_prefix_stripped(computed_ewc, device):
    """Penalty must match when model is wrapped so names get module. prefix.

    Simulates DDP: nn.Module with a sub-module named 'module' causes
    named_parameters() to return 'module.<original>' names.
    """
    ewc, model = computed_ewc

    # Introduce a drift so penalty is non-zero
    param_name = next(iter(ewc._fisher))
    param = dict(model.named_parameters())[param_name]
    baseline = ewc._baseline[param_name]
    with torch.no_grad():
        param.copy_(baseline + 1e-3)

    p_unwrapped = ewc.penalty(model).item()

    class FakeDDP(torch.nn.Module):
        def __init__(self, inner):
            super().__init__()
            self.module = inner  # → named_params: "module.<name>"

    wrapped = FakeDDP(model)
    p_wrapped = ewc.penalty(wrapped).item()

    # Restore
    with torch.no_grad():
        param.copy_(baseline)

    assert p_unwrapped > 0
    assert p_wrapped == pytest.approx(p_unwrapped, rel=1e-9), (
        f"DDP-wrapped penalty {p_wrapped} != unwrapped {p_unwrapped}"
    )


def test_penalty_warns_on_missing_param(computed_ewc, caplog):
    """A WARNING must be emitted when a Fisher param is absent from model."""
    ewc, model = computed_ewc

    phantom_name = "__phantom__.weight"
    ewc._fisher[phantom_name] = torch.zeros(1)
    ewc._baseline[phantom_name] = torch.zeros(1)

    with caplog.at_level(logging.WARNING, logger="root"):
        ewc.penalty(model)

    # Clean up
    del ewc._fisher[phantom_name]
    del ewc._baseline[phantom_name]

    warning_texts = [r.message for r in caplog.records
                     if r.levelno == logging.WARNING]
    assert any(phantom_name in t for t in warning_texts), (
        f"Expected WARNING mentioning '{phantom_name}', "
        f"got: {warning_texts}"
    )


# ---------------------------------------------------------------------------
# Tests: TrainingConfig validation
# ---------------------------------------------------------------------------


_REQUIRED_TRAINING_FIELDS = dict(
    batch_size=4,
    valid_batch_size=4,
    lr=1e-3,
    num_epochs=1,
    path_to_train_data="train.xyz",
)


def test_ewc_config_requires_fisher_data():
    """use_ewc=True without ewc_fisher_data must raise ValidationError."""
    from so3krates_torch.config.training_config import TrainingConfig

    with pytest.raises(ValidationError, match="ewc_fisher_data"):
        TrainingConfig(
            **_REQUIRED_TRAINING_FIELDS,
            use_ewc=True,
            ewc_fisher_data=None,
        )


@pytest.mark.parametrize(
    "adapter_choice",
    ["lora", "dora", "vera", "lora+mlp"],
)
def test_ewc_config_rejects_adapter_finetuning(adapter_choice):
    """EWC + adapter fine-tuning must raise ValidationError."""
    from so3krates_torch.config.training_config import TrainingConfig

    with pytest.raises(ValidationError):
        TrainingConfig(
            **_REQUIRED_TRAINING_FIELDS,
            use_ewc=True,
            ewc_fisher_data="fisher.xyz",
            finetune_choice=adapter_choice,
        )


@pytest.mark.parametrize(
    "full_choice",
    ["naive", "last_layer", "qkv", "mlp", "qkv+mlp", "last_layer+mlp"],
)
def test_ewc_config_accepts_full_finetuning(full_choice):
    """EWC + partial-base fine-tuning must not raise."""
    from so3krates_torch.config.training_config import TrainingConfig

    cfg = TrainingConfig(
        **_REQUIRED_TRAINING_FIELDS,
        use_ewc=True,
        ewc_fisher_data="fisher.xyz",
        finetune_choice=full_choice,
    )
    assert cfg.use_ewc is True
    assert cfg.finetune_choice == full_choice


def test_ewc_config_disabled_with_adapter_finetuning_ok():
    """use_ewc=False with finetune_choice='lora' must not raise."""
    from so3krates_torch.config.training_config import TrainingConfig

    cfg = TrainingConfig(
        **_REQUIRED_TRAINING_FIELDS,
        use_ewc=False,
        finetune_choice="lora",
    )
    assert cfg.use_ewc is False
