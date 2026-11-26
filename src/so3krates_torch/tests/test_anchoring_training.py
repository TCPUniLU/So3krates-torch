"""Long-range anchoring wired into the training setup: loss functions,
startup checks, validation metrics and ``run_training``."""

import copy

import pytest
import torch

from so3krates_torch.cli.run_train import setup_config_from_yaml
from so3krates_torch.modules.anchoring import (
    AnchoredLoss,
    make_anchor_teacher,
)
from so3krates_torch.modules.models import SO3LR
from so3krates_torch.tests.test_anchoring import (
    EXAMPLE,
    _batch,
    _pred,
)
from so3krates_torch.tools.eval import ModelEval
from so3krates_torch.tools.training_setup import (
    check_loss_outputs,
    setup_loss_functions,
    setup_optimizer_and_scheduler,
)
from ase.build import molecule


@pytest.fixture(autouse=True)
def _float64():
    before = torch.get_default_dtype()
    torch.set_default_dtype(torch.float64)
    yield
    torch.set_default_dtype(before)


@pytest.fixture
def config():
    cfg = setup_config_from_yaml(str(EXAMPLE))
    cfg["GENERAL"]["default_dtype"] = "float64"
    return cfg


def _with(cfg, **sections):
    cfg = copy.deepcopy(cfg)
    for section, values in sections.items():
        cfg[section].update(values)
    return cfg


@pytest.fixture
def config_with_anchors(config):
    return _with(
        config,
        TRAINING=dict(
            pretrained_model="pretrained.model",
            long_range_anchor_weights={"partial_charges": 1.0},
        ),
    )


@pytest.fixture
def lr_model(so3lr_model_config):
    return SO3LR(
        **so3lr_model_config, c6_ratios_bool=True, output_partial_charges=True
    )


@pytest.fixture
def batch():
    return _batch(molecule("H2O"))


def _anchored(fns):
    return [f for f in fns.modules() if isinstance(f, AnchoredLoss)]


def test_setup_wraps_all_losses_including_stage_two(
    config_with_anchors, lr_model
):
    cfg = _with(
        config_with_anchors,
        TRAINING=dict(
            stage_two_start_epoch=1,
            stage_two_loss_weights={"energy_weight": 1.0},
            num_epochs=3,
        ),
    )
    fns = setup_loss_functions(
        cfg, teacher=make_anchor_teacher(lr_model, "float64")
    )
    for fn in (
        fns.train,
        fns.valid,
        fns.stage_two.loss_fn,
        fns.stage_two.valid_loss_fn,
    ):
        assert isinstance(fn, AnchoredLoss)
        assert fn.anchor_weights == {"charges": 1.0}


def test_setup_without_anchors_unchanged(config, lr_model):
    fns = setup_loss_functions(
        config, teacher=make_anchor_teacher(lr_model, "float64")
    )
    assert not _anchored(fns)


def test_setup_requires_teacher(config_with_anchors):
    with pytest.raises(ValueError, match="teacher"):
        setup_loss_functions(config_with_anchors)


def test_adaptive_anchor_params_shared_across_losses(
    config_with_anchors, lr_model
):
    cfg = _with(
        config_with_anchors,
        TRAINING=dict(
            use_adaptive_robust_loss=True,
            stage_two_start_epoch=1,
            stage_two_loss_weights={"energy_weight": 1.0},
            num_epochs=3,
        ),
    )
    fns = setup_loss_functions(
        cfg, teacher=make_anchor_teacher(lr_model, "float64")
    )
    adaptives = {id(f.anchor_adaptive) for f in _anchored(fns)}
    assert len(adaptives) == 1 and None not in {
        f.anchor_adaptive for f in _anchored(fns)
    }


def test_optimizer_holds_no_teacher_params(config_with_anchors, lr_model):
    student_only = setup_optimizer_and_scheduler(
        lr_model, config_with_anchors, loss_fn=None
    )[0]
    fns = setup_loss_functions(
        config_with_anchors, teacher=make_anchor_teacher(lr_model, "float64")
    )
    with_loss = setup_optimizer_and_scheduler(
        lr_model, config_with_anchors, loss_fn=fns.train
    )[0]

    def count(opt):
        return sum(
            p.numel() for group in opt.param_groups for p in group["params"]
        )

    assert count(with_loss) == count(student_only)


def test_check_loss_outputs_rejects_missing_teacher_output(
    config, so3lr_model_config, batch
):
    no_c6 = SO3LR(**so3lr_model_config)
    cfg = _with(
        config,
        TRAINING=dict(
            pretrained_model="pretrained.model",
            long_range_anchor_weights={"c6_ratios": 1.0},
        ),
    )
    fns = setup_loss_functions(
        cfg, teacher=make_anchor_teacher(no_c6, "float64")
    )
    with pytest.raises((ValueError, KeyError), match="c6_ratios"):
        check_loss_outputs(no_c6, fns, batch)


def test_model_eval_reports_anchor_metrics(
    config_with_anchors, lr_model, batch
):
    fns = setup_loss_functions(
        config_with_anchors, teacher=make_anchor_teacher(lr_model, "float64")
    )
    ev = ModelEval(loss_fn=fns.valid)
    ev.update(batch, _pred(lr_model, batch))
    _, aux = ev.compute()
    assert aux["rmse_anchor_charges"] == pytest.approx(0.0, abs=1e-10)
    assert aux["mae_anchor_charges"] == pytest.approx(0.0, abs=1e-10)


def test_model_eval_anchor_metrics_see_drift(
    config_with_anchors, lr_model, batch
):
    fns = setup_loss_functions(
        config_with_anchors, teacher=make_anchor_teacher(lr_model, "float64")
    )
    with torch.no_grad():
        for p in lr_model.partial_charges_output_block.parameters():
            p.add_(0.1)
    ev = ModelEval(loss_fn=fns.valid)
    ev.update(batch, _pred(lr_model, batch))
    _, aux = ev.compute()
    assert aux["rmse_anchor_charges"] > 0


def test_model_eval_without_anchors_has_no_anchor_metrics(
    config, lr_model, batch
):
    fns = setup_loss_functions(config)
    ev = ModelEval(loss_fn=fns.valid)
    ev.update(batch, _pred(lr_model, batch))
    _, aux = ev.compute()
    assert not [k for k in aux if "anchor" in k]
