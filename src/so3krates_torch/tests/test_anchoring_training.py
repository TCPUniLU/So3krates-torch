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


# ---------------------------------------------------------------------------
# End to end: torchkrates-train (run_training)
# ---------------------------------------------------------------------------


def _write_e2e_config(tmp_path, name="anchor", **training):
    """Energies/forces-only data, a pretrained SO3LR with a charge head
    saved as a pickled .model, and a fine-tuning config pointing at it."""
    import ase.io
    import numpy as np
    import yaml

    from so3krates_torch.tools.model_setup import create_model

    rng = np.random.default_rng(0)
    atoms_list = []
    for i, mol in enumerate(["H2O", "NH3", "CH4", "H2O", "NH3", "CH4"]):
        atoms = molecule(mol)
        atoms.rattle(0.05, seed=i)
        atoms.info["REF_energy"] = -10.0 * len(atoms) + rng.normal()
        atoms.arrays["REF_forces"] = rng.normal(size=(len(atoms), 3))
        atoms_list.append(atoms)
    data_path = tmp_path / "train.xyz"
    ase.io.write(data_path, atoms_list)

    raw = yaml.safe_load(EXAMPLE.read_text())
    raw["GENERAL"].update(
        checkpoints_dir=str(tmp_path / f"ckpt_{name}"),
        model_dir=str(tmp_path / "model"),
        log_dir=str(tmp_path / f"logs_{name}"),
        name_exp=name,
        default_dtype="float64",
        seed=0,
    )
    raw["ARCHITECTURE"].update(
        degrees=[1],
        num_features=8,
        num_heads=2,
        num_layers=1,
        num_radial_basis_fn=4,
        energy_regression_dim=8,
        r_max=4.0,
        r_max_lr=6.0,
        electrostatic_energy_bool=True,
        dispersion_energy_bool=True,
        dispersion_energy_cutoff_lr_damping=2.0,
        zbl_repulsion_bool=False,
    )
    raw["MISC"].update(device="cpu")
    raw["TRAINING"].update(
        path_to_train_data=str(data_path),
        path_to_val_data=str(data_path),
        batch_size=2,
        valid_batch_size=2,
        num_epochs=1,
        energy_weight=1.0,
        forces_weight=1.0,
    )
    pretrained_path = tmp_path / "pretrained.model"
    if not pretrained_path.exists():
        cfg = setup_config_from_yaml_dict(raw)
        torch.save(create_model(cfg, torch.device("cpu")), pretrained_path)
    raw["TRAINING"].update(pretrained_model=str(pretrained_path), **training)
    config_path = tmp_path / f"{name}.yaml"
    config_path.write_text(yaml.dump(raw))
    return config_path


def setup_config_from_yaml_dict(raw):
    from so3krates_torch.config import TrainConfig

    return TrainConfig.model_validate(raw).model_dump()


def _run(config_path, monkeypatch, tmp_path):
    from so3krates_torch.cli.run_train import run_training

    monkeypatch.chdir(tmp_path)
    run_training(setup_config_from_yaml(str(config_path)))


def _log_text(tmp_path, name):
    return "".join(
        p.read_text() for p in (tmp_path / f"logs_{name}").rglob("*.log")
    )


ANCHORS = {"partial_charges": 1.0, "a0_ratios": 1.0}


def test_finetune_with_anchors_runs_and_logs_metrics(tmp_path, monkeypatch):
    path = _write_e2e_config(tmp_path, long_range_anchor_weights=ANCHORS)
    _run(path, monkeypatch, tmp_path)
    log = _log_text(tmp_path, "anchor")
    assert "RMSE_anchor_charges=" in log
    assert "RMSE_anchor_hirshfeld_ratios=" in log
    assert "Long-range anchoring" in log
    assert "nan" not in log.lower().replace("nan_", "")
    assert (tmp_path / "anchor.model").exists()


def _charges(model, atoms):
    model = model.double().eval()
    out = model(_batch(atoms).to_dict(), training=False, compute_force=False)
    return out["partial_charges"].detach()


def test_anchoring_keeps_charges_closer(tmp_path, monkeypatch):
    common = dict(num_epochs=3, lr=0.05, finetune_choice="naive")
    free = _write_e2e_config(tmp_path, name="free", **common)
    anchored = _write_e2e_config(
        tmp_path,
        name="anchored",
        long_range_anchor_weights={"partial_charges": 100.0},
        **common,
    )
    _run(free, monkeypatch, tmp_path)
    _run(anchored, monkeypatch, tmp_path)

    load = lambda p: torch.load(p, weights_only=False)  # noqa: E731
    probe = molecule("CH3OH")
    ref = _charges(load(tmp_path / "pretrained.model"), probe)
    drift_free = (_charges(load(tmp_path / "free.model"), probe) - ref).norm()
    drift_anchored = (
        _charges(load(tmp_path / "anchored.model"), probe) - ref
    ).norm()
    assert drift_free > 0
    assert drift_anchored < drift_free


def test_anchoring_with_multihead_conversion(tmp_path, monkeypatch):
    path = _write_e2e_config(tmp_path, long_range_anchor_weights=ANCHORS)
    import yaml

    raw = yaml.safe_load(path.read_text())
    raw["ARCHITECTURE"].update(convert_to_multihead=True, num_output_heads=2)
    path.write_text(yaml.dump(raw))
    _run(path, monkeypatch, tmp_path)
    assert "RMSE_anchor_charges=" in _log_text(tmp_path, "anchor")


def test_anchoring_with_lora(tmp_path, monkeypatch):
    path = _write_e2e_config(
        tmp_path, long_range_anchor_weights=ANCHORS, finetune_choice="lora"
    )
    _run(path, monkeypatch, tmp_path)
    assert "RMSE_anchor_charges=" in _log_text(tmp_path, "anchor")


def test_restart_rebuilds_teacher_from_pretrained(tmp_path, monkeypatch):
    import yaml

    import so3krates_torch.cli.run_train as run_train_module

    teachers = []
    original = run_train_module.make_anchor_teacher

    def capture(model, dtype):
        teacher = original(model, dtype)
        teachers.append(teacher)
        return teacher

    monkeypatch.setattr(run_train_module, "make_anchor_teacher", capture)
    path = _write_e2e_config(
        tmp_path, long_range_anchor_weights=ANCHORS, lr=0.05
    )
    _run(path, monkeypatch, tmp_path)
    raw = yaml.safe_load(path.read_text())
    raw["TRAINING"]["num_epochs"] = 2
    path.write_text(yaml.dump(raw))
    _run(path, monkeypatch, tmp_path)

    assert len(teachers) == 2
    pretrained = torch.load(tmp_path / "pretrained.model", weights_only=False)
    expected = pretrained.double().state_dict()
    restarted = teachers[1].state_dict()
    assert all(torch.equal(expected[k], restarted[k]) for k in expected)

    for ckpt_path in (tmp_path / "ckpt_anchor").rglob("*.pt"):
        ckpt = torch.load(ckpt_path, weights_only=False)
        loss_state = ckpt.get("loss_fn", {}) or {}
        assert all("adaptive" in k for k in loss_state), list(loss_state)


def test_anchors_without_pretrained_model_rejected(tmp_path):
    import yaml

    path = _write_e2e_config(tmp_path, long_range_anchor_weights=ANCHORS)
    raw = yaml.safe_load(path.read_text())
    del raw["TRAINING"]["pretrained_model"]
    path.write_text(yaml.dump(raw))
    with pytest.raises(ValueError, match="pretrained"):
        setup_config_from_yaml(str(path))
