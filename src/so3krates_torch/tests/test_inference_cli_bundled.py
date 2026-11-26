"""``torchkrates-eval`` / ``torchkrates-metric`` with the bundled models
(v1, and the multi-theory-level tauri models): same results as
``SO3LRCalculator``, theory level selection, long-range cutoffs."""

import sys
from pathlib import Path

import ase.io
import numpy as np
import pytest
import torch

from so3krates_torch.calculator.so3 import SO3LRCalculator
from so3krates_torch.cli import run_metric
from so3krates_torch.cli.run_eval import run_evaluation
from so3krates_torch.modules.models import SO3LR
from so3krates_torch.tools.eval import test_ensemble as run_test_ensemble

REPO = Path(__file__).resolve().parents[3]
ETHANOL = REPO / "examples/training/md17_ethanol_small.xyz"
TAURI = "tauri-s-2606"


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(dtype)


def _frames(theory_level=None):
    frames = ase.io.read(ETHANOL, ":3")
    if theory_level is not None:
        for atoms in frames:
            atoms.info["theory_level"] = theory_level
    return frames


@pytest.fixture
def data_path(tmp_path):
    path = tmp_path / "ethanol.xyz"
    ase.io.write(path, _frames())
    return str(path)


def _calculator(model, **kwargs):
    calc = SO3LRCalculator(model=model, **kwargs)
    energies, forces = [], []
    for atoms in _frames():
        atoms.calc = calc
        energies.append(atoms.get_potential_energy())
        forces.append(atoms.get_forces())
    return np.array(energies), forces


def _evaluate(model, data_path, **kwargs):
    result, _ = run_evaluation(
        model_path=model,
        data_path=data_path,
        device="cpu",
        dtype="float32",
        multispecies=True,
        **kwargs,
    )
    return np.array(result["energies"]).ravel(), result["forces"]


def _assert_same(got, expected):
    (energies, forces), (ref_energies, ref_forces) = got, expected
    np.testing.assert_allclose(energies, ref_energies, rtol=1e-5, atol=1e-3)
    np.testing.assert_allclose(
        np.concatenate([np.asarray(f).reshape(-1, 3) for f in forces]),
        np.concatenate(ref_forces),
        atol=1e-3,
    )


def test_eval_tauri_with_theory_level_matches_calculator(data_path):
    _assert_same(
        _evaluate(TAURI, data_path, theory_level=5),
        _calculator(TAURI, theory_level=5),
    )


def test_eval_tauri_uses_theory_level_of_structures(tmp_path):
    path = tmp_path / "with_level.xyz"
    ase.io.write(path, _frames(theory_level=5))
    _assert_same(
        _evaluate(TAURI, str(path)), _calculator(TAURI, theory_level=5)
    )


def test_eval_tauri_without_theory_level_raises(data_path):
    with pytest.raises(ValueError, match="--theory_level"):
        _evaluate(TAURI, data_path)


def test_eval_theory_level_on_single_level_model_raises(data_path):
    with pytest.raises(ValueError, match="single theory level"):
        _evaluate("v1", data_path, theory_level=5)


def test_eval_bundled_model_defaults_to_calculator_cutoffs(data_path):
    """No --r_max_lr: the bundled models use SO3LRCalculator's long-range
    cutoff and damping (12 A, 2.0)."""
    _assert_same(_evaluate("v1", data_path), _calculator("v1"))


def test_metric_tauri_with_theory_level(data_path, tmp_path, monkeypatch):
    _, ref_forces = _calculator(TAURI, theory_level=5)
    frames = _frames()
    ref_mae_f = np.mean(
        [
            np.abs(f - a.arrays["REF_forces"]).mean()
            for f, a in zip(ref_forces, frames)
        ]
    )
    argv = [
        "torchkrates-metric",
        "--models",
        TAURI,
        "--data",
        data_path,
        "--theory_level",
        "5",
        "--save",
        f"{tmp_path}/",
        "--log_file",
        str(tmp_path / "metric.log"),
    ]
    monkeypatch.setattr(sys, "argv", argv)

    run_metric.main()

    results = np.load(
        tmp_path / "ensemble_test_results.npz", allow_pickle=True
    )
    mae_f = results["avg_ensemble_metrics"].item()["mae_f"]
    assert mae_f == pytest.approx(ref_mae_f, rel=1e-3)


def test_metric_uses_the_model_long_range_cutoff(
    so3lr_model_config, example_xyz_with_data
):
    """Without --r_max_lr, a long-range model is evaluated with its own
    r_max_lr (as torchkrates-eval does), instead of crashing."""
    model = SO3LR(**so3lr_model_config).float()
    assert model.r_max_lr is not None
    torch.set_default_dtype(torch.float32)

    avg, _ = run_test_ensemble(
        ensemble={"m": model},
        path_to_data=example_xyz_with_data,
        batch_size=2,
        output_args=run_metric.parse_output_args(["energy", "forces"]),
        device="cpu",
    )
    assert np.isfinite(avg["mae_f"])


def _create_lammps(monkeypatch, tmp_path, *args):
    from so3krates_torch.cli import create_lammps_model

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(
        sys, "argv", ["torchkrates-create-lammps-model", *args]
    )
    create_lammps_model.main()


def test_lammps_export_of_tauri_keeps_one_theory_level(tmp_path, monkeypatch):
    _create_lammps(
        monkeypatch,
        tmp_path,
        TAURI,
        "--elements",
        "C",
        "H",
        "O",
        "--theory-level",
        "5",
    )
    exported = torch.load(
        tmp_path / f"{TAURI}-mliap_lammps.pt", weights_only=False
    )
    model = exported.model.model
    assert model.atomic_energy_output_block.num_theory_levels == 1
    assert model.r_max_lr == 12.0
    assert model.dispersion_energy_cutoff_lr_damping == 2.0


def test_lammps_export_of_tauri_without_theory_level_raises(
    tmp_path, monkeypatch
):
    with pytest.raises(ValueError, match="--theory-level"):
        _create_lammps(
            monkeypatch, tmp_path, TAURI, "--elements", "C", "H", "O"
        )
