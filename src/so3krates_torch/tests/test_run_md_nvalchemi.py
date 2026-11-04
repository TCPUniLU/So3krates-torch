"""Tests for the torchkrates-md-nvalchemi CLI helpers."""

import types

import numpy as np
import pytest
import torch

pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit not installed")
pytest.importorskip("nvalchemiops", reason="nvalchemi-toolkit-ops missing")


def test_cli_module_importable_from_cli_package():
    from so3krates_torch.cli import run_md_nvalchemi

    assert callable(run_md_nvalchemi.main)


def _fake_batch(stress, vel_scale=0.0):
    n = 2
    cell = (torch.eye(3, dtype=torch.float64) * 5.0).unsqueeze(0)
    return types.SimpleNamespace(
        cell=cell,
        stress=stress.unsqueeze(0),
        velocities=torch.full((n, 3), vel_scale, dtype=torch.float64),
        atomic_masses=torch.ones(n, dtype=torch.float64),
        batch_idx=torch.zeros(n, dtype=torch.long),
        num_graphs=1,
        num_nodes_per_graph=torch.tensor([n]),
    )


def test_pressure_bar_sign_matches_ase():
    """ASE: P = -tr(sigma)/3. A compressed cell has negative (compressive)
    stress and must report positive pressure."""
    from so3krates_torch.cli.run_md_nvalchemi import (
        BAR_TO_EV_PER_A3,
        _pressure_bar_per_graph,
    )

    sigma = -1e-3 * torch.eye(3, dtype=torch.float64)  # compressive, eV/A^3
    p = _pressure_bar_per_graph(_fake_batch(sigma))
    expected = 1e-3 / BAR_TO_EV_PER_A3  # zero velocities -> no kinetic term
    torch.testing.assert_close(
        p, torch.tensor([expected], dtype=torch.float64)
    )


def test_atoms_to_data_keeps_ase_velocities():
    # No nvalchemi import needed for the reference KE: ASE defines
    # KE = 0.5 * sum_i m_i |v_i|^2, computed here directly in numpy.
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from nvalchemi.data.batch import Batch

    from so3krates_torch.cli.run_md_nvalchemi import atoms_to_data

    atoms = bulk("NaCl", "rocksalt", a=5.64).repeat(2)
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=300, rng=np.random.default_rng(0)
    )
    data = atoms_to_data(atoms, device="cpu", dtype=torch.float64)
    b = Batch.from_data_list([data])
    masses = b.atomic_masses.numpy()
    v = b.velocities.numpy()
    ke = 0.5 * (masses * (v**2).sum(-1)).sum()
    assert ke == pytest.approx(atoms.get_kinetic_energy(), rel=1e-8)


def _fake_batch_multi():
    """A 2-graph fake batch: graph 0 has 2 atoms, graph 1 has 3 atoms."""
    cell = torch.stack(
        [
            torch.eye(3, dtype=torch.float64) * 5.0,
            torch.eye(3, dtype=torch.float64) * 6.0,
        ]
    )
    masses = torch.tensor([1.0, 2.0, 1.0, 1.0, 3.0], dtype=torch.float64)
    velocities = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [2.0, 0.0, 0.0],
        ],
        dtype=torch.float64,
    )
    batch_idx = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)
    stress = torch.stack(
        [
            -1e-3 * torch.eye(3, dtype=torch.float64),
            -2e-3 * torch.eye(3, dtype=torch.float64),
        ]
    )
    return types.SimpleNamespace(
        cell=cell,
        stress=stress,
        velocities=velocities,
        atomic_masses=masses,
        batch_idx=batch_idx,
        num_graphs=2,
        num_nodes_per_graph=torch.tensor([2, 3]),
    )


def test_temperature_and_pressure_per_graph_shapes_and_values():
    """Guards the [B, 1] / [B] broadcast bug: both helpers must return a
    1-D [B] tensor (not [B, B]) for a multi-graph batch, with per-graph
    values matching an independent hand computation."""
    from so3krates_torch.cli.run_md_nvalchemi import (
        BAR_TO_EV_PER_A3,
        KB_EV_PER_K,
        _pressure_bar_per_graph,
        _temperature_per_graph,
    )

    batch = _fake_batch_multi()

    # Hand computation: KE = 0.5 * sum_i m_i |v_i|^2 per graph.
    # graph 0: m=[1, 2], |v|^2=[1, 1] -> 0.5*(1*1 + 2*1) = 1.5
    # graph 1: m=[1, 1, 3], |v|^2=[2, 1, 4] -> 0.5*(1*2+1*1+3*4) = 7.5
    ke0 = 1.5
    ke1 = 7.5
    t0 = 2.0 * ke0 / (3.0 * 2 * KB_EV_PER_K)
    t1 = 2.0 * ke1 / (3.0 * 3 * KB_EV_PER_K)

    temps = _temperature_per_graph(batch)
    assert temps.shape == (2,)
    torch.testing.assert_close(
        temps, torch.tensor([t0, t1], dtype=torch.float64)
    )

    vols = (125.0, 216.0)  # det(5*I) = 5^3, det(6*I) = 6^3
    n_atoms = (2.0, 3.0)
    p_vir = (1e-3, 2e-3)  # -tr(sigma)/3 for -1e-3*I and -2e-3*I
    expected = [
        (n * KB_EV_PER_K * t / v + pv) / BAR_TO_EV_PER_A3
        for n, t, v, pv in zip(n_atoms, (t0, t1), vols, p_vir)
    ]

    press = _pressure_bar_per_graph(batch)
    assert press.shape == (2,)
    torch.testing.assert_close(
        press, torch.tensor(expected, dtype=torch.float64)
    )


def test_compile_guard_cpu_lr(tmp_path):
    from so3krates_torch.cli.run_md_nvalchemi import load_model
    from so3krates_torch.tests.test_nvalchemi_so3 import _make_so3lr

    path = tmp_path / "m.model"
    # electrostatics_bool=True (real-space, use_pme=False) sets
    # model.use_lr = True (so3krates_torch.modules.models.SO3LR.__init__);
    # no manual `model.use_lr = True` override is needed.
    torch.save(_make_so3lr(electrostatics_bool=True), path)
    with pytest.raises(RuntimeError, match="torch.compile is not supported"):
        load_model(str(path), "cpu", torch.float64, compile_model=True)


# ---------------------------------------------------------------------------
# End-to-end CLI runs (main() with a patched sys.argv, CPU, 2 steps)
# ---------------------------------------------------------------------------


def _save_tiny_model(tmp_path, dtype=torch.float64, strip_lr_cutoff=False):
    """Tiny long-range model.  ``strip_lr_cutoff`` stores no long-range
    cutoff, like the packaged pre-trained ``so3lr.model``."""
    from so3krates_torch.tests.test_nvalchemi_so3 import _make_so3lr

    path = tmp_path / "tiny.model"
    model = _make_so3lr(electrostatics_bool=True).to(dtype)
    if strip_lr_cutoff:
        model.r_max_lr = None
    torch.save(model, path)
    return path


def _h2o():
    from ase.build import molecule

    atoms = molecule("H2O")
    atoms.center(vacuum=4.0)
    atoms.pbc = False
    return atoms


def _nacl_crystal():
    from ase.build import bulk

    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)
    atoms.rattle(0.02, seed=3)
    return atoms


def _run_cli(
    monkeypatch,
    tmp_path,
    frames,
    *extra,
    model_dtype=None,
    strip_lr_cutoff=False,
):
    from ase.io import write

    from so3krates_torch.cli import run_md_nvalchemi

    model = _save_tiny_model(
        tmp_path,
        dtype=model_dtype or torch.float64,
        strip_lr_cutoff=strip_lr_cutoff,
    )
    xyz = tmp_path / "in.xyz"
    write(xyz, frames, format="extxyz")
    log = tmp_path / "md_log.csv"
    argv = [
        "torchkrates-md-nvalchemi",
        "--model",
        str(model),
        "--atoms",
        str(xyz),
        "--steps",
        "2",
        "--log-freq",
        "1",
        "--log-file",
        str(log),
        "--device",
        "cpu",
        *extra,
    ]
    monkeypatch.setattr("sys.argv", argv)
    run_md_nvalchemi.main()
    return log


def _read_csv_rows(log):
    import csv

    with open(log) as fh:
        rows = list(csv.DictReader(fh))
    assert rows, "CSV log is empty"
    return rows


def _assert_finite_rows(rows, required):
    for row in rows:
        for key in required:
            assert key in row, f"missing column {key!r} in {list(row)}"
        for key, val in row.items():
            try:
                x = float(val)
            except (TypeError, ValueError):
                continue
            assert np.isfinite(x), f"{key}={val} not finite"


def test_cli_nvt_molecule_runs(monkeypatch, tmp_path, capsys):
    """NVT on a non-periodic molecule (no cell) completes; no cell-derived
    columns are logged."""
    log = _run_cli(monkeypatch, tmp_path, _h2o(), "--ensemble", "nvt")
    rows = _read_csv_rows(log)
    _assert_finite_rows(rows, ["energy", "elapsed_s"])
    assert "volume_A3" not in rows[0]
    assert "pressure_bar" not in rows[0]


def test_cli_nvt_crystal_runs(monkeypatch, tmp_path):
    log = _run_cli(monkeypatch, tmp_path, _nacl_crystal(), "--ensemble", "nvt")
    rows = _read_csv_rows(log)
    _assert_finite_rows(rows, ["energy", "volume_A3", "density_g_cm3"])
    # NVT does not compute stress, so no pressure column.
    assert "pressure_bar" not in rows[0]


def test_cli_npt_crystal_runs(monkeypatch, tmp_path):
    log = _run_cli(monkeypatch, tmp_path, _nacl_crystal(), "--ensemble", "npt")
    rows = _read_csv_rows(log)
    _assert_finite_rows(
        rows, ["energy", "volume_A3", "density_g_cm3", "pressure_bar"]
    )


def test_cli_nvt_sets_active_outputs_without_stress(monkeypatch, tmp_path):
    """The CLI narrows NVT to energy+forces (nvalchemi 0.2 integrators do
    not), so stress is never computed during NVT."""
    from so3krates_torch.cli import run_md_nvalchemi

    seen = {}
    orig = run_md_nvalchemi.NVTLangevin

    def _spy(*args, **kwargs):
        seen["active"] = set(kwargs["model"].model_config.active_outputs)
        return orig(*args, **kwargs)

    monkeypatch.setattr(run_md_nvalchemi, "NVTLangevin", _spy)
    _run_cli(monkeypatch, tmp_path, _nacl_crystal(), "--ensemble", "nvt")
    assert seen["active"] == {"energy", "forces"}


def test_cli_nvt_passes_seed_to_langevin(monkeypatch, tmp_path):
    """``--seed`` must reach the Langevin thermostat's RNG."""
    from so3krates_torch.cli import run_md_nvalchemi

    seen = {}
    orig = run_md_nvalchemi.NVTLangevin

    def _spy(*args, **kwargs):
        seen["random_seed"] = kwargs["random_seed"]
        return orig(*args, **kwargs)

    monkeypatch.setattr(run_md_nvalchemi, "NVTLangevin", _spy)
    _run_cli(
        monkeypatch,
        tmp_path,
        _nacl_crystal(),
        "--ensemble",
        "nvt",
        "--seed",
        "1234",
    )
    assert seen["random_seed"] == 1234


def test_cli_npt_keeps_stress_active(monkeypatch, tmp_path):
    from so3krates_torch.cli import run_md_nvalchemi

    seen = {}
    orig = run_md_nvalchemi.NPT

    def _spy(*args, **kwargs):
        seen["active"] = set(kwargs["model"].model_config.active_outputs)
        return orig(*args, **kwargs)

    monkeypatch.setattr(run_md_nvalchemi, "NPT", _spy)
    _run_cli(monkeypatch, tmp_path, _nacl_crystal(), "--ensemble", "npt")
    assert seen["active"] == {"energy", "forces", "stress"}


def test_cli_float32_model_float64_run(monkeypatch, tmp_path):
    """A float32 checkpoint run with the default --dtype float64."""
    log = _run_cli(
        monkeypatch,
        tmp_path,
        _nacl_crystal(),
        "--ensemble",
        "nvt",
        model_dtype=torch.float32,
    )
    _assert_finite_rows(_read_csv_rows(log), ["energy"])


def test_cli_mixed_pbc_frames_rejected(monkeypatch, tmp_path):
    with pytest.raises(SystemExit, match="mix of periodic and non-periodic"):
        _run_cli(
            monkeypatch,
            tmp_path,
            [_h2o(), _nacl_crystal()],
            "--index",
            ":",
            "--ensemble",
            "nvt",
        )


# ---------------------------------------------------------------------------
# load_model: dtype, frozen parameters, diagnostics
# ---------------------------------------------------------------------------


def _energy_forces_via_wrapper(model, atoms):
    from so3krates_torch.tests.test_nvalchemi_so3 import _make_batch_with_nl

    out = model(_make_batch_with_nl(model, atoms))
    return out["energy"], out["forces"]


@pytest.mark.parametrize(
    "ckpt_dtype,run_dtype",
    [(torch.float32, torch.float64), (torch.float64, torch.float32)],
    ids=["f32ckpt-f64run", "f64ckpt-f32run"],
)
def test_load_model_applies_dtype(tmp_path, ckpt_dtype, run_dtype):
    """load_model(dtype=...) converts the checkpoint before wrapping: all
    parameters and the inputs handed to the model use ``dtype``, and the
    result matches a calculator reference run at that dtype."""
    from ase.build import bulk

    from so3krates_torch.calculator.so3 import TorchkratesCalculator
    from so3krates_torch.cli.run_md_nvalchemi import load_model
    from so3krates_torch.tests.test_nvalchemi_so3 import _make_so3lr

    raw = _make_so3lr(electrostatics_bool=True).to(ckpt_dtype)
    path = tmp_path / "m.model"
    torch.save(raw, path)
    model = load_model(str(path), "cpu", run_dtype)
    params = list(model.model.parameters())
    assert all(p.dtype == run_dtype for p in params)
    assert not any(p.requires_grad for p in params)

    atoms = bulk("NaCl", "rocksalt", a=5.64)
    atoms.rattle(0.05, seed=4)
    e, f = _energy_forces_via_wrapper(model, atoms)
    assert e.dtype == run_dtype

    ref_model = torch.load(path, weights_only=False)
    # The calculator converts the model to default_dtype but also sets the
    # global torch default dtype; restore it so other tests are unaffected.
    old_default = torch.get_default_dtype()
    try:
        atoms.calc = TorchkratesCalculator(
            models=[ref_model],
            device="cpu",
            default_dtype=str(run_dtype).split(".")[-1],
            r_max=ref_model.r_max,
            r_max_lr=ref_model.r_max_lr,
        )
        e_ref = atoms.get_potential_energy()
        f_ref = atoms.get_forces()
    finally:
        torch.set_default_dtype(old_default)
    tol = 1e-8 if run_dtype == torch.float64 else 1e-4
    assert e.item() == pytest.approx(e_ref, abs=tol)
    np.testing.assert_allclose(f.detach().double().numpy(), f_ref, atol=tol)


@pytest.mark.parametrize(
    "mode,expected",
    [
        ("model", "Electrostatics: yes"),
        ("none", "Electrostatics: NO"),
        ("nvalchemi", "Electrostatics: NVAlchemi PME"),
    ],
)
def test_load_model_diagnostics_reflect_electrostatics_flag(
    tmp_path, capsys, mode, expected
):
    from so3krates_torch.cli.run_md_nvalchemi import load_model
    from so3krates_torch.tests.test_nvalchemi_so3 import _make_so3lr

    path = tmp_path / "m.model"
    torch.save(_make_so3lr(electrostatics_bool=True), path)
    load_model(str(path), "cpu", torch.float64, electrostatics=mode)
    assert expected in capsys.readouterr().out


def test_compile_help_text_has_no_graph_break_claim():
    from so3krates_torch.cli import run_md_nvalchemi

    parser_help = run_md_nvalchemi._build_parser().format_help()
    assert "intended graph-breaks" not in parser_help
    assert "fullgraph" in parser_help


# ---------------------------------------------------------------------------
# Friendly error when nvalchemi is not importable (e.g. Python 3.10)
# ---------------------------------------------------------------------------


@pytest.fixture
def restore_cli_modules():
    """Snapshot and restore the CLI and wrapper modules.

    Re-importing a module rebinds it in ``sys.modules`` *and* as an
    attribute of its parent package. ``monkeypatch`` only undoes the
    former, so a test that re-imports the CLI against a blocked
    ``nvalchemi`` would leave ``so3krates_torch.cli.run_md_nvalchemi``
    pointing at the broken copy for every later test.
    """
    import importlib
    import sys

    missing = object()
    targets = (
        (
            "so3krates_torch.cli.run_md_nvalchemi",
            "so3krates_torch.cli",
            "run_md_nvalchemi",
        ),
        (
            "so3krates_torch.calculator.nvalchemi_so3",
            "so3krates_torch.calculator",
            "nvalchemi_so3",
        ),
    )
    saved = []
    for mod_name, pkg_name, attr in targets:
        pkg = importlib.import_module(pkg_name)
        saved.append(
            (
                mod_name,
                sys.modules.get(mod_name, missing),
                pkg,
                attr,
                getattr(pkg, attr, missing),
            )
        )
    yield
    for mod_name, mod, pkg, attr, attr_val in saved:
        if mod is missing:
            sys.modules.pop(mod_name, None)
        else:
            sys.modules[mod_name] = mod
        if attr_val is missing:
            if hasattr(pkg, attr):
                delattr(pkg, attr)
        else:
            setattr(pkg, attr, attr_val)


def test_cli_friendly_error_without_nvalchemi(
    monkeypatch, restore_cli_modules
):
    import importlib
    import sys

    for name in list(sys.modules):
        if name == "nvalchemi" or name.startswith("nvalchemi."):
            monkeypatch.setitem(sys.modules, name, None)
    monkeypatch.setitem(sys.modules, "nvalchemi", None)
    for name in (
        "so3krates_torch.cli.run_md_nvalchemi",
        "so3krates_torch.calculator.nvalchemi_so3",
    ):
        if name in sys.modules:
            monkeypatch.delitem(sys.modules, name)

    mod = importlib.import_module("so3krates_torch.cli.run_md_nvalchemi")
    monkeypatch.setattr(sys, "argv", ["torchkrates-md-nvalchemi", "--help"])
    with pytest.raises(SystemExit) as exc:
        mod.main()
    msg = str(exc.value)
    assert "3.11" in msg
    assert ".[nvalchemi]" in msg


# ---------------------------------------------------------------------------
# Native PME (--electrostatics nvalchemi)
# ---------------------------------------------------------------------------

_NATIVE_PME = ("--electrostatics", "nvalchemi", "--pme-alpha", "0.6")


def test_cli_native_pme_rejects_non_periodic(monkeypatch, tmp_path):
    """PME needs a periodic cell: a molecule exits with a clear message."""
    with pytest.raises(SystemExit) as exc:
        _run_cli(
            monkeypatch,
            tmp_path,
            _h2o(),
            "--ensemble",
            "nvt",
            "--electrostatics",
            "nvalchemi",
        )
    assert "periodic" in str(exc.value)


def test_configure_active_outputs_npt_native_pme_stress():
    """NPT on the native-PME pipeline: the stress it reports equals the
    strain derivative of the energy (Coulomb virial included)."""
    from so3krates_torch.cli.run_md_nvalchemi import _configure_active_outputs
    from so3krates_torch.tests.test_nvalchemi_so3 import (
        _pme_pipeline,
        _rocksalt,
        _stress_vs_strain_fd,
    )

    pipe = _pme_pipeline()
    _configure_active_outputs(pipe, "npt")
    assert pipe.model_config.active_outputs == {"energy", "forces", "stress"}
    analytic, fd = _stress_vs_strain_fd(pipe, _rocksalt(), 0, 0)
    assert analytic == pytest.approx(fd, rel=1e-3, abs=1e-7)


def test_configure_active_outputs_nvt_native_pme():
    from so3krates_torch.cli.run_md_nvalchemi import _configure_active_outputs
    from so3krates_torch.tests.test_nvalchemi_so3 import (
        _pipeline_eval,
        _pme_pipeline,
        _rocksalt,
    )

    pipe = _pme_pipeline()
    _configure_active_outputs(pipe, "nvt")
    assert pipe.model_config.active_outputs == {"energy", "forces"}
    out, _ = _pipeline_eval(pipe, _rocksalt())
    assert "stress" not in out
    assert torch.isfinite(out["forces"]).all()


def test_cli_native_pme_nvt_runs(monkeypatch, tmp_path):
    log = _run_cli(
        monkeypatch,
        tmp_path,
        _nacl_crystal(),
        "--ensemble",
        "nvt",
        *_NATIVE_PME,
    )
    rows = _read_csv_rows(log)
    _assert_finite_rows(rows, ["energy", "volume_A3", "density_g_cm3"])
    assert "pressure_bar" not in rows[0]


def test_cli_native_pme_npt_runs(monkeypatch, tmp_path):
    log = _run_cli(
        monkeypatch,
        tmp_path,
        _nacl_crystal(),
        "--ensemble",
        "npt",
        *_NATIVE_PME,
    )
    rows = _read_csv_rows(log)
    _assert_finite_rows(
        rows, ["energy", "volume_A3", "density_g_cm3", "pressure_bar"]
    )


def test_cli_has_no_pme_mesh_spacing_option():
    """``--pme-mesh-spacing`` did nothing (the wrapper ignores it), so it is
    gone from both the parser and ``load_model``."""
    import inspect

    from so3krates_torch.cli.run_md_nvalchemi import (
        _build_parser,
        load_model,
    )

    assert "pme_mesh_spacing" not in inspect.signature(load_model).parameters
    with pytest.raises(SystemExit):
        _build_parser().parse_args(
            ["--model", "m", "--atoms", "a", "--pme-mesh-spacing", "0.5"]
        )


def test_cli_pme_mesh_dimensions_option():
    from so3krates_torch.cli.run_md_nvalchemi import (
        _build_parser,
        load_model,
    )

    import inspect

    assert "pme_mesh_dimensions" in inspect.signature(load_model).parameters
    parser = _build_parser()
    base = ["--model", "m", "--atoms", "a"]
    assert parser.parse_args(base).pme_mesh_dimensions is None
    args = parser.parse_args(
        base + ["--pme-mesh-dimensions", "16", "20", "24"]
    )
    assert args.pme_mesh_dimensions == [16, 20, 24]
    with pytest.raises(SystemExit):  # exactly three integers
        parser.parse_args(base + ["--pme-mesh-dimensions", "16", "16"])


def test_cli_native_pme_mesh_dimensions_reach_the_builder(
    monkeypatch, tmp_path
):
    from so3krates_torch.cli import run_md_nvalchemi

    seen = {}
    orig = run_md_nvalchemi.build_nvalchemi_pme_model

    def _spy(*args, **kwargs):
        seen["mesh_dimensions"] = kwargs["mesh_dimensions"]
        return orig(*args, **kwargs)

    monkeypatch.setattr(run_md_nvalchemi, "build_nvalchemi_pme_model", _spy)
    log = _run_cli(
        monkeypatch,
        tmp_path,
        _nacl_crystal(),
        "--ensemble",
        "nvt",
        *_NATIVE_PME,
        "--pme-mesh-dimensions",
        "16",
        "16",
        "16",
    )
    assert tuple(seen["mesh_dimensions"]) == (16, 16, 16)
    _assert_finite_rows(_read_csv_rows(log), ["energy"])


# ---------------------------------------------------------------------------
# Models that store no long-range cutoff (e.g. the packaged so3lr.model)
# ---------------------------------------------------------------------------


def test_cli_r_max_lr_option():
    import inspect

    from so3krates_torch.cli.run_md_nvalchemi import (
        _build_parser,
        load_model,
    )

    assert "r_max_lr" in inspect.signature(load_model).parameters
    parser = _build_parser()
    base = ["--model", "m", "--atoms", "a"]
    assert parser.parse_args(base).r_max_lr is None
    assert parser.parse_args(base + ["--r-max-lr", "12"]).r_max_lr == 12.0
    # the other CLIs spell it with underscores
    assert parser.parse_args(base + ["--r_max_lr", "12"]).r_max_lr == 12.0


def test_load_model_needs_a_long_range_cutoff(tmp_path):
    """A long-range model without a stored cutoff fails early with a message
    that names the flag; --r-max-lr supplies it."""
    from so3krates_torch.cli.run_md_nvalchemi import load_model

    path = _save_tiny_model(tmp_path, strip_lr_cutoff=True)
    with pytest.raises(ValueError, match="--r-max-lr"):
        load_model(str(path), "cpu", torch.float64)
    model = load_model(str(path), "cpu", torch.float64, r_max_lr=6.0)
    assert model.has_lr
    assert model.r_lr == 6.0


def test_load_model_r_max_lr_overrides_the_stored_cutoff(tmp_path):
    from so3krates_torch.cli.run_md_nvalchemi import load_model

    path = _save_tiny_model(tmp_path)  # stores r_max_lr = 6.0
    model = load_model(str(path), "cpu", torch.float64, r_max_lr=5.0)
    assert model.r_lr == 5.0


@pytest.mark.parametrize("ensemble", ["nvt", "npt"])
def test_cli_runs_model_without_stored_cutoff(monkeypatch, tmp_path, ensemble):
    log = _run_cli(
        monkeypatch,
        tmp_path,
        _nacl_crystal(),
        "--ensemble",
        ensemble,
        "--r-max-lr",
        "6.0",
        strip_lr_cutoff=True,
    )
    required = ["energy", "volume_A3"]
    if ensemble == "npt":
        required.append("pressure_bar")
    _assert_finite_rows(_read_csv_rows(log), required)


def test_cli_without_cutoff_flag_fails_with_a_clear_message(
    monkeypatch, tmp_path
):
    with pytest.raises(ValueError, match="--r-max-lr"):
        _run_cli(
            monkeypatch,
            tmp_path,
            _nacl_crystal(),
            "--ensemble",
            "nvt",
            strip_lr_cutoff=True,
        )
