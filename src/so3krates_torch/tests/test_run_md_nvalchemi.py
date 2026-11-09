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


def _v1_checkpoint_path():
    import importlib.resources as resources

    return str(
        resources.files("so3krates_torch.pretrained.so3lr") / "v1" / "so3lr.pt"
    )


@pytest.mark.parametrize("source", ["keyword", "pt_path"])
def test_load_model_accepts_bundled_checkpoints(source):
    """The packaged model is now a state_dict + settings (v1), not a pickled
    model: ``--model v1`` and ``--model <path>/so3lr.pt`` must both load."""
    from so3krates_torch.calculator.nvalchemi_so3 import NVAlchemiSO3LR
    from so3krates_torch.cli.run_md_nvalchemi import load_model

    path = "v1" if source == "keyword" else _v1_checkpoint_path()
    model = load_model(
        path,
        "cpu",
        torch.float64,
        r_max_lr=12.0,
        dispersion_energy_cutoff_lr_damping=2.0,
    )
    assert isinstance(model, NVAlchemiSO3LR)
    assert model.model.r_max_lr == pytest.approx(12.0)
    assert next(model.model.parameters()).dtype == torch.float64
    assert not any(p.requires_grad for p in model.model.parameters())


def _save_multi_head_model(tmp_path, num_theory_levels, **extra):
    from so3krates_torch.tests.test_nvalchemi_so3 import _make_so3lr

    path = tmp_path / f"heads{num_theory_levels}.model"
    torch.manual_seed(0)
    torch.save(
        _make_so3lr(
            electrostatics_bool=True,
            num_theory_levels=num_theory_levels,
            **extra,
        ),
        path,
    )
    return path


def test_load_model_multi_head_defaults_to_theory_level_5(tmp_path, capsys):
    """Like so3lr_dev: level 5 unless --theory-level says otherwise, and the
    default is announced."""
    from so3krates_torch.cli.run_md_nvalchemi import load_model

    path = _save_multi_head_model(tmp_path, 6)
    model = load_model(str(path), "cpu", torch.float64)
    assert model.theory_level == 5
    assert "theory level 5" in capsys.readouterr().out
    assert (
        load_model(
            str(path), "cpu", torch.float64, theory_level=2
        ).theory_level
        == 2
    )


def test_load_model_theory_level_out_of_range_names_the_flag(tmp_path):
    from so3krates_torch.cli.run_md_nvalchemi import load_model

    path = _save_multi_head_model(tmp_path, 2)  # default level 5 is invalid
    with pytest.raises(ValueError, match="--theory-level"):
        load_model(str(path), "cpu", torch.float64)
    assert (
        load_model(
            str(path), "cpu", torch.float64, theory_level=1
        ).theory_level
        == 1
    )


def test_cli_theory_level_selects_the_head(monkeypatch, tmp_path):
    """Through the CLI, --theory-level changes the energy of a multi-head
    model (the heads of a fresh random model differ)."""
    from so3krates_torch.cli import run_md_nvalchemi

    energies = []
    for level in ("0", "1"):
        run_dir = tmp_path / f"level{level}"
        run_dir.mkdir()
        model = _save_multi_head_model(run_dir, 2)
        xyz = run_dir / "in.xyz"
        from ase.io import write

        write(xyz, _nacl_crystal(), format="extxyz")
        log = run_dir / "md_log.csv"
        monkeypatch.setattr(
            "sys.argv",
            [
                "torchkrates-md-nvalchemi",
                "--model",
                str(model),
                "--atoms",
                str(xyz),
                "--steps",
                "1",
                "--log-freq",
                "1",
                "--log-file",
                str(log),
                "--device",
                "cpu",
                "--theory-level",
                level,
            ],
        )
        run_md_nvalchemi.main()
        energies.append(float(_read_csv_rows(log)[0]["energy"]))
    assert energies[0] != pytest.approx(energies[1], abs=1e-9)


def test_load_model_bundled_checkpoint_still_needs_long_range_flags():
    """v1 stores no long-range cutoff: the early error names the flag."""
    from so3krates_torch.cli.run_md_nvalchemi import load_model

    with pytest.raises(ValueError, match="--r-max-lr"):
        load_model("v1", "cpu", torch.float64)


class _StopAfterLoad(Exception):
    pass


@pytest.mark.parametrize(
    "extra, expected",
    [(("--compile",), None), (("--compile", "--no-compile-fullgraph"), False)],
)
def test_no_compile_fullgraph_flag_reaches_load_model(
    monkeypatch, tmp_path, extra, expected
):
    """``--no-compile-fullgraph`` passes ``fullgraph=False`` to load_model;
    without it ``fullgraph`` stays ``None`` (load_model picks True on CUDA)."""
    from so3krates_torch.cli import run_md_nvalchemi

    seen = {}

    def _spy(*args, **kwargs):
        seen.update(kwargs)
        raise _StopAfterLoad

    monkeypatch.setattr(run_md_nvalchemi, "load_model", _spy)
    with pytest.raises(_StopAfterLoad):
        _run_cli(monkeypatch, tmp_path, _nacl_crystal(), *extra)
    assert seen["compile_model"] is True
    assert seen["fullgraph"] is expected


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


def test_cli_r_max_lr_defaults_to_12(monkeypatch, tmp_path):
    """No --r-max-lr: the long-range cutoff is 12 A (SO3LRCalculator's
    default), also for a model that stores none."""
    from so3krates_torch.cli import run_md_nvalchemi

    seen = {}
    orig = run_md_nvalchemi.NVTLangevin

    def _spy(*args, **kwargs):
        seen["r_lr"] = kwargs["model"].r_lr
        return orig(*args, **kwargs)

    monkeypatch.setattr(run_md_nvalchemi, "NVTLangevin", _spy)
    _run_cli(
        monkeypatch,
        tmp_path,
        _nacl_crystal(),
        "--ensemble",
        "nvt",
        strip_lr_cutoff=True,
    )
    assert seen["r_lr"] == 12.0


def test_cli_explicit_r_max_lr_wins_over_the_default(monkeypatch, tmp_path):
    from so3krates_torch.cli import run_md_nvalchemi

    seen = {}
    orig = run_md_nvalchemi.NVTLangevin

    def _spy(*args, **kwargs):
        seen["r_lr"] = kwargs["model"].r_lr
        return orig(*args, **kwargs)

    monkeypatch.setattr(run_md_nvalchemi, "NVTLangevin", _spy)
    _run_cli(
        monkeypatch,
        tmp_path,
        _nacl_crystal(),
        "--ensemble",
        "nvt",
        "--r-max-lr",
        "6.0",
    )
    assert seen["r_lr"] == 6.0


def test_load_model_default_r_max_lr_rules(tmp_path):
    """default_r_max_lr applies when no explicit value is given; it warns
    when it replaces a finite trained cutoff, and stays quiet for a missing
    one or the 1000 A 'unbounded' value the tauri models store."""
    import warnings

    from so3krates_torch.cli.run_md_nvalchemi import load_model
    from so3krates_torch.tests.test_nvalchemi_so3 import _make_so3lr

    stored_6 = _save_tiny_model(tmp_path)  # stores r_max_lr = 6.0
    with pytest.warns(UserWarning, match="r_max_lr=6.0"):
        model = load_model(
            str(stored_6), "cpu", torch.float64, default_r_max_lr=12.0
        )
    assert model.r_lr == 12.0

    with warnings.catch_warnings():
        warnings.simplefilter("error")
        explicit = load_model(
            str(stored_6),
            "cpu",
            torch.float64,
            r_max_lr=6.0,
            default_r_max_lr=12.0,
        )
    assert explicit.r_lr == 6.0

    for stored, name in ((None, "none"), (1000.0, "unbounded")):
        path = tmp_path / f"{name}.model"
        m = _make_so3lr(electrostatics_bool=True)
        m.r_max_lr = stored
        torch.save(m, path)
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            loaded = load_model(
                str(path), "cpu", torch.float64, default_r_max_lr=12.0
            )
        assert loaded.r_lr == 12.0


def test_load_model_default_damping_only_fills_a_missing_width(tmp_path):
    from so3krates_torch.cli.run_md_nvalchemi import load_model
    from so3krates_torch.tests.test_nvalchemi_so3 import _make_so3lr

    path = tmp_path / "nodamp.model"
    m = _make_so3lr(electrostatics_bool=True, dispersion_bool=True)
    m.dispersion_energy_cutoff_lr_damping = None
    torch.save(m, path)
    model = load_model(
        str(path),
        "cpu",
        torch.float64,
        default_dispersion_damping=2.0,
    )
    assert model.model.dispersion_energy_cutoff_lr_damping == 2.0

    stored = tmp_path / "damp.model"  # stores 1.0 (test value)
    torch.save(
        _make_so3lr(electrostatics_bool=True, dispersion_bool=True), stored
    )
    model = load_model(
        str(stored), "cpu", torch.float64, default_dispersion_damping=2.0
    )
    assert model.model.dispersion_energy_cutoff_lr_damping == 1.0


# ---------------------------------------------------------------------------
# The model's own PME (use_pme / use_pme_dispersion) from the CLI
# ---------------------------------------------------------------------------

_V1_PME_ARGS = (
    "--r-max-lr",
    "12.0",
    "--dispersion-energy-cutoff-lr-damping",
    "2.0",
)


def _run_cli_with_model(monkeypatch, tmp_path, frames, model_arg, *extra):
    from ase.io import write

    from so3krates_torch.cli import run_md_nvalchemi

    xyz = tmp_path / "in.xyz"
    write(xyz, frames, format="extxyz")
    log = tmp_path / "md_log.csv"
    monkeypatch.setattr(
        "sys.argv",
        [
            "torchkrates-md-nvalchemi",
            "--model",
            str(model_arg),
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
        ],
    )
    run_md_nvalchemi.main()
    return log


def test_load_model_applies_model_pme_overrides():
    pytest.importorskip("torchpme", reason="torch-pme not installed")
    from so3krates_torch.cli.run_md_nvalchemi import load_model

    model = load_model(
        "v1",
        "cpu",
        torch.float64,
        r_max_lr=12.0,
        dispersion_energy_cutoff_lr_damping=2.0,
        architecture_overrides={
            "use_pme": True,
            "pme_smearing": 1.0,
            "pme_mesh_spacing": 0.5,
        },
    )
    assert model.model.use_pme is True
    assert model.model.pme_smearing == 1.0
    assert model.model.pme_mesh_spacing == 0.5


def test_model_pme_flags_change_the_energy():
    """The PME flags must be live, not just accepted."""
    pytest.importorskip("torchpme", reason="torch-pme not installed")
    from so3krates_torch.cli.run_md_nvalchemi import atoms_to_data, load_model
    from nvalchemi.data.batch import Batch

    common = dict(r_max_lr=12.0, dispersion_energy_cutoff_lr_damping=2.0)
    energies = {}
    for name, overrides in {
        "plain": None,
        "pme": {"use_pme": True, "pme_smearing": 1.0, "pme_mesh_spacing": 0.5},
        "pme+disp": {
            "use_pme": True,
            "pme_smearing": 1.0,
            "pme_mesh_spacing": 0.5,
            "use_pme_dispersion": True,
            "pme_dispersion_smearing": 2.4,
            "pme_dispersion_mesh_spacing": 1.2,
        },
    }.items():
        model = load_model(
            "v1",
            "cpu",
            torch.float64,
            architecture_overrides=overrides,
            **common,
        )
        model.model_config.active_outputs = {"energy"}
        hooks = model.make_neighbor_hooks()
        batch = Batch.from_data_list(
            [atoms_to_data(_nacl_crystal(), "cpu", torch.float64)]
        )
        from nvalchemi.dynamics.base import DynamicsStage
        from nvalchemi.hooks import HookContext

        for hook in hooks:
            hook(HookContext(batch=batch), DynamicsStage.BEFORE_COMPUTE)
        energies[name] = float(model(batch)["energy"].sum())
    assert energies["pme"] != pytest.approx(energies["plain"], abs=1e-6)
    assert energies["pme+disp"] != pytest.approx(energies["pme"], abs=1e-6)


def test_load_model_rejects_pme_overrides_for_pickled_models(tmp_path):
    from so3krates_torch.cli.run_md_nvalchemi import load_model

    path = _save_tiny_model(tmp_path)
    with pytest.raises(ValueError, match="pickled"):
        load_model(
            str(path),
            "cpu",
            torch.float64,
            architecture_overrides={"use_pme": True},
        )


@pytest.mark.parametrize("ensemble", ["nvt", "npt"])
def test_cli_runs_a_bundled_model_with_its_own_pme(
    monkeypatch, tmp_path, ensemble
):
    """v1 + --model-pme (electrostatics) + --model-pme-dispersion."""
    pytest.importorskip("torchpme", reason="torch-pme not installed")
    log = _run_cli_with_model(
        monkeypatch,
        tmp_path,
        _nacl_crystal(),
        "v1",
        "--ensemble",
        ensemble,
        "--model-pme",
        "--model-pme-smearing",
        "1.0",
        "--model-pme-mesh-spacing",
        "0.5",
        "--model-pme-dispersion",
        "--model-pme-dispersion-smearing",
        "2.4",
        "--model-pme-dispersion-mesh-spacing",
        "1.2",
        *_V1_PME_ARGS,
    )
    required = ["energy", "volume_A3"]
    if ensemble == "npt":
        required.append("pressure_bar")
    _assert_finite_rows(_read_csv_rows(log), required)


@pytest.mark.parametrize(
    "extra, message",
    [
        (["--model-pme", "--electrostatics", "nvalchemi"], "nvalchemi"),
        (["--model-pme", "--electrostatics", "none"], "--electrostatics"),
        (["--model-pme-smearing", "1.0"], "--model-pme"),
        (["--model-pme-dispersion-smearing", "2.4"], "--model-pme-dispersion"),
    ],
)
def test_cli_model_pme_flag_conflicts(monkeypatch, tmp_path, extra, message):
    with pytest.raises(SystemExit) as excinfo:
        _run_cli_with_model(
            monkeypatch, tmp_path, _nacl_crystal(), "v1", *extra, *_V1_PME_ARGS
        )
    assert message in str(excinfo.value)


def test_cli_model_pme_needs_a_periodic_cell(monkeypatch, tmp_path):
    with pytest.raises(SystemExit) as excinfo:
        _run_cli_with_model(
            monkeypatch,
            tmp_path,
            _h2o(),
            "v1",
            "--model-pme",
            *_V1_PME_ARGS,
        )
    assert "periodic" in str(excinfo.value)
