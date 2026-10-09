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
