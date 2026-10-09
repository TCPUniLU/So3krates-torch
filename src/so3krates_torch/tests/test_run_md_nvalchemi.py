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
    from ase.build import bulk
    from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
    from nvalchemi.data.batch import Batch

    # NOTE: the task brief's test imports `from nvalchemi.dynamics import
    # KB_EV` and `from nvalchemi.dynamics.hooks import
    # kinetic_energy_per_graph`; neither name is re-exported at those
    # levels in nvalchemi-toolkit 0.2.0 (installed). Both live in
    # nvalchemi.dynamics.hooks._utils, the same module nvalchemi's own
    # integrators import them from. KB_EV is unused by this test (it only
    # compares the raw kinetic energy to ASE's), so it is dropped here.
    from nvalchemi.dynamics.hooks._utils import kinetic_energy_per_graph

    from so3krates_torch.cli.run_md_nvalchemi import atoms_to_data

    atoms = bulk("NaCl", "rocksalt", a=5.64).repeat(2)
    MaxwellBoltzmannDistribution(
        atoms, temperature_K=300, rng=np.random.default_rng(0)
    )
    data = atoms_to_data(atoms, device="cpu", dtype=torch.float64)
    b = Batch.from_data_list([data])
    ke = kinetic_energy_per_graph(
        b.velocities, b.atomic_masses, b.batch_idx, b.num_graphs
    )
    assert ke.item() == pytest.approx(atoms.get_kinetic_energy(), rel=1e-8)


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
