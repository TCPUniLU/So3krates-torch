"""Tests for the torchkrates-md-nvalchemi CLI helpers."""

import pytest

pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit not installed")
pytest.importorskip("nvalchemiops", reason="nvalchemi-toolkit-ops missing")


def test_cli_module_importable_from_cli_package():
    from so3krates_torch.cli import run_md_nvalchemi

    assert callable(run_md_nvalchemi.main)


import types

import torch


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
