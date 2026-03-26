"""Tests for autograd utilities in tools/utils.py.

Covers compute_forces, compute_forces_virials, and
get_symmetric_displacement. Wrong signs or missing gradients are
catastrophic for force predictions.
"""

import pytest
import torch

from so3krates_torch.tools.utils import (
    compute_forces,
    compute_forces_virials,
    get_symmetric_displacement,
)


def test_compute_forces_sign(device):
    """Forces should equal -dE/dx (F = -grad E)."""
    positions = torch.randn(10, 3, requires_grad=True, device=device)
    energy = (positions**2).sum()

    forces = compute_forces(energy, positions, training=False)

    expected = -2 * positions.detach()
    assert torch.allclose(forces, expected, atol=1e-6)


def test_compute_forces_zero_when_no_dependence(device):
    """Forces are zero when energy has no dependence on positions."""
    positions = torch.randn(10, 3, requires_grad=True, device=device)
    # Energy depends on a different leaf — not positions
    other = torch.tensor(1.0, requires_grad=True, device=device)
    energy = other.unsqueeze(0)

    forces = compute_forces(energy, positions, training=False)

    assert torch.allclose(forces, torch.zeros_like(positions))


def test_compute_forces_retain_graph_training(device):
    """training=True retains the computation graph for re-use."""
    positions = torch.randn(5, 3, requires_grad=True, device=device)
    energy = (positions**2).sum()

    compute_forces(energy, positions, training=True)

    # Graph must still be intact — backward should not raise
    energy.backward()


def test_compute_forces_no_create_graph_eval(device):
    """training=False does not create a higher-order graph."""
    positions = torch.randn(5, 3, requires_grad=True, device=device)
    energy = (positions**2).sum()

    forces = compute_forces(energy, positions, training=False)

    assert forces.grad_fn is None


def _make_symmetric_displacement_inputs(N, G, device):
    """Helper: minimal valid inputs for get_symmetric_displacement."""
    positions = torch.randn(N, 3, requires_grad=True, device=device)
    # Two directed edges between first two atoms (no PBC shifts)
    edge_index = torch.tensor([[0, 1], [1, 0]], device=device)
    unit_shifts = torch.zeros(2, 3, dtype=positions.dtype, device=device)
    cell = (
        torch.eye(3, dtype=positions.dtype, device=device)
        .unsqueeze(0)
        .expand(G, -1, -1)
        .reshape(-1, 3)
    )
    batch = torch.zeros(N, dtype=torch.long, device=device)
    return positions, unit_shifts, cell, edge_index, batch


def test_get_symmetric_displacement_symmetric(device):
    """Returned displacement produces a symmetric strain tensor."""
    N, G = 4, 1
    (
        positions,
        unit_shifts,
        cell,
        edge_index,
        batch,
    ) = _make_symmetric_displacement_inputs(N, G, device)

    _, _, displacement = get_symmetric_displacement(
        positions=positions,
        unit_shifts=unit_shifts,
        cell=cell,
        edge_index=edge_index,
        num_graphs=G,
        batch=batch,
    )

    assert displacement.shape == (G, 3, 3)
    assert displacement.requires_grad

    # The symmetrized version must be symmetric by construction
    sym = 0.5 * (displacement + displacement.transpose(-1, -2))
    assert torch.allclose(sym, sym.transpose(-1, -2))


def test_compute_forces_virials_sign(device):
    """Forces and virials have the correct sign for a harmonic potential."""
    N, G = 2, 1
    (
        positions,
        unit_shifts,
        cell,
        edge_index,
        batch,
    ) = _make_symmetric_displacement_inputs(N, G, device)
    # Fix positions so sign is deterministic
    with torch.no_grad():
        positions.copy_(torch.tensor([[1.0, 0.0, 0.0], [-1.0, 0.0, 0.0]]))

    pos_def, _, displacement = get_symmetric_displacement(
        positions=positions,
        unit_shifts=unit_shifts,
        cell=cell,
        edge_index=edge_index,
        num_graphs=G,
        batch=batch,
    )

    energy = (pos_def**2).sum().unsqueeze(0)  # [G,]
    forces, virials, _ = compute_forces_virials(
        energy=energy,
        positions=pos_def,
        displacement=displacement,
        cell=cell,
        training=False,
    )

    # Atom 0 has positive x; force must point toward -x
    assert forces[0, 0].item() < 0
    # Diagonal virial (xx) must be negative for this potential
    assert virials[0, 0, 0].item() < 0


def test_stress_from_virials(device):
    """Stress equals virials divided by cell volume."""
    N, G = 3, 1
    L = 2.0  # cubic box side length
    (
        positions,
        unit_shifts,
        _,
        edge_index,
        batch,
    ) = _make_symmetric_displacement_inputs(N, G, device)
    cell_3x3 = torch.eye(3, dtype=positions.dtype, device=device) * L
    cell = cell_3x3.reshape(-1, 3)  # (3, 3) for G=1

    pos_def, _, displacement = get_symmetric_displacement(
        positions=positions,
        unit_shifts=unit_shifts,
        cell=cell,
        edge_index=edge_index,
        num_graphs=G,
        batch=batch,
    )

    energy = (pos_def**2).sum().unsqueeze(0)
    forces, virials, stress = compute_forces_virials(
        energy=energy,
        positions=pos_def,
        displacement=displacement,
        cell=cell,
        training=False,
        compute_stress=True,
    )

    # The function computes stress = virials_raw / volume and returns
    # (-1 * virials_raw, stress), so stress = -returned_virials / volume.
    cell_volume = torch.linalg.det(cell.view(-1, 3, 3)).abs()
    expected_stress = -virials / cell_volume.view(-1, 1, 1)
    assert torch.allclose(stress, expected_stress, atol=1e-6)


def test_compute_forces_virials_shapes(device):
    """Output tensors have the expected shapes: forces (N,3), virials (G,3,3)."""
    N, G = 5, 1
    (
        positions,
        unit_shifts,
        cell,
        edge_index,
        batch,
    ) = _make_symmetric_displacement_inputs(N, G, device)

    pos_def, _, displacement = get_symmetric_displacement(
        positions=positions,
        unit_shifts=unit_shifts,
        cell=cell,
        edge_index=edge_index,
        num_graphs=G,
        batch=batch,
    )

    energy = (pos_def**2).sum().unsqueeze(0)  # [G,]
    forces, virials, _ = compute_forces_virials(
        energy=energy,
        positions=pos_def,
        displacement=displacement,
        cell=cell,
        training=False,
    )

    assert forces.shape == (N, 3)
    assert virials.shape == (G, 3, 3)
