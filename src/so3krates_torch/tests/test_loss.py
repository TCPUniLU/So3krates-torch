"""Tests for loss functions in modules/loss.py.

Tests verify correctness of individual loss components and composite losses
using hand-computed expected values.
"""

import pytest
import torch
from so3krates_torch.modules.loss import (
    weighted_mean_squared_error_energy,
    mean_squared_error_forces,
    weighted_mean_squared_error_dipole,
    weighted_mean_squared_error_hirshfeld,
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesHirshfeldLoss,
    WeightedEnergyForcesDipoleHirshfeldLoss,
)


def test_weighted_mse_energy_single_graph(mock_batch_for_loss):
    """Verify energy loss for single graph with per-atom normalization.

    Hand calculation for graph 0 (3 atoms):
    loss = weight * energy_weight * ((ref - pred) / num_atoms)^2
         = 1.0 * 1.0 * ((10.0 - 8.0) / 3)^2
         = 1.0 * 1.0 * (2.0 / 3)^2
         = 0.44444...
    """
    batch = mock_batch_for_loss
    # Create single-graph batch (first 3 atoms only)
    batch.ptr = torch.tensor([0, 3])
    batch.energy = torch.tensor([10.0])
    batch.weight = torch.tensor([1.0])
    batch.energy_weight = torch.tensor([1.0])

    pred = {"energy": torch.tensor([8.0])}

    loss = weighted_mean_squared_error_energy(batch, pred, ddp=False)

    expected = 1.0 * 1.0 * ((10.0 - 8.0) / 3.0) ** 2
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_weighted_mse_energy_multi_graph(mock_batch_for_loss):
    """Verify per-graph normalization doesn't leak across graphs.

    Graph 0 (3 atoms): weight=1.0, energy_weight=1.0
        loss_0 = 1.0 * 1.0 * ((10.0 - 8.0) / 3)^2 = 4/9
    Graph 1 (4 atoms): weight=2.0, energy_weight=1.5
        loss_1 = 2.0 * 1.5 * ((20.0 - 16.0) / 4)^2 = 3.0 * 1.0 = 3.0
    Mean: (4/9 + 3.0) / 2 = 1.722...
    """
    batch = mock_batch_for_loss
    pred = {"energy": torch.tensor([8.0, 16.0])}

    loss = weighted_mean_squared_error_energy(batch, pred, ddp=False)

    loss_0 = 1.0 * 1.0 * ((10.0 - 8.0) / 3.0) ** 2
    loss_1 = 2.0 * 1.5 * ((20.0 - 16.0) / 4.0) ** 2
    expected = (loss_0 + loss_1) / 2
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_weighted_mse_energy_zero_weight(mock_batch_for_loss):
    """Verify zero weight makes graph contribute zero to loss.

    Graph 0: weight=0.0, should contribute 0
    Graph 1: weight=2.0, energy_weight=1.5
        loss_1 = 2.0 * 1.5 * ((20.0 - 16.0) / 4)^2 = 3.0
    Mean: (0.0 + 3.0) / 2 = 1.5
    """
    batch = mock_batch_for_loss
    batch.weight = torch.tensor([0.0, 2.0])
    pred = {"energy": torch.tensor([8.0, 16.0])}

    loss = weighted_mean_squared_error_energy(batch, pred, ddp=False)

    loss_0 = 0.0
    loss_1 = 2.0 * 1.5 * ((20.0 - 16.0) / 4.0) ** 2
    expected = (loss_0 + loss_1) / 2
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_mse_forces_weight_broadcasting(mock_batch_for_loss):
    """Verify weight broadcasting from graph-level to atom-level.

    Graph 0 (3 atoms): weight=1.0, forces_weight=10.0
    Graph 1 (4 atoms): weight=2.0, forces_weight=20.0
    Each atom in a graph should share the same broadcasted weight.
    """
    batch = mock_batch_for_loss
    pred = {"forces": batch.forces.clone()}  # pred = ref

    # When pred = ref, loss should be zero regardless of weights
    loss = mean_squared_error_forces(batch, pred, ddp=False)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-6)

    # Now verify non-zero case with known values
    pred = {"forces": torch.zeros_like(batch.forces)}
    loss = mean_squared_error_forces(batch, pred, ddp=False)

    # Manual calculation:
    # Graph 0: 3 atoms, weight=1.0, forces_weight=10.0
    #   atom 0: 1.0 * 10.0 * (1^2 + 0^2 + 0^2) = 10.0
    #   atom 1: 1.0 * 10.0 * (0^2 + 1^2 + 0^2) = 10.0
    #   atom 2: 1.0 * 10.0 * (0^2 + 0^2 + 1^2) = 10.0
    # Graph 1: 4 atoms, weight=2.0, forces_weight=20.0
    #   atom 3: 2.0 * 20.0 * (2^2 + 0^2 + 0^2) = 160.0
    #   atom 4: 2.0 * 20.0 * (0^2 + 2^2 + 0^2) = 160.0
    #   atom 5: 2.0 * 20.0 * (0^2 + 0^2 + 2^2) = 160.0
    #   atom 6: 2.0 * 20.0 * (1^2 + 1^2 + 1^2) = 120.0
    # Total elements: 3 * 3 + 4 * 3 = 21
    # Mean: (30 + 600) / 21 = 30.0
    expected = (30.0 + 600.0) / 21.0
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_mse_forces_value(mock_batch_for_loss):
    """Verify forces MSE calculation with hand-computed value.

    Using pred = ref * 0.5 to create known differences.
    The loss function computes element-wise squared errors (7 atoms * 3
    components = 21 elements), each weighted, then takes the mean.
    """
    batch = mock_batch_for_loss
    pred = {"forces": batch.forces * 0.5}

    loss = mean_squared_error_forces(batch, pred, ddp=False)

    # Compute manually: each element has difference = ref * 0.5
    # squared_diff = (ref - pred)^2 = (ref * 0.5)^2 = ref^2 * 0.25
    raw_loss_elements = []
    for i, (g_idx, ref_f) in enumerate(
        zip(batch.batch.tolist(), batch.forces)
    ):
        weight = batch.weight[g_idx].item()
        forces_weight = batch.forces_weight[g_idx].item()
        # Each of 3 force components contributes separately
        for component_diff_sq in (ref_f - pred["forces"][i]) ** 2:
            raw_loss_elements.append(
                weight * forces_weight * component_diff_sq.item()
            )

    expected = sum(raw_loss_elements) / len(raw_loss_elements)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_weighted_mse_dipole(mock_batch_for_loss):
    """Verify dipole loss uses per-atom normalization.

    Graph 0 (3 atoms): dipole_ref=[1,0,0], dipole_pred=[0.5,0,0]
        Component losses: ((1-0.5)/3)^2, ((0-0)/3)^2, ((0-0)/3)^2
    Graph 1 (4 atoms): dipole_ref=[0,1,0], dipole_pred=[0,0.5,0]
        Component losses: ((0-0)/4)^2, ((1-0.5)/4)^2, ((0-0)/4)^2
    Mean over all 6 elements (2 graphs * 3 components).
    """
    batch = mock_batch_for_loss
    pred = {"dipole": batch.dipole * 0.5}

    loss = weighted_mean_squared_error_dipole(batch, pred, ddp=False)

    # Compute element-wise: 2 graphs * 3 components = 6 elements
    raw_loss_elements = []

    # Graph 0: 3 atoms
    num_atoms_0 = 3.0
    diff_0 = (batch.dipole[0] - pred["dipole"][0]) / num_atoms_0
    for component in diff_0**2:
        raw_loss_elements.append(component.item())

    # Graph 1: 4 atoms
    num_atoms_1 = 4.0
    diff_1 = (batch.dipole[1] - pred["dipole"][1]) / num_atoms_1
    for component in diff_1**2:
        raw_loss_elements.append(component.item())

    expected = sum(raw_loss_elements) / len(raw_loss_elements)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_weighted_mse_hirshfeld(mock_batch_for_loss):
    """Verify Hirshfeld loss weight broadcasting from graph to atom.

    Graph 0 (3 atoms): weight=1.0, hirshfeld_weight=1.0
    Graph 1 (4 atoms): weight=2.0, hirshfeld_weight=2.0
    All reference values are 1.0, pred = 0.5
    """
    batch = mock_batch_for_loss
    pred = {"hirshfeld_ratios": batch.hirshfeld_ratios * 0.5}

    loss = weighted_mean_squared_error_hirshfeld(batch, pred, ddp=False)

    # Manual: each atom has ref=1.0, pred=0.5, diff^2=0.25
    # Graph 0: 3 atoms * 1.0 * 1.0 * 0.25 = 0.75
    # Graph 1: 4 atoms * 2.0 * 2.0 * 0.25 = 4.0
    # Mean over 7 elements: (0.75 + 4.0) / 7 = 0.678...
    raw_loss = []
    for i, g_idx in enumerate(batch.batch.tolist()):
        weight = batch.weight[g_idx].item()
        hirsh_weight = batch.hirshfeld_ratios_weight[g_idx].item()
        diff_squared = (
            (batch.hirshfeld_ratios[i] - pred["hirshfeld_ratios"][i]) ** 2
        ).item()
        raw_loss.append(weight * hirsh_weight * diff_squared)

    expected = sum(raw_loss) / len(raw_loss)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_loss_zero_when_pred_equals_ref(mock_batch_for_loss):
    """Verify all loss functions return exactly 0 when pred=ref."""
    batch = mock_batch_for_loss
    pred = {
        "energy": batch.energy.clone(),
        "forces": batch.forces.clone(),
        "dipole": batch.dipole.clone(),
        "hirshfeld_ratios": batch.hirshfeld_ratios.clone(),
    }

    loss_energy = weighted_mean_squared_error_energy(batch, pred, ddp=False)
    loss_forces = mean_squared_error_forces(batch, pred, ddp=False)
    loss_dipole = weighted_mean_squared_error_dipole(batch, pred, ddp=False)
    loss_hirshfeld = weighted_mean_squared_error_hirshfeld(
        batch, pred, ddp=False
    )

    assert torch.allclose(loss_energy, torch.tensor(0.0), atol=1e-8)
    assert torch.allclose(loss_forces, torch.tensor(0.0), atol=1e-8)
    assert torch.allclose(loss_dipole, torch.tensor(0.0), atol=1e-8)
    assert torch.allclose(loss_hirshfeld, torch.tensor(0.0), atol=1e-8)


def test_WeightedEnergyForcesLoss_weight_control(mock_batch_for_loss):
    """Verify energy_weight and forces_weight control contributions.

    Test that setting one weight to zero excludes that component.
    """
    batch = mock_batch_for_loss
    pred = {
        "energy": torch.tensor([8.0, 16.0]),
        "forces": torch.zeros_like(batch.forces),
    }

    # Case 1: Only energy (forces_weight=0)
    loss_fn = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=0.0)
    loss = loss_fn(batch, pred, ddp=False)
    loss_energy_only = weighted_mean_squared_error_energy(
        batch, pred, ddp=False
    )
    assert torch.allclose(loss, loss_energy_only, atol=1e-6)

    # Case 2: Only forces (energy_weight=0)
    loss_fn = WeightedEnergyForcesLoss(energy_weight=0.0, forces_weight=1.0)
    loss = loss_fn(batch, pred, ddp=False)
    loss_forces_only = mean_squared_error_forces(batch, pred, ddp=False)
    assert torch.allclose(loss, loss_forces_only, atol=1e-6)

    # Case 3: Both components with equal weights
    loss_fn = WeightedEnergyForcesLoss(energy_weight=1.0, forces_weight=1.0)
    loss = loss_fn(batch, pred, ddp=False)
    expected = loss_energy_only + loss_forces_only
    assert torch.allclose(loss, expected, atol=1e-6)


def test_WeightedEnergyForcesDipoleLoss(mock_batch_for_loss):
    """Verify three-component loss sums correctly.

    Total loss should equal:
    energy_weight * energy_loss + forces_weight * forces_loss +
    dipole_weight * dipole_loss
    """
    batch = mock_batch_for_loss
    pred = {
        "energy": torch.tensor([8.0, 16.0]),
        "forces": torch.zeros_like(batch.forces),
        "dipole": batch.dipole * 0.5,
    }

    # Compute individual losses
    loss_energy = weighted_mean_squared_error_energy(batch, pred, ddp=False)
    loss_forces = mean_squared_error_forces(batch, pred, ddp=False)
    loss_dipole = weighted_mean_squared_error_dipole(batch, pred, ddp=False)

    # Test with unit weights
    loss_fn = WeightedEnergyForcesDipoleLoss(
        energy_weight=1.0, forces_weight=1.0, dipole_weight=1.0
    )
    loss = loss_fn(batch, pred, ddp=False)
    expected = loss_energy + loss_forces + loss_dipole
    assert torch.allclose(loss, expected, atol=1e-6)

    # Test with custom weights
    loss_fn = WeightedEnergyForcesDipoleLoss(
        energy_weight=2.0, forces_weight=3.0, dipole_weight=0.5
    )
    loss = loss_fn(batch, pred, ddp=False)
    expected = 2.0 * loss_energy + 3.0 * loss_forces + 0.5 * loss_dipole
    assert torch.allclose(loss, expected, atol=1e-6)


def test_WeightedEnergyForcesHirshfeldLoss(mock_batch_for_loss):
    """Verify three-component independence (energy+forces+Hirshfeld).

    Zero out one weight at a time and verify other components unaffected.
    """
    batch = mock_batch_for_loss
    pred = {
        "energy": torch.tensor([8.0, 16.0]),
        "forces": torch.zeros_like(batch.forces),
        "hirshfeld_ratios": batch.hirshfeld_ratios * 0.5,
    }

    # Compute individual losses
    loss_energy = weighted_mean_squared_error_energy(batch, pred, ddp=False)
    loss_forces = mean_squared_error_forces(batch, pred, ddp=False)
    loss_hirshfeld = weighted_mean_squared_error_hirshfeld(
        batch, pred, ddp=False
    )

    # Case 1: Zero energy_weight
    loss_fn = WeightedEnergyForcesHirshfeldLoss(
        energy_weight=0.0, forces_weight=1.0, hirshfeld_weight=1.0
    )
    loss = loss_fn(batch, pred, ddp=False)
    expected = loss_forces + loss_hirshfeld
    assert torch.allclose(loss, expected, atol=1e-6)

    # Case 2: Zero forces_weight
    loss_fn = WeightedEnergyForcesHirshfeldLoss(
        energy_weight=1.0, forces_weight=0.0, hirshfeld_weight=1.0
    )
    loss = loss_fn(batch, pred, ddp=False)
    expected = loss_energy + loss_hirshfeld
    assert torch.allclose(loss, expected, atol=1e-6)

    # Case 3: Zero hirshfeld_weight
    loss_fn = WeightedEnergyForcesHirshfeldLoss(
        energy_weight=1.0, forces_weight=1.0, hirshfeld_weight=0.0
    )
    loss = loss_fn(batch, pred, ddp=False)
    expected = loss_energy + loss_forces
    assert torch.allclose(loss, expected, atol=1e-6)


def test_WeightedEnergyForcesDipoleHirshfeldLoss(mock_batch_for_loss):
    """Verify four-component loss composition.

    Set each weight to zero individually and verify remaining three sum
    correctly.
    """
    batch = mock_batch_for_loss
    pred = {
        "energy": torch.tensor([8.0, 16.0]),
        "forces": torch.zeros_like(batch.forces),
        "dipole": batch.dipole * 0.5,
        "hirshfeld_ratios": batch.hirshfeld_ratios * 0.5,
    }

    # Compute individual losses
    loss_energy = weighted_mean_squared_error_energy(batch, pred, ddp=False)
    loss_forces = mean_squared_error_forces(batch, pred, ddp=False)
    loss_dipole = weighted_mean_squared_error_dipole(batch, pred, ddp=False)
    loss_hirshfeld = weighted_mean_squared_error_hirshfeld(
        batch, pred, ddp=False
    )

    # Case 1: All weights = 1.0
    loss_fn = WeightedEnergyForcesDipoleHirshfeldLoss(
        energy_weight=1.0,
        forces_weight=1.0,
        dipole_weight=1.0,
        hirshfeld_weight=1.0,
    )
    loss = loss_fn(batch, pred, ddp=False)
    expected = loss_energy + loss_forces + loss_dipole + loss_hirshfeld
    assert torch.allclose(loss, expected, atol=1e-6)

    # Case 2: Zero energy_weight
    loss_fn = WeightedEnergyForcesDipoleHirshfeldLoss(
        energy_weight=0.0,
        forces_weight=1.0,
        dipole_weight=1.0,
        hirshfeld_weight=1.0,
    )
    loss = loss_fn(batch, pred, ddp=False)
    expected = loss_forces + loss_dipole + loss_hirshfeld
    assert torch.allclose(loss, expected, atol=1e-6)

    # Case 3: Zero forces_weight
    loss_fn = WeightedEnergyForcesDipoleHirshfeldLoss(
        energy_weight=1.0,
        forces_weight=0.0,
        dipole_weight=1.0,
        hirshfeld_weight=1.0,
    )
    loss = loss_fn(batch, pred, ddp=False)
    expected = loss_energy + loss_dipole + loss_hirshfeld
    assert torch.allclose(loss, expected, atol=1e-6)

    # Case 4: Zero dipole_weight
    loss_fn = WeightedEnergyForcesDipoleHirshfeldLoss(
        energy_weight=1.0,
        forces_weight=1.0,
        dipole_weight=0.0,
        hirshfeld_weight=1.0,
    )
    loss = loss_fn(batch, pred, ddp=False)
    expected = loss_energy + loss_forces + loss_hirshfeld
    assert torch.allclose(loss, expected, atol=1e-6)

    # Case 5: Zero hirshfeld_weight
    loss_fn = WeightedEnergyForcesDipoleHirshfeldLoss(
        energy_weight=1.0,
        forces_weight=1.0,
        dipole_weight=1.0,
        hirshfeld_weight=0.0,
    )
    loss = loss_fn(batch, pred, ddp=False)
    expected = loss_energy + loss_forces + loss_dipole
    assert torch.allclose(loss, expected, atol=1e-6)
