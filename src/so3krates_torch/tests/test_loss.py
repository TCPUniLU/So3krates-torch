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
    weighted_mean_squared_error_charges,
    WeightedEnergyForcesLoss,
    WeightedEnergyForcesDipoleLoss,
    WeightedEnergyForcesHirshfeldLoss,
    WeightedEnergyForcesDipoleHirshfeldLoss,
    WeightedEnergyForcesChargesLoss,
    WeightedEnergyForcesChargesHirshfeldLoss,
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


def test_weighted_mse_charges(mock_batch_for_loss):
    """Verify charge loss broadcasts per-graph weights correctly to atoms.

    Graph 0 (3 atoms): weight=1.0, charges_weight=1.0
    Graph 1 (4 atoms): weight=2.0, charges_weight=2.0
    pred = ref * 0.5 so diff^2 = (ref * 0.5)^2 = ref^2 * 0.25
    Mean over 7 atom-level terms.
    """
    batch = mock_batch_for_loss
    pred = {"partial_charges": batch.charges * 0.5}

    loss = weighted_mean_squared_error_charges(batch, pred, ddp=False)

    raw_loss = []
    for i, g_idx in enumerate(batch.batch.tolist()):
        weight = batch.weight[g_idx].item()
        ch_weight = batch.charges_weight[g_idx].item()
        diff_sq = ((batch.charges[i] - pred["partial_charges"][i]) ** 2).item()
        raw_loss.append(weight * ch_weight * diff_sq)

    expected = sum(raw_loss) / len(raw_loss)
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_weighted_mse_charges_zero_when_equal(mock_batch_for_loss):
    """Verify charge loss is zero when pred matches ref."""
    batch = mock_batch_for_loss
    pred = {"partial_charges": batch.charges.clone()}
    loss = weighted_mean_squared_error_charges(batch, pred, ddp=False)
    assert torch.allclose(loss, torch.tensor(0.0), atol=1e-8)


def test_WeightedEnergyForcesChargesLoss(mock_batch_for_loss):
    """Verify charges loss class sums energy + forces + charges correctly."""
    batch = mock_batch_for_loss
    pred = {
        "energy": torch.tensor([8.0, 16.0]),
        "forces": torch.zeros_like(batch.forces),
        "partial_charges": batch.charges * 0.5,
    }

    loss_energy = weighted_mean_squared_error_energy(batch, pred, ddp=False)
    loss_forces = mean_squared_error_forces(batch, pred, ddp=False)
    loss_charges = weighted_mean_squared_error_charges(batch, pred, ddp=False)

    loss_fn = WeightedEnergyForcesChargesLoss(
        energy_weight=1.0, forces_weight=1.0, charges_weight=1.0
    )
    loss = loss_fn(batch, pred, ddp=False)
    expected = loss_energy + loss_forces + loss_charges
    assert torch.allclose(loss, expected, atol=1e-6)

    # Zero charges_weight excludes charges term
    loss_fn = WeightedEnergyForcesChargesLoss(
        energy_weight=1.0, forces_weight=1.0, charges_weight=0.0
    )
    loss = loss_fn(batch, pred, ddp=False)
    expected = loss_energy + loss_forces
    assert torch.allclose(loss, expected, atol=1e-6)


def test_WeightedEnergyForcesChargesHirshfeldLoss(mock_batch_for_loss):
    """Verify four-component charges+hirshfeld loss."""
    batch = mock_batch_for_loss
    pred = {
        "energy": torch.tensor([8.0, 16.0]),
        "forces": torch.zeros_like(batch.forces),
        "partial_charges": batch.charges * 0.5,
        "hirshfeld_ratios": batch.hirshfeld_ratios * 0.5,
    }

    loss_energy = weighted_mean_squared_error_energy(batch, pred, ddp=False)
    loss_forces = mean_squared_error_forces(batch, pred, ddp=False)
    loss_charges = weighted_mean_squared_error_charges(batch, pred, ddp=False)
    loss_hirshfeld = weighted_mean_squared_error_hirshfeld(
        batch, pred, ddp=False
    )

    loss_fn = WeightedEnergyForcesChargesHirshfeldLoss(
        energy_weight=1.0,
        forces_weight=1.0,
        charges_weight=1.0,
        hirshfeld_weight=1.0,
    )
    loss = loss_fn(batch, pred, ddp=False)
    expected = loss_energy + loss_forces + loss_charges + loss_hirshfeld
    assert torch.allclose(loss, expected, atol=1e-6)


# ── Masked per-atom ratio losses (c6 / a0) and generic loss ────


def _ratio_pred(batch):
    return {
        "energy": torch.tensor([8.0, 16.0]),
        "forces": torch.zeros_like(batch.forces),
        "dipole": batch.dipole * 0.5,
        "partial_charges": batch.charges * 0.5,
        "hirshfeld_ratios": batch.hirshfeld_ratios * 0.5,
        "c6_ratios": batch.c6_ratios * 0.5,
    }


def _manual_ratio_loss(batch, pred, key, weight_key, mask=None):
    total, count = 0.0, 0
    for i, g in enumerate(batch.batch.tolist()):
        w = batch.weight[g].item() * batch[weight_key][g].item()
        ref = batch[key][i].item()
        if mask is not None and not mask[i]:
            continue
        total += w * (ref - pred[key][i].item()) ** 2
        count += 1
    return total / count


def test_weighted_mse_c6_ratios(mock_batch_for_loss):
    from so3krates_torch.modules.loss import (
        weighted_mean_squared_error_c6_ratios,
    )

    batch = mock_batch_for_loss
    pred = _ratio_pred(batch)
    loss = weighted_mean_squared_error_c6_ratios(batch, pred, ddp=False)
    # graph 0: 3 atoms, w=1*1, (2-1)^2=1 ; graph 1: 4 atoms, w=2*2, 0.25
    expected = (3 * 1.0 + 4 * 4.0 * 0.25) / 7
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_masked_ratio_loss_nan_times_zero_weight_is_finite(
    mock_batch_for_loss,
):
    from so3krates_torch.modules.loss import (
        weighted_mean_squared_error_c6_ratios,
    )

    batch = mock_batch_for_loss
    pred = _ratio_pred(batch)
    batch.c6_ratios[3:] = float("nan")
    batch.c6_ratios_weight = torch.tensor([1.0, 0.0])
    loss = weighted_mean_squared_error_c6_ratios(batch, pred, ddp=False)
    assert torch.isfinite(loss)
    # only graph 0's three atoms count
    assert torch.allclose(loss, torch.tensor(1.0), atol=1e-6)


def test_masked_ratio_loss_all_missing_is_zero(mock_batch_for_loss):
    from so3krates_torch.modules.loss import (
        weighted_mean_squared_error_c6_ratios,
    )

    batch = mock_batch_for_loss
    batch.c6_ratios[:] = float("nan")
    pred_c6 = torch.ones(7, requires_grad=True)
    loss = weighted_mean_squared_error_c6_ratios(
        batch, {"c6_ratios": pred_c6}, ddp=False
    )
    assert loss.item() == 0.0
    loss.backward()
    assert torch.isfinite(pred_c6.grad).all()
    assert (pred_c6.grad == 0).all()


def test_masked_ratio_loss_partial_nan_atoms(mock_batch_for_loss):
    from so3krates_torch.modules.loss import (
        weighted_mean_squared_error_hirshfeld,
    )

    batch = mock_batch_for_loss
    pred = _ratio_pred(batch)
    batch.hirshfeld_ratios[[1, 5]] = float("nan")
    pred["hirshfeld_ratios"] = torch.full((7,), 0.5)
    loss = weighted_mean_squared_error_hirshfeld(batch, pred, ddp=False)
    mask = [True, False, True, True, True, False, True]
    expected = _manual_ratio_loss(
        batch, pred, "hirshfeld_ratios", "hirshfeld_ratios_weight", mask
    )
    assert torch.allclose(loss, torch.tensor(expected), atol=1e-6)


def test_masked_ratio_loss_zero_weight_structure_excluded(
    mock_batch_for_loss,
):
    """Old preprocessed files zero-fill missing labels with weight 0;
    those atoms must not dilute the mean."""
    from so3krates_torch.modules.loss import (
        weighted_mean_squared_error_hirshfeld,
    )

    batch = mock_batch_for_loss
    pred = {"hirshfeld_ratios": batch.hirshfeld_ratios * 0.5}
    batch.hirshfeld_ratios_weight = torch.tensor([1.0, 0.0])
    batch.hirshfeld_ratios[3:] = 0.0
    loss = weighted_mean_squared_error_hirshfeld(batch, pred, ddp=False)
    assert torch.allclose(loss, torch.tensor(0.25), atol=1e-6)


_LEGACY_CASES = [
    (WeightedEnergyForcesLoss, {"energy": 2.0, "forces": 3.0}),
    (
        WeightedEnergyForcesDipoleLoss,
        {"energy": 2.0, "forces": 3.0, "dipole": 0.5},
    ),
    (
        WeightedEnergyForcesHirshfeldLoss,
        {"energy": 2.0, "forces": 3.0, "hirshfeld": 0.5},
    ),
    (
        WeightedEnergyForcesDipoleHirshfeldLoss,
        {"energy": 2.0, "forces": 3.0, "dipole": 0.5, "hirshfeld": 0.7},
    ),
    (
        WeightedEnergyForcesChargesLoss,
        {"energy": 2.0, "forces": 3.0, "charges": 0.5},
    ),
    (
        WeightedEnergyForcesChargesHirshfeldLoss,
        {"energy": 2.0, "forces": 3.0, "charges": 0.5, "hirshfeld": 0.7},
    ),
]
_LEGACY_TO_PROPERTY = {
    "energy": "energy",
    "forces": "forces",
    "dipole": "dipole",
    "hirshfeld": "hirshfeld_ratios",
    "charges": "charges",
}


@pytest.mark.parametrize("legacy_cls,kw", _LEGACY_CASES)
def test_properties_loss_matches_legacy_classes(
    mock_batch_for_loss, legacy_cls, kw
):
    from so3krates_torch.modules.loss import WeightedPropertiesLoss

    batch = mock_batch_for_loss
    pred = _ratio_pred(batch)
    legacy = legacy_cls(**{f"{k}_weight": v for k, v in kw.items()})
    generic = WeightedPropertiesLoss(
        {_LEGACY_TO_PROPERTY[k]: v for k, v in kw.items()}
    )
    assert torch.allclose(
        generic(batch, pred, ddp=False),
        legacy(batch, pred, ddp=False),
        atol=1e-10,
    )


def test_properties_loss_c6_term_adds_linearly(mock_batch_for_loss):
    from so3krates_torch.modules.loss import (
        WeightedPropertiesLoss,
        weighted_mean_squared_error_c6_ratios,
    )

    batch = mock_batch_for_loss
    pred = _ratio_pred(batch)
    base = {"energy": 2.0, "forces": 3.0, "hirshfeld_ratios": 0.1}
    without = WeightedPropertiesLoss(base)(batch, pred, ddp=False)
    with_c6 = WeightedPropertiesLoss({**base, "c6_ratios": 0.1})(
        batch, pred, ddp=False
    )
    c6 = weighted_mean_squared_error_c6_ratios(batch, pred, ddp=False)
    assert torch.allclose(with_c6, without + 0.1 * c6, atol=1e-10)


def test_properties_loss_skips_zero_weight_terms(mock_batch_for_loss):
    from so3krates_torch.modules.loss import WeightedPropertiesLoss

    batch = mock_batch_for_loss
    pred = _ratio_pred(batch)
    del pred["c6_ratios"]  # not needed when its weight is 0
    loss = WeightedPropertiesLoss(
        {"energy": 1.0, "forces": 1.0, "c6_ratios": 0.0}
    )(batch, pred, ddp=False)
    assert torch.isfinite(loss)


def test_properties_loss_missing_pred_key_raises(mock_batch_for_loss):
    from so3krates_torch.modules.loss import WeightedPropertiesLoss

    batch = mock_batch_for_loss
    pred = _ratio_pred(batch)
    del pred["c6_ratios"]
    loss_fn = WeightedPropertiesLoss({"c6_ratios": 0.1})
    with pytest.raises(KeyError, match="c6_ratios_bool"):
        loss_fn(batch, pred, ddp=False)


def test_properties_loss_unknown_property_raises():
    from so3krates_torch.modules.loss import WeightedPropertiesLoss

    with pytest.raises(ValueError, match="stress"):
        WeightedPropertiesLoss({"stress": 1.0})


def test_setup_loss_auto_returns_properties_loss():
    from so3krates_torch.modules.loss import WeightedPropertiesLoss
    from so3krates_torch.tools.training_setup import setup_loss_function

    loss = setup_loss_function(
        {
            "TRAINING": {
                "energy_weight": 0.01,
                "forces_weight": 1.0,
                "dipole_weight": 0.1,
                "hirshfeld_weight": 0.1,
                "c6_ratios_weight": 0.1,
            }
        }
    )
    assert isinstance(loss, WeightedPropertiesLoss)
    assert abs(loss.c6_ratios_weight.item() - 0.1) < 1e-6
    explicit = setup_loss_function(
        {"TRAINING": {"loss_type": "properties", "c6_ratios_weight": 0.1}}
    )
    assert isinstance(explicit, WeightedPropertiesLoss)


@pytest.mark.parametrize(
    "loss_type",
    [
        "energy_forces",
        "energy_forces_hirshfeld",
        "energy_forces_dipole_hirshfeld",
        "energy_forces_charges_hirshfeld",
    ],
)
def test_legacy_loss_type_rejects_c6_weight(loss_type):
    from so3krates_torch.tools.training_setup import setup_loss_function

    with pytest.raises(ValueError, match="c6_ratios_weight"):
        setup_loss_function(
            {"TRAINING": {"loss_type": loss_type, "c6_ratios_weight": 0.1}}
        )


# ── loss_reduction: per_graph (JAX so3lr parity) ───────────────


_ALL_W = {
    "energy": 0.5,
    "forces": 2.0,
    "dipole": 0.3,
    "hirshfeld_ratios": 0.1,
    "c6_ratios": 0.2,
    "charges": 0.4,
}


def test_flat_reduction_unchanged(mock_batch_for_loss):
    from so3krates_torch.modules.loss import WeightedPropertiesLoss

    batch = mock_batch_for_loss
    pred = _ratio_pred(batch)
    default = WeightedPropertiesLoss(_ALL_W)(batch, pred, ddp=False)
    flat = WeightedPropertiesLoss(_ALL_W, reduction="flat")(
        batch, pred, ddp=False
    )
    assert torch.equal(default, flat)


def test_per_graph_forces_equal_weight_per_graph(mock_batch_for_loss):
    from so3krates_torch.modules.loss import mean_squared_error_forces

    batch = mock_batch_for_loss
    pred = {"forces": torch.zeros_like(batch.forces)}
    loss = mean_squared_error_forces(
        batch, pred, ddp=False, reduction="per_graph"
    )
    # per-atom mean of f^2 over xyz: graph 0 atoms 1/3 each -> 1/3;
    # graph 1: three atoms 4/3, one atom 1 -> (4/3*3 + 1)/4 = 5/4
    w = (batch.weight * batch.forces_weight).tolist()  # [10, 40]
    expected = (w[0] * (1 / 3) + w[1] * (5 / 4)) / 2
    assert loss.item() == pytest.approx(expected, rel=1e-6)


def test_per_graph_partial_nan_graph_excluded(mock_batch_for_loss):
    from so3krates_torch.modules.loss import (
        weighted_mean_squared_error_c6_ratios,
    )

    batch = mock_batch_for_loss
    pred = _ratio_pred(batch)
    graph0_only = (1.0 * 1.0 * 1.0) / 1  # graph 0: w=1, (2-1)^2 = 1
    batch.c6_ratios[4] = float("nan")
    loss = weighted_mean_squared_error_c6_ratios(
        batch, pred, ddp=False, reduction="per_graph"
    )
    assert loss.item() == pytest.approx(graph0_only)


def test_per_graph_energy_and_dipole(mock_batch_for_loss):
    from so3krates_torch.modules.loss import (
        weighted_mean_squared_error_dipole,
        weighted_mean_squared_error_energy,
    )

    batch = mock_batch_for_loss
    pred = _ratio_pred(batch)  # energy [8, 16] vs ref [10, 20]
    e = weighted_mean_squared_error_energy(
        batch, pred, ddp=False, reduction="per_graph"
    )
    w = (batch.weight * batch.energy_weight).tolist()  # [1, 3]
    expected_e = (w[0] * (2 / 3) ** 2 + w[1] * (4 / 4) ** 2) / 2
    assert e.item() == pytest.approx(expected_e, rel=1e-6)

    d = weighted_mean_squared_error_dipole(
        batch, pred, ddp=False, reduction="per_graph"
    )
    # ref dipoles unit vectors, pred = 0.5*ref -> residual 0.5/N on one
    # component; weights weight*dipole_weight = [1, 4] per component
    expected_d = (1 * (0.5 / 3) ** 2 + 4 * (0.5 / 4) ** 2) / 6
    assert d.item() == pytest.approx(expected_d, rel=1e-6)


def test_unknown_reduction_raises():
    from so3krates_torch.modules.loss import WeightedPropertiesLoss

    with pytest.raises(ValueError, match="reduction"):
        WeightedPropertiesLoss({"forces": 1.0}, reduction="mean")


def _jax_args(batch):
    import jax.numpy as jnp
    import numpy as np

    return dict(
        batch_segments=jnp.asarray(batch.batch.numpy()),
        graph_mask=jnp.ones(batch.ptr.numel() - 1, dtype=bool),
        scale=jnp.ones(1),
    )


def test_per_graph_matches_jax(mock_batch_for_loss, jax_so3lr):
    """forces, c6 (with a partially NaN graph) and energy vs the JAX
    reference terms, all structure weights 1."""
    import jax.numpy as jnp
    from so3krates_torch.modules.loss import (
        mean_squared_error_forces,
        weighted_mean_squared_error_c6_ratios,
        weighted_mean_squared_error_energy,
    )

    tu = jax_so3lr.training_utils
    batch = mock_batch_for_loss
    for key in (
        "weight",
        "energy_weight",
        "forces_weight",
        "c6_ratios_weight",
    ):
        setattr(batch, key, torch.ones(2))
    torch.manual_seed(0)
    pred = {
        "forces": torch.randn(7, 3),
        "c6_ratios": torch.rand(7),
        "energy": torch.tensor([9.0, 23.0]),
    }
    batch.c6_ratios[5] = float("nan")
    args = _jax_args(batch)

    def j(t):
        return jnp.asarray(t.numpy())

    ref_f = tu.node_mse_loss(j(pred["forces"]), j(batch.forces), **args)
    ref_c6 = tu.node_mse_loss(j(pred["c6_ratios"]), j(batch.c6_ratios), **args)
    ref_e = tu.graph_mse_loss_per_atom(
        j(pred["energy"]), j(batch.energy), **args
    )
    kw = dict(ddp=False, reduction="per_graph")
    assert mean_squared_error_forces(batch, pred, **kw).item() == (
        pytest.approx(float(ref_f.squeeze()), rel=1e-5)
    )
    assert weighted_mean_squared_error_c6_ratios(batch, pred, **kw).item() == (
        pytest.approx(float(ref_c6.squeeze()), rel=1e-5)
    )
    assert weighted_mean_squared_error_energy(batch, pred, **kw).item() == (
        pytest.approx(float(ref_e.squeeze()), rel=1e-5)
    )


def test_legacy_loss_type_rejects_per_graph():
    from so3krates_torch.tools.training_setup import setup_loss_function

    with pytest.raises(ValueError, match="loss_reduction"):
        setup_loss_function(
            {
                "TRAINING": {
                    "loss_type": "energy_forces",
                    "loss_reduction": "per_graph",
                }
            }
        )


def test_setup_loss_passes_reduction():
    from so3krates_torch.tools.training_setup import setup_loss_function

    loss = setup_loss_function({"TRAINING": {"loss_reduction": "per_graph"}})
    assert loss.reduction == "per_graph"


# ── Robust loss (fixed / adaptive) ─────────────────────────────


@pytest.mark.parametrize("reduction", ["flat", "per_graph"])
def test_fixed_robust_alpha2_equals_l2(mock_batch_for_loss, reduction):
    from so3krates_torch.modules.loss import WeightedPropertiesLoss

    batch = mock_batch_for_loss
    pred = _ratio_pred(batch)
    l2 = WeightedPropertiesLoss(_ALL_W, reduction=reduction)
    robust = WeightedPropertiesLoss(
        _ALL_W, reduction=reduction, robust="fixed", robust_alpha=2.0
    )
    # 2 * rho(x, 2, 1) = 2 * 0.5 x^2 = x^2
    assert torch.allclose(
        robust(batch, pred, ddp=False), l2(batch, pred, ddp=False)
    )


def test_adaptive_init_values():
    from so3krates_torch.modules.loss import AdaptiveRobustParams

    params = AdaptiveRobustParams(["forces", "energy"], 1.0, 1.0)
    alpha, scale = params.alpha_scale("forces")
    assert alpha.item() == pytest.approx(1.0, abs=1e-6)
    assert scale.item() == pytest.approx(1.0, abs=1e-5)
    alpha, scale = AdaptiveRobustParams(["forces"], 2.5, 0.01).alpha_scale(
        "forces"
    )
    assert alpha.item() == pytest.approx(1.99, abs=1e-5)  # clipped
    assert scale.item() == pytest.approx(0.1, abs=1e-5)  # floored


def test_adaptive_params_receive_grads(mock_batch_for_loss):
    from so3krates_torch.modules.loss import WeightedPropertiesLoss

    batch = mock_batch_for_loss
    loss_fn = WeightedPropertiesLoss(
        {"energy": 1.0, "forces": 1.0}, robust="adaptive"
    )
    loss_fn(batch, _ratio_pred(batch), ddp=False).backward()
    grads = [p.grad for p in loss_fn.parameters()]
    assert len(grads) == 4
    assert all(g is not None and torch.isfinite(g).all() for g in grads)
    assert any(g.abs().sum() > 0 for g in grads)


def test_adaptive_only_weighted_properties_have_params():
    from so3krates_torch.modules.loss import WeightedPropertiesLoss

    loss_fn = WeightedPropertiesLoss(
        {"energy": 1.0, "forces": 1.0, "dipole": 0.0}, robust="adaptive"
    )
    assert set(loss_fn.adaptive.alpha_raw.keys()) == {"energy", "forces"}
    assert WeightedPropertiesLoss({"forces": 1.0}).adaptive is None


def test_robust_and_explicit_penalty_conflict():
    from so3krates_torch.modules.loss import (
        WeightedPropertiesLoss,
        squared_penalty,
    )

    with pytest.raises(ValueError, match="penalty"):
        WeightedPropertiesLoss(
            {"forces": 1.0}, penalty=squared_penalty, robust="fixed"
        )


@pytest.mark.parametrize("mode", ["fixed", "adaptive"])
def test_robust_matches_jax(mock_batch_for_loss, jax_so3lr, mode):
    import jax.numpy as jnp
    from so3krates_torch.modules.loss import (
        AdaptiveRobustParams,
        fixed_robust_penalty,
        mean_squared_error_forces,
        weighted_mean_squared_error_energy,
    )

    tu = jax_so3lr.training_utils
    batch = mock_batch_for_loss
    for key in ("weight", "energy_weight", "forces_weight"):
        setattr(batch, key, torch.ones(2))
    torch.manual_seed(1)
    pred = {"forces": torch.randn(7, 3), "energy": torch.tensor([9.0, 23.0])}
    args = _jax_args(batch)

    def j(t):
        return jnp.asarray(t.detach().numpy())

    if mode == "fixed":
        penalty = fixed_robust_penalty(1.0)
        jkw = dict(use_robust_loss=True, robust_loss_alpha=1.0)
        jkw_f, jkw_e = jkw, jkw
    else:
        params = AdaptiveRobustParams(["forces", "energy"], 1.3, 0.4)
        penalty = params.penalty
        af, sf = (t.detach().numpy() for t in params.alpha_scale("forces"))
        ae, se = (t.detach().numpy() for t in params.alpha_scale("energy"))
        jkw_f = dict(adaptive_alpha=jnp.asarray(af), adaptive_scale=sf)
        jkw_e = dict(adaptive_alpha=jnp.asarray(ae), adaptive_scale=se)

    ref_f = tu.node_mse_loss(
        j(pred["forces"]), j(batch.forces), **args, **jkw_f
    )
    ref_e = tu.graph_mse_loss_per_atom(
        j(pred["energy"]), j(batch.energy), **args, **jkw_e
    )
    kw = dict(ddp=False, reduction="per_graph", penalty=penalty)
    f = mean_squared_error_forces(batch, pred, **kw)
    e = weighted_mean_squared_error_energy(batch, pred, **kw)
    assert f.item() == pytest.approx(float(ref_f.squeeze()), rel=1e-5)
    assert e.item() == pytest.approx(float(ref_e.squeeze()), rel=1e-5)


def test_legacy_loss_type_rejects_robust():
    from so3krates_torch.tools.training_setup import setup_loss_function

    with pytest.raises(ValueError, match="robust"):
        setup_loss_function(
            {
                "TRAINING": {
                    "loss_type": "energy_forces",
                    "use_adaptive_robust_loss": True,
                }
            }
        )


def test_setup_loss_builds_adaptive():
    from so3krates_torch.tools.training_setup import setup_loss_function

    loss = setup_loss_function(
        {
            "TRAINING": {
                "use_adaptive_robust_loss": True,
                "adaptive_robust_loss_init_alpha": 1.5,
                "adaptive_robust_loss_init_scale": 0.5,
            }
        }
    )
    alpha, scale = loss.adaptive.alpha_scale("forces")
    assert alpha.item() == pytest.approx(1.5, abs=1e-5)
    assert scale.item() == pytest.approx(0.5, abs=1e-5)
