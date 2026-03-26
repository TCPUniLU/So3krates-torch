"""Tests for physics-based potential energy components.

Tests verify correctness of ZBL repulsion, electrostatic interactions,
and dispersion energies using hand-computed reference values and
physical constraints.
"""

import pytest
import torch
import math
from so3krates_torch.blocks.physical_potentials import (
    ZBLRepulsion,
    CoulombErf,
    CoulombErfShiftedForceSmooth,
    ElectrostaticInteraction,
    DispersionInteraction,
    softplus_inverse,
    mixing_rules,
)


# ============================================================
# ZBL Repulsion Tests (4 tests)
# ============================================================


def test_zbl_repulsion_positive(device):
    """Verify ZBL repulsion energy is always positive.

    ZBL is a short-range repulsive potential by definition,
    so energy must be positive at all distances.
    """
    zbl = ZBLRepulsion().to(device)

    # Test at multiple distances
    distances = torch.tensor([0.5, 1.0, 2.0], device=device).unsqueeze(1)
    num_dists = len(distances)

    # Two hydrogen atoms (Z=1)
    atomic_numbers = torch.tensor([1, 1], device=device)
    senders = torch.zeros(num_dists, dtype=torch.long, device=device)
    receivers = torch.ones(num_dists, dtype=torch.long, device=device)
    cutoffs = torch.ones(num_dists, 1, device=device)

    energy = zbl(
        atomic_numbers,
        cutoffs,
        senders,
        receivers,
        distances,
        num_nodes=2,
    )

    # All repulsion energies should be positive
    assert torch.all(energy >= 0), "ZBL repulsion must be positive"


def test_zbl_repulsion_decays_with_distance(device):
    """Verify repulsion energy decreases as distance increases.

    Repulsive potential should decay: E(r1) > E(r2) for r1 < r2.
    """
    zbl = ZBLRepulsion().to(device)

    distances = [0.5, 1.0, 2.0]
    energies = []

    for r in distances:
        atomic_numbers = torch.tensor([1, 1], device=device)
        senders = torch.tensor([0], dtype=torch.long, device=device)
        receivers = torch.tensor([1], dtype=torch.long, device=device)
        cutoffs = torch.ones(1, 1, device=device)
        lengths = torch.tensor([[r]], device=device)

        energy = zbl(
            atomic_numbers,
            cutoffs,
            senders,
            receivers,
            lengths,
            num_nodes=2,
        )
        energies.append(energy[1].item())  # Energy on receiving atom

    # Energy should decay with increasing distance
    assert (
        energies[0] > energies[1] > energies[2]
    ), f"ZBL must decay: {energies}"


def test_zbl_repulsion_switching_to_zero(device):
    """Verify switching function drives energy to zero at large r.

    The switching function activates at x_off=1.5 in ZBL,
    smoothly reducing energy to zero at large distances.
    """
    zbl = ZBLRepulsion().to(device)

    # Test at distances beyond switching region
    large_distances = torch.tensor([5.0, 10.0], device=device).unsqueeze(1)
    num_dists = len(large_distances)

    atomic_numbers = torch.tensor([1, 1], device=device)
    senders = torch.zeros(num_dists, dtype=torch.long, device=device)
    receivers = torch.ones(num_dists, dtype=torch.long, device=device)
    cutoffs = torch.ones(num_dists, 1, device=device)

    energy = zbl(
        atomic_numbers,
        cutoffs,
        senders,
        receivers,
        large_distances,
        num_nodes=2,
    )

    # Energy should be very close to zero at large distances
    assert torch.all(energy < 0.01), f"ZBL should be ~0 at large r: {energy}"


def test_zbl_repulsion_learnable_params(device):
    """Verify all ZBL parameters are trainable.

    All *_raw parameters should have requires_grad=True for training.
    """
    zbl = ZBLRepulsion().to(device)

    trainable_params = [
        "a1_raw",
        "a2_raw",
        "a3_raw",
        "a4_raw",
        "c1_raw",
        "c2_raw",
        "c3_raw",
        "c4_raw",
        "p_raw",
        "d_raw",
    ]

    for param_name in trainable_params:
        param = getattr(zbl, param_name)
        assert param.requires_grad, f"{param_name} should be trainable"


# ============================================================
# Electrostatics Tests (6 tests)
# ============================================================


def test_coulomb_erf_positive_charges_repulsive(device):
    """Verify like charges produce positive (repulsive) energy.

    Two positive charges should repel: E > 0.
    """
    ke = 14.399645351950548  # eV·Å/e²
    sigma = 1.0
    coulomb = CoulombErf(ke=ke, sigma=sigma).to(device)

    # Two positive charges (q1=1.0, q2=1.0)
    q = torch.tensor([1.0, 1.0], device=device)
    rij = torch.tensor([2.0], device=device)
    senders = torch.tensor([0], dtype=torch.long, device=device)
    receivers = torch.tensor([1], dtype=torch.long, device=device)

    energy = coulomb(q, rij, senders, receivers)

    assert torch.all(energy > 0), "Like charges should repel (E > 0)"


def test_coulomb_erf_opposite_charges_attractive(device):
    """Verify opposite charges produce negative (attractive) energy.

    Positive and negative charges should attract: E < 0.
    """
    ke = 14.399645351950548
    sigma = 1.0
    coulomb = CoulombErf(ke=ke, sigma=sigma).to(device)

    # Opposite charges (q1=1.0, q2=-1.0)
    q = torch.tensor([1.0, -1.0], device=device)
    rij = torch.tensor([2.0], device=device)
    senders = torch.tensor([0], dtype=torch.long, device=device)
    receivers = torch.tensor([1], dtype=torch.long, device=device)

    energy = coulomb(q, rij, senders, receivers)

    assert torch.all(energy < 0), "Opposite charges should attract (E < 0)"


def test_coulomb_erf_1_over_r_decay(device):
    """Verify Coulomb law approaches ke*q1*q2/r at large distances.

    At r >> sigma, erf(r/sigma) → 1, so potential → ke*q1*q2/r.
    """
    ke = 14.399645351950548
    sigma = 0.5  # Small sigma so erf saturates quickly
    coulomb = CoulombErf(ke=ke, sigma=sigma).to(device)

    # Use r >> sigma for erf saturation
    r = 10.0
    q = torch.tensor([1.0, 2.0], device=device)
    rij = torch.tensor([r], device=device)
    senders = torch.tensor([0], dtype=torch.long, device=device)
    receivers = torch.tensor([1], dtype=torch.long, device=device)

    energy = coulomb(q, rij, senders, receivers)

    # Factor 0.5 comes from sparse neighborlist_format_lr="sparse"
    expected = 0.5 * ke * q[0] * q[1] / r
    # erf(10/0.5) ≈ 1, so energy should be very close to Coulomb law
    assert torch.allclose(
        energy, expected, rtol=1e-2
    ), f"Expected {expected}, got {energy}"


def test_coulomb_shifted_force_zero_at_cutoff(device):
    """Verify energy is exactly zero at and beyond cutoff.

    Shifted-force cutoff ensures E = 0 for r >= cutoff.
    """
    ke = 14.399645351950548
    sigma = 1.0
    cutoff = 5.0
    cuton = 2.25
    coulomb = CoulombErfShiftedForceSmooth(
        ke=ke, sigma=sigma, cutoff=cutoff, cuton=cuton
    ).to(device)

    # Test at cutoff and beyond
    q = torch.tensor([1.0, 1.0], device=device)
    rij_values = torch.tensor([5.0, 6.0, 10.0], device=device)

    for r in rij_values:
        rij = torch.tensor([r.item()], device=device)
        senders = torch.tensor([0], dtype=torch.long, device=device)
        receivers = torch.tensor([1], dtype=torch.long, device=device)

        energy = coulomb(q, rij, senders, receivers)

        assert torch.allclose(
            energy, torch.zeros_like(energy), atol=1e-6
        ), f"Energy should be 0 at r={r.item()}"


def test_coulomb_shifted_force_smooth_near_cutoff(device):
    """Verify energy is smooth near cuton/cutoff boundary.

    Sample energies in switching region to check no discontinuities.
    """
    ke = 14.399645351950548
    sigma = 1.0
    cutoff = 5.0
    cuton = 2.25
    coulomb = CoulombErfShiftedForceSmooth(
        ke=ke, sigma=sigma, cutoff=cutoff, cuton=cuton
    ).to(device)

    q = torch.tensor([1.0, 1.0], device=device)

    # Sample points across switching region
    r_values = torch.linspace(1.0, 5.5, 20, device=device)
    energies = []

    for r in r_values:
        rij = torch.tensor([r.item()], device=device)
        senders = torch.tensor([0], dtype=torch.long, device=device)
        receivers = torch.tensor([1], dtype=torch.long, device=device)
        energy = coulomb(q, rij, senders, receivers)
        energies.append(energy.item())

    energies_t = torch.tensor(energies, device=device)

    # Check monotonic decay (no jumps)
    diffs = energies_t[1:] - energies_t[:-1]
    assert torch.all(
        diffs <= 0
    ), "Energy should decay monotonically (no jumps)"


def test_electrostatic_interaction_end_to_end(device):
    """Integration test for full ElectrostaticInteraction module.

    Verify correct sign, shape, and aggregation from edge to node.
    """
    ke = 14.399645351950548
    elec = ElectrostaticInteraction(ke=ke).to(device)

    # 4 atoms with different charge configurations
    # [+1, +1, -1, +1] to test both attraction and repulsion
    partial_charges = torch.tensor([1.0, 1.0, -1.0, 1.0], device=device)
    # Edge 0: atom 0 (+1) → atom 1 (+1): repulsive
    # Edge 1: atom 2 (-1) → atom 3 (+1): attractive
    senders_lr = torch.tensor([0, 2], dtype=torch.long, device=device)
    receivers_lr = torch.tensor([1, 3], dtype=torch.long, device=device)
    lengths_lr = torch.tensor([2.0, 2.0], device=device)
    num_nodes = 4

    energy = elec(
        partial_charges,
        senders_lr,
        receivers_lr,
        lengths_lr,
        num_nodes,
        cutoff_lr=None,
        electrostatic_energy_scale=1.0,
    )

    # Check shape
    assert energy.shape == (4, 1), f"Expected (4,1), got {energy.shape}"

    # Check sign consistency
    # Atom 1 receives from atom 0: both +1 → repel (E > 0)
    assert energy[1] > 0, "Like charges should repel (E > 0)"
    # Atom 3 receives from atom 2: (+1)*(-1) → attract (E < 0)
    assert energy[3] < 0, "Opposite charges should attract (E < 0)"


# ============================================================
# Dispersion Tests (4 tests)
# ============================================================


def test_dispersion_energy_attractive(device):
    """Verify dispersion energy is attractive (negative).

    Dispersion interactions are always attractive by definition.
    """
    disp = DispersionInteraction().to(device)

    # 2 atoms with typical Hirshfeld ratios
    hirshfeld_ratios = torch.tensor([1.0, 1.0], device=device)
    atomic_numbers = torch.tensor([6, 6], device=device)  # Carbon
    senders_lr = torch.tensor([0], dtype=torch.long, device=device)
    receivers_lr = torch.tensor([1], dtype=torch.long, device=device)
    lengths_lr = torch.tensor([3.5], device=device)  # Typical distance
    num_nodes = 2

    energy = disp(
        hirshfeld_ratios,
        atomic_numbers,
        senders_lr,
        receivers_lr,
        lengths_lr,
        num_nodes,
        cutoff_lr=None,
        cutoff_lr_damping=None,
        dispersion_energy_scale=1.0,
    )

    # Dispersion is attractive: E < 0 for receiving atom
    # (sender atom has no interaction, so energy[0] = 0)
    assert energy[1] < 0, "Dispersion energy should be negative"


def test_dispersion_cutoff_smoothing(device):
    """Verify dispersion smoothly approaches zero near cutoff.

    Switching function should smoothly damp energy to zero.
    """
    disp = DispersionInteraction().to(device)

    cutoff_lr = 8.0
    cutoff_lr_damping = 2.0

    hirshfeld_ratios = torch.tensor([1.0, 1.0], device=device)
    atomic_numbers = torch.tensor([1, 1], device=device)  # Hydrogen

    # Sample distances approaching cutoff
    r_values = torch.linspace(4.0, cutoff_lr + 0.5, 15, device=device)
    energies = []

    for r in r_values:
        senders_lr = torch.tensor([0], dtype=torch.long, device=device)
        receivers_lr = torch.tensor([1], dtype=torch.long, device=device)
        lengths_lr = torch.tensor([r.item()], device=device)

        energy = disp(
            hirshfeld_ratios,
            atomic_numbers,
            senders_lr,
            receivers_lr,
            lengths_lr,
            num_nodes=2,
            cutoff_lr=cutoff_lr,
            cutoff_lr_damping=cutoff_lr_damping,
            dispersion_energy_scale=1.0,
        )
        energies.append(energy[1].item())  # Receiving atom

    # Energy should approach zero at cutoff
    assert (
        abs(energies[-1]) < 1e-3
    ), f"Energy should be ~0 at cutoff: {energies[-1]}"

    # Energy should be smooth (no jumps)
    energies_t = torch.tensor(energies, device=device)
    diffs = torch.abs(energies_t[1:] - energies_t[:-1])
    max_jump = torch.max(diffs).item()
    assert max_jump < 0.5, f"Discontinuity detected: max jump = {max_jump}"


def test_dispersion_missing_cutoff_damping_raises(device):
    """Verify error when cutoff_lr is set without cutoff_lr_damping.

    Configuration validation should catch missing damping parameter.
    """
    disp = DispersionInteraction().to(device)

    hirshfeld_ratios = torch.tensor([1.0, 1.0], device=device)
    atomic_numbers = torch.tensor([1, 1], device=device)
    senders_lr = torch.tensor([0], dtype=torch.long, device=device)
    receivers_lr = torch.tensor([1], dtype=torch.long, device=device)
    lengths_lr = torch.tensor([3.0], device=device)

    with pytest.raises(
        ValueError, match="cutoff_lr is set but cutoff_lr_damping is not"
    ):
        disp(
            hirshfeld_ratios,
            atomic_numbers,
            senders_lr,
            receivers_lr,
            lengths_lr,
            num_nodes=2,
            cutoff_lr=8.0,
            cutoff_lr_damping=None,  # Missing damping
            dispersion_energy_scale=1.0,
        )


def test_mixing_rules_symmetry(device):
    """Verify mixing rules are symmetric under atom exchange.

    alpha_ij and C6_ij should be identical for (i,j) and (j,i).
    """
    atomic_numbers = torch.tensor([1, 6], device=device)  # H and C
    hirshfeld_ratios = torch.tensor([1.0, 1.2], device=device)

    # Compute (i=0, j=1)
    idx_i = torch.tensor([0], dtype=torch.long, device=device)
    idx_j = torch.tensor([1], dtype=torch.long, device=device)
    alpha_ij, C6_ij = mixing_rules(
        atomic_numbers, idx_i, idx_j, hirshfeld_ratios
    )

    # Compute (i=1, j=0) - swapped
    idx_i_swap = torch.tensor([1], dtype=torch.long, device=device)
    idx_j_swap = torch.tensor([0], dtype=torch.long, device=device)
    alpha_ji, C6_ji = mixing_rules(
        atomic_numbers, idx_i_swap, idx_j_swap, hirshfeld_ratios
    )

    assert torch.allclose(
        alpha_ij, alpha_ji, atol=1e-6
    ), "alpha_ij should be symmetric"
    assert torch.allclose(C6_ij, C6_ji, atol=1e-6), "C6_ij should be symmetric"


# ============================================================
# Helper Function Tests (1 test)
# ============================================================


def test_softplus_inverse_roundtrip(device):
    """Verify softplus_inverse is the true inverse of softplus.

    For any x > 0: softplus(softplus_inverse(x)) ≈ x.
    """
    import torch.nn.functional as F

    # Test multiple positive values
    x_values = torch.tensor([0.1, 1.0, 3.0, 10.0, 100.0], device=device)

    for x in x_values:
        x_inv = softplus_inverse(x)
        x_reconstructed = F.softplus(x_inv)

        assert torch.allclose(
            x_reconstructed, x, rtol=1e-5
        ), f"Roundtrip failed for x={x.item()}"
