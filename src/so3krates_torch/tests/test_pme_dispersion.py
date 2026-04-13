"""Tests for PMEDispersionInteraction.

Tests verify:
  1. Sign and magnitude of dispersion energy (attractive, rough SR sanity).
  2. Gradient consistency via torch.autograd.gradcheck.
  3. Correct batch isolation for multi-system batches.

Note on smearing: InversePowerLawPotential(exponent=6) requires smearing
on the order of the nearest-neighbour distance (here ~3.7 Å × 0.5 ≈ 1.85 Å)
for the PME LR/SR split to work correctly and produce a negative (attractive)
dispersion energy.  smearing=2.0 Å is used throughout these tests.
"""

import numpy as np
import torch
from so3krates_torch.blocks.physical_potentials import (
    BOHR,
    C6_COEF,
    HARTREE,
    PMEDispersionInteraction,
    _make_fixed_inverse_power_law_potential,
)


def build_nl(positions_np, cell_np, cutoff):
    """Build a periodic neighbor list using matscipy.

    Returns:
        edge_index: (2, E) numpy array of sender/receiver indices
        lengths: (E,) numpy array of pair distances
    """
    from matscipy.neighbours import neighbour_list as matscipy_nl

    i, j, d = matscipy_nl(
        "ijd",
        atoms=None,
        cutoff=cutoff,
        positions=positions_np,
        cell=cell_np,
        pbc=[True, True, True],
    )
    edge_index = np.stack([i, j], axis=0)  # (2, E)
    return edge_index, d


# ============================================================
# Test 1: Sign and magnitude sanity check on Ar FCC 3x3x3
# ============================================================

# Smearing and mesh parameters used throughout the tests.
# For InversePowerLawPotential(exponent=6), the smearing must be
# comparable to the nearest-neighbour distance so that the SR/LR split
# is well-conditioned; smearing=0.5 Å is too small and produces a
# positive (repulsive) energy artefact.
_SMEARING = 2.0
_MESH_SPACING = 0.5


def test_pme_dispersion_sign_and_magnitude():
    """PME dispersion energy of Ar FCC supercell is negative and
    within a factor of 1.5 of the SR pairwise direct sum.

    Uses Ar FCC 3x3x3 supercell (27 atoms) with unit Hirshfeld ratios.
    Dispersion is attractive so E_total < 0.
    The direct SR sum (geometric mean rule, same factorisation as PME)
    provides a rough lower-bound sanity check; PME includes long-range
    mesh contributions so the two won't match exactly.
    """
    from ase.build import bulk

    dtype = torch.float64

    ar = bulk("Ar", "fcc", a=5.26)
    atoms = ar.repeat((3, 3, 3))  # 27 atoms
    positions_np = atoms.positions
    cell_np = atoms.cell.array
    n = len(atoms)  # 27

    cutoff = 5.0  # Å — captures nearest neighbours at ~3.72 Å

    edge_index_np, lengths_np = build_nl(positions_np, cell_np, cutoff)

    pme = PMEDispersionInteraction(
        smearing=_SMEARING,
        mesh_spacing=_MESH_SPACING,
    )

    positions = torch.tensor(positions_np, dtype=dtype)
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    hirshfeld_ratios = torch.ones(n, dtype=dtype)
    cell = torch.tensor(cell_np, dtype=dtype).unsqueeze(0)  # (1, 3, 3)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    lengths = torch.tensor(lengths_np, dtype=dtype)
    batch_segments = torch.zeros(n, dtype=torch.long)

    atomic_e = pme(
        hirshfeld_ratios=hirshfeld_ratios,
        atomic_numbers=atomic_numbers,
        positions=positions,
        cell=cell,
        edge_index=edge_index,
        lengths=lengths,
        batch_segments=batch_segments,
        num_graphs=1,
        num_nodes=n,
    )

    E_total = atomic_e.sum().item()
    assert E_total < 0, (
        f"Dispersion energy should be negative (attractive), "
        f"got {E_total:.4f} eV"
    )

    # Rough direct pairwise sum using geometric mean C6 rule (same as PME).
    # C6_i for Ar (Z=18): C6_COEF[17] is Hartree·Bohr^6; convert to eV·Å^6.
    c6_ar = pme.c6_coef[17].item() * HARTREE * (BOHR**6)  # eV·Å^6
    c_ar = c6_ar**0.5  # sqrt(C6) — same "charge" as in PME path

    # Geometric mean: C6_ij = c_i * c_j = c_ar^2 (homogeneous system)
    c6_ij = c_ar * c_ar
    # E_direct = -0.5 * sum_{i,j pairs} C6_ij / r^6
    E_direct = -0.5 * np.sum(c6_ij / lengths_np**6)

    rel_diff = abs(E_total - E_direct) / max(abs(E_total), 1e-10)
    assert rel_diff < 0.5, (
        f"PME dispersion energy {E_total:.4f} eV deviates from SR direct "
        f"sum {E_direct:.4f} eV by relative error {rel_diff:.3f} (> 0.5)"
    )


# ============================================================
# Test 2: Force consistency via torch.autograd.gradcheck
# ============================================================


def test_pme_dispersion_gradcheck():
    """Verify PME dispersion energy gradients w.r.t. positions via gradcheck.

    Uses a small Ar FCC 2x2x2 supercell (8 atoms) with slightly perturbed
    positions (seed 42) to break the perfect FCC symmetry so that numerical
    finite differences are non-trivially non-zero.
    torch.autograd.gradcheck confirms analytic vs. numeric Jacobians agree
    to within atol=1e-4.
    """
    from ase.build import bulk

    dtype = torch.float64

    ar = bulk("Ar", "fcc", a=5.26)
    atoms = ar.repeat((2, 2, 2))  # 8 atoms
    # Perturb slightly to break perfect symmetry so finite differences are
    # non-negligible and gradcheck can compare analytic vs. numeric Jacobian.
    rng = np.random.default_rng(42)
    positions_np = atoms.positions + rng.normal(0, 0.05, atoms.positions.shape)
    cell_np = atoms.cell.array
    n = len(atoms)  # 8

    cutoff = 4.5  # Å

    edge_index_np, _ = build_nl(positions_np, cell_np, cutoff)

    pme = PMEDispersionInteraction(
        smearing=_SMEARING,
        mesh_spacing=_MESH_SPACING,
    )

    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    hirshfeld_ratios = torch.ones(n, dtype=dtype)
    cell = torch.tensor(cell_np, dtype=dtype).unsqueeze(0)  # (1, 3, 3)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    batch_segments = torch.zeros(n, dtype=torch.long)

    positions = torch.tensor(positions_np, dtype=dtype, requires_grad=True)

    def energy_fn(pos):
        # Recompute lengths from current positions for gradcheck.
        # We pass the pre-built neighbor list topology but recompute
        # distances so the computation graph flows through pos.
        src = edge_index[0]
        dst = edge_index[1]
        # Minimum-image distances under the periodic box
        cell_mat = cell[0]
        inv_cell = torch.linalg.inv(cell_mat)
        dr = pos[dst] - pos[src]  # (E, 3)
        # Fractional displacements, wrap to [-0.5, 0.5]
        dr_frac = dr @ inv_cell
        dr_frac = dr_frac - torch.round(dr_frac)
        dr_cart = dr_frac @ cell_mat
        d = torch.linalg.norm(dr_cart, dim=1)  # (E,)

        return pme(
            hirshfeld_ratios=hirshfeld_ratios,
            atomic_numbers=atomic_numbers,
            positions=pos,
            cell=cell,
            edge_index=edge_index,
            lengths=d,
            batch_segments=batch_segments,
            num_graphs=1,
            num_nodes=n,
        ).sum()

    assert torch.autograd.gradcheck(
        energy_fn,
        (positions,),
        eps=1e-4,
        atol=1e-4,
    ), "gradcheck failed: analytic and numeric forces disagree"


# ============================================================
# Test 3: Batch isolation — two Ar systems in one batch
# ============================================================


def test_pme_dispersion_batch_isolation():
    """Per-atom energies from a batched call equal individual calls.

    System A: Ar FCC 2x2x2 supercell (8 atoms, a=5.26 Å).
    System B: Ar FCC 2x2x2 supercell (8 atoms, a=4.5 Å, denser box).

    Stacking both systems into a single batch and calling PME once
    must produce the same per-atom energies as two separate calls.
    """
    from ase.build import bulk

    dtype = torch.float64

    # --- System A ---
    ar_A = bulk("Ar", "fcc", a=5.26).repeat((2, 2, 2))  # 8 atoms
    pos_A_np = ar_A.positions
    cell_A_np = ar_A.cell.array
    n_A = len(ar_A)  # 8
    cutoff_A = 4.5  # Å

    ei_A_np, d_A_np = build_nl(pos_A_np, cell_A_np, cutoff_A)

    # --- System B ---
    ar_B = bulk("Ar", "fcc", a=4.5).repeat((2, 2, 2))  # 8 atoms
    pos_B_np = ar_B.positions
    cell_B_np = ar_B.cell.array
    n_B = len(ar_B)  # 8
    cutoff_B = 4.0  # Å

    ei_B_np, d_B_np = build_nl(pos_B_np, cell_B_np, cutoff_B)

    pme = PMEDispersionInteraction(
        smearing=_SMEARING,
        mesh_spacing=_MESH_SPACING,
    )

    # --- Individual calls ---
    def _call_single(atoms, pos_np, cell_np, ei_np, d_np, n):
        pos = torch.tensor(pos_np, dtype=dtype)
        cell = torch.tensor(cell_np, dtype=dtype).unsqueeze(0)
        atomic_numbers = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.long
        )
        hirshfeld_ratios = torch.ones(n, dtype=dtype)
        edge_index = torch.tensor(ei_np, dtype=torch.long)
        lengths = torch.tensor(d_np, dtype=dtype)
        batch_seg = torch.zeros(n, dtype=torch.long)
        return pme(
            hirshfeld_ratios=hirshfeld_ratios,
            atomic_numbers=atomic_numbers,
            positions=pos,
            cell=cell,
            edge_index=edge_index,
            lengths=lengths,
            batch_segments=batch_seg,
            num_graphs=1,
            num_nodes=n,
        )

    e_A = _call_single(ar_A, pos_A_np, cell_A_np, ei_A_np, d_A_np, n_A)
    e_B = _call_single(ar_B, pos_B_np, cell_B_np, ei_B_np, d_B_np, n_B)

    # --- Batched call ---
    n_total = n_A + n_B

    pos_batch_np = np.concatenate([pos_A_np, pos_B_np], axis=0)
    z_batch = np.concatenate(
        [ar_A.get_atomic_numbers(), ar_B.get_atomic_numbers()]
    )

    # Shift B edge indices by n_A
    ei_B_shifted = ei_B_np + n_A
    ei_batch_np = np.concatenate([ei_A_np, ei_B_shifted], axis=1)
    d_batch_np = np.concatenate([d_A_np, d_B_np])

    cell_batch_np = np.stack([cell_A_np, cell_B_np], axis=0)  # (2, 3, 3)

    batch_seg = torch.cat(
        [
            torch.zeros(n_A, dtype=torch.long),
            torch.ones(n_B, dtype=torch.long),
        ]
    )

    pos_batch = torch.tensor(pos_batch_np, dtype=dtype)
    z_batch_t = torch.tensor(z_batch, dtype=torch.long)
    hr_batch = torch.ones(n_total, dtype=dtype)
    cell_batch = torch.tensor(cell_batch_np, dtype=dtype)  # (2, 3, 3)
    edge_index_batch = torch.tensor(ei_batch_np, dtype=torch.long)
    lengths_batch = torch.tensor(d_batch_np, dtype=dtype)

    e_batch = pme(
        hirshfeld_ratios=hr_batch,
        atomic_numbers=z_batch_t,
        positions=pos_batch,
        cell=cell_batch,
        edge_index=edge_index_batch,
        lengths=lengths_batch,
        batch_segments=batch_seg,
        num_graphs=2,
        num_nodes=n_total,
    )

    # Compare per-atom energies
    assert torch.allclose(e_batch[:n_A], e_A, atol=1e-10), (
        f"Batch isolation failed for system A:\n"
        f"  batched: {e_batch[:n_A].squeeze().tolist()}\n"
        f"  single:  {e_A.squeeze().tolist()}"
    )
    assert torch.allclose(e_batch[n_A:], e_B, atol=1e-10), (
        f"Batch isolation failed for system B:\n"
        f"  batched: {e_batch[n_A:].squeeze().tolist()}\n"
        f"  single:  {e_B.squeeze().tolist()}"
    )


# ============================================================
# Test 4: PME convergence vs exact direct lattice sum
# ============================================================


def test_pme_dispersion_convergence_vs_direct_sum():
    """PME dispersion energy matches exact direct lattice sum to < 1 meV/atom.

    Verifies that the k=0 Fourier-term fix and the factor-of-2 energy
    formula fix together make PME converge to the exact C6 sum across
    a range of smearing values (0.3, 0.4, 0.5 A) appropriate for the
    ~11 A box of the 3x3x3 Ar FCC supercell.

    Without the k=0 fix, the unfixed InversePowerLawPotential gives
    large positive energies (e.g. +72 eV at sigma=0.5 vs E_ref=-2.83 eV)
    due to the missing k=0 Fourier term.  With the fix: < 0.2 meV/atom.

    Notes on mesh_spacing:
      For sigma < 0.5 A, mesh_spacing=sigma/4 is not fine enough —
      aliasing in the B-spline interpolation grows. sigma/8 is used
      here to ensure < 1 meV/atom for all tested smearing values.
    """
    import torchpme
    from ase.build import bulk

    from so3krates_torch.cli.run_tune_pme import _direct_lattice_sum_c6

    dtype = torch.float64

    ar = bulk("Ar", "fcc", a=5.26)
    atoms = ar.repeat((3, 3, 3))  # 27 atoms, box ~11 A
    positions_np = atoms.positions
    cell_np = atoms.cell.array
    n = len(atoms)  # 27
    cutoff = 6.0  # A

    edge_index_np, lengths_np = build_nl(positions_np, cell_np, cutoff)

    positions = torch.tensor(positions_np, dtype=dtype)
    cell = torch.tensor(cell_np, dtype=dtype)  # (3, 3)
    cell_batched = cell.unsqueeze(0)  # (1, 3, 3)
    atomic_numbers = torch.tensor(atoms.get_atomic_numbers(), dtype=torch.long)
    hirshfeld_ratios = torch.ones(n, dtype=dtype)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    lengths = torch.tensor(lengths_np, dtype=dtype)
    batch_segments = torch.zeros(n, dtype=torch.long)

    # Build c_charges for direct sum (same formula as PMEDispersionInteraction)
    c6_ar = C6_COEF[17].item() * HARTREE * (BOHR**6)  # eV*Ang^6
    c_charges = torch.full((n,), c6_ar**0.5, dtype=dtype)

    E_ref = _direct_lattice_sum_c6(
        positions=positions,
        cell=cell,
        c_charges=c_charges,
        n_max=5,
    )

    # sigma in {0.3, 0.4, 0.5} A — realistic for an ~11 A periodic box.
    # mesh_spacing = sigma/8 ensures aliasing errors are < 0.2 meV/atom.
    pme_energies = {}
    for smearing in [0.3, 0.4, 0.5]:
        pme = PMEDispersionInteraction(
            smearing=smearing,
            mesh_spacing=smearing / 8,
        )
        atomic_e = pme(
            hirshfeld_ratios=hirshfeld_ratios,
            atomic_numbers=atomic_numbers,
            positions=positions,
            cell=cell_batched,
            edge_index=edge_index,
            lengths=lengths,
            batch_segments=batch_segments,
            num_graphs=1,
            num_nodes=n,
        )
        E_pme = atomic_e.sum().item()
        pme_energies[smearing] = E_pme
        err_mev = abs(E_pme - E_ref) / n * 1000  # meV/atom
        assert err_mev < 1.0, (
            f"PME dispersion (sigma={smearing}) deviates from direct "
            f"lattice sum by {err_mev:.3f} meV/atom (tolerance: 1.0). "
            f"E_pme={E_pme:.6f} eV, E_ref={E_ref:.6f} eV"
        )

    # sigma-independence: energies at sigma=0.3 and sigma=0.5 must agree
    # within 1% — they would differ by thousands of percent without the fix.
    rel_spread = abs(pme_energies[0.3] - pme_energies[0.5]) / abs(
        pme_energies[0.4]
    )
    assert rel_spread < 0.01, (
        f"PME energy is not sigma-independent: "
        f"E(sigma=0.3)={pme_energies[0.3]:.6f}, "
        f"E(sigma=0.5)={pme_energies[0.5]:.6f}, "
        f"relative spread={rel_spread:.4f} (tolerance: 0.01)"
    )

    # Regression guard: the unfixed potential gives a wrong answer
    # (large positive energy due to missing k=0 term).
    pot_unfixed = torchpme.InversePowerLawPotential(exponent=6, smearing=0.5)
    calc_unfixed = torchpme.PMECalculator(
        potential=pot_unfixed,
        mesh_spacing=0.5 / 8,
        interpolation_nodes=4,
    )
    phi_unfixed = calc_unfixed.forward(
        charges=c_charges.unsqueeze(1),
        cell=cell,
        positions=positions,
        neighbor_indices=edge_index.T,
        neighbor_distances=lengths,
    )
    E_unfixed = float(
        (-1.0 * c_charges.unsqueeze(1) * phi_unfixed).sum().item()
    )
    err_unfixed_mev = abs(E_unfixed - E_ref) / n * 1000
    assert err_unfixed_mev > 100.0, (
        f"Unfixed InversePowerLawPotential unexpectedly gave a correct "
        f"answer ({err_unfixed_mev:.1f} meV/atom). The regression guard "
        f"assumes it is very wrong (> 100 meV/atom due to missing k=0 "
        f"term) — check whether torchpme was updated upstream."
    )
