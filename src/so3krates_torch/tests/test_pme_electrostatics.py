"""Tests for PMEElectrostaticInteraction.

Tests verify correctness of the PME-based electrostatic energy against
analytical references, gradient consistency via autograd, and correct
batch isolation for multi-system batches.
"""

import numpy as np
import pytest
import torch
from so3krates_torch.blocks.physical_potentials import (
    PMEElectrostaticInteraction,
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
# Test 1: Madelung energy of NaCl rock-salt unit cell
# ============================================================


def test_pme_madelung_nacl():
    """PME total energy of NaCl primitive cell matches Madelung reference.

    The analytical Madelung energy per ion pair for rock-salt NaCl is
        E = -ke * M / a
    where M = 1.7476 is the NaCl Madelung constant and a = 5.6402 Å.
    The FCC primitive cell (2 atoms) with cell vectors [0,a/2,a/2],
    [a/2,0,a/2], [a/2,a/2,0] is used — this is the minimal periodic
    cell of the rock-salt structure.
    Tolerance is 0.05 eV to account for finite mesh / smearing errors.
    """
    from ase.build import bulk

    dtype = torch.float64

    ke = 14.399645351950548  # eV·Å/e²
    a = 5.6402  # Å, NaCl conventional cubic lattice constant
    M = 1.7476  # Madelung constant for NaCl rock-salt
    E_ref = -ke * M / a  # eV (negative: binding)

    # Smearing must be small enough (~d_nn/5 ≈ 0.56 Å) that the real-
    # space sum converges within the cutoff.  mesh_spacing = smearing/2
    # is fine for interpolation accuracy.
    smearing = 0.5  # Å
    mesh_spacing = 0.25  # Å
    cutoff = a  # Å (nearest-neighbor cutoff suffices)

    pme = PMEElectrostaticInteraction(
        smearing=smearing,
        mesh_spacing=mesh_spacing,
        ke=ke,
    )

    # Use ASE to build the FCC primitive cell of NaCl rock-salt (2 atoms):
    # cell vectors = [0, a/2, a/2], [a/2, 0, a/2], [a/2, a/2, 0]
    # Na at (0, 0, 0), Cl at (a/2, 0, 0)
    nacl = bulk("NaCl", crystalstructure="rocksalt", a=a)
    positions_np = nacl.positions
    cell_np = nacl.cell.array

    edge_index_np, lengths_np = build_nl(positions_np, cell_np, cutoff)

    positions = torch.tensor(positions_np, dtype=dtype)
    charges = torch.tensor([1.0, -1.0], dtype=dtype)
    cell = torch.tensor(cell_np, dtype=dtype).unsqueeze(0)  # (1, 3, 3)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    lengths = torch.tensor(lengths_np, dtype=dtype)
    batch_segments = torch.zeros(2, dtype=torch.long)

    atomic_e = pme(
        partial_charges=charges,
        positions=positions,
        cell=cell,
        edge_index=edge_index,
        lengths=lengths,
        batch_segments=batch_segments,
        num_graphs=1,
        num_nodes=2,
    )

    E_pme = atomic_e.sum().item()
    assert abs(E_pme - E_ref) < 0.05, (
        f"PME Madelung energy {E_pme:.4f} eV deviates from "
        f"reference {E_ref:.4f} eV by {abs(E_pme - E_ref):.4f} eV"
    )


# ============================================================
# Test 2: Force consistency via torch.autograd.gradcheck
# ============================================================


def test_pme_gradcheck():
    """Verify PME energy gradients w.r.t. positions via gradcheck.

    Uses a small 4-atom periodic box with charge-neutral configuration.
    torch.autograd.gradcheck confirms analytic vs. numeric Jacobians agree
    to within atol=1e-4.
    """
    dtype = torch.float64
    torch.manual_seed(0)

    ke = 14.399645351950548
    box = 5.0  # Å, cubic cell side
    cutoff = 3.5  # Å, SR cutoff

    smearing = cutoff / 5.0
    mesh_spacing = smearing / 2.0

    pme = PMEElectrostaticInteraction(
        smearing=smearing,
        mesh_spacing=mesh_spacing,
        ke=ke,
    )

    # 4 atoms, charge-neutral: sum = 0
    charges_np = np.array([-0.5, 0.5, -0.25, 0.25])
    cell_np = np.diag([box, box, box])

    # Fixed random positions inside the box, seeded for reproducibility
    rng = np.random.default_rng(42)
    positions_np = rng.uniform(0.5, box - 0.5, size=(4, 3))

    edge_index_np, lengths_np = build_nl(positions_np, cell_np, cutoff)

    charges = torch.tensor(charges_np, dtype=dtype)
    cell = torch.tensor(cell_np, dtype=dtype).unsqueeze(0)  # (1, 3, 3)
    edge_index = torch.tensor(edge_index_np, dtype=torch.long)
    batch_segments = torch.zeros(4, dtype=torch.long)

    # positions must require grad for gradcheck
    positions = torch.tensor(
        positions_np, dtype=dtype, requires_grad=True
    )

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
            partial_charges=charges,
            positions=pos,
            cell=cell,
            edge_index=edge_index,
            lengths=d,
            batch_segments=batch_segments,
            num_graphs=1,
            num_nodes=4,
        ).sum()

    assert torch.autograd.gradcheck(
        energy_fn,
        (positions,),
        eps=1e-4,
        atol=1e-4,
    ), "gradcheck failed: analytic and numeric forces disagree"


# ============================================================
# Test 3: Batch isolation — two systems in one batch
# ============================================================


def test_pme_batch_isolation():
    """Per-atom energies from a batched call equal individual calls.

    System A: 2-atom NaCl rock-salt FCC primitive cell (a = 5.6402 Å).
    System B: 3-atom cell (a = 4.0 Å, charges [0.5, -0.5, 0.0]).

    Stacking both systems into a single batch and calling PME once
    must produce the same per-atom energies as two separate calls.
    """
    from ase.build import bulk

    dtype = torch.float64
    ke = 14.399645351950548

    # --- System A: NaCl FCC primitive cell (2 atoms) ---
    a_A = 5.6402
    smearing_A = a_A / 5.0
    mesh_spacing_A = smearing_A / 2.0
    cutoff_A = a_A

    nacl = bulk("NaCl", crystalstructure="rocksalt", a=a_A)
    pos_A_np = nacl.positions
    cell_A_np = nacl.cell.array
    charges_A_np = np.array([1.0, -1.0])

    ei_A_np, d_A_np = build_nl(pos_A_np, cell_A_np, cutoff_A)

    # --- System B: 3-atom cell ---
    a_B = 4.0
    smearing_B = a_B / 5.0
    mesh_spacing_B = smearing_B / 2.0
    cutoff_B = a_B

    rng = np.random.default_rng(7)
    pos_B_np = rng.uniform(0.3, a_B - 0.3, size=(3, 3))
    cell_B_np = np.diag([a_B, a_B, a_B])
    charges_B_np = np.array([0.5, -0.5, 0.0])

    ei_B_np, d_B_np = build_nl(pos_B_np, cell_B_np, cutoff_B)

    # Use a common smearing / mesh_spacing (average) for the shared PME
    # module; individual calls also use this same module for consistency.
    smearing = (smearing_A + smearing_B) / 2.0
    mesh_spacing = (mesh_spacing_A + mesh_spacing_B) / 2.0

    pme = PMEElectrostaticInteraction(
        smearing=smearing,
        mesh_spacing=mesh_spacing,
        ke=ke,
    )

    # --- Individual calls ---
    def _call_single(pos_np, cell_np, charges_np, ei_np, d_np, n):
        pos = torch.tensor(pos_np, dtype=dtype)
        cell = torch.tensor(cell_np, dtype=dtype).unsqueeze(0)
        charges = torch.tensor(charges_np, dtype=dtype)
        edge_index = torch.tensor(ei_np, dtype=torch.long)
        lengths = torch.tensor(d_np, dtype=dtype)
        batch_seg = torch.zeros(n, dtype=torch.long)
        return pme(
            partial_charges=charges,
            positions=pos,
            cell=cell,
            edge_index=edge_index,
            lengths=lengths,
            batch_segments=batch_seg,
            num_graphs=1,
            num_nodes=n,
        )

    e_A = _call_single(
        pos_A_np, cell_A_np, charges_A_np, ei_A_np, d_A_np, n=2
    )
    e_B = _call_single(
        pos_B_np, cell_B_np, charges_B_np, ei_B_np, d_B_np, n=3
    )

    # --- Batched call ---
    n_A, n_B = 2, 3
    n_total = n_A + n_B

    pos_batch_np = np.concatenate([pos_A_np, pos_B_np], axis=0)
    charges_batch_np = np.concatenate([charges_A_np, charges_B_np])

    # Shift B indices by n_A
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
    charges_batch = torch.tensor(charges_batch_np, dtype=dtype)
    cell_batch = torch.tensor(cell_batch_np, dtype=dtype)  # (2, 3, 3)
    edge_index_batch = torch.tensor(ei_batch_np, dtype=torch.long)
    lengths_batch = torch.tensor(d_batch_np, dtype=dtype)

    e_batch = pme(
        partial_charges=charges_batch,
        positions=pos_batch,
        cell=cell_batch,
        edge_index=edge_index_batch,
        lengths=lengths_batch,
        batch_segments=batch_seg,
        num_graphs=2,
        num_nodes=n_total,
    )

    # Compare per-atom energies
    assert torch.allclose(
        e_batch[:n_A], e_A, atol=1e-10
    ), (
        f"Batch isolation failed for system A:\n"
        f"  batched: {e_batch[:n_A].squeeze().tolist()}\n"
        f"  single:  {e_A.squeeze().tolist()}"
    )
    assert torch.allclose(
        e_batch[n_A:], e_B, atol=1e-10
    ), (
        f"Batch isolation failed for system B:\n"
        f"  batched: {e_batch[n_A:].squeeze().tolist()}\n"
        f"  single:  {e_B.squeeze().tolist()}"
    )
