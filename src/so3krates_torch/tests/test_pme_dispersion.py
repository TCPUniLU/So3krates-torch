"""Tests for PMEDispersionInteraction (k-space-only C6 dispersion via PME/Ewald).

Tests verify energy sanity against an independent numerical method (Ewald
direct sum vs. FFT-interpolated mesh), gradient consistency via autograd,
correct batch isolation for multi-system batches, and — most importantly —
that the real-space residual (from `DispersionInteraction`) plus the
k-space contribution (from `PMEDispersionInteraction`) reproduces the plain
non-PME dispersion result. This last check (Test 4) is the permanent
regression guard for two bugs found and fixed during this plan's execution:
a k-space-only fix (commit 8830494) and a `background_correction`
double-counting fix (commit 31c0716).
"""

import numpy as np
import torch
from so3krates_torch.blocks.physical_potentials import (
    DispersionInteraction,
    PMEDispersionInteraction,
    atomic_c6_pseudo_charges,
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
# Test 1: Energy sanity check — Ewald direct sum vs. PME mesh
# ============================================================


def test_pme_dispersion_energy_sanity():
    """PME-dispersion k-space energy agrees between two independent
    numerical methods: the exact reciprocal-space Ewald sum
    (``do_ewald=True``) and the FFT-interpolated PME mesh
    (``do_ewald=False``, the default used everywhere else in this codebase).

    Judgment call: the brief's primary suggestion for this test was a
    fully from-scratch reciprocal-space reference built by hand from
    torch-pme's public ``generate_kvectors_for_ewald``/``lr_from_k_sq``
    API. A prior attempt at this exact task tried exactly that, swept the
    k-space cutoff to "check convergence", and had a k-vector grid balloon
    to ~1e9 vectors on this same 27-atom system, which drove one Python
    process to ~60GB RSS and got it killed by the OOM-killer. Inspecting
    torch-pme's source (``EwaldCalculator._compute_kspace``) shows that a
    hand-rolled version calling the same public functions with the same
    parameters would just be re-executing that exact code path, not an
    independent check — so it would not actually buy additional
    protection against the real historical bugs in this plan (both fixed
    long before this task) beyond what's exercised below.

    Instead this test uses the brief's explicitly-sanctioned simpler
    alternative: ``do_ewald=True`` (exact k-vector summation) vs.
    ``do_ewald=False`` (FFT-based mesh interpolation) are genuinely
    different algorithms living in different torch-pme modules
    (``calculators/ewald.py`` vs. ``calculators/pme.py``), so agreement
    between them is a real cross-check of the underlying k-space physics
    the way Task 1's development notes describe (~1e-11 relative on an
    8-atom NaCl cell).

    Before running, the k-vector count for this system/parameters was
    checked by hand: with smearing=0.5 Angstrom, mesh_spacing=0.25
    Angstrom on the 27-atom Ar supercell (box ~15.78 Angstrom / ~29.8
    Bohr), ``ns`` per dimension is 45, giving 91,125 k-vectors total —
    comfortably in the "thousands to low tens of thousands" safe range
    the task's OOM-avoidance guidance calls for (not millions+), and the
    subsequent ``kvectors @ positions.T`` matmul for only 27 atoms is a
    few tens of MB. No sweeping/tightening of this density is done.

    Why ``interpolation_nodes=7`` for the mesh path (diagnosed, not
    guessed): under torchpme>=0.5.0 (which correctly includes the k=0
    reciprocal-space limit for exponent>3 dispersion potentials — see
    Test 4's docstring), the k-space-only dispersion energy for this
    non-neutral (all-positive) pseudo-charge system is a near-total
    cancellation between a large k=0 background term and the sum over
    nonzero k-vectors, leaving a small net total (~-2.7 to -2.8 eV for
    this system). At the class's default ``interpolation_nodes=4``
    (cubic Lagrange interpolation), the FFT-mesh path has an absolute
    mismatch against the exact Ewald sum of ~0.094 eV at this
    ``mesh_spacing`` — this is a genuine, expected mesh-interpolation
    truncation error, NOT a bug: it was present identically under
    torchpme 0.4.0 too, but was invisible there because both paths
    lacked the k=0 term identically, making the ~93 eV uncorrected
    totals agree to ~1.3e-3 relative even though their absolute
    difference was unchanged. Now that the true (much smaller) net
    energy is exposed, that same ~0.094 eV absolute mismatch is ~3.3%
    of the total — nothing changed about the mesh's accuracy, only
    about how large a total it's being compared against.
    A parameter sweep (mesh_spacing in {0.25, 0.2, 0.15}, interpolation_nodes
    in {4, 5, 6, 7}, all bounds-checked for k-vector count before running,
    max ~3.4M vectors) showed the fix is to raise ``interpolation_nodes``
    (free — same mesh size, higher-order polynomial stencil) rather than
    shrink ``mesh_spacing`` (expensive — grows the FFT grid and k-vector
    count): nodes=5 -> 5.3e-4 relative, nodes=6 -> 8.5e-4, nodes=7 ->
    2.1e-5. `interpolation_nodes=7` (the max torch-pme supports) is used
    here, at zero extra memory cost, to make this an apples-to-apples
    comparison instead of loosening the assertion to paper over an
    under-converged default.
    """
    from ase.build import bulk

    dtype = torch.float64

    ar = bulk("Ar", "fcc", a=5.26).repeat((3, 3, 3))
    N = len(ar)
    pos_np = ar.positions
    cell_np = ar.cell.array

    atomic_numbers = torch.full((N,), 18, dtype=torch.long)  # Ar
    hirshfeld_ratios = torch.ones(N, dtype=dtype)

    # q_i = sqrt(C6_i), tracked via the actual legacy table (not a
    # hardcoded magic number).
    C6_i = atomic_c6_pseudo_charges(atomic_numbers, hirshfeld_ratios)
    q = torch.sqrt(torch.clamp(C6_i, min=0.0))

    positions = torch.tensor(pos_np, dtype=dtype)
    cell = torch.tensor(cell_np, dtype=dtype).unsqueeze(0)  # (1, 3, 3)
    batch_segments = torch.zeros(N, dtype=torch.long)

    smearing = 0.5
    mesh_spacing = 0.25

    # interpolation_nodes=7 (max supported by torch-pme) on the mesh path
    # only — see docstring above for why this is needed for a fair
    # comparison against the exact Ewald sum, at zero extra memory cost
    # (same FFT mesh size, just a higher-order interpolation stencil).
    # Production code / other callers are unaffected: this is a local,
    # test-only choice of an already-public constructor parameter, not a
    # change to any default.
    pme_mesh = PMEDispersionInteraction(
        smearing=smearing,
        mesh_spacing=mesh_spacing,
        interpolation_nodes=7,
        do_ewald=False,
    )
    pme_ewald = PMEDispersionInteraction(
        smearing=smearing, mesh_spacing=mesh_spacing, do_ewald=True
    )

    kwargs = dict(
        c6_pseudo_charges=q,
        positions=positions,
        cell=cell,
        batch_segments=batch_segments,
        num_graphs=1,
        num_nodes=N,
    )

    E_mesh = pme_mesh(**kwargs).sum().item()
    E_ewald = pme_ewald(**kwargs).sum().item()

    rel_diff = abs(E_mesh - E_ewald) / abs(E_ewald)
    # Observed on this run: with interpolation_nodes=7, mesh vs. Ewald
    # agree to ~2.1e-5 relative (see this test's docstring for the full
    # diagnosis of why interpolation_nodes=4, the class default, is not
    # tight enough for a *relative* comparison at these parameters). 1e-3
    # gives ample margin above the observed value.
    assert rel_diff < 1e-3, (
        f"PME mesh energy {E_mesh:.8f} eV vs. Ewald-sum energy "
        f"{E_ewald:.8f} eV disagree by {rel_diff:.3e} relative "
        f"(expected sub-percent agreement between the two k-space "
        f"methods)"
    )


# ============================================================
# Test 2: torch.autograd.gradcheck on PMEDispersionInteraction
# ============================================================


def test_pme_dispersion_gradcheck():
    """Verify PME-dispersion k-space energy gradients w.r.t. positions.

    Unlike PME electrostatics, `PMEDispersionInteraction.forward` is
    k-space-only and takes no edge_index/lengths — only `positions` and
    `cell` feed the reciprocal-space FFT/sum, so `energy_fn` needs no
    neighbor-list rebuilding or minimum-image distance recomputation.
    """
    dtype = torch.float64
    torch.manual_seed(0)

    box = 5.0  # Angstrom, cubic cell side
    smearing = 0.5
    mesh_spacing = 0.25

    pme_disp = PMEDispersionInteraction(
        smearing=smearing, mesh_spacing=mesh_spacing
    )

    N = 4
    atomic_numbers = torch.full((N,), 18, dtype=torch.long)  # Ar
    hirshfeld_ratios = torch.ones(N, dtype=dtype)
    C6_i = atomic_c6_pseudo_charges(atomic_numbers, hirshfeld_ratios)
    q = torch.sqrt(torch.clamp(C6_i, min=0.0))

    cell_np = np.diag([box, box, box])
    rng = np.random.default_rng(42)
    positions_np = rng.uniform(0.5, box - 0.5, size=(N, 3))

    cell = torch.tensor(cell_np, dtype=dtype).unsqueeze(0)  # (1, 3, 3)
    batch_segments = torch.zeros(N, dtype=torch.long)

    positions = torch.tensor(positions_np, dtype=dtype, requires_grad=True)

    def energy_fn(pos):
        return pme_disp(
            c6_pseudo_charges=q,
            positions=pos,
            cell=cell,
            batch_segments=batch_segments,
            num_graphs=1,
            num_nodes=N,
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


def test_pme_dispersion_batch_isolation():
    """Per-atom PME-dispersion energies from a batched call equal
    individual calls.

    System A: 2-atom NaCl rock-salt FCC primitive cell (a = 5.6402 Angstrom).
    System B: 3-atom random cell (a = 4.0 Angstrom).

    Dispersion pseudo-charges are always positive (unlike electrostatic
    charges), so `atomic_c6_pseudo_charges` with unit Hirshfeld ratios is
    used for both systems' actual atomic numbers.
    """
    from ase.build import bulk

    dtype = torch.float64

    smearing = 0.5
    mesh_spacing = 0.25
    pme_disp = PMEDispersionInteraction(
        smearing=smearing, mesh_spacing=mesh_spacing
    )

    def _charges(atomic_numbers):
        ratios = torch.ones(atomic_numbers.shape[0], dtype=dtype)
        C6_i = atomic_c6_pseudo_charges(atomic_numbers, ratios)
        return torch.sqrt(torch.clamp(C6_i, min=0.0))

    # --- System A: NaCl FCC primitive cell (2 atoms) ---
    a_A = 5.6402
    nacl = bulk("NaCl", crystalstructure="rocksalt", a=a_A)
    pos_A_np = nacl.positions
    cell_A_np = nacl.cell.array
    Z_A = torch.tensor([11, 17], dtype=torch.long)  # Na, Cl
    q_A = _charges(Z_A)

    # --- System B: 3-atom cell ---
    a_B = 4.0
    rng = np.random.default_rng(7)
    pos_B_np = rng.uniform(0.3, a_B - 0.3, size=(3, 3))
    cell_B_np = np.diag([a_B, a_B, a_B])
    Z_B = torch.tensor([18, 10, 6], dtype=torch.long)  # Ar, Ne, C
    q_B = _charges(Z_B)

    def _call_single(pos_np, cell_np, q, n):
        pos = torch.tensor(pos_np, dtype=dtype)
        cell = torch.tensor(cell_np, dtype=dtype).unsqueeze(0)
        batch_seg = torch.zeros(n, dtype=torch.long)
        return pme_disp(
            c6_pseudo_charges=q,
            positions=pos,
            cell=cell,
            batch_segments=batch_seg,
            num_graphs=1,
            num_nodes=n,
        )

    e_A = _call_single(pos_A_np, cell_A_np, q_A, n=2)
    e_B = _call_single(pos_B_np, cell_B_np, q_B, n=3)

    # --- Batched call ---
    n_A, n_B = 2, 3
    n_total = n_A + n_B

    pos_batch_np = np.concatenate([pos_A_np, pos_B_np], axis=0)
    q_batch = torch.cat([q_A, q_B])
    cell_batch_np = np.stack([cell_A_np, cell_B_np], axis=0)  # (2, 3, 3)

    batch_seg = torch.cat(
        [
            torch.zeros(n_A, dtype=torch.long),
            torch.ones(n_B, dtype=torch.long),
        ]
    )

    pos_batch = torch.tensor(pos_batch_np, dtype=dtype)
    cell_batch = torch.tensor(cell_batch_np, dtype=dtype)  # (2, 3, 3)

    e_batch = pme_disp(
        c6_pseudo_charges=q_batch,
        positions=pos_batch,
        cell=cell_batch,
        batch_segments=batch_seg,
        num_graphs=2,
        num_nodes=n_total,
    )

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
# Test 4: Residual-split self-consistency (the most important test)
# ============================================================


def test_pme_dispersion_residual_split_self_consistency():
    """Real-space residual + k-space must reproduce non-PME dispersion.

    With PME dispersion active, `DispersionInteraction`'s real-space
    output (residual C6 + unaffected C8/C10) plus
    `PMEDispersionInteraction`'s k-space output must reproduce the plain
    non-PME `DispersionInteraction(use_pme_dispersion=False)` result, in
    the limit of a well-converged PME setup on a large-enough cell.

    This is the permanent regression guard for two real bugs found and
    fixed during this plan's execution, both after an earlier review had
    already approved `PMEDispersionInteraction`:
      - commit 8830494: the JAX reference is k-space-ONLY; an earlier
        version wrongly also added a real-space SR sum via torch-pme's
        generic `Calculator.forward`.
      - commit 31c0716: a custom potential subclass wrongly patched
        `background_correction()` for exponent=6, causing a spurious
        charge_tot^2 * smearing^-3 term; fixed by deleting the subclass
        entirely in favor of a plain, unmodified
        `torchpme.InversePowerLawPotential(exponent=6, ...)`.
    Both bugs would have been caught immediately by this test.

    Uses the exact, already-validated setup from this plan's development:
    ase.build.bulk("Ar", "fcc", a=5.26).repeat((3, 3, 3)) (27 atoms), unit
    Hirshfeld ratios, r_max_lr=10.0 Angstrom, cutoff_lr_damping=2.0,
    dispersion_energy_scale=1.2, pme_dispersion_smearing=0.5 Angstrom,
    mesh_spacing=0.25 Angstrom.
    """
    from ase.build import bulk

    dtype = torch.float64

    ar = bulk("Ar", "fcc", a=5.26).repeat((3, 3, 3))
    N = len(ar)
    pos_np = ar.positions
    cell_np = ar.cell.array

    r_max_lr = 10.0
    cutoff_lr_damping = 2.0
    dispersion_energy_scale = 1.2
    pme_dispersion_smearing = 0.5
    mesh_spacing = 0.25

    edge_index_np, lengths_np = build_nl(pos_np, cell_np, r_max_lr)

    atomic_numbers = torch.full((N,), 18, dtype=torch.long)  # Ar
    hirshfeld_ratios = torch.ones(N, dtype=dtype)
    receivers_lr = torch.tensor(edge_index_np[0], dtype=torch.long)
    senders_lr = torch.tensor(edge_index_np[1], dtype=torch.long)
    lengths_lr = torch.tensor(lengths_np, dtype=dtype)
    positions = torch.tensor(pos_np, dtype=dtype)
    cell = torch.tensor(cell_np, dtype=dtype).unsqueeze(0)  # (1, 3, 3)
    batch_segments = torch.zeros(N, dtype=torch.long)

    disp = DispersionInteraction()

    E_nonpme = disp(
        hirshfeld_ratios=hirshfeld_ratios,
        atomic_numbers=atomic_numbers,
        senders_lr=senders_lr,
        receivers_lr=receivers_lr,
        lengths_lr=lengths_lr,
        num_nodes=N,
        cutoff_lr=r_max_lr,
        cutoff_lr_damping=cutoff_lr_damping,
        dispersion_energy_scale=dispersion_energy_scale,
    ).sum()

    E_residual = disp(
        hirshfeld_ratios=hirshfeld_ratios,
        atomic_numbers=atomic_numbers,
        senders_lr=senders_lr,
        receivers_lr=receivers_lr,
        lengths_lr=lengths_lr,
        num_nodes=N,
        cutoff_lr=r_max_lr,
        cutoff_lr_damping=cutoff_lr_damping,
        dispersion_energy_scale=dispersion_energy_scale,
        use_pme_dispersion=True,
        pme_dispersion_smearing=pme_dispersion_smearing,
    ).sum()

    C6_i = atomic_c6_pseudo_charges(atomic_numbers, hirshfeld_ratios)
    q = torch.sqrt(torch.clamp(C6_i, min=0.0))
    pme_disp = PMEDispersionInteraction(
        smearing=pme_dispersion_smearing, mesh_spacing=mesh_spacing
    )
    E_kspace = pme_disp(
        c6_pseudo_charges=q,
        positions=positions,
        cell=cell,
        batch_segments=batch_segments,
        num_graphs=1,
        num_nodes=N,
    ).sum()

    E_pme_total = E_residual + E_kspace
    per_atom_diff = abs(E_pme_total - E_nonpme).item() / N
    # Observed on this run: see report for the exact meV/atom figure;
    # the brief's own development baseline was ~0.4 meV/atom on this
    # exact setup. 2e-3 eV/atom (2 meV/atom) gives sensible margin above
    # that baseline while still tightly constraining the residual-split
    # identity this test guards.
    assert per_atom_diff < 2e-3, (
        f"Residual-split self-consistency failed: PME total "
        f"{E_pme_total:.6f} eV vs non-PME {E_nonpme:.6f} eV, "
        f"{per_atom_diff * 1000:.3f} meV/atom "
        f"(observed baseline during development: ~0.4 meV/atom)"
    )
