"""Tests for the NVAlchemi wrapper (NVAlchemiSO3LR).

Skipped automatically when nvalchemi is not installed.
"""

import pytest
import torch

pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit not installed")

from ase.build import bulk, molecule

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch

from so3krates_torch.calculator.nvalchemi_so3 import NVAlchemiSO3LR
from so3krates_torch.calculator.so3 import TorchkratesCalculator
from so3krates_torch.modules.models import SO3LR


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

DTYPE = torch.float64
DEVICE = torch.device("cpu")


def _small_nacl():
    """2-atom NaCl primitive cell, periodic."""
    return bulk("NaCl", crystalstructure="rocksalt", a=5.6402)


def _make_so3lr(use_pme: bool = False) -> SO3LR:
    """Minimal SO3LR model (short-range only or with PME) for testing."""
    model = SO3LR(
        r_max=4.5,
        r_max_lr=6.0,
        num_radial_basis_fn=8,
        degrees=[1, 2],
        num_features=16,
        num_heads=2,
        num_layers=1,
        num_elements=118,
        avg_num_neighbors=10.0,
        energy_regression_dim=16,
        use_pme=use_pme,
        pme_smearing=1.0 if use_pme else None,
        pme_mesh_spacing=0.5 if use_pme else None,
        electrostatic_energy_bool=use_pme,
        dispersion_energy_bool=False,
        zbl_repulsion_bool=False,
    ).to(dtype=DTYPE, device=DEVICE)
    model.eval()
    return model


def _make_batch(atoms) -> Batch:
    """Create a nvalchemi Batch (with neighbor list) from ASE Atoms."""
    data = AtomicData.from_atoms(atoms, device=str(DEVICE), dtype=DTYPE)
    return Batch.from_data_list([data])


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_wrapper_instantiation():
    """NVAlchemiSO3LR can be constructed without error."""
    model = _make_so3lr()
    wrapper = NVAlchemiSO3LR(model)
    assert wrapper.r_sr == pytest.approx(4.5)
    assert wrapper.r_lr == pytest.approx(6.0)
    assert wrapper.has_lr is True


def test_adapt_input_shapes():
    """adapt_input returns dict with correct tensor shapes."""
    atoms = _small_nacl()
    model = _make_so3lr()
    wrapper = NVAlchemiSO3LR(model)
    batch = _make_batch_with_nl(wrapper, atoms)
    d = wrapper.adapt_input(batch)

    assert d["node_attrs"].shape == (len(atoms), 118)
    assert d["atomic_numbers"].shape == (len(atoms),)
    assert d["shifts"].dim() == 2 and d["shifts"].shape[1] == 3
    assert d["edge_index"].shape[0] == 2
    assert d["positions"].shape == (len(atoms), 3)
    # positions must not be zeros (real coordinates)
    assert d["positions"].abs().sum() > 0


def test_positions_not_zeros():
    """adapt_input passes real atomic coordinates, not zeros."""
    atoms = _small_nacl()
    model = _make_so3lr()
    wrapper = NVAlchemiSO3LR(model)
    batch = _make_batch_with_nl(wrapper, atoms)
    d = wrapper.adapt_input(batch)
    assert not torch.allclose(d["positions"], torch.zeros_like(d["positions"]))


def test_energy_forces_finite():
    """Full forward pass returns finite energy and forces."""
    atoms = _small_nacl()
    model = _make_so3lr()
    wrapper = NVAlchemiSO3LR(model)
    batch = _make_batch_with_nl(wrapper, atoms)
    d = wrapper.adapt_input(batch)

    out = wrapper.model(d, compute_force=True)
    energy = out["energy"]
    forces = out["forces"]

    assert torch.isfinite(energy).all(), "energy is not finite"
    assert torch.isfinite(forces).all(), "forces are not finite"
    assert forces.shape == (len(atoms), 3)


def test_energy_matches_ase_calc():
    """Energy from wrapper matches TorchkratesCalculator to float64 tolerance."""
    pytest.importorskip("matscipy", reason="matscipy needed for neighbor list")

    atoms = _small_nacl()
    model = _make_so3lr(use_pme=False)

    # Reference energy via the ASE calculator path
    calc = TorchkratesCalculator(
        models=[model],
        device=str(DEVICE),
        dtype=DTYPE,
        r_max=model.r_max,
        r_max_lr=model.r_max_lr,
    )
    atoms.calc = calc
    e_ref = atoms.get_potential_energy()

    # Wrapper energy
    wrapper = NVAlchemiSO3LR(model)
    batch = _make_batch_with_nl(wrapper, atoms)
    d = wrapper.adapt_input(batch)
    out = wrapper.model(d, compute_force=True)
    e_wrap = out["energy"].item()

    assert (
        abs(e_wrap - e_ref) < 1e-4
    ), f"Energy mismatch: wrapper={e_wrap:.6f} eV, ASE={e_ref:.6f} eV"


# ---------------------------------------------------------------------------
# Helper: build AtomicData with neighbor list set, then batch
# ---------------------------------------------------------------------------


def _make_batch_with_nl(wrapper: NVAlchemiSO3LR, atoms) -> Batch:
    """Build a nvalchemi Batch that includes a pre-built neighbor list."""
    import numpy as np
    from nvalchemiops.torch.neighbors.cell_list import cell_list

    pos = torch.tensor(atoms.positions, dtype=DTYPE)
    cell_np = np.array(atoms.cell)
    cell_t = torch.tensor(cell_np, dtype=DTYPE).unsqueeze(0)  # [1, 3, 3]
    pbc_t = torch.tensor(atoms.pbc, dtype=torch.bool).unsqueeze(0)  # [1, 3]

    nm, _num_nb, nm_shifts = cell_list(
        pos,
        wrapper.r_lr,
        cell=cell_t,
        pbc=pbc_t,
        return_neighbor_list=True,
    )
    # nm: [E, 2] COO pairs, nm_shifts: [E, 3] integer image offsets

    # Build AtomicData with neighbor list set at construction time so the
    # 'edges' segmented group exists when Batch.from_data_list is called.
    data = AtomicData.from_atoms(atoms, device=str(DEVICE), dtype=DTYPE)
    # cell_list returns nm as [2, E] (sender/receiver rows);
    # AtomicData.neighbor_list expects [E, 2].
    data = data.model_copy(
        update={
            "neighbor_list": nm.long().T,  # [E, 2]
            "neighbor_list_shifts": nm_shifts.float(),  # [E, 3]
        }
    )
    return Batch.from_data_list([data])
