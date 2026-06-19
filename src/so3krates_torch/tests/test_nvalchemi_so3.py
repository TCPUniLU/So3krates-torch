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


def _make_so3lr(
    use_pme: bool = False,
    electrostatics_bool: bool = None,
    output_partial_charges: bool = False,
) -> SO3LR:
    """Minimal SO3LR model for testing.

    ``electrostatics_bool`` defaults to ``use_pme`` (so a short-range-only
    model has electrostatics off); pass it explicitly to enable the
    real-space erf-Coulomb electrostatics with ``use_pme=False``.
    """
    es = use_pme if electrostatics_bool is None else electrostatics_bool
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
        electrostatic_energy_bool=es,
        output_partial_charges=output_partial_charges,
        dispersion_energy_bool=False,
        zbl_repulsion_bool=False,
    ).to(dtype=DTYPE, device=DEVICE)
    model.eval()
    return model


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


def test_batched_md_independence():
    """A multi-graph batch yields per-system results identical to running
    each system alone (graphs are independent — no cross-graph edges)."""
    a1 = bulk("NaCl", crystalstructure="rocksalt", a=5.6402)
    a2 = bulk("NaCl", crystalstructure="rocksalt", a=5.8000)
    model = _make_so3lr()
    wrapper = NVAlchemiSO3LR(model)

    batch = _make_batch_with_nl_multi(wrapper, [a1, a2])
    out = wrapper(batch)

    assert out["energy"].shape == (2, 1)
    assert out["forces"].shape == (len(a1) + len(a2), 3)
    assert torch.isfinite(out["energy"]).all()
    assert torch.isfinite(out["forces"]).all()

    # Reference: each system run on its own.
    e_refs, f_refs = [], []
    for atoms in (a1, a2):
        single = wrapper(_make_batch_with_nl(wrapper, atoms))
        e_refs.append(single["energy"])
        f_refs.append(single["forces"])

    assert torch.allclose(out["energy"][0], e_refs[0][0], atol=1e-6)
    assert torch.allclose(out["energy"][1], e_refs[1][0], atol=1e-6)
    assert torch.allclose(out["forces"][: len(a1)], f_refs[0], atol=1e-6)
    assert torch.allclose(out["forces"][len(a1) :], f_refs[1], atol=1e-6)


def test_stress_sign_matches_nvalchemi_virial_convention():
    """The wrapper's `stress` must be the NEGATIVE of the ASE-convention
    stress.

    The model emits ASE-convention stress (+dE/de / V), which the ASE
    calculator surfaces verbatim.  NVAlchemi's NPT treats the `stress` key as
    Cauchy W/V with virial W = -dE/de (the opposite sign), so the adapter
    negates it.  A regression here silently drives the barostat the wrong way
    (density falls instead of rising), so it is asserted explicitly.
    """
    pytest.importorskip("matscipy", reason="matscipy needed for neighbor list")
    from ase.stress import voigt_6_to_full_3x3_stress

    atoms = _small_nacl()
    model = _make_so3lr(use_pme=False, electrostatics_bool=True)

    calc = TorchkratesCalculator(
        models=[model],
        device=str(DEVICE),
        dtype=DTYPE,
        r_max=model.r_max,
        r_max_lr=model.r_max_lr,
        compute_stress=True,
    )
    atoms.calc = calc
    ase_stress_3x3 = voigt_6_to_full_3x3_stress(atoms.get_stress())

    wrapper = NVAlchemiSO3LR(model)
    out = wrapper(_make_batch_with_nl(wrapper, atoms))
    nv_stress = out["stress"][0].detach().cpu().numpy()

    # Sanity: stress must actually be non-trivial, else the sign test is vacuous
    assert abs(ase_stress_3x3).max() > 1e-8, "ASE stress is ~zero; test vacuous"

    # The adapter must flip the sign relative to the ASE convention.
    assert torch.allclose(
        torch.tensor(nv_stress),
        torch.tensor(-ase_stress_3x3),
        atol=1e-8,
        rtol=1e-6,
    ), (
        "NVAlchemi wrapper stress is not -(ASE stress); barostat sign bug.\n"
        f"  wrapper stress:\n{nv_stress}\n"
        f"  -(ASE stress):\n{-ase_stress_3x3}"
    )


# ---------------------------------------------------------------------------
# Electrostatics: real-space (model) and NVAlchemi-native PME
# ---------------------------------------------------------------------------


def test_model_emits_charges_without_electrostatic_energy():
    """output_partial_charges=True yields charges even when the model's own
    electrostatic energy term is disabled."""
    model = _make_so3lr(electrostatics_bool=False, output_partial_charges=True)
    wrapper = NVAlchemiSO3LR(model)  # plain "model" path
    d = wrapper.adapt_input(_make_batch_with_nl(wrapper, _small_nacl()))
    out = wrapper.model(d, compute_force=True)
    assert out["partial_charges"] is not None
    assert out["partial_charges"].reshape(-1).shape[0] == 2
    assert torch.isfinite(out["partial_charges"]).all()


def test_real_space_electrostatics_through_wrapper():
    """The model's real-space (non-PME) electrostatics runs through the
    wrapper and matches the ASE calculator path."""
    pytest.importorskip("matscipy", reason="matscipy needed for neighbor list")
    atoms = _small_nacl()
    model = _make_so3lr(use_pme=False, electrostatics_bool=True)

    calc = TorchkratesCalculator(
        models=[model],
        device=str(DEVICE),
        dtype=DTYPE,
        r_max=model.r_max,
        r_max_lr=model.r_max_lr,
    )
    atoms.calc = calc
    e_ref = atoms.get_potential_energy()

    wrapper = NVAlchemiSO3LR(model)
    d = wrapper.adapt_input(_make_batch_with_nl(wrapper, atoms))
    out = wrapper.model(d, compute_force=True)
    e_wrap = out["energy"].item()
    assert torch.isfinite(out["energy"]).all()
    assert (
        abs(e_wrap - e_ref) < 1e-4
    ), f"Energy mismatch: wrapper={e_wrap:.6f} eV, ASE={e_ref:.6f} eV"


def test_native_pme_wrapper_requires_charges_and_no_electrostatics():
    """The 'nvalchemi' electrostatics mode validates its preconditions."""
    # electrostatics still enabled on the model -> error
    with pytest.raises(ValueError, match="electrostatic_energy_bool"):
        NVAlchemiSO3LR(
            _make_so3lr(electrostatics_bool=True), electrostatics="nvalchemi"
        )
    # charges not emitted -> error
    with pytest.raises(ValueError, match="output_partial_charges"):
        NVAlchemiSO3LR(
            _make_so3lr(electrostatics_bool=False), electrostatics="nvalchemi"
        )


def test_build_nvalchemi_pme_model_emits_charges():
    """build_nvalchemi_pme_model composes a pipeline; the SR step emits
    charges and the model's own electrostatics are disabled."""
    from so3krates_torch.calculator.nvalchemi_so3 import (
        build_nvalchemi_pme_model,
    )

    model = _make_so3lr(electrostatics_bool=True)
    pipe = build_nvalchemi_pme_model(model)
    assert model.electrostatic_energy_bool is False
    assert model.output_partial_charges is True
    assert pipe.model_config.active_outputs == {"energy", "forces", "stress"}

    sr = NVAlchemiSO3LR(model, electrostatics="nvalchemi")
    out = sr(_make_batch_with_nl(sr, _small_nacl()))
    assert "charges" in out
    assert out["charges"].shape == (2,)


def test_native_pme_md_runs_on_cpu():
    """A short NVT run with NVAlchemi-native PME completes and stays finite
    (single + batched)."""
    from nvalchemi.dynamics.base import DynamicsStage
    from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
    from nvalchemi.hooks.neighbor_list import NeighborListHook

    from so3krates_torch.calculator.nvalchemi_so3 import (
        build_nvalchemi_pme_model,
    )

    pipe = build_nvalchemi_pme_model(_make_so3lr(electrostatics_bool=True))

    def _data(atoms):
        n = len(atoms)
        return AtomicData.from_atoms(
            atoms, device=str(DEVICE), dtype=DTYPE
        ).model_copy(
            update={
                "forces": torch.zeros(n, 3, dtype=DTYPE),
                "energy": torch.zeros(1, 1, dtype=DTYPE),
                "stress": torch.zeros(1, 3, 3, dtype=DTYPE),
            }
        )

    a1 = bulk("NaCl", crystalstructure="rocksalt", a=5.6402)
    a2 = bulk("NaCl", crystalstructure="rocksalt", a=5.8000)
    batch = Batch.from_data_list([_data(a1), _data(a2)])

    hooks = [
        NeighborListHook(
            config=pipe.model_config.neighbor_config,
            skin=0.5,
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
    ]
    integ = NVTLangevin(
        model=pipe,
        dt=0.5,
        temperature=300.0,
        friction=1e-2,
        n_steps=1,
        hooks=hooks,
    )
    out = integ.run(batch)
    assert out.energy.reshape(-1).shape[0] == 2
    assert torch.isfinite(out.energy).all()


# ---------------------------------------------------------------------------
# Helper: build AtomicData with neighbor list set, then batch
# ---------------------------------------------------------------------------


def _make_data_with_nl(wrapper: NVAlchemiSO3LR, atoms) -> AtomicData:
    """Build a single nvalchemi AtomicData with a pre-built neighbor list."""
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

    # Set the neighbor list at construction time so the 'edges' segmented
    # group exists when Batch.from_data_list is called.
    data = AtomicData.from_atoms(atoms, device=str(DEVICE), dtype=DTYPE)
    # cell_list returns nm as [2, E] (sender/receiver rows);
    # AtomicData.neighbor_list expects [E, 2].
    return data.model_copy(
        update={
            "neighbor_list": nm.long().T,  # [E, 2]
            "neighbor_list_shifts": nm_shifts.float(),  # [E, 3]
        }
    )


def _make_batch_with_nl(wrapper: NVAlchemiSO3LR, atoms) -> Batch:
    """Build a single-graph nvalchemi Batch with a pre-built neighbor list."""
    return Batch.from_data_list([_make_data_with_nl(wrapper, atoms)])


def _make_batch_with_nl_multi(wrapper: NVAlchemiSO3LR, atoms_list) -> Batch:
    """Build a multi-graph Batch (one entry per Atoms) with neighbor lists."""
    return Batch.from_data_list(
        [_make_data_with_nl(wrapper, atoms) for atoms in atoms_list]
    )
