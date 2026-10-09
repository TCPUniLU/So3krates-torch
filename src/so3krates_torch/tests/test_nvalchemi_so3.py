"""Tests for the NVAlchemi wrapper (NVAlchemiSO3LR).

Skipped automatically when nvalchemi is not installed.
"""

import numpy as np
import pytest
import torch

pytest.importorskip("nvalchemi", reason="nvalchemi-toolkit not installed")
pytest.importorskip("nvalchemiops", reason="nvalchemi-toolkit-ops missing")

from ase.build import bulk

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
    dispersion_bool: bool = False,
    zbl_bool: bool = False,
) -> SO3LR:
    """Minimal SO3LR model for testing.

    ``electrostatics_bool`` defaults to ``use_pme`` (so a short-range-only
    model has electrostatics off); pass it explicitly to enable the
    real-space erf-Coulomb electrostatics with ``use_pme=False``.
    ``dispersion_bool`` / ``zbl_bool`` switch on the real-space dispersion
    and ZBL repulsion terms (off by default).
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
        dispersion_energy_bool=dispersion_bool,
        # DispersionInteraction requires a damping width whenever
        # cutoff_lr is set (SO3LR always passes r_max_lr).
        dispersion_energy_cutoff_lr_damping=1.0 if dispersion_bool else None,
        zbl_repulsion_bool=zbl_bool,
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


def test_stress_matches_ase_convention():
    """nvalchemi-toolkit>=0.2 treats `stress` as tensile-positive Cauchy
    stress sigma = -W/V (npt.py: virial = -stress * V), which is exactly
    ASE's +dE/deps / V. The wrapper must pass the model stress through
    unchanged; negating it makes the barostat drive the cell the wrong way.
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
    assert (
        abs(ase_stress_3x3).max() > 1e-8
    ), "ASE stress is ~zero; test vacuous"

    # The adapter must pass the ASE-convention stress through unchanged.
    torch.testing.assert_close(
        torch.tensor(nv_stress),
        torch.tensor(ase_stress_3x3),
        atol=1e-6,
        rtol=1e-5,
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
# Active outputs, skin, charge/spin, energy-only mode
# ---------------------------------------------------------------------------


def _parity_calc(model, compute_stress: bool = False):
    """TorchkratesCalculator with the same args as the parity tests."""
    return TorchkratesCalculator(
        models=[model],
        device=str(DEVICE),
        dtype=DTYPE,
        r_max=model.r_max,
        r_max_lr=model.r_max_lr,
        compute_stress=compute_stress,
    )


def test_skin_does_not_change_energy():
    """End-to-end energy-invariance smoke test: a list built at r_lr + skin
    gives the same energy as one built at r_lr.

    This alone does not prove the skin mask works — the model's LR kernels
    also zero pairs beyond cutoff_lr internally. The discriminating guard
    for the skin mask is ``test_adapt_input_drops_skin_shell_edges``.
    """
    model = _make_so3lr(electrostatics_bool=True)
    w = NVAlchemiSO3LR(model).to(DEVICE)
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    e0 = w(_make_batch_with_nl(w, atoms, cutoff=w.r_lr))["energy"]
    e1 = w(_make_batch_with_nl(w, atoms, cutoff=w.r_lr + 1.0))["energy"]
    torch.testing.assert_close(e0, e1)


def test_adapt_input_drops_skin_shell_edges():
    """The LR edge set handed to the model is identical whether the list
    was built at r_lr or at r_lr + skin. (The model's LR kernels also zero
    pairs beyond cutoff_lr internally, so the energy test above cannot
    detect a missing mask on its own.)"""
    w = NVAlchemiSO3LR(_make_so3lr(electrostatics_bool=True))
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    d0 = w.adapt_input(_make_batch_with_nl(w, atoms, cutoff=w.r_lr))
    d1 = w.adapt_input(_make_batch_with_nl(w, atoms, cutoff=w.r_lr + 1.0))
    assert d1["edge_index_lr"].shape == d0["edge_index_lr"].shape
    assert d1["shifts_lr"].shape == d0["shifts_lr"].shape
    assert d1["unit_shifts_lr"].shape == d0["unit_shifts_lr"].shape
    ii, jj = d1["edge_index_lr"]
    pos = d1["positions"]
    lengths = torch.linalg.norm(pos[jj] - pos[ii] + d1["shifts_lr"], dim=1)
    assert (lengths < w.r_lr).all()


def test_skin_is_forwarded_to_neighbor_config():
    """``skin`` reaches NeighborConfig so make_neighbor_hooks uses it."""
    w = NVAlchemiSO3LR(_make_so3lr(), skin=0.7)
    assert w.model_config.neighbor_config.skin == pytest.approx(0.7)
    (hook,) = w.make_neighbor_hooks()
    assert hook.skin == pytest.approx(0.7)
    assert NVAlchemiSO3LR(_make_so3lr()).model_config.neighbor_config.skin == 0


def test_total_charge_is_forwarded():
    """atoms.info['charge'] reaches the model like in TorchkratesCalculator."""
    model = _make_so3lr(electrostatics_bool=True)
    w = NVAlchemiSO3LR(model).to(DEVICE)
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    neutral = w(_make_batch_with_nl(w, atoms))["energy"]
    atoms.info["charge"] = 1
    charged = w(_make_batch_with_nl(w, atoms))["energy"]
    assert not torch.allclose(neutral, charged)
    atoms.calc = _parity_calc(model)
    assert charged.item() == pytest.approx(
        atoms.get_potential_energy(), abs=1e-4
    )


def test_nvt_outputs_skip_stress():
    """With stress not active, stress is not requested from the model."""
    model = _make_so3lr(electrostatics_bool=False)
    w = NVAlchemiSO3LR(model).to(DEVICE)
    w.model_config.active_outputs = {"energy", "forces"}
    calls = []
    orig = w.model.forward
    w.model.forward = lambda *a, **k: calls.append(k) or orig(*a, **k)
    out = w(_make_batch_with_nl(w, bulk("NaCl", "rocksalt", a=5.64)))
    assert calls[-1]["compute_stress"] is False
    assert out.get("stress") is None
    assert out["forces"] is not None


def test_energy_only_mode_keeps_graph_to_positions():
    """Autograd pipeline groups differentiate energy w.r.t. data.positions;
    the wrapper must not clone/detach them."""
    model = _make_so3lr(electrostatics_bool=False)
    w = NVAlchemiSO3LR(model).to(DEVICE)
    w.model_config.active_outputs = {"energy"}
    batch = _make_batch_with_nl(w, bulk("NaCl", "rocksalt", a=5.64))
    batch["positions"] = batch.positions.detach().requires_grad_(True)
    out = w(batch)
    # compute_force=False path: no forces/stress in the output.
    assert out.get("forces") is None
    assert out.get("stress") is None
    e = out["energy"].sum()
    (g,) = torch.autograd.grad(e, batch.positions)
    assert torch.isfinite(g).all()


def test_multi_graph_per_graph_charge_matches_calculator():
    """A 2-graph batch with charges 0 and 1 gives per-graph energies equal
    to the calculator run on each system alone (atoms.info['charge'])."""
    model = _make_so3lr(electrostatics_bool=True)
    w = NVAlchemiSO3LR(model).to(DEVICE)
    a1 = bulk("NaCl", "rocksalt", a=5.6402)
    a2 = bulk("NaCl", "rocksalt", a=5.8000)
    a1.info["charge"] = 0
    a2.info["charge"] = 1
    out = w(_make_batch_with_nl_multi(w, [a1, a2]))
    assert out["energy"].shape == (2, 1)

    # Charge must matter for the check to be meaningful.
    a2_neutral = a2.copy()
    a2_neutral.info["charge"] = 0
    e2_neutral = w(_make_batch_with_nl(w, a2_neutral))["energy"]
    assert not torch.allclose(out["energy"][1], e2_neutral[0])

    for i, atoms in enumerate((a1, a2)):
        atoms.calc = _parity_calc(model)
        assert out["energy"][i].item() == pytest.approx(
            atoms.get_potential_energy(), abs=1e-5
        )


@pytest.mark.parametrize(
    "lr_kwargs",
    [
        dict(electrostatics_bool=True),  # erf-Coulomb
        dict(electrostatics_bool=True, dispersion_bool=True),
        dict(electrostatics_bool=False, dispersion_bool=True),
        dict(electrostatics_bool=True, zbl_bool=True),
    ],
    ids=["coulomb", "coulomb+disp", "disp", "coulomb+zbl"],
)
@pytest.mark.parametrize("periodic", [True, False], ids=["pbc", "molecule"])
def test_lr_terms_match_calculator(lr_kwargs, periodic):
    """Real-space LR terms through the wrapper == ASE calculator path
    (energy, forces, and stress for PBC), with a nonzero skin."""
    pytest.importorskip("matscipy", reason="matscipy needed for neighbor list")
    model = _make_so3lr(**lr_kwargs)
    w = NVAlchemiSO3LR(model, skin=0.5).to(DEVICE)
    if periodic:
        atoms = bulk("NaCl", "rocksalt", a=5.64).repeat(2)
        atoms.rattle(0.05, seed=1)
    else:
        # Non-periodic 16-atom NaCl cluster: has pairs in (r_sr, r_lr)
        # (LR-only edges) and in [r_lr, r_lr + skin) (skin shell), unlike
        # a single small molecule where every pair is inside r_sr.
        atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True).repeat((2, 1, 1))
        atoms.pbc = False
        atoms.center(vacuum=6.0)
        atoms.rattle(0.05, seed=1)
    batch = _make_batch_with_nl(w, atoms, cutoff=w.r_lr + 0.5)
    # Guard against the case silently becoming vacuous: the built list
    # must contain LR-only edges and skin-shell edges.
    d = w.adapt_input(batch)
    n_all = batch.neighbor_list.shape[0]
    n_lr = d["edge_index_lr"].shape[1]
    n_sr = d["edge_index"].shape[1]
    assert n_lr > n_sr, "no LR-only edges (r_sr <= r < r_lr)"
    assert n_all > n_lr, "no skin-shell edges (r_lr <= r < r_lr + skin)"

    out = w(batch)
    atoms.calc = _parity_calc(model, compute_stress=periodic)
    assert out["energy"].item() == pytest.approx(
        atoms.get_potential_energy(), abs=1e-5
    )
    f_ref = atoms.get_forces()
    # Forces must be well above atol, else the force check is vacuous.
    assert np.abs(f_ref).max() > 1e-3
    np.testing.assert_allclose(
        out["forces"].detach().numpy(), f_ref, atol=1e-5
    )
    if periodic:
        from ase.stress import voigt_6_to_full_3x3_stress

        np.testing.assert_allclose(
            out["stress"].detach().numpy()[0],
            voigt_6_to_full_3x3_stress(atoms.get_stress()),
            atol=1e-6,
        )


# ---------------------------------------------------------------------------
# Helper: build AtomicData with neighbor list set, then batch
# ---------------------------------------------------------------------------


def _make_data_with_nl(
    wrapper: NVAlchemiSO3LR, atoms, cutoff: float | None = None
) -> AtomicData:
    """Build a single nvalchemi AtomicData with a pre-built neighbor list.

    ``cutoff`` defaults to ``wrapper.r_lr``; pass a larger value to emulate
    the ``cutoff + skin`` list that ``NeighborListHook`` builds.
    """
    from nvalchemiops.torch.neighbors.cell_list import cell_list

    pos = torch.tensor(atoms.positions, dtype=DTYPE)
    cell_np = np.array(atoms.cell)
    cell_t = torch.tensor(cell_np, dtype=DTYPE).unsqueeze(0)  # [1, 3, 3]
    pbc_t = torch.tensor(atoms.pbc, dtype=torch.bool).unsqueeze(0)  # [1, 3]

    nm, _num_nb, nm_shifts = cell_list(
        pos,
        wrapper.r_lr if cutoff is None else cutoff,
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


def _make_batch_with_nl(
    wrapper: NVAlchemiSO3LR, atoms, cutoff: float | None = None
) -> Batch:
    """Build a single-graph nvalchemi Batch with a pre-built neighbor list."""
    return Batch.from_data_list(
        [_make_data_with_nl(wrapper, atoms, cutoff=cutoff)]
    )


def _make_batch_with_nl_multi(
    wrapper: NVAlchemiSO3LR, atoms_list, cutoff: float | None = None
) -> Batch:
    """Build a multi-graph Batch (one entry per Atoms) with neighbor lists."""
    return Batch.from_data_list(
        [
            _make_data_with_nl(wrapper, atoms, cutoff=cutoff)
            for atoms in atoms_list
        ]
    )
