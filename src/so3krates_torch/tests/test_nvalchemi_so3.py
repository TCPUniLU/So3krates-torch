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
    **extra,
) -> SO3LR:
    """Minimal SO3LR model for testing.

    ``electrostatics_bool`` defaults to ``use_pme`` (so a short-range-only
    model has electrostatics off); pass it explicitly to enable the
    real-space erf-Coulomb electrostatics with ``use_pme=False``.
    ``dispersion_bool`` / ``zbl_bool`` switch on the real-space dispersion
    and ZBL repulsion terms (off by default).  ``extra`` is forwarded to
    ``SO3LR`` (e.g. ``use_spin_embed=True``).
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
        # cutoff_lr is set (SO3LR always passes r_max_lr). The value 1.0 is
        # an arbitrary test value, not a physical parameter.
        dispersion_energy_cutoff_lr_damping=1.0 if dispersion_bool else None,
        zbl_repulsion_bool=zbl_bool,
        **extra,
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
    pipe = build_nvalchemi_pme_model(model, alpha=0.6)
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
    from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin

    from so3krates_torch.calculator.nvalchemi_so3 import (
        build_nvalchemi_pme_model,
    )

    pipe = build_nvalchemi_pme_model(
        _make_so3lr(electrostatics_bool=True), alpha=0.6
    )

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

    integ = NVTLangevin(
        model=pipe,
        dt=0.5,
        temperature=300.0,
        friction=1e-2,
        n_steps=1,
        hooks=pipe.make_neighbor_hooks(),
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
# Dtype follows the model; active-output narrowing; spin
# ---------------------------------------------------------------------------


def test_wrapper_dtype_follows_model_after_to():
    """A float32 model wrapped and then moved to float64 runs in float64
    (no silent float32 cast of positions/shifts/cell) and matches a
    float64 reference built from the same weights."""
    import copy

    model32 = _make_so3lr(electrostatics_bool=True).float()
    ref = NVAlchemiSO3LR(copy.deepcopy(model32).double())
    w = NVAlchemiSO3LR(model32).to(torch.float64)
    atoms = bulk("NaCl", "rocksalt", a=5.64)
    atoms.rattle(0.05, seed=2)
    batch = _make_batch_with_nl(w, atoms)
    d = w.adapt_input(batch)
    assert d["positions"].dtype == torch.float64
    assert d["shifts_lr"].dtype == torch.float64
    assert d["cell"].dtype == torch.float64
    assert d["node_attrs"].dtype == torch.float64
    out, out_ref = w(batch), ref(batch)
    assert out["energy"].dtype == torch.float64
    torch.testing.assert_close(
        out["energy"], out_ref["energy"], rtol=0, atol=1e-12
    )
    torch.testing.assert_close(
        out["forces"], out_ref["forces"], rtol=0, atol=1e-12
    )


def test_wrapper_float64_model_to_float32_does_not_crash():
    """The reverse (float64 model -> float32) must run, not raise
    'mat1 and mat2 must have the same dtype'."""
    model = _make_so3lr(electrostatics_bool=True)
    e64 = NVAlchemiSO3LR(model)(
        _make_batch_with_nl(NVAlchemiSO3LR(model), _small_nacl())
    )["energy"]
    w = NVAlchemiSO3LR(model).to(torch.float32)
    out = w(_make_batch_with_nl(w, _small_nacl()))
    assert out["energy"].dtype == torch.float32
    assert out["energy"].item() == pytest.approx(e64.item(), abs=1e-3)


def test_set_active_outputs_bare_wrapper():
    from so3krates_torch.calculator.nvalchemi_so3 import set_active_outputs

    w = NVAlchemiSO3LR(_make_so3lr())
    set_active_outputs(w, {"energy", "forces"})
    assert w.model_config.active_outputs == {"energy", "forces"}
    set_active_outputs(w, {"energy", "forces", "stress"})
    assert w.model_config.active_outputs == {"energy", "forces", "stress"}


def test_set_active_outputs_pme_pipeline_nvt():
    """NVT narrowing on the native-PME pipeline: the pipeline reports only
    energy and forces, and (autograd group) the short-range step computes
    neither forces nor stress itself; forces come from the pipeline's
    autograd on the summed energy."""
    from so3krates_torch.calculator.nvalchemi_so3 import (
        build_nvalchemi_pme_model,
        set_active_outputs,
    )

    pipe = build_nvalchemi_pme_model(
        _make_so3lr(electrostatics_bool=True), alpha=0.6
    )
    set_active_outputs(pipe, {"energy", "forces"})
    assert pipe.model_config.active_outputs == {"energy", "forces"}
    (group,) = pipe.groups
    sr, _pme = (step.model for step in group.steps)

    calls = []
    orig = sr.model.forward
    sr.model.forward = lambda *a, **k: calls.append(k) or orig(*a, **k)
    # One NVT step through the pipeline (PME needs the hook-built list).
    from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin

    from so3krates_torch.cli.run_md_nvalchemi import atoms_to_data

    batch = Batch.from_data_list(
        [atoms_to_data(_small_nacl(), str(DEVICE), DTYPE)]
    )
    out = NVTLangevin(
        model=pipe,
        dt=0.5,
        temperature=300.0,
        friction=1e-2,
        n_steps=1,
        hooks=pipe.make_neighbor_hooks(),
    ).run(batch)
    assert calls and calls[-1]["compute_stress"] is False
    assert calls[-1]["compute_force"] is False
    assert torch.isfinite(out.forces).all()
    assert torch.isfinite(out.energy).all()


def test_total_spin_is_forwarded():
    """atoms.info['total_spin'] (calculator key) reaches the model via
    atoms_to_data and matches TorchkratesCalculator."""
    from so3krates_torch.cli.run_md_nvalchemi import atoms_to_data

    model = _make_so3lr(electrostatics_bool=True, use_spin_embed=True)
    w = NVAlchemiSO3LR(model).to(DEVICE)
    atoms = bulk("NaCl", "rocksalt", a=5.64)

    def _energy(at):
        nl = _make_data_with_nl(w, at)
        data = atoms_to_data(at, str(DEVICE), DTYPE).model_copy(
            update={
                "neighbor_list": nl.neighbor_list,
                "neighbor_list_shifts": nl.neighbor_list_shifts,
            }
        )
        return data, w(Batch.from_data_list([data]))["energy"]

    _, e0 = _energy(atoms)
    atoms.info["total_spin"] = 1.0
    data, e1 = _energy(atoms)
    assert data.spin is not None
    assert data.spin.shape == (1, 1)
    assert data.spin.item() == pytest.approx(1.0)
    assert not torch.allclose(e0, e1), "model energy must depend on spin"
    atoms.calc = _parity_calc(model)
    assert e1.item() == pytest.approx(atoms.get_potential_energy(), abs=1e-5)


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


# ---------------------------------------------------------------------------
# Native PME as an autograd pipeline (phase 2)
# ---------------------------------------------------------------------------

_MADELUNG_NACL = 1.747564594633
_ALL_COMPONENTS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _rocksalt(a: float = 5.64):
    """16-atom NaCl cell, primitive (rhombohedral, non-orthogonal) lattice,
    rattled by a small amount: finite differences are only trustworthy for
    distortions of this size."""
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=a).repeat(2)
    atoms.rattle(0.05, seed=1)
    return atoms


def _pme_pipeline(
    alpha: float = 0.6, outputs=None, geometry_charges: bool = False
):
    """Native-PME pipeline on the tiny test model.  ``alpha`` is explicit
    because the auto-estimated Ewald alpha is under-converged at the test
    model's 6 A cutoff, and (when auto) is re-estimated whenever the cell
    changes, which makes strain finite differences noisy.  ``outputs`` (if
    given) is applied through ``set_active_outputs``.

    ``geometry_charges=True`` adds ``0.3 * sin(x + 1.3 y + 0.7 z)`` (minus its per-graph mean, so the total charge is conserved) to the emitted partial
    charges.  The random test model's charges depend only weakly on the
    geometry, which hides the charge-response terms (dq/dr, dq/d strain);
    this makes them large so tests can tell a conservative pipeline from
    one that ignores them."""
    from so3krates_torch.calculator.nvalchemi_so3 import (
        build_nvalchemi_pme_model,
        set_active_outputs,
    )

    # Dispersion and ZBL make the short-range model's own forces and stress
    # non-negligible; the random network alone contributes ~1e-8 and could
    # not tell a pipeline that mishandles the strained cell from a correct one.
    pipe = build_nvalchemi_pme_model(
        _make_so3lr(
            electrostatics_bool=True, dispersion_bool=True, zbl_bool=True
        ),
        alpha=alpha,
    )
    if geometry_charges:
        (group,) = pipe.groups
        inner = group.steps[0].model.model
        orig_forward = inner.forward

        def forward(inputs, *args, **kwargs):
            out = orig_forward(inputs, *args, **kwargs)
            pos = inputs["positions"]
            phase = pos @ pos.new_tensor([1.0, 1.3, 0.7])
            delta = 0.3 * torch.sin(phase)
            # Remove the per-graph mean so the perturbation conserves the
            # total charge (a charged cell would add PME's neutralising
            # background, which real models never produce).
            graph = inputs["batch"].long()
            count = torch.bincount(graph).to(delta.dtype)
            total = torch.zeros_like(count).index_add(0, graph, delta)
            delta = delta - (total / count)[graph]
            out["partial_charges"] = out["partial_charges"] + delta.reshape_as(
                out["partial_charges"]
            )
            return out

        inner.forward = forward
    if outputs is not None:
        set_active_outputs(pipe, outputs)
    return pipe


def _build_neighbor_lists(pipe, batch):
    from nvalchemi.dynamics.base import DynamicsStage
    from nvalchemi.hooks import HookContext

    for hook in pipe.make_neighbor_hooks():
        hook(HookContext(batch=batch), DynamicsStage.BEFORE_COMPUTE)


def _pipeline_eval(pipe, atoms):
    """Evaluate a pipeline outside an integrator: build the batch, let the
    pipeline's own neighbor hooks fill the lists, run the forward pass."""
    from so3krates_torch.cli.run_md_nvalchemi import atoms_to_data

    batch = Batch.from_data_list([atoms_to_data(atoms, str(DEVICE), DTYPE)])
    _build_neighbor_lists(pipe, batch)
    return pipe(batch), batch


def _pipeline_energy(pipe, atoms) -> float:
    return _pipeline_eval(pipe, atoms)[0]["energy"].sum().item()


def _stress_vs_strain_fd(pipe, atoms, i: int, j: int, h: float = 1e-4):
    """(analytic stress[i, j], strain-derivative dE/d(eps_ij) / V).

    Diagonal strain: eps[i, i] = h.  Shear: eps[i, j] = eps[j, i] = h / 2,
    so that dE = V * sigma_ij * h.  Positions scale with the cell."""
    stress = _pipeline_eval(pipe, atoms)[0]["stress"][0].detach()
    eps = np.zeros((3, 3))
    if i == j:
        eps[i, i] = h
    else:
        eps[i, j] = eps[j, i] = h / 2

    def _strained(e):
        out = atoms.copy()
        out.set_cell(atoms.cell.array @ (np.eye(3) + e).T, scale_atoms=True)
        return out

    de = _pipeline_energy(pipe, _strained(eps)) - _pipeline_energy(
        pipe, _strained(-eps)
    )
    return stress[i, j].item(), de / (2 * h) / atoms.get_volume()


def test_pme_pipeline_forces_match_finite_difference():
    """Forces equal -dE/dr (central FD), including the charge response
    dq/dr (large here because the charges are made geometry-dependent)."""
    pipe = _pme_pipeline(
        outputs={"energy", "forces", "stress"}, geometry_charges=True
    )
    atoms = _rocksalt()
    forces = _pipeline_eval(pipe, atoms)[0]["forces"].detach()
    h = 1e-4
    for i in (0, 3):
        for k in range(3):
            plus, minus = atoms.copy(), atoms.copy()
            plus.positions[i, k] += h
            minus.positions[i, k] -= h
            fd = -(
                _pipeline_energy(pipe, plus) - _pipeline_energy(pipe, minus)
            ) / (2 * h)
            # |F| reaches ~2 eV/A here, where the FD noise is ~3e-5.
            assert forces[i, k].item() == pytest.approx(
                fd, rel=1e-4, abs=1e-4
            ), (i, k)


@pytest.mark.parametrize("ij", _ALL_COMPONENTS)
def test_pme_pipeline_stress_matches_strain_derivative(ij):
    """All six stress components equal the strain derivative of the total
    energy (this pins the sign convention and the Coulomb virial, including
    the charge response to strain)."""
    pipe = _pme_pipeline(
        outputs={"energy", "forces", "stress"}, geometry_charges=True
    )
    analytic, fd = _stress_vs_strain_fd(pipe, _rocksalt(), *ij)
    assert analytic == pytest.approx(fd, rel=1e-3, abs=1e-7), ij


def _bare_pme_energy(atoms, charges, **pme_kwargs) -> float:
    """Total energy of a bare PMEModelWrapper for fixed ``charges``."""
    from nvalchemi.models.pme import PMEModelWrapper

    n = len(atoms)
    data = AtomicData.from_atoms(
        atoms, device=str(DEVICE), dtype=DTYPE
    ).model_copy(
        update={
            "charges": charges.to(DTYPE),
            "forces": torch.zeros(n, 3, dtype=DTYPE),
            "energy": torch.zeros(1, 1, dtype=DTYPE),
            "stress": torch.zeros(1, 3, 3, dtype=DTYPE),
        }
    )
    from so3krates_torch.calculator.nvalchemi_so3 import _KE_EV_ANG

    # Upstream's default constant is 14.3996; the builder passes the
    # model's ke, so compare like with like.
    pme_kwargs.setdefault("coulomb_constant", _KE_EV_ANG)
    pme = PMEModelWrapper(hybrid_forces=False, **pme_kwargs)
    pme.set_config("active_outputs", {"energy", "forces", "stress"})
    batch = Batch.from_data_list([data])
    _build_neighbor_lists(pme, batch)
    return pme(batch)["energy"].sum().item()


def _derived_alpha(cutoff: float, accuracy: float = 1e-6) -> float:
    import math

    return math.sqrt(-math.log(accuracy)) / cutoff


def _madelung_error(repeat: int = 2, **pme_kwargs) -> float:
    """Relative error of the PME energy per NaCl ion pair against the
    analytical Madelung value -M * ke / r0 (r0 = a / 2, +/-1 charges)."""
    from so3krates_torch.calculator.nvalchemi_so3 import _KE_EV_ANG

    a = 5.64
    atoms = bulk("NaCl", crystalstructure="rocksalt", a=a, cubic=True)
    atoms = atoms.repeat(repeat)
    q = torch.tensor(
        [1.0 if s == "Na" else -1.0 for s in atoms.get_chemical_symbols()]
    )
    per_pair = _bare_pme_energy(atoms, q, **pme_kwargs) / (len(atoms) / 2)
    ref = -_MADELUNG_NACL * _KE_EV_ANG / (a / 2)
    return abs(per_pair / ref - 1.0)


def test_pme_wrapper_energy_matches_madelung():
    """Guard on the upstream PME and our Coulomb constant: with a converged
    alpha even a 6 A cutoff reproduces the Madelung energy."""
    err = _madelung_error(cutoff=6.0, alpha=_derived_alpha(6.0))
    assert err < 1e-4


@pytest.mark.parametrize("repeat", [2, 4])
def test_pme_wrapper_derived_alpha_converged_for_larger_cells(repeat):
    """The alpha the builder derives from the cutoff, sqrt(-ln acc) / rc,
    keeps the real-space sum converged as the cell grows (measured at 12 A:
    2e-7 for 64 and 2e-7 for 512 atoms, 4e-7 for 4096).  Upstream's own
    auto alpha shrinks like N**(-1/6) and is off by 3e-4 at 512 and 1.5e-2
    at 4096 atoms, which is why the builder does not use it."""
    assert (
        _madelung_error(cutoff=12.0, alpha=_derived_alpha(12.0), repeat=repeat)
        < 1e-5
    )


def test_pme_pipeline_energy_is_sr_plus_pme():
    """The pipeline energy is the short-range energy plus the bare PME
    energy of the charges the short-range step emits (wiring + summation)."""
    from so3krates_torch.calculator.nvalchemi_so3 import (
        build_nvalchemi_pme_model,
    )

    model = _make_so3lr(electrostatics_bool=True)
    pipe = build_nvalchemi_pme_model(model, alpha=0.6)
    sr = NVAlchemiSO3LR(model, electrostatics="nvalchemi")
    atoms = _rocksalt()
    sr_out = sr(_make_batch_with_nl(sr, atoms))
    pme_energy = _bare_pme_energy(
        atoms,
        sr_out["charges"].detach(),
        cutoff=float(model.r_max_lr),
        alpha=0.6,
    )
    expected = sr_out["energy"].sum().item() + pme_energy
    assert _pipeline_energy(pipe, atoms) == pytest.approx(expected, rel=1e-6)


def test_pme_pipeline_sr_step_is_energy_only():
    """In the autograd group the short-range step returns energies only;
    forces and stress come from the pipeline's autograd."""
    pipe = _pme_pipeline(outputs={"energy", "forces", "stress"})
    (group,) = pipe.groups
    sr, _pme = (step.model for step in group.steps)
    calls = []
    orig = sr.model.forward
    sr.model.forward = lambda *a, **k: calls.append(k) or orig(*a, **k)
    out, _ = _pipeline_eval(pipe, _rocksalt())
    assert calls
    assert calls[-1]["compute_force"] is False
    assert calls[-1]["compute_stress"] is False
    assert torch.isfinite(out["forces"]).all()
    assert torch.isfinite(out["stress"]).all()


def test_pme_pipeline_two_graph_batch_finite():
    """Two crystals with different cells in one batch: finite outputs for
    both (upstream shares one mean alpha per batch only when alpha is
    auto; alpha is explicit here)."""
    from so3krates_torch.cli.run_md_nvalchemi import atoms_to_data

    pipe = _pme_pipeline(outputs={"energy", "forces", "stress"})
    frames = [_rocksalt(5.64), _rocksalt(5.9)]
    batch = Batch.from_data_list(
        [atoms_to_data(a, str(DEVICE), DTYPE) for a in frames]
    )
    _build_neighbor_lists(pipe, batch)
    out = pipe(batch)
    assert out["energy"].shape == (2, 1)
    assert out["forces"].shape == (32, 3)
    assert out["stress"].shape == (2, 3, 3)
    for key in ("energy", "forces", "stress"):
        assert torch.isfinite(out[key]).all(), key


def _alpha_warnings(*args):
    import warnings

    from so3krates_torch.calculator.nvalchemi_so3 import (
        _warn_if_alpha_unconverged,
    )

    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        _warn_if_alpha_unconverged(*args)
    return [w for w in rec if issubclass(w.category, UserWarning)]


def test_warn_if_alpha_unconverged_rule():
    """Only an explicit alpha is judged: the real-space truncation error
    is erfc(alpha * cutoff), which must stay below 10x the accuracy."""
    (w,) = _alpha_warnings(6.0, 0.3, 1e-6)
    assert "--pme-alpha" in str(w.message)
    assert "0.62" in str(w.message)  # sqrt(-ln 1e-6) / 6
    assert _alpha_warnings(12.0, 0.15, 1e-6)
    assert not _alpha_warnings(6.0, 0.6, 1e-6)
    assert not _alpha_warnings(12.0, 0.31, 1e-6)
    assert not _alpha_warnings(6.0, None, 1e-6)  # derived, not judged
    assert _alpha_warnings(12.0, 0.31, 1e-9)  # the rule scales with accuracy


def test_build_nvalchemi_pme_model_alpha():
    """alpha=None is derived from the cutoff and accuracy (not left to
    upstream's size-dependent estimate); an explicit but too small alpha
    warns; a sufficient one does not."""
    import math

    from so3krates_torch.calculator.nvalchemi_so3 import (
        build_nvalchemi_pme_model,
    )

    def _pme_alpha(pipe):
        (group,) = pipe.groups
        return group.steps[1].model.alpha

    def _build(**kw):
        return build_nvalchemi_pme_model(
            _make_so3lr(electrostatics_bool=True), **kw
        )

    with pytest.warns(UserWarning, match="pme-alpha"):
        _build(alpha=0.3)
    for kw, expected in (
        ({}, math.sqrt(-math.log(1e-6)) / 6.0),
        ({"accuracy": 1e-8}, math.sqrt(-math.log(1e-8)) / 6.0),
        ({"alpha": 0.6}, 0.6),
    ):
        import warnings

        with warnings.catch_warnings(record=True) as rec:
            warnings.simplefilter("always")
            pipe = _build(**kw)
        assert not [w for w in rec if "pme-alpha" in str(w.message)], kw
        assert _pme_alpha(pipe) == pytest.approx(expected), kw


def test_build_nvalchemi_pme_model_has_no_mesh_spacing():
    """nvalchemi-toolkit 0.2.0's PMEModelWrapper stores ``mesh_spacing`` but
    derives the mesh from its own accuracy-based estimate (the energy is
    identical to 12 digits for 0.3, 1.0 and 3.0 A), so the builder must not
    offer a knob that does nothing."""
    import inspect

    from so3krates_torch.calculator.nvalchemi_so3 import (
        build_nvalchemi_pme_model,
    )

    assert (
        "mesh_spacing"
        not in inspect.signature(build_nvalchemi_pme_model).parameters
    )


def _pme_step(pipe):
    (group,) = pipe.groups
    return group.steps[1].model


def test_build_nvalchemi_pme_model_forwards_mesh_dimensions():
    from so3krates_torch.calculator.nvalchemi_so3 import (
        build_nvalchemi_pme_model,
    )

    default = build_nvalchemi_pme_model(
        _make_so3lr(electrostatics_bool=True), alpha=0.6
    )
    assert _pme_step(default).mesh_dimensions is None  # upstream's estimate
    fixed = build_nvalchemi_pme_model(
        _make_so3lr(electrostatics_bool=True),
        alpha=0.6,
        mesh_dimensions=[16, 16, 16],
    )
    assert _pme_step(fixed).mesh_dimensions == (16, 16, 16)


@pytest.mark.parametrize("bad", [(16, 16), (0, 16, 16), (16, -2, 16), 16])
def test_build_nvalchemi_pme_model_rejects_bad_mesh_dimensions(bad):
    from so3krates_torch.calculator.nvalchemi_so3 import (
        build_nvalchemi_pme_model,
    )

    with pytest.raises(ValueError, match="mesh_dimensions"):
        build_nvalchemi_pme_model(
            _make_so3lr(electrostatics_bool=True),
            alpha=0.6,
            mesh_dimensions=bad,
        )


def test_mesh_dimensions_changes_pme_energy():
    """The FFT grid is the one reciprocal-space accuracy control that works
    in nvalchemi-toolkit 0.2.0: a coarse mesh and a fine one give clearly
    different energies for the same structure and weights."""
    from so3krates_torch.calculator.nvalchemi_so3 import (
        build_nvalchemi_pme_model,
    )

    model = _make_so3lr(electrostatics_bool=True)
    coarse = build_nvalchemi_pme_model(
        model, alpha=0.6, mesh_dimensions=(8, 8, 8)
    )
    fine = build_nvalchemi_pme_model(
        model, alpha=0.6, mesh_dimensions=(48, 48, 48)
    )
    atoms = _rocksalt()
    gap = abs(_pipeline_energy(coarse, atoms) - _pipeline_energy(fine, atoms))
    assert gap > 1e-3


def test_geometry_charges_keep_the_system_neutral():
    """The test-only charge perturbation must not add net charge: a charged
    periodic cell makes PME add a volume-dependent neutralising background
    that charge-conserving models never produce."""
    pipe = _pme_pipeline(geometry_charges=True)
    inner = pipe.groups[0].steps[0].model.model
    perturbed_forward = inner.forward
    totals = []

    def spy(inputs, *args, **kwargs):
        out = perturbed_forward(inputs, *args, **kwargs)
        totals.append(out["partial_charges"].detach().sum().item())
        return out

    inner.forward = spy
    _pipeline_eval(pipe, _rocksalt())
    assert totals
    assert abs(totals[-1]) < 1e-9


# ---------------------------------------------------------------------------
# Long-range settings the model must carry (or be given)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "kwargs", [{"electrostatics_bool": True}, {"dispersion_bool": True}]
)
def test_wrapper_requires_lr_cutoff_for_long_range_terms(kwargs):
    model = _make_so3lr(**kwargs)
    model.r_max_lr = None
    with pytest.raises(ValueError, match="r_max_lr") as exc:
        NVAlchemiSO3LR(model)
    assert "--r-max-lr" in str(exc.value)


def test_wrapper_requires_dispersion_damping_when_dispersion_is_on():
    model = _make_so3lr(dispersion_bool=True)
    model.dispersion_energy_cutoff_lr_damping = None
    with pytest.raises(
        ValueError, match="dispersion_energy_cutoff_lr_damping"
    ) as exc:
        NVAlchemiSO3LR(model)
    assert "--dispersion-energy-cutoff-lr-damping" in str(exc.value)


def test_wrapper_without_long_range_terms_needs_no_lr_cutoff():
    model = _make_so3lr()  # electrostatics and dispersion off
    model.r_max_lr = None
    wrapper = NVAlchemiSO3LR(model)
    assert not wrapper.has_lr


@pytest.mark.parametrize("n_graphs", [1, 2])
def test_adapt_input_shift_backward_has_no_cell_index_put(n_graphs):
    """The strained cell (native-PME NPT) reaches ``adapt_input`` with
    ``requires_grad``; gathering it per edge with ``cell[idx]`` would
    accumulate every edge into the cell rows via ``index_put_`` (very slow on
    CUDA).  Boolean-mask selections (unique indices) are fine, so only an
    ``index_put_`` onto a cell-shaped ``[G, 3, 3]`` tensor counts."""
    atoms_list = [_small_nacl().repeat(2) for _ in range(n_graphs)]
    wrapper = NVAlchemiSO3LR(_make_so3lr(electrostatics_bool=True))
    batch = _make_batch_with_nl_multi(wrapper, atoms_list)
    batch.cell.requires_grad_(True)
    shifts = wrapper.adapt_input(batch)["shifts"]
    assert shifts.requires_grad

    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU], record_shapes=True
    ) as prof:
        shifts.sum().backward()
    cell_shape = [n_graphs, 3, 3]
    onto_cell = [
        e
        for e in prof.key_averages(group_by_input_shape=True)
        if e.key in ("aten::_index_put_impl_", "aten::index_put_")
        and e.input_shapes
        and e.input_shapes[0] == cell_shape
    ]
    assert not onto_cell, onto_cell


# ---------------------------------------------------------------------------
# The model's own PME (use_pme=True: real-space residual + k-space) driven
# by the NVAlchemi wrapper.
# ---------------------------------------------------------------------------


def _pme_model():
    pytest.importorskip("torchpme", reason="torch-pme not installed")
    torch.manual_seed(0)
    return _make_so3lr(
        use_pme=True,
        output_partial_charges=True,
        dispersion_bool=True,
        # the random test model's charges barely depend on geometry
    )


def test_wrapper_model_pme_matches_ase_calculator():
    """Energy, forces and stress of a ``use_pme=True`` model through the
    NVAlchemi wrapper equal the ASE calculator's."""
    pytest.importorskip("matscipy", reason="matscipy needed for neighbor list")
    from ase.stress import voigt_6_to_full_3x3_stress

    atoms = _rocksalt()
    model = _pme_model()
    atoms.calc = TorchkratesCalculator(
        models=[model],
        device=str(DEVICE),
        dtype=DTYPE,
        r_max=model.r_max,
        r_max_lr=model.r_max_lr,
        compute_stress=True,
    )
    e_ref = atoms.get_potential_energy()
    f_ref = atoms.get_forces()
    s_ref = voigt_6_to_full_3x3_stress(atoms.get_stress())
    assert abs(s_ref).max() > 1e-8, "stress is ~zero; test vacuous"

    wrapper = NVAlchemiSO3LR(model)
    out = wrapper(_make_batch_with_nl(wrapper, atoms))
    assert abs(float(out["energy"].sum()) - e_ref) < 1e-6
    np.testing.assert_allclose(
        out["forces"].detach().numpy(), f_ref, atol=1e-6, rtol=1e-5
    )
    np.testing.assert_allclose(
        out["stress"][0].detach().numpy(), s_ref, atol=1e-7, rtol=1e-5
    )


@pytest.mark.parametrize("component", [(0, 0), (1, 2)])
def test_wrapper_model_pme_stress_matches_finite_difference(component):
    """sigma = (1/V) dE/d(strain) for the PME model: guards the k-space term
    seeing the strained positions and cell."""
    atoms = _rocksalt()
    wrapper = NVAlchemiSO3LR(_pme_model())
    stress = (
        wrapper(_make_batch_with_nl(wrapper, atoms))["stress"][0]
        .detach()
        .numpy()
    )

    def energy(eps):
        strained = atoms.copy()
        strain = np.eye(3) + eps
        strained.set_cell(atoms.cell.array @ strain, scale_atoms=True)
        out = wrapper(_make_batch_with_nl(wrapper, strained))
        return float(out["energy"].sum())

    i, j = component
    h = 1e-5
    d = np.zeros((3, 3))
    d[i, j] = d[j, i] = h if i != j else h
    fd = (energy(d) - energy(-d)) / (2 * h) / atoms.get_volume()
    if i != j:
        fd /= 2  # symmetric strain moves both off-diagonal entries
    assert fd == pytest.approx(stress[i, j], abs=2e-6, rel=1e-4)


# ---------------------------------------------------------------------------
# Multi-theory-level models
# ---------------------------------------------------------------------------


def _multi_head_model(num_theory_levels=3):
    torch.manual_seed(0)
    return _make_so3lr(
        electrostatics_bool=True, num_theory_levels=num_theory_levels
    )


def test_wrapper_multi_head_model_requires_theory_level():
    with pytest.raises(ValueError, match="theory_level"):
        NVAlchemiSO3LR(_multi_head_model())


@pytest.mark.parametrize("level", [-1, 3])
def test_wrapper_theory_level_out_of_range(level):
    with pytest.raises(ValueError, match="out of range"):
        NVAlchemiSO3LR(_multi_head_model(3), theory_level=level)


def test_wrapper_theory_level_selects_the_head():
    atoms = _rocksalt()
    model = _multi_head_model(3)
    energies = []
    for level in range(3):
        wrapper = NVAlchemiSO3LR(model, theory_level=level)
        out = wrapper(_make_batch_with_nl(wrapper, atoms))
        energies.append(float(out["energy"].sum()))
    assert len({round(e, 9) for e in energies}) == 3


@pytest.mark.parametrize("level", [0, 2])
def test_wrapper_theory_level_matches_ase_calculator(level):
    pytest.importorskip("matscipy", reason="matscipy needed for neighbor list")
    atoms = _small_nacl()
    model = _multi_head_model(3)
    atoms.calc = TorchkratesCalculator(
        models=[model],
        device=str(DEVICE),
        dtype=DTYPE,
        r_max=model.r_max,
        r_max_lr=model.r_max_lr,
        theory_level=level,
    )
    e_ref = atoms.get_potential_energy()
    wrapper = NVAlchemiSO3LR(model, theory_level=level)
    out = wrapper(_make_batch_with_nl(wrapper, atoms))
    assert float(out["energy"].sum()) == pytest.approx(e_ref, abs=1e-6)


def test_wrapper_single_head_model_needs_no_theory_level():
    wrapper = NVAlchemiSO3LR(_make_so3lr(electrostatics_bool=True))
    assert wrapper.theory_level is None
    assert "theory_level" not in wrapper.adapt_input(
        _make_batch_with_nl(wrapper, _small_nacl())
    )
