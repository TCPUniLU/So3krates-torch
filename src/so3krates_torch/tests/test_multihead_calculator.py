"""Tests for MultiHeadSO3LRCalculator (ASE calculator for models with
several independent energy output heads, ``MultiHeadSO3LR``)."""

import numpy as np
import pytest
import torch
from ase.build import bulk

from so3krates_torch.calculator.so3 import MultiHeadSO3LRCalculator
from so3krates_torch.modules.models import MultiHeadSO3LR

DTYPE = torch.float64
NUM_OUTPUT_HEADS = 3


def _make_multihead(long_range: bool = True, **extra) -> MultiHeadSO3LR:
    """Tiny MultiHeadSO3LR (settings as in test_nvalchemi_so3._make_so3lr).

    ``long_range`` switches on electrostatics (with partial charges and
    dipole) and dispersion (with Hirshfeld ratios).
    """
    settings = dict(
        num_output_heads=NUM_OUTPUT_HEADS,
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
        electrostatic_energy_bool=long_range,
        output_partial_charges=long_range,
        dispersion_energy_bool=long_range,
        dispersion_energy_cutoff_lr_damping=1.0 if long_range else None,
        zbl_repulsion_bool=False,
    )
    settings.update(extra)
    model = MultiHeadSO3LR(**settings).to(dtype=DTYPE)
    model.eval()
    return model


def _nacl():
    atoms = bulk("NaCl", "rocksalt", a=5.64, cubic=True)  # 8 atoms
    atoms.rattle(0.05, seed=1)
    return atoms


def _calculate(model, atoms):
    calc = MultiHeadSO3LRCalculator(
        model=model,
        device="cpu",
        default_dtype="float64",
        compute_stress=True,
        r_max_lr=6.0,
    )
    atoms.calc = calc
    atoms.get_potential_energy()
    return calc


def _reference(model, calc, atoms):
    """Raw per-head model output on the calculator's own input batch."""
    batch = calc._atoms_to_batch(atoms)
    batch.positions.requires_grad_(True)
    out = model(batch.to_dict(), compute_stress=True)
    return {k: v.detach() for k, v in out.items() if torch.is_tensor(v)}


@pytest.fixture
def mh_model():
    return _make_multihead()


def test_shared_outputs_are_per_atom(mh_model):
    atoms = _nacl()
    calc = _calculate(mh_model, atoms)
    ref = _reference(mh_model, calc, atoms)
    res = calc.results

    assert res["partial_charges"].shape == (8,)
    np.testing.assert_allclose(
        res["partial_charges"], ref["partial_charges"].numpy(), atol=1e-10
    )
    assert res["hirshfeld_ratios"].shape == (8,)
    np.testing.assert_allclose(
        res["hirshfeld_ratios"], ref["hirshfeld_ratios"].numpy(), atol=1e-10
    )
    assert res["dipole"].shape == (3,)
    np.testing.assert_allclose(
        res["dipole"], ref["dipole"][0].numpy(), atol=1e-10
    )
    for key, shape in (
        ("partial_charges_var", (8,)),
        ("hirshfeld_ratios_var", (8,)),
        ("dipole_var", (3,)),
    ):
        assert res[key].shape == shape, key
        assert np.all(res[key] == 0.0), key


def test_energy_forces_are_head_means(mh_model):
    atoms = _nacl()
    calc = _calculate(mh_model, atoms)
    ref = _reference(mh_model, calc, atoms)
    res = calc.results
    head_energies = ref["energy"][:, 0]  # (H,)

    assert res["energies"].shape == (NUM_OUTPUT_HEADS, 1)
    np.testing.assert_allclose(
        res["energy"], head_energies.mean().item(), atol=1e-10
    )
    np.testing.assert_allclose(
        res["energy_var"],
        head_energies.var(unbiased=False).item(),
        atol=1e-10,
    )
    np.testing.assert_allclose(
        res["forces"], ref["forces"].mean(0).numpy(), atol=1e-10
    )
    np.testing.assert_allclose(
        res["forces_comm"], ref["forces"].numpy(), atol=1e-10
    )


def test_shared_outputs_absent_keep_shapes():
    atoms = _nacl()
    calc = _calculate(_make_multihead(long_range=False), atoms)
    res = calc.results

    assert res["dipole"].shape == (3,)
    assert np.all(res["dipole"] == 0.0)
    assert res["partial_charges"].shape == (8,)
    assert np.all(res["partial_charges"] == 0.0)


def test_num_heads_is_output_heads(mh_model):
    # 2 attention heads (num_heads=2), 3 output heads.
    calc = _calculate(mh_model, _nacl())
    assert calc.num_heads == NUM_OUTPUT_HEADS
