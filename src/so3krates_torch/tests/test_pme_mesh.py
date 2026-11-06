"""Tests for the PME mesh-size rule.

so3lr_dev's jax-pme fork picks the mesh as
``2**ceil(log2(2*ceil(L/h)+1))`` per cell axis; stock torch-pme omits the
inner ``ceil``. The torch PME wrappers use the JAX rule by default
(``pme_mesh_rule="jax"``) and keep the stock rule available
(``pme_mesh_rule="torchpme"``).
"""

import numpy as np
import pytest
import torch

torchpme = pytest.importorskip("torchpme")


from so3krates_torch.blocks.physical_potentials import (  # noqa: E402
    BOHR,
    PMEElectrostaticInteraction,
)
from so3krates_torch.blocks.pme_mesh import (  # noqa: E402
    JaxMeshPMECalculator,
    jax_ns_mesh,
    make_pme_calculator,
)

TESTED_TORCHPME_VERSION = "0.5.0"

CUBIC = np.diag([7.849, 7.849, 7.849])
ORTHO = np.diag([6.1, 7.3, 9.7])
TRICLINIC = np.array(
    [[7.2, 0.0, 0.0], [1.9, 6.8, 0.0], [-1.3, 2.2, 8.4]],
)
SPACINGS = [0.05, 0.1, 0.125, 0.2, 0.25, 0.3, 0.4, 0.5]


def _jax_mesh(cell: np.ndarray, spacing: float):
    jax = pytest.importorskip("jax")
    pytest.importorskip("jaxpme")
    from jaxpme.kspace import get_kgrid_mesh_shape

    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    return tuple(
        get_kgrid_mesh_shape(jnp.asarray(cell, dtype=jnp.float64), spacing)
    )


def test_torchpme_version_pinned():
    assert torchpme.__version__ == TESTED_TORCHPME_VERSION, (
        "JaxMeshPMECalculator._compute_kspace is a copy of torch-pme "
        f"{TESTED_TORCHPME_VERSION}'s PMECalculator._compute_kspace; "
        "re-verify the copy against the installed version, then update "
        "TESTED_TORCHPME_VERSION."
    )


def test_stock_torchpme_rule_differs_from_jax_on_boundary_cell():
    """Documents the bug being fixed: 7.849 A at 0.25 A gives a 64^3
    mesh in torch-pme and a 128^3 mesh in jax-pme."""
    from torchpme.lib.kvectors import get_ns_mesh

    cell = torch.tensor(CUBIC, dtype=torch.float64)
    assert tuple(get_ns_mesh(cell, 0.25).tolist()) == (64, 64, 64)
    assert _jax_mesh(CUBIC, 0.25) == (128, 128, 128)
    assert tuple(jax_ns_mesh(cell, 0.25).tolist()) == (128, 128, 128)


@pytest.mark.parametrize("cell", [CUBIC, ORTHO, TRICLINIC])
@pytest.mark.parametrize("spacing", SPACINGS)
def test_jax_ns_mesh_matches_jaxpme(cell, spacing):
    c = torch.tensor(cell, dtype=torch.float64)
    ours = jax_ns_mesh(c, spacing)
    assert ours.shape == (3,)
    assert ours.dtype == torch.long
    assert tuple(ours.tolist()) == _jax_mesh(cell, spacing)


@pytest.mark.parametrize("scale", [1.0, 1.0 / BOHR])
def test_jax_ns_mesh_exact_integer_ratio(scale):
    """L/h exactly an integer (8.0 / 0.25 = 32): both rules must agree
    in Angstrom and after the Angstrom -> Bohr conversion the dispersion
    path applies."""
    cell = np.diag([8.0, 8.0, 8.0]) * scale
    spacing = 0.25 * scale
    c = torch.tensor(cell, dtype=torch.float64)
    assert tuple(jax_ns_mesh(c, spacing).tolist()) == _jax_mesh(cell, spacing)
    # exactly 32 -> 2*32+1 = 65 -> 128
    assert tuple(jax_ns_mesh(c, spacing).tolist()) == (128, 128, 128)


def _random_kspace_inputs(cell: np.ndarray):
    gen = torch.Generator().manual_seed(0)
    n = 8
    frac = torch.rand(n, 3, generator=gen, dtype=torch.float64)
    cell_t = torch.tensor(cell, dtype=torch.float64)
    positions = frac @ cell_t
    charges = torch.randn(n, 1, generator=gen, dtype=torch.float64)
    charges = charges - charges.mean()
    return charges, cell_t, positions


def test_calculator_reproduces_stock_when_rules_agree():
    """At 0.125 A on the cubic cell both rules give 128^3; the copied
    ``_compute_kspace`` must then match stock torch-pme exactly."""
    from torchpme.lib.kvectors import get_ns_mesh

    cell = torch.tensor(CUBIC, dtype=torch.float64)
    assert torch.equal(get_ns_mesh(cell, 0.125), jax_ns_mesh(cell, 0.125))

    charges, cell, positions = _random_kspace_inputs(CUBIC)
    potential = torchpme.CoulombPotential(smearing=0.5)
    stock = torchpme.PMECalculator(potential=potential, mesh_spacing=0.125)
    ours = JaxMeshPMECalculator(potential=potential, mesh_spacing=0.125)
    kw = dict(charges=charges, cell=cell, positions=positions)
    torch.testing.assert_close(
        ours._compute_kspace(**kw),
        stock._compute_kspace(**kw),
        atol=1e-12,
        rtol=0,
    )


def test_calculator_rejects_batching_like_stock():
    charges, cell, positions = _random_kspace_inputs(CUBIC)
    calc = JaxMeshPMECalculator(
        potential=torchpme.CoulombPotential(smearing=0.5), mesh_spacing=0.25
    )
    with pytest.raises(NotImplementedError):
        calc._compute_kspace(
            charges=charges,
            cell=cell,
            positions=positions,
            node_mask=torch.ones(len(positions), dtype=torch.bool),
        )


def test_make_pme_calculator_rules():
    potential = torchpme.CoulombPotential(smearing=0.5)
    jax_calc = make_pme_calculator(
        "jax", potential=potential, mesh_spacing=0.25
    )
    stock_calc = make_pme_calculator(
        "torchpme", potential=potential, mesh_spacing=0.25
    )
    assert isinstance(jax_calc, JaxMeshPMECalculator)
    assert type(stock_calc) is torchpme.PMECalculator
    with pytest.raises(ValueError, match="pme_mesh_rule"):
        make_pme_calculator("nope", potential=potential, mesh_spacing=0.25)


def _pme_energy(rule):
    charges, cell, positions = _random_kspace_inputs(CUBIC)
    pme = PMEElectrostaticInteraction(
        smearing=0.5, mesh_spacing=0.25, pme_mesh_rule=rule
    )
    n = len(positions)
    e = pme(
        partial_charges=charges[:, 0],
        positions=positions,
        cell=cell,
        batch_segments=torch.zeros(n, dtype=torch.long),
        num_graphs=1,
        num_nodes=n,
    )
    return e.sum().item()


def test_pme_electrostatics_default_is_jax_rule_and_flag_restores_stock():
    from torchpme.lib.kvectors import get_ns_mesh

    cell = torch.tensor(CUBIC, dtype=torch.float64)
    assert not torch.equal(get_ns_mesh(cell, 0.25), jax_ns_mesh(cell, 0.25))

    default = _pme_energy("jax")
    stock = _pme_energy("torchpme")
    assert default != pytest.approx(stock, abs=1e-6)

    charges, cell, positions = _random_kspace_inputs(CUBIC)
    n = len(positions)
    implicit = PMEElectrostaticInteraction(smearing=0.5, mesh_spacing=0.25)(
        partial_charges=charges[:, 0],
        positions=positions,
        cell=cell,
        batch_segments=torch.zeros(n, dtype=torch.long),
        num_graphs=1,
        num_nodes=n,
    )
    assert implicit.sum().item() == pytest.approx(default, abs=1e-12)


def test_so3lr_model_threads_pme_mesh_rule():
    from so3krates_torch.modules.models import SO3LR

    def build(**extra):
        return SO3LR(
            r_max=4.5,
            r_max_lr=8.0,
            num_radial_basis_fn=8,
            degrees=[1, 2],
            num_features=16,
            num_heads=2,
            num_layers=1,
            num_elements=20,
            energy_regression_dim=16,
            seed=42,
            device=torch.device("cpu"),
            dtype="float64",
            use_pme=True,
            pme_smearing=0.5,
            pme_mesh_spacing=0.25,
            use_pme_dispersion=True,
            pme_dispersion_smearing=0.5,
            pme_dispersion_mesh_spacing=0.25,
            **extra,
        )

    default = build()
    assert default.pme_mesh_rule == "jax"
    assert isinstance(
        default.pme_electrostatic_potential.calculator, JaxMeshPMECalculator
    )
    assert isinstance(
        default.pme_dispersion_potential.calculator, JaxMeshPMECalculator
    )

    stock = build(pme_mesh_rule="torchpme")
    assert stock.pme_mesh_rule == "torchpme"
    assert (
        type(stock.pme_electrostatic_potential.calculator)
        is torchpme.PMECalculator
    )
    assert (
        type(stock.pme_dispersion_potential.calculator)
        is torchpme.PMECalculator
    )


def test_so3lr_calculator_forwards_pme_mesh_rule():
    from so3krates_torch.calculator.so3 import SO3LRCalculator

    kwargs = dict(
        model="v2-s",
        theory_level=5,
        use_pme=True,
        pme_smearing=0.5,
        pme_mesh_spacing=0.25,
    )
    default = SO3LRCalculator(**kwargs).models[0]
    assert isinstance(
        default.pme_electrostatic_potential.calculator, JaxMeshPMECalculator
    )
    stock = SO3LRCalculator(**kwargs, pme_mesh_rule="torchpme").models[0]
    assert (
        type(stock.pme_electrostatic_potential.calculator)
        is torchpme.PMECalculator
    )


@pytest.mark.parametrize(
    "edge,spacing",
    [(7.75, 0.25), (15.5, 0.5), (3.875, 0.125), (9.3, 0.3)],
)
def test_pme_dispersion_mesh_matches_jax_when_ratio_is_an_exact_integer(
    edge, spacing
):
    """The dispersion path runs torch-pme in Bohr, but so3lr_dev derives
    the mesh from the Angstrom cell and spacing. L/h integer cells sit
    exactly on the inner ceil, where the Angstrom -> Bohr round trip used
    to flip the mesh size (e.g. 7.75 A at 0.25 A: JAX 64^3, torch 128^3)."""
    from so3krates_torch.blocks.physical_potentials import (
        PMEDispersionInteraction,
    )

    cell_np = np.diag([edge] * 3)
    expected = _jax_mesh(cell_np, spacing)

    gen = torch.Generator().manual_seed(1)
    n = 6
    positions = torch.rand(n, 3, generator=gen, dtype=torch.float64) * edge
    c6 = torch.rand(n, generator=gen, dtype=torch.float64) + 0.5
    cell = torch.tensor(cell_np, dtype=torch.float64)

    pme = PMEDispersionInteraction(smearing=0.5, mesh_spacing=spacing)
    seen = []
    update = pme.calculator.mesh_interpolator.update

    def spy(cell_arg, ns_arg):
        seen.append(tuple(ns_arg.tolist()))
        return update(cell_arg, ns_arg)

    pme.calculator.mesh_interpolator.update = spy
    pme(
        c6_pseudo_charges=c6,
        positions=positions,
        cell=cell,
        batch_segments=torch.zeros(n, dtype=torch.long),
        num_graphs=1,
        num_nodes=n,
    )
    assert seen == [expected]
