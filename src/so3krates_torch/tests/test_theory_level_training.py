"""Training multi-theory-level models (e.g. the bundled tauri models):
per-element energy shifts, selecting one level for fine-tuning, and
long-range anchoring."""

import copy

import pytest
import torch
from ase.build import molecule

from so3krates_torch.modules.anchoring import make_anchor_teacher
from so3krates_torch.modules.models import SO3LR
from so3krates_torch.tests.test_multihead_conversion import (
    _energy_forces,
    _trained_like,
)
from so3krates_torch.tests.test_multihead_training_setup import (
    NUM_ELEMENTS,
    _prepare,
    _with,
    config,  # noqa: F401  (fixture)
)
from so3krates_torch.tools.model_setup import set_atomic_energy_shifts_in_model

LEVELS = 3
LR_KEYS = ("partial_charges", "hirshfeld_ratios", "c6_ratios")


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    before = torch.get_default_dtype()
    yield
    torch.set_default_dtype(before)


@pytest.fixture
def multi_level(so3lr_model_config):
    torch.manual_seed(0)
    return _trained_like(
        SO3LR(
            **so3lr_model_config,
            num_theory_levels=LEVELS,
            c6_ratios_bool=True,
            output_partial_charges=True,
        )
    ).double()


def _atoms(theory_level=None):
    atoms = molecule("CH3CH2OH")
    atoms.rattle(0.02, seed=1)
    if theory_level is not None:
        atoms.info["theory_level"] = theory_level
    return atoms


def _batch(make_batch, atoms):
    return make_batch(atoms, r_max=5.0, cutoff_lr=10.0)


def test_teacher_of_multi_level_model_needs_no_theory_level(
    multi_level, make_batch
):
    teacher = make_anchor_teacher(multi_level, "float64")

    assert teacher.atomic_energy_output_block.num_theory_levels == 1
    with_level = _batch(make_batch, _atoms(theory_level=2)).to_dict()
    expected = multi_level(with_level, training=False, compute_force=False)
    got = teacher(
        _batch(make_batch, _atoms()).to_dict(),
        training=False,
        compute_force=False,
    )
    for key in LR_KEYS:
        torch.testing.assert_close(got[key], expected[key].detach())
    # The model being fine-tuned is not modified.
    assert multi_level.atomic_energy_output_block.num_theory_levels == LEVELS


@pytest.mark.parametrize("kind", ["tensor", "dict"])
def test_per_element_shifts_apply_to_every_level(multi_level, kind):
    shifts = torch.arange(NUM_ELEMENTS, dtype=torch.float64)
    given = (
        shifts
        if kind == "tensor"
        else {z: float(shifts[z - 1]) for z in range(1, NUM_ELEMENTS + 1)}
    )

    set_atomic_energy_shifts_in_model(multi_level, given)

    got = multi_level.atomic_energy_output_block.energy_shifts
    assert got.shape == (NUM_ELEMENTS, LEVELS)
    for level in range(LEVELS):
        assert torch.equal(got[:, level], shifts)


def test_multi_level_training_from_scratch_uses_data_shifts(
    config, multi_level, make_batch  # noqa: F811
):
    model = _prepare(config, multi_level, warm_start=False)
    energy, _ = _energy_forces(
        model, _batch(make_batch, _atoms(theory_level=1))
    )
    assert torch.isfinite(energy).all()
    assert model.atomic_energy_output_block.energy_shifts.shape == (
        NUM_ELEMENTS,
        LEVELS,
    )


def test_convert_theory_level_selects_one_level_without_multihead(
    config, multi_level, make_batch  # noqa: F811
):
    source = copy.deepcopy(multi_level)
    cfg = _with(
        config,
        ARCHITECTURE=dict(convert_theory_level=2),
        TRAINING=dict(finetune_choice="naive"),
    )

    model = _prepare(cfg, multi_level)

    assert type(model) is SO3LR
    assert model.atomic_energy_output_block.num_theory_levels == 1
    energy, forces = _energy_forces(model, _batch(make_batch, _atoms()))
    ref_energy, ref_forces = _energy_forces(
        source, _batch(make_batch, _atoms(theory_level=2))
    )
    torch.testing.assert_close(energy, ref_energy)
    torch.testing.assert_close(forces, ref_forces)


def test_selected_level_takes_data_shifts(
    config, multi_level, make_batch  # noqa: F811
):
    cfg = _with(
        config,
        ARCHITECTURE=dict(convert_theory_level=0),
        TRAINING=dict(force_use_average_shifts=True),
    )
    model = _prepare(cfg, multi_level)
    head = model.atomic_energy_output_block
    assert head.num_theory_levels == 1
    assert torch.equal(
        head.energy_shifts, torch.zeros(NUM_ELEMENTS, dtype=torch.float64)
    )
    energy, _ = _energy_forces(model, _batch(make_batch, _atoms()))
    assert torch.isfinite(energy).all()


def test_configured_shifts_keep_other_levels_of_multi_level_model(
    config, multi_level  # noqa: F811
):
    before = multi_level.atomic_energy_output_block.energy_shifts.clone()
    cfg = _with(config, ARCHITECTURE=dict(atomic_energy_shifts={1: -13.6}))

    model = _prepare(cfg, multi_level)

    got = model.atomic_energy_output_block.energy_shifts
    assert got.shape == (NUM_ELEMENTS, LEVELS)
    assert torch.equal(got[0], torch.full((LEVELS,), -13.6).double())
    assert torch.equal(got[1:], before[1:])
