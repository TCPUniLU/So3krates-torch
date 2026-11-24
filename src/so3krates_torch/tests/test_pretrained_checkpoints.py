"""The released SO3LR tauri checkpoints bundled with the package."""

import numpy as np
import pytest
import torch
from ase.build import molecule

from so3krates_torch.calculator.so3 import (
    _PRETRAINED_SO3LR_MODELS,
    SO3LRCalculator,
    load_pretrained_so3lr,
)

TAURI_KEYWORDS = [
    f"tauri-{size}-{gen}" for gen in ("2602", "2606") for size in "sml"
]


@pytest.mark.parametrize("keyword", TAURI_KEYWORDS)
def test_tauri_checkpoint_loads_and_predicts(keyword):
    """Each bundled tauri model loads (strictly) and gives finite energy and
    forces for a water molecule."""
    assert keyword in _PRETRAINED_SO3LR_MODELS
    atoms = molecule("H2O")
    atoms.calc = SO3LRCalculator(model=keyword, theory_level=5, device="cpu")
    assert np.isfinite(atoms.get_potential_energy())
    assert np.isfinite(atoms.get_forces()).all()


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_v2_names_are_aliases_of_the_2602_generation(size):
    old = load_pretrained_so3lr(f"v2-{size}").state_dict()
    new = load_pretrained_so3lr(f"tauri-{size}-2602").state_dict()
    assert old.keys() == new.keys()
    assert all(torch.equal(old[k], new[k]) for k in old)


@pytest.mark.parametrize("size", ["s", "m", "l"])
def test_generations_are_different_models(size):
    a = load_pretrained_so3lr(f"tauri-{size}-2602").state_dict()
    b = load_pretrained_so3lr(f"tauri-{size}-2606").state_dict()
    assert a.keys() == b.keys()
    assert any(not torch.equal(a[k], b[k]) for k in a)


def test_tauri_2606_stores_its_own_dispersion_scale():
    """2606 was trained with dispersion_energy_scale 1.0 (2602: 1.2); the
    converted settings carry that, so the model must not fall back to the
    class default."""
    assert load_pretrained_so3lr("tauri-s-2606").dispersion_energy_scale == 1.0
    assert load_pretrained_so3lr("tauri-s-2602").dispersion_energy_scale == 1.2
