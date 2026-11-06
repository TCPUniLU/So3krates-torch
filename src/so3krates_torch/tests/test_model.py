import numpy as np
import pytest
from ase.build import molecule
from so3krates_torch.calculator.so3 import SO3LRCalculator


def test_so3lr_calculator_prediction():
    """Test SO3LR pre-trained model loading and prediction via ASE.

    Uses the bundled "v1" checkpoint (``pretrained/so3lr/v1/``, loaded
    via ``load_pretrained_so3lr`` -- state_dict + settings.yaml, not a
    pickled model). Reference values captured fresh from this exact
    checkpoint/structure, not carried over from the old pickled
    ``so3lr.model`` artifact (which has since been removed).
    """
    model = SO3LRCalculator(
        model="v1",
        default_dtype="float64",
    )
    mol = molecule("H2O")
    model.calculate(mol)
    results = model.results
    ref_energy = -5.01616383247536
    ref_forces = np.array(
        [
            [0.0, 1.48318857e-15, -0.14374645878607717],
            [0.0, -0.292303374, 0.0718732294],
            [0.0, 0.292303374, 0.0718732294],
        ]
    )
    assert np.allclose(results["forces"], ref_forces, rtol=1e-4)
    assert np.isclose(results["energy"], ref_energy, rtol=1e-4)


def test_so3lr_calculator_requires_theory_level_for_multihead_model():
    """v2 checkpoints have 16 theory-level heads; silently picking one
    would give energies that differ from so3lr_dev (which uses 5)."""
    import pytest

    with pytest.raises(ValueError, match="theory_level") as excinfo:
        SO3LRCalculator(model="v2-s")
    assert "5" in str(excinfo.value)
    assert "so3lr_dev" in str(excinfo.value)

    for level in (0, 5):
        calc = SO3LRCalculator(model="v2-s", theory_level=level)
        assert calc.theory_level == level


def test_so3lr_calculator_single_head_model_needs_no_theory_level():
    calc = SO3LRCalculator(model="v1")
    assert calc.theory_level == 0


def test_so3lr_calculator_rejects_out_of_range_theory_level():
    import pytest

    for level in (16, -1):
        with pytest.raises(ValueError, match="theory_level"):
            SO3LRCalculator(model="v2-s", theory_level=level)
    # v1 has a single head: only level 0 exists
    with pytest.raises(ValueError, match="theory_level"):
        SO3LRCalculator(model="v1", theory_level=5)
    assert SO3LRCalculator(model="v1", theory_level=0).theory_level == 0


def test_so3lr_calculator_warns_for_shared_untrained_theory_levels():
    """Only levels 0, 4 and 5 of the bundled v2 models are trained; the
    other 13 heads share identical parameters and return one value."""
    import warnings

    import pytest

    with pytest.warns(UserWarning, match="theory_level=6"):
        SO3LRCalculator(model="v2-s", theory_level=6)
    for level in (0, 4, 5):
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            SO3LRCalculator(model="v2-s", theory_level=level)
