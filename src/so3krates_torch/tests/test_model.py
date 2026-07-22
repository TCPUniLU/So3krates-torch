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
