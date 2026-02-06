import unittest
from ase.build import molecule
from so3krates_torch.calculator.so3 import SO3LRCalculator
import numpy as np


class TestSO3LRCalculator(unittest.TestCase):
    def test_prediction(self):
        model = SO3LRCalculator(
            default_dtype="float64",
        )
        mol = molecule("H2O")
        model.calculate(mol)
        results = model.results
        ref_energy = -495.8705186908738
        ref_forces = np.array(
            [
                [0.0, -1.0547118733938987e-15, -0.14374645878607717],
                [0.0, -0.2923033736335291, 0.071873229393038196],
                [0.0, 0.29230337363353009, 0.071873229393039084],
            ]
        )
        assert np.allclose(results["forces"], ref_forces, rtol=1e-4)
        assert np.isclose(results["energy"], ref_energy, rtol=1e-4)
