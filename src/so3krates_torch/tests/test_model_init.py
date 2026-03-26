"""Tests for model initialization."""

import pytest
import torch
from so3krates_torch.modules.models import (
    So3krates,
    SO3LR,
    MultiHeadSO3LR,
)


class TestSo3kratesInit:
    """Test So3krates model initialization."""

    def test_init_layer_count_matches_config(self, default_model_config):
        """Test number of transformer layers matches config."""
        config = {**default_model_config, "num_layers": 3}
        model = So3krates(**config)

        assert len(model.euclidean_transformers) == 3

    def test_init_with_different_degrees(self, default_model_config):
        """Test model with various spherical harmonic degrees."""
        for degrees in [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]:
            config = {
                **default_model_config,
                "degrees": degrees,
            }
            model = So3krates(**config)
            assert model.degrees == degrees

    def test_init_with_different_dtypes(self, default_model_config):
        """Test model with float32 and float64."""
        for dtype in [torch.float32, torch.float64]:
            config = {**default_model_config, "dtype": dtype}
            model = So3krates(**config)
            param = next(model.parameters())
            assert param.dtype == dtype


class TestMultiHeadSO3LRInit:
    """Test MultiHeadSO3LR model initialization."""

    def test_init_creates_multihead_output(self, multihead_model_config):
        """Test MultiHeadSO3LR creates multi-head output."""
        model = MultiHeadSO3LR(**multihead_model_config)
        assert hasattr(model, "atomic_energy_output_block")
        from so3krates_torch.blocks.output_block import (
            MultiAtomicEnergyOutputHead,
        )

        assert isinstance(
            model.atomic_energy_output_block,
            MultiAtomicEnergyOutputHead,
        )
