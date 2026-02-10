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

    def test_minimal_init(self, default_model_config):
        """Test model initializes with minimal config."""
        model = So3krates(**default_model_config)
        assert model is not None
        assert isinstance(model, torch.nn.Module)

    def test_init_creates_expected_layers(self, default_model_config):
        """Test model creates expected sub-modules."""
        model = So3krates(**default_model_config)

        assert hasattr(model, "inv_feature_embedding")
        assert hasattr(model, "ev_embedding")
        assert hasattr(model, "radial_embedding")
        assert hasattr(model, "euclidean_transformers")
        assert hasattr(model, "atomic_energy_output_block")

    def test_init_layer_count_matches_config(
        self, default_model_config
    ):
        """Test number of transformer layers matches config."""
        config = {**default_model_config, "num_layers": 3}
        model = So3krates(**config)

        assert len(model.euclidean_transformers) == 3

    def test_init_with_different_degrees(
        self, default_model_config
    ):
        """Test model with various spherical harmonic degrees."""
        for degrees in [[1], [1, 2], [1, 2, 3], [1, 2, 3, 4]]:
            config = {
                **default_model_config,
                "degrees": degrees,
            }
            model = So3krates(**config)
            assert model.degrees == degrees

    def test_init_with_different_radial_basis(
        self, default_model_config
    ):
        """Test different radial basis function types."""
        for rbf_type in ["gaussian", "bernstein", "bessel"]:
            config = {
                **default_model_config,
                "radial_basis_fn": rbf_type,
            }
            model = So3krates(**config)
            assert model is not None

    def test_init_with_different_cutoff_fn(
        self, default_model_config
    ):
        """Test different cutoff function types."""
        for cutoff_fn in ["cosine", "polynomial"]:
            config = {
                **default_model_config,
                "cutoff_fn": cutoff_fn,
            }
            model = So3krates(**config)
            assert model is not None

    def test_init_with_different_dtypes(
        self, default_model_config
    ):
        """Test model with float32 and float64."""
        for dtype in [torch.float32, torch.float64]:
            config = {**default_model_config, "dtype": dtype}
            model = So3krates(**config)
            param = next(model.parameters())
            assert param.dtype == dtype



class TestSO3LRInit:
    """Test SO3LR model initialization."""

    def test_minimal_init(self, so3lr_model_config):
        """Test SO3LR initializes with config."""
        model = SO3LR(**so3lr_model_config)
        assert model is not None
        assert isinstance(model, So3krates)

    def test_init_creates_lr_components(
        self, so3lr_model_config
    ):
        """Test SO3LR creates long-range components."""
        model = SO3LR(**so3lr_model_config)

        assert hasattr(model, "zbl_repulsion")
        assert hasattr(model, "partial_charges_output_block")
        assert hasattr(model, "dipole_output_head")
        assert hasattr(model, "electrostatic_potential")
        assert hasattr(model, "hirshfeld_output_block")
        assert hasattr(model, "dispersion_potential")



class TestMultiHeadSO3LRInit:
    """Test MultiHeadSO3LR model initialization."""

    def test_minimal_init(self, multihead_model_config):
        """Test MultiHeadSO3LR initializes with config."""
        model = MultiHeadSO3LR(**multihead_model_config)
        assert model is not None
        assert isinstance(model, SO3LR)

    def test_init_creates_multihead_output(
        self, multihead_model_config
    ):
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

