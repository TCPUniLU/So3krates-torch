"""Tests for model save/load functionality."""

import pytest
import torch
import tempfile
import os
from so3krates_torch.modules.models import So3krates, SO3LR


class TestModelSaveLoad:
    """Test model serialization using state_dict (PyTorch best practice)."""

    def test_save_load_state_dict(self, default_model_config):
        """Test saving and loading model state dict."""
        model = So3krates(**default_model_config)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)

            # Load into new model
            model2 = So3krates(**default_model_config)
            model2.load_state_dict(torch.load(f.name, weights_only=True))

            os.unlink(f.name)

        # Compare parameters
        for (n1, p1), (n2, p2) in zip(
            model.named_parameters(), model2.named_parameters()
        ):
            assert n1 == n2
            assert torch.allclose(p1, p2)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_load_preserves_dtype(self, default_model_config, dtype):
        """Test that loading preserves model dtype."""
        config = {**default_model_config, "dtype": dtype}
        model = So3krates(**config)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)

            model2 = So3krates(**config)
            model2.load_state_dict(torch.load(f.name, weights_only=True))

            os.unlink(f.name)

        param = next(model2.parameters())
        assert param.dtype == dtype

    def test_so3lr_save_load(self, so3lr_model_config):
        """Test SO3LR model save/load preserves long-range components."""
        model = SO3LR(**so3lr_model_config)

        with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as f:
            torch.save(model.state_dict(), f.name)

            model2 = SO3LR(**so3lr_model_config)
            model2.load_state_dict(torch.load(f.name, weights_only=True))

            os.unlink(f.name)

        # Verify LR components preserved
        assert model2.zbl_repulsion is not None
        assert model2.electrostatic_potential is not None
        assert model2.dispersion_potential is not None
