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

    def test_save_load_inference_roundtrip(
        self,
        default_model_config,
        make_batch,
        h2o_atoms,
        tmp_path,
    ):
        """Saved and loaded So3krates model produces identical predictions."""
        model1 = So3krates(**default_model_config)
        model1.eval()
        batch = make_batch(h2o_atoms, r_max=5.0)

        # Forces are computed via autograd.grad; cannot use torch.no_grad()
        out1 = model1(batch.to_dict(), compute_stress=False)
        energy1 = out1["energy"].detach().clone()
        forces1 = out1["forces"].detach().clone()

        save_path = tmp_path / "model.pt"
        torch.save(model1.state_dict(), save_path)

        model2 = So3krates(**default_model_config)
        model2.load_state_dict(torch.load(save_path, weights_only=True))
        model2.eval()
        out2 = model2(batch.to_dict(), compute_stress=False)

        assert torch.allclose(energy1, out2["energy"].detach())
        assert torch.allclose(forces1, out2["forces"].detach())

    def test_so3lr_save_load_inference(
        self,
        so3lr_model_config,
        make_batch,
        h2o_atoms,
        tmp_path,
    ):
        """SO3LR model with long-range components produces identical
        predictions after save/load."""
        model1 = SO3LR(**so3lr_model_config)
        model1.eval()
        batch = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)

        out1 = model1(batch.to_dict(), compute_stress=False)
        energy1 = out1["energy"].detach().clone()
        forces1 = out1["forces"].detach().clone()

        save_path = tmp_path / "so3lr_model.pt"
        torch.save(model1.state_dict(), save_path)

        model2 = SO3LR(**so3lr_model_config)
        # strict=False: Coulomb constants are registered as buffers during
        # the first forward pass so they appear in model1's state_dict but
        # not in fresh model2; they are deterministic and always identical.
        model2.load_state_dict(
            torch.load(save_path, weights_only=True), strict=False
        )
        model2.eval()
        out2 = model2(batch.to_dict(), compute_stress=False)

        assert torch.allclose(energy1, out2["energy"].detach())
        assert torch.allclose(forces1, out2["forces"].detach())
