"""Tests for multi-head <-> single-head conversion in
``tools/multihead_utils.py``.

A conversion deep-copies the source model and swaps only the energy
output head, so everything else (backbone weights, ``r_max_lr``, dtype,
device, non-weight settings) must come through unchanged.
"""

import pytest
import torch

from so3krates_torch.modules.models import (
    MultiHeadSO3LR,
    SO3LR,
    So3krates,
)
from so3krates_torch.tools.multihead_utils import to_multihead_model

HEAD_PREFIX = "atomic_energy_output_block."


# ---------------------------------------------------------------------------
# Helpers (also imported by test_convert_heads_cli.py)
# ---------------------------------------------------------------------------


def _trained_like(model):
    """Give the energy head non-trivial shifts and scales, as a trained
    model would have, so that copying them is observable."""
    head = model.atomic_energy_output_block
    with torch.no_grad():
        head.energy_shifts.normal_()
        head.energy_scales.weight.uniform_(0.5, 2.0)
    return model


def _energy_forces(model, batch):
    """Detached (energy, forces) of ``model`` on a batch."""
    model.eval()
    out = model(batch.to_dict(), compute_stress=False)
    return out["energy"].detach(), out["forces"].detach()


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    before = torch.get_default_dtype()
    yield
    torch.set_default_dtype(before)


@pytest.fixture(params=[1, 2], ids=["mlp1", "mlp2"])
def sh(request, so3lr_model_config):
    """Single-head SO3LR with 1- or 2-layer energy MLP."""
    config = {**so3lr_model_config, "final_mlp_layers": request.param}
    return _trained_like(SO3LR(**config))


# ---------------------------------------------------------------------------
# to_multihead_model
# ---------------------------------------------------------------------------


class TestToMultihead:
    def test_backbone_identical(self, sh):
        mh = to_multihead_model(sh, 3, seed=0)
        a, b = sh.state_dict(), mh.state_dict()
        keys = [k for k in a if not k.startswith(HEAD_PREFIX)]
        assert keys
        assert all(torch.equal(a[k], b[k]) for k in keys)

    def test_heads_independent_and_fresh(self, sh):
        head = to_multihead_model(sh, 3, seed=0).atomic_energy_output_block
        w = head.final_layer_weights
        assert w.shape[0] == 3
        assert not torch.equal(w[0], w[1])
        assert torch.count_nonzero(head.final_layer_bias) == 0

    def test_shifts_and_scales_copied(self, sh):
        src = sh.atomic_energy_output_block
        dst = to_multihead_model(sh, 3).atomic_energy_output_block
        assert torch.equal(dst.energy_shifts, src.energy_shifts)
        assert torch.equal(dst.energy_scales, src.energy_scales.weight.T)
        assert (
            dst.energy_scales.requires_grad
            == src.energy_scales.weight.requires_grad
        )

    def test_learned_scales_stay_trainable(self, so3lr_model_config):
        sh = SO3LR(
            **so3lr_model_config,
            energy_learn_atomic_type_shifts=True,
            energy_learn_atomic_type_scales=True,
        )
        head = to_multihead_model(sh, 2).atomic_energy_output_block
        assert head.energy_shifts.requires_grad
        assert head.energy_scales.requires_grad

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_settings_dtype_and_globals_preserved(
        self, so3lr_model_config, dtype
    ):
        sh = SO3LR(**{**so3lr_model_config, "dtype": dtype})
        other = torch.float32 if dtype == torch.float64 else torch.float64
        torch.set_default_dtype(other)
        rng_before = torch.get_rng_state()

        mh = to_multihead_model(sh, 2, seed=1)

        assert type(mh) is MultiHeadSO3LR
        assert mh.num_output_heads == 2
        assert mh.r_max_lr == 10.0
        assert mh.electrostatic_energy_scale == sh.electrostatic_energy_scale
        assert all(p.dtype == dtype for p in mh.parameters())
        assert torch.get_default_dtype() == other
        assert torch.equal(torch.get_rng_state(), rng_before)
        assert mh.select_heads is False
        assert mh.return_mean is False

    def test_input_not_mutated(self, sh):
        before = {k: v.clone() for k, v in sh.state_dict().items()}
        to_multihead_model(sh, 2)
        assert type(sh) is SO3LR
        after = sh.state_dict()
        assert before.keys() == after.keys()
        assert all(torch.equal(before[k], after[k]) for k in before)

    def test_seed_reproducible(self, sh):
        def weights(seed):
            mh = to_multihead_model(sh, 2, seed=seed)
            return mh.atomic_energy_output_block.final_layer_weights

        assert torch.equal(weights(7), weights(7))
        assert not torch.equal(weights(7), weights(8))

    def test_forward_shape(self, sh, make_batch, h2o_atoms):
        mh = to_multihead_model(sh, 3)
        batch = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
        energy, forces = _energy_forces(mh, batch)
        assert energy.shape == (3, 1)
        assert forces.shape == (3, 3, 3)

    def test_rejects_so3krates(self, default_model_config):
        with pytest.raises(ValueError, match="SO3LR"):
            to_multihead_model(So3krates(**default_model_config), 2)

    def test_rejects_multihead(self, multihead_model_config):
        mh = MultiHeadSO3LR(**multihead_model_config)
        with pytest.raises(ValueError, match="already"):
            to_multihead_model(mh, 2)

    @pytest.mark.parametrize("n", [0, -1])
    def test_rejects_bad_num_heads(self, sh, n):
        with pytest.raises(ValueError, match="num_output_heads"):
            to_multihead_model(sh, n)

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
    def test_cuda_device_preserved(self, sh):
        mh = to_multihead_model(sh.cuda(), 2)
        assert all(p.is_cuda for p in mh.parameters())
