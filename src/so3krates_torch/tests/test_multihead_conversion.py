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
from so3krates_torch.tools.finetune import model_to_lora, setup_finetuning
from so3krates_torch.tools.multihead_utils import (
    select_theory_level,
    to_multihead_model,
    to_single_head_model,
)

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

    def test_converts_after_forward(self, sh, make_batch, h2o_atoms):
        # A forward pass leaves graph-attached geometry tensors cached on
        # the model; conversion must still work and leave them intact.
        batch = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
        _energy_forces(sh, batch)
        cached = sh.vectors
        mh = to_multihead_model(sh, 2)
        assert type(mh) is MultiHeadSO3LR
        assert sh.vectors is cached

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
    def test_cuda_rng_untouched_for_cpu_model(self, sh):
        before = torch.cuda.get_rng_state_all()
        to_multihead_model(sh, 2, seed=3)
        after = torch.cuda.get_rng_state_all()
        assert all(torch.equal(a, b) for a, b in zip(before, after))

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA only")
    def test_cuda_device_preserved(self, sh):
        mh = to_multihead_model(sh.cuda(), 2)
        assert all(p.is_cuda for p in mh.parameters())


# ---------------------------------------------------------------------------
# to_single_head_model
# ---------------------------------------------------------------------------


def _copy_single_head_into(mh, sh, head_idx):
    """Write ``sh``'s energy MLP into head ``head_idx`` of ``mh`` (the
    inverse of the layout ``to_single_head_model`` reads)."""
    src = sh.atomic_energy_output_block
    dst = mh.atomic_energy_output_block
    with torch.no_grad():
        for i, layer in enumerate(src.layers):
            dst.layers_weights[i][head_idx].copy_(layer.weight.T)
            dst.layers_bias[i][head_idx].copy_(layer.bias)
        dst.final_layer_weights[head_idx].copy_(src.final_layer.weight.T)
        dst.final_layer_bias[head_idx].copy_(src.final_layer.bias)


class TestToSingleHead:
    @pytest.mark.parametrize("k", [0, 1, 2])
    def test_extracted_head_matches_multihead(
        self, sh, make_batch, h2o_atoms, k
    ):
        mh = to_multihead_model(sh, 3, seed=0)
        batch = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
        e_mh, f_mh = _energy_forces(mh, batch)
        single = to_single_head_model(mh, k)
        e, f = _energy_forces(single, batch)
        assert type(single) is SO3LR
        assert torch.allclose(e, e_mh[k], atol=1e-10)
        assert torch.allclose(f, f_mh[k], atol=1e-10)

    def test_original_head_round_trips(self, sh, make_batch, h2o_atoms):
        mh = to_multihead_model(sh, 3)
        _copy_single_head_into(mh, sh, head_idx=1)
        batch = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
        e_ref, f_ref = _energy_forces(sh, batch)
        e, f = _energy_forces(to_single_head_model(mh, 1), batch)
        assert torch.allclose(e, e_ref, atol=1e-10)
        assert torch.allclose(f, f_ref, atol=1e-10)

    def test_keeps_settings(self, sh):
        single = to_single_head_model(to_multihead_model(sh, 2), 0)
        assert single.r_max_lr == 10.0
        assert not hasattr(single, "num_output_heads")
        assert not hasattr(single, "select_heads")
        assert single.atomic_energy_output_block.num_theory_levels == 1
        assert all(p.dtype == torch.float64 for p in single.parameters())

    def test_shifts_and_scales_copied(self, sh):
        mh = to_multihead_model(sh, 2)
        src = mh.atomic_energy_output_block
        dst = to_single_head_model(mh, 1).atomic_energy_output_block
        assert torch.equal(dst.energy_shifts, src.energy_shifts)
        assert torch.equal(dst.energy_scales.weight, src.energy_scales.T)

    def test_lora_wrapped_multihead(self, sh, make_batch, h2o_atoms):
        mh = model_to_lora(
            to_multihead_model(sh, 2), rank=2, alpha=4.0, device="cpu"
        )
        lora_b = [p for n, p in mh.named_parameters() if "lora_B" in n]
        assert lora_b
        with torch.no_grad():
            lora_b[0].normal_()  # make the adapter non-trivial
        batch = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
        e_mh, f_mh = _energy_forces(mh, batch)
        e, f = _energy_forces(to_single_head_model(mh, 1), batch)
        assert torch.allclose(e, e_mh[1], atol=1e-10)
        assert torch.allclose(f, f_mh[1], atol=1e-10)

    def test_input_not_mutated(self, sh):
        mh = to_multihead_model(sh, 2)
        before = {k: v.clone() for k, v in mh.state_dict().items()}
        to_single_head_model(mh, 0)
        assert type(mh) is MultiHeadSO3LR
        assert mh.num_output_heads == 2
        after = mh.state_dict()
        assert all(torch.equal(before[k], after[k]) for k in before)

    @pytest.mark.parametrize("k", [-1, 2])
    def test_rejects_bad_head_idx(self, sh, k):
        with pytest.raises(ValueError, match="head_idx"):
            to_single_head_model(to_multihead_model(sh, 2), k)

    def test_rejects_single_head_model(self, sh):
        with pytest.raises(ValueError, match="MultiHeadSO3LR"):
            to_single_head_model(sh, 0)


# ---------------------------------------------------------------------------
# select_theory_level
# ---------------------------------------------------------------------------

NUM_LEVELS = 4


@pytest.fixture(params=[True, False], ids=["bias", "nobias"])
def sh_tl(request, so3lr_model_config):
    """SO3LR with several theory levels, as the bundled tauri models."""
    model = SO3LR(
        **so3lr_model_config,
        num_theory_levels=NUM_LEVELS,
        final_layer_bias=request.param,
    )
    return _trained_like(model)


class TestSelectTheoryLevel:
    @pytest.mark.parametrize("t", [0, NUM_LEVELS - 1])
    def test_matches_original_at_level(self, sh_tl, make_batch, h2o_atoms, t):
        atoms = h2o_atoms.copy()
        atoms.info["theory_level"] = t
        e_ref, f_ref = _energy_forces(
            sh_tl, make_batch(atoms, r_max=5.0, cutoff_lr=10.0)
        )
        reduced = select_theory_level(sh_tl, t)
        # The reduced model needs no theory_level in the data.
        e, f = _energy_forces(
            reduced, make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
        )
        assert reduced.atomic_energy_output_block.num_theory_levels == 1
        assert torch.allclose(e, e_ref, atol=1e-10)
        assert torch.allclose(f, f_ref, atol=1e-10)

    def test_input_not_mutated(self, sh_tl):
        select_theory_level(sh_tl, 1)
        assert sh_tl.atomic_energy_output_block.num_theory_levels == (
            NUM_LEVELS
        )

    def test_multihead_from_theory_level_model(self, sh_tl):
        mh = to_multihead_model(sh_tl, 2, theory_level=1)
        src = sh_tl.atomic_energy_output_block
        dst = mh.atomic_energy_output_block
        assert type(mh) is MultiHeadSO3LR
        assert torch.equal(dst.energy_shifts, src.energy_shifts[:, 1])
        assert torch.equal(
            dst.energy_scales.flatten(), src.energy_scales.weight[1]
        )

    def test_multihead_requires_theory_level(self, sh_tl):
        with pytest.raises(ValueError, match="theory_level"):
            to_multihead_model(sh_tl, 2)

    @pytest.mark.parametrize("t", [-1, NUM_LEVELS])
    def test_rejects_out_of_range(self, sh_tl, t):
        with pytest.raises(ValueError, match="theory_level"):
            select_theory_level(sh_tl, t)

    def test_rejects_single_level_model(self, so3lr_model_config):
        sh = SO3LR(**so3lr_model_config)
        with pytest.raises(ValueError, match="theory_level"):
            select_theory_level(sh, 0)
        with pytest.raises(ValueError, match="theory_level"):
            to_multihead_model(sh, 2, theory_level=0)

    def test_rejects_multihead(self, multihead_model_config):
        mh = MultiHeadSO3LR(**multihead_model_config)
        with pytest.raises(ValueError, match="SO3LR"):
            select_theory_level(mh, 0)


# ---------------------------------------------------------------------------
# Fine-tuning path and persistence
# ---------------------------------------------------------------------------


class TestFinetuneConversion:
    def test_setup_finetuning_converts_with_validated_config(self, sh):
        # A validated ARCHITECTURE section carries keys the model
        # constructor does not accept, and an r_max_lr that differs from
        # the pretrained model's.
        arch = {
            "convert_to_multihead": True,
            "num_output_heads": 3,
            "compute_avg_num_neighbors": True,
            "r_max_lr": 12.0,
        }
        mh = setup_finetuning(
            sh,
            finetune_choice="naive",
            device_name="cpu",
            convert_to_multihead=True,
            architecture_settings=arch,
        )
        assert type(mh) is MultiHeadSO3LR
        assert mh.num_output_heads == 3
        assert mh.r_max_lr == 10.0  # from the model, not the config
        assert all(p.dtype == torch.float64 for p in mh.parameters())

    def test_setup_finetuning_theory_level(self, so3lr_model_config):
        sh = SO3LR(**so3lr_model_config, num_theory_levels=3)
        arch = {"num_output_heads": 2, "convert_theory_level": 2}
        mh = setup_finetuning(
            sh,
            finetune_choice="naive",
            device_name="cpu",
            convert_to_multihead=True,
            architecture_settings=arch,
        )
        assert type(mh) is MultiHeadSO3LR

    def test_converted_models_pickle(self, sh, tmp_path):
        mh = to_multihead_model(sh, 2)
        for model in (mh, to_single_head_model(mh, 0)):
            path = tmp_path / "m.model"
            torch.save(model, path)
            loaded = torch.load(path, weights_only=False)
            assert type(loaded) is type(model)
            a, b = model.state_dict(), loaded.state_dict()
            assert all(torch.equal(a[k], b[k]) for k in a)
