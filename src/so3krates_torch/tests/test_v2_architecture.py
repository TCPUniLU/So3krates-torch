"""Tests for the v2 architecture additions: use_rms_norm, qk_norm, and
use_residual_scalars.

These three flags close the last architectural gap between this torch
port and the JAX `so3lr_dev` ("v2") reference — all three ship as `True`
in the real so3lr-s/-m/-l checkpoints:
  - so3lr_dev/so3lr/mlff/nn/layer/so3krates_layer_sparse.py
    (functional_rms_norm, use_rms_norm, qk_norm)
  - so3lr_dev/so3lr/mlff/nn/stacknet/stacknet_sparse.py
    (use_residual_scalars: resid_lambdas/x0_lambdas)

This file is fully isolated from checkpoint loading, JAX, and parity
tooling (later tasks in the same plan) — it only exercises the torch-side
architecture in isolation, per this task's brief.
"""

import pytest
import torch

from so3krates_torch.blocks.euclidean_transformer import (
    EuclideanAttentionBlock,
    EuclideanAttentionBlockLORA,
    EuclideanAttentionBlockDoRA,
    EuclideanAttentionBlockVeRA,
    FilterNet,
    _rms_norm_no_params,
)
from so3krates_torch.modules.models import So3krates


DEGREES = [1, 2]
NUM_HEADS = 2
NUM_FEATURES = 8
NUM_RADIAL_BASIS_FN = 4
LORA_RANK = 4


def _build_filter_nets():
    filter_net_inv = FilterNet(
        degrees=DEGREES,
        num_radial_basis_fn=NUM_RADIAL_BASIS_FN,
        num_features=NUM_FEATURES,
    )
    filter_net_ev = FilterNet(
        degrees=DEGREES,
        num_radial_basis_fn=NUM_RADIAL_BASIS_FN,
        num_features=NUM_FEATURES,
    )
    return filter_net_inv, filter_net_ev


def _make_block(block_cls, qk_norm, seed, **extra_kwargs):
    """Build an attention block with deterministic weights.

    Re-seeding right before construction (rather than relying on
    whatever RNG state happens to be ambient) lets two blocks built with
    the same seed end up with byte-identical weights, so that qk_norm is
    the only difference between them.
    """
    torch.manual_seed(seed)
    filter_net_inv, filter_net_ev = _build_filter_nets()
    kwargs = dict(
        degrees=DEGREES,
        num_heads=NUM_HEADS,
        num_features=NUM_FEATURES,
        filter_net_inv=filter_net_inv,
        filter_net_ev=filter_net_ev,
        # Identity non-linearity: qk_non_linearity defaults to None on
        # the block itself (which is not callable) -- tests need a real
        # callable here since _get_qkv always applies it.
        qk_non_linearity=torch.nn.Identity,
        qk_norm=qk_norm,
    )
    kwargs.update(extra_kwargs)
    return block_cls(**kwargs)


def _make_vera_matrices():
    inv_heads = NUM_HEADS
    inv_head_dim = NUM_FEATURES // NUM_HEADS
    ev_heads = len(DEGREES)
    ev_head_dim = NUM_FEATURES // len(DEGREES)
    return dict(
        vera_A_matrix_inv=torch.randn(inv_heads, inv_head_dim, LORA_RANK),
        vera_B_matrix_inv=torch.randn(inv_heads, LORA_RANK, inv_head_dim),
        vera_A_matrix_ev=torch.randn(ev_heads, ev_head_dim, LORA_RANK),
        vera_B_matrix_ev=torch.randn(ev_heads, LORA_RANK, ev_head_dim),
    )


def _random_qkv_inputs(num_nodes=5, num_edges=6, seed=0):
    g = torch.Generator().manual_seed(seed)
    inv_heads = NUM_HEADS
    inv_head_dim = NUM_FEATURES // NUM_HEADS
    ev_heads = len(DEGREES)
    ev_head_dim = NUM_FEATURES // len(DEGREES)
    inv_features_inv = torch.randn(
        num_nodes, inv_heads, inv_head_dim, generator=g
    )
    inv_features_ev = torch.randn(
        num_nodes, ev_heads, ev_head_dim, generator=g
    )
    receivers = torch.randint(0, num_nodes, (num_edges,), generator=g)
    senders = torch.randint(0, num_nodes, (num_edges,), generator=g)
    return inv_features_inv, inv_features_ev, receivers, senders


def _assert_rms_one(x, atol=1e-5):
    rms_sq = x.pow(2).mean(dim=-1)
    assert torch.allclose(rms_sq, torch.ones_like(rms_sq), atol=atol), (
        f"expected mean-square along last dim ~1.0 (RMS-normalized), "
        f"got {rms_sq}"
    )


# ============================================================
# use_rms_norm: which normalization module gets built
# ============================================================


class TestUseRmsNorm:
    def test_rms_norm_module_used_when_flag_set(self, default_model_config):
        config = {
            **default_model_config,
            "layer_normalization_1": True,
            "layer_normalization_2": True,
            "use_rms_norm": True,
        }
        model = So3krates(**config)
        for transformer in model.euclidean_transformers:
            assert isinstance(transformer.layer_norm_inv_1, torch.nn.RMSNorm)
            assert isinstance(transformer.layer_norm_inv_2, torch.nn.RMSNorm)
            # JAX uses use_scale=False -- no learnable affine parameters.
            assert transformer.layer_norm_inv_1.weight is None
            assert transformer.layer_norm_inv_2.weight is None
            assert list(transformer.layer_norm_inv_1.parameters()) == []
            assert list(transformer.layer_norm_inv_2.parameters()) == []

    def test_layer_norm_used_by_default(self, default_model_config):
        config = {
            **default_model_config,
            "layer_normalization_1": True,
            "layer_normalization_2": True,
        }
        model = So3krates(**config)
        for transformer in model.euclidean_transformers:
            assert isinstance(transformer.layer_norm_inv_1, torch.nn.LayerNorm)
            assert isinstance(transformer.layer_norm_inv_2, torch.nn.LayerNorm)

    def test_use_rms_norm_alone_builds_no_normalization_module(
        self, default_model_config
    ):
        """use_rms_norm only selects *which* module to build -- it must
        not implicitly turn normalization on when layer_normalization_1/2
        are False."""
        config = {**default_model_config, "use_rms_norm": True}
        model = So3krates(**config)
        for transformer in model.euclidean_transformers:
            assert not hasattr(transformer, "layer_norm_inv_1")
            assert not hasattr(transformer, "layer_norm_inv_2")

    def test_forward_finite_with_rms_norm(
        self, default_model_config, make_batch, h2o_atoms
    ):
        config = {
            **default_model_config,
            "layer_normalization_1": True,
            "layer_normalization_2": True,
            "use_rms_norm": True,
        }
        model = So3krates(**config)
        model.eval()
        batch = make_batch(h2o_atoms, r_max=config["r_max"])
        out = model(batch.to_dict(), compute_stress=False)
        assert torch.isfinite(out["energy"]).all()
        assert torch.isfinite(out["forces"]).all()


# ============================================================
# qk_norm: base EuclideanAttentionBlock
# ============================================================


class TestQkNormBase:
    def test_qk_norm_normalizes_q_and_k_not_v(self):
        torch.set_default_dtype(torch.float64)
        (
            inv_features_inv,
            inv_features_ev,
            receivers,
            senders,
        ) = _random_qkv_inputs()

        block_norm = _make_block(EuclideanAttentionBlock, True, seed=1)
        block_plain = _make_block(EuclideanAttentionBlock, False, seed=1)

        q_inv_n, k_inv_n, v_inv_n, q_ev_n, k_ev_n = block_norm._get_qkv(
            inv_features_inv, inv_features_ev, receivers, senders
        )
        q_inv_p, k_inv_p, v_inv_p, q_ev_p, k_ev_p = block_plain._get_qkv(
            inv_features_inv, inv_features_ev, receivers, senders
        )

        # v is never touched by qk_norm.
        assert torch.equal(v_inv_n, v_inv_p)

        # q/k under qk_norm are exactly the RMS-normalized un-normalized
        # outputs.
        assert torch.allclose(q_inv_n, _rms_norm_no_params(q_inv_p))
        assert torch.allclose(k_inv_n, _rms_norm_no_params(k_inv_p))
        assert torch.allclose(q_ev_n, _rms_norm_no_params(q_ev_p))
        assert torch.allclose(k_ev_n, _rms_norm_no_params(k_ev_p))

        _assert_rms_one(q_inv_n)
        _assert_rms_one(k_inv_n)
        _assert_rms_one(q_ev_n)
        _assert_rms_one(k_ev_n)


# ============================================================
# qk_norm: LoRA/DoRA/VeRA non-fused branches
# ============================================================


@pytest.mark.parametrize(
    "block_cls,needs_vera",
    [
        (EuclideanAttentionBlockLORA, False),
        (EuclideanAttentionBlockDoRA, False),
        (EuclideanAttentionBlockVeRA, True),
    ],
)
def test_qk_norm_applied_in_finetuning_subclass_non_fused_branch(
    block_cls, needs_vera
):
    """qk_norm must be applied identically in the non-fused branch of
    each fine-tuning subclass's `_get_qkv` override (the fused branch
    already delegates to the base class, which is covered separately)."""
    torch.set_default_dtype(torch.float64)
    (
        inv_features_inv,
        inv_features_ev,
        receivers,
        senders,
    ) = _random_qkv_inputs()

    torch.manual_seed(3)
    extra_kwargs = _make_vera_matrices() if needs_vera else {}

    block_norm = _make_block(block_cls, True, seed=2, **extra_kwargs)
    block_plain = _make_block(block_cls, False, seed=2, **extra_kwargs)
    assert block_norm.weights_fused is False
    assert block_plain.weights_fused is False

    q_inv_n, k_inv_n, v_inv_n, q_ev_n, k_ev_n = block_norm._get_qkv(
        inv_features_inv, inv_features_ev, receivers, senders
    )
    q_inv_p, k_inv_p, v_inv_p, q_ev_p, k_ev_p = block_plain._get_qkv(
        inv_features_inv, inv_features_ev, receivers, senders
    )

    assert torch.equal(v_inv_n, v_inv_p)
    assert torch.allclose(q_inv_n, _rms_norm_no_params(q_inv_p))
    assert torch.allclose(k_inv_n, _rms_norm_no_params(k_inv_p))
    assert torch.allclose(q_ev_n, _rms_norm_no_params(q_ev_p))
    assert torch.allclose(k_ev_n, _rms_norm_no_params(k_ev_p))

    _assert_rms_one(q_inv_n)
    _assert_rms_one(k_inv_n)
    _assert_rms_one(q_ev_n)
    _assert_rms_one(k_ev_n)


def test_rms_norm_helper_gradcheck():
    """Exact gradcheck on the bare qk_norm helper function itself."""
    torch.manual_seed(0)
    x = torch.randn(3, 4, 5, dtype=torch.float64, requires_grad=True)
    assert torch.autograd.gradcheck(
        _rms_norm_no_params, (x,)
    ), "gradcheck failed for _rms_norm_no_params"


# ============================================================
# use_residual_scalars
# ============================================================


class TestUseResidualScalars:
    def test_resid_lambdas_in_state_dict_with_correct_init(
        self, default_model_config
    ):
        num_layers = 3
        config = {
            **default_model_config,
            "num_layers": num_layers,
            "use_residual_scalars": True,
        }
        model = So3krates(**config)
        sd = model.state_dict()
        assert "resid_lambdas" in sd
        assert "x0_lambdas" in sd
        assert sd["resid_lambdas"].shape == (num_layers,)
        assert sd["x0_lambdas"].shape == (num_layers,)
        assert torch.allclose(
            sd["resid_lambdas"],
            torch.ones(num_layers, dtype=sd["resid_lambdas"].dtype),
        )
        assert torch.allclose(
            sd["x0_lambdas"],
            torch.full((num_layers,), 0.1, dtype=sd["x0_lambdas"].dtype),
        )
        assert model.resid_lambdas.requires_grad
        assert model.x0_lambdas.requires_grad

    def test_resid_lambdas_absent_when_flag_false(self, default_model_config):
        model = So3krates(**default_model_config)
        sd = model.state_dict()
        assert "resid_lambdas" not in sd
        assert "x0_lambdas" not in sd

    def test_old_state_dict_without_resid_lambdas_still_loads(
        self, default_model_config
    ):
        """A state dict saved before this feature existed (missing
        resid_lambdas/x0_lambdas) must still load with strict=True
        (the existing backward-compatible load_state_dict override
        falls back to strict=False internally for such keys)."""
        old_model = So3krates(**default_model_config)
        old_sd = old_model.state_dict()

        new_config = {**default_model_config, "use_residual_scalars": True}
        new_model = So3krates(**new_config)

        # Should not raise despite old_sd missing resid_lambdas/x0_lambdas.
        new_model.load_state_dict(old_sd, strict=True)

    def test_resid_lambdas_receive_gradients(
        self, default_model_config, make_batch, h2o_atoms
    ):
        config = {
            **default_model_config,
            "num_layers": 2,
            "use_residual_scalars": True,
        }
        model = So3krates(**config)
        model.train()
        batch = make_batch(h2o_atoms, r_max=config["r_max"])
        out = model(batch.to_dict(), compute_stress=False, compute_force=False)
        energy = out["energy"].sum()
        energy.backward()

        assert model.resid_lambdas.grad is not None
        assert torch.isfinite(model.resid_lambdas.grad).all()
        assert model.x0_lambdas.grad is not None
        assert torch.isfinite(model.x0_lambdas.grad).all()

    def test_ev_features_input_to_transformer_unaffected_by_residual_scalars(
        self, default_model_config, make_batch, h2o_atoms
    ):
        """use_residual_scalars must only scale the invariant stream fed
        into each transformer layer -- the ev_features tensor handed to
        `transformer(...)` must be bit-identical regardless of
        resid_lambdas/x0_lambdas values.

        Note: this does NOT mean the model's final ev_features output is
        independent of resid_lambdas -- attention/interaction blocks
        couple the inv and ev streams internally, so a changed invariant
        stream legitimately changes ev outputs downstream. What must
        hold, and what this test isolates via monkeypatching, is that
        the residual-scalar mixing itself (inserted in
        So3krates.get_representation) never touches ev_features
        directly -- exactly mirroring the JAX reference, where only
        quantities['x'] is reassigned before each layer.
        """
        config = {
            **default_model_config,
            "num_layers": 2,
            "use_residual_scalars": True,
        }
        model = So3krates(**config)
        model.eval()
        batch = make_batch(h2o_atoms, r_max=config["r_max"])

        captured = {}
        orig_forward = model.euclidean_transformers[0].forward

        def spy_forward(*args, **kwargs):
            captured["ev_features"] = kwargs["ev_features"].clone()
            return orig_forward(*args, **kwargs)

        model.euclidean_transformers[0].forward = spy_forward

        with torch.no_grad():
            model.resid_lambdas.copy_(torch.tensor([2.0, 3.0]))
            model.x0_lambdas.copy_(torch.tensor([0.5, -0.5]))
        model(batch.to_dict(), compute_stress=False, compute_force=False)
        ev_first = captured["ev_features"]

        with torch.no_grad():
            model.resid_lambdas.copy_(torch.tensor([-7.0, 4.0]))
            model.x0_lambdas.copy_(torch.tensor([9.0, -3.0]))
        model(batch.to_dict(), compute_stress=False, compute_force=False)
        ev_second = captured["ev_features"]

        assert torch.equal(ev_first, ev_second)


# ============================================================
# Full-model integration: all three flags together
# ============================================================


class TestFullModelIntegration:
    def test_all_three_flags_construct_and_forward_finite(
        self, default_model_config, make_batch, h2o_atoms
    ):
        config = {
            **default_model_config,
            "num_layers": 2,
            "layer_normalization_1": True,
            "layer_normalization_2": True,
            "use_rms_norm": True,
            "qk_norm": True,
            "use_residual_scalars": True,
        }
        model = So3krates(**config)
        model.eval()
        batch = make_batch(h2o_atoms, r_max=config["r_max"])
        out = model(batch.to_dict(), compute_stress=False)
        assert out["energy"].shape == (1,)
        assert torch.isfinite(out["energy"]).all()
        assert out["forces"].shape == (3, 3)
        assert torch.isfinite(out["forces"]).all()

    def test_all_three_flags_forces_match_finite_differences(
        self, default_model_config, make_batch, h2o_atoms
    ):
        """End-to-end gradient-flow check (mirrors
        TestForcesFiniteDifferences in test_model_inference.py):
        autograd forces from a model with all three v2 flags enabled
        must match central finite differences of the energy. This
        exercises gradients through qk_norm's RMS normalization and
        through the resid_lambdas/x0_lambdas residual-scalar mixing at
        every layer."""
        config = {
            **default_model_config,
            "num_layers": 2,
            "layer_normalization_1": True,
            "layer_normalization_2": True,
            "use_rms_norm": True,
            "qk_norm": True,
            "use_residual_scalars": True,
        }
        model = So3krates(**config)
        model.eval()
        eps = 1e-4
        positions = h2o_atoms.get_positions().copy()
        N = len(h2o_atoms)

        batch_ref = make_batch(h2o_atoms, r_max=config["r_max"])
        F_autograd = model(batch_ref.to_dict(), compute_stress=False)[
            "forces"
        ].detach()

        F_fd = torch.zeros(N, 3, dtype=torch.float64)
        for i in range(N):
            for j in range(3):
                for sign, delta in [(+1, eps), (-1, -eps)]:
                    pos = positions.copy()
                    pos[i, j] += delta
                    atoms_pert = h2o_atoms.copy()
                    atoms_pert.set_positions(pos)
                    batch_pert = make_batch(atoms_pert, r_max=config["r_max"])
                    E_pert = (
                        model(
                            batch_pert.to_dict(),
                            compute_stress=False,
                        )["energy"]
                        .detach()
                        .item()
                    )
                    F_fd[i, j] -= sign * E_pert / (2 * eps)

        assert torch.allclose(F_autograd, F_fd, atol=1e-3)


# ============================================================
# Backward compatibility: default (all-False) path unchanged
# ============================================================


class TestDefaultBehaviorUnchanged:
    def test_explicit_false_flags_match_omitted_defaults(
        self, default_model_config, make_batch, h2o_atoms
    ):
        """Explicitly passing use_rms_norm=False, qk_norm=False,
        use_residual_scalars=False must be indistinguishable from
        omitting them entirely -- the new flags must not alter the
        default (all-False) code path at all."""
        seed = default_model_config.get("seed", 42)

        torch.manual_seed(seed)
        model_omitted = So3krates(**default_model_config)

        torch.manual_seed(seed)
        model_explicit = So3krates(
            **default_model_config,
            use_rms_norm=False,
            qk_norm=False,
            use_residual_scalars=False,
        )

        model_omitted.eval()
        model_explicit.eval()

        sd_omitted = model_omitted.state_dict()
        sd_explicit = model_explicit.state_dict()
        assert sd_omitted.keys() == sd_explicit.keys()
        assert "resid_lambdas" not in sd_omitted
        for key in sd_omitted:
            assert torch.equal(sd_omitted[key], sd_explicit[key]), key

        batch = make_batch(h2o_atoms, r_max=default_model_config["r_max"])
        out_omitted = model_omitted(batch.to_dict(), compute_stress=False)
        out_explicit = model_explicit(batch.to_dict(), compute_stress=False)
        assert torch.equal(out_omitted["energy"], out_explicit["energy"])
        assert torch.equal(out_omitted["forces"], out_explicit["forces"])
