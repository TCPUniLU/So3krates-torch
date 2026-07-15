"""torch.compile parity tests for So3krates with charge/spin embedding.

Historically, ChargeSpinEmbedding produced different results between
eager and compiled modes due to (a) the `psi // inf` sign trick being
fragile under Inductor and (b) `@torch.compiler.disable()` papering over
it. These tests verify that the rewritten embedding is numerically
identical compiled vs eager, including the "trap" case where the compiled
callable is first warmed up on a neutral (zero charge/spin) batch and then
called with a charged/spin-polarized batch on the same callable.
"""

import pytest
import torch
from so3krates_torch.modules.models import So3krates, SO3LR


def _make_model(default_model_config):
    cfg = {
        **default_model_config,
        "use_charge_embed": True,
        "use_spin_embed": True,
    }
    model = So3krates(**cfg)
    model.eval()
    return model


def _run(model, batch, charge: float, spin: float):
    d = batch.to_dict()
    d["total_charge"] = torch.tensor(
        [charge], dtype=torch.float64, device=batch.positions.device
    )
    d["total_spin"] = torch.tensor(
        [spin], dtype=torch.float64, device=batch.positions.device
    )
    # No torch.no_grad() — the model needs autograd.grad for force computation.
    return model(d, compute_stress=False)


class TestChargeSpinEmbedCompile:
    """Eager-vs-compiled parity for ChargeSpinEmbedding."""

    def test_neutral_parity(self, default_model_config, make_batch, h2o_atoms):
        """Neutral (zero charge/spin) case matches eager."""
        model = _make_model(default_model_config)
        batch = make_batch(h2o_atoms, r_max=5.0)
        compiled = torch.compile(model, dynamic=True)

        out_eager = _run(model, batch, 0.0, 0.0)
        out_compiled = _run(compiled, batch, 0.0, 0.0)

        assert torch.allclose(
            out_eager["energy"], out_compiled["energy"], atol=1e-8, rtol=1e-6
        )
        assert torch.allclose(
            out_eager["forces"], out_compiled["forces"], atol=1e-8, rtol=1e-6
        )

    def test_positive_charge_parity(
        self, default_model_config, make_batch, h2o_atoms
    ):
        """Positive charge case matches eager."""
        model = _make_model(default_model_config)
        batch = make_batch(h2o_atoms, r_max=5.0)
        compiled = torch.compile(model, dynamic=True)

        out_eager = _run(model, batch, 1.0, 0.0)
        out_compiled = _run(compiled, batch, 1.0, 0.0)

        assert torch.allclose(
            out_eager["energy"], out_compiled["energy"], atol=1e-8, rtol=1e-6
        )
        assert torch.allclose(
            out_eager["forces"], out_compiled["forces"], atol=1e-8, rtol=1e-6
        )

    def test_negative_charge_parity(
        self, default_model_config, make_batch, h2o_atoms
    ):
        """Negative charge case matches eager (uses row 1 of Wk/Wv)."""
        model = _make_model(default_model_config)
        batch = make_batch(h2o_atoms, r_max=5.0)
        compiled = torch.compile(model, dynamic=True)

        out_eager = _run(model, batch, -1.0, 0.0)
        out_compiled = _run(compiled, batch, -1.0, 0.0)

        assert torch.allclose(
            out_eager["energy"], out_compiled["energy"], atol=1e-8, rtol=1e-6
        )
        assert torch.allclose(
            out_eager["forces"], out_compiled["forces"], atol=1e-8, rtol=1e-6
        )

    def test_spin_parity(self, default_model_config, make_batch, h2o_atoms):
        """Spin-polarized case matches eager."""
        model = _make_model(default_model_config)
        batch = make_batch(h2o_atoms, r_max=5.0)
        compiled = torch.compile(model, dynamic=True)

        out_eager = _run(model, batch, 0.0, 1.0)
        out_compiled = _run(compiled, batch, 0.0, 1.0)

        assert torch.allclose(
            out_eager["energy"], out_compiled["energy"], atol=1e-8, rtol=1e-6
        )
        assert torch.allclose(
            out_eager["forces"], out_compiled["forces"], atol=1e-8, rtol=1e-6
        )

    def test_neutral_then_charged_trap(
        self, default_model_config, make_batch, h2o_atoms
    ):
        """Warm up compiled on neutral, then verify charged gives correct result.

        This is the critical regression test: the old `psi // inf` sign
        trick let Dynamo specialize on psi=0 → bake in idx=0 for all rows
        → wrong result when psi≠0 on the same compiled callable.
        """
        model = _make_model(default_model_config)
        batch = make_batch(h2o_atoms, r_max=5.0)
        compiled = torch.compile(model, dynamic=True)

        # Eager references
        out_charge_eager = _run(model, batch, 1.0, 0.0)
        out_spin_eager = _run(model, batch, 0.0, 1.0)
        out_neg_eager = _run(model, batch, -1.0, 0.0)

        # Warm compiled callable up on neutral first — old code would
        # specialize here and produce wrong results on the charged cases.
        _run(compiled, batch, 0.0, 0.0)

        # Now run the non-neutral cases on the SAME compiled callable.
        for eager_out, charge, spin in [
            (out_charge_eager, 1.0, 0.0),
            (out_spin_eager, 0.0, 1.0),
            (out_neg_eager, -1.0, 0.0),
        ]:
            out = _run(compiled, batch, charge, spin)
            assert torch.allclose(
                eager_out["energy"], out["energy"], atol=1e-8, rtol=1e-6
            ), (
                f"Energy mismatch after neutral warm-up "
                f"(charge={charge}, spin={spin}): "
                f"eager={eager_out['energy'].item():.10f}, "
                f"compiled={out['energy'].item():.10f}"
            )
            assert torch.allclose(
                eager_out["forces"], out["forces"], atol=1e-8, rtol=1e-6
            ), f"Forces mismatch after neutral warm-up (charge={charge}, spin={spin})"

    def test_charge_spin_changes_output(
        self, default_model_config, make_batch, h2o_atoms
    ):
        """Charge/spin embedding must actually influence the output."""
        model = _make_model(default_model_config)
        batch = make_batch(h2o_atoms, r_max=5.0)

        out_neutral = _run(model, batch, 0.0, 0.0)
        out_charged = _run(model, batch, 1.0, 0.0)
        out_spin = _run(model, batch, 0.0, 1.0)
        out_neg = _run(model, batch, -1.0, 0.0)

        assert not torch.allclose(
            out_neutral["energy"], out_charged["energy"]
        ), "Positive charge did not change energy."
        assert not torch.allclose(
            out_neutral["energy"], out_spin["energy"]
        ), "Spin did not change energy."
        assert not torch.allclose(
            out_neutral["energy"], out_neg["energy"]
        ), "Negative charge did not change energy."
        # Negative and positive charge should differ (row 0 vs row 1 of Wk/Wv)
        assert not torch.allclose(
            out_charged["energy"], out_neg["energy"]
        ), "Positive and negative charge gave same energy."


def _make_so3lr_model(so3lr_model_config):
    model = SO3LR(**so3lr_model_config)
    model.eval()
    return model


class TestSO3LRCompile:
    """Eager-vs-compiled parity for the full SO3LR long-range path
    (ZBL repulsion + electrostatics + dispersion).

    `requires_grad_()` on positions/vectors/displacement is unsupported
    by Dynamo tracing (regardless of backend). `prepare_graph` now
    guards those calls and callers pre-set `requires_grad_(True)`
    before invoking the model, so compiled calls never hit the
    unsupported call. These tests exercise both the guarded (compiled)
    and default (eager, not pre-set) calling conventions.
    """

    def test_fullgraph_false_parity(
        self, so3lr_model_config, make_batch, h2o_atoms
    ):
        """fullgraph=False must match eager on whatever device is
        available (CPU in typical CI, GPU if available). This is the
        combination already confirmed working end-to-end and it fully
        exercises the requires_grad_ guard fix for the long-range
        path, which the base So3krates tests above never touch."""
        model = _make_so3lr_model(so3lr_model_config)
        batch = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
        compiled = torch.compile(model, dynamic=True, fullgraph=False)

        out_eager = model(batch.to_dict(), compute_stress=False)
        out_compiled = compiled(batch.to_dict(), compute_stress=False)

        assert torch.allclose(
            out_eager["energy"], out_compiled["energy"], atol=1e-8, rtol=1e-6
        )
        assert torch.allclose(
            out_eager["forces"], out_compiled["forces"], atol=1e-8, rtol=1e-6
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(),
        reason="fullgraph=True SO3LR compilation requires CUDA (the "
        "CPU Inductor backend crashes on the long-range scatter-sums)",
    )
    def test_fullgraph_true_parity_gpu(
        self, so3lr_model_config, make_batch, h2o_atoms
    ):
        """fullgraph=True must succeed and match eager on GPU — the
        combination this fix targets."""
        device = torch.device("cuda")
        model = _make_so3lr_model(so3lr_model_config).to(device)
        # `make_batch` already resolves to the `device` fixture, which
        # picks cuda when available (guaranteed here by the skipif).
        batch = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
        compiled = torch.compile(model, dynamic=True, fullgraph=True)

        out_eager = model(batch.to_dict(), compute_stress=False)
        out_compiled = compiled(batch.to_dict(), compute_stress=False)

        assert torch.allclose(
            out_eager["energy"], out_compiled["energy"], atol=1e-8, rtol=1e-6
        )
        assert torch.allclose(
            out_eager["forces"], out_compiled["forces"], atol=1e-8, rtol=1e-6
        )

    def test_eager_default_path_unaffected(
        self, so3lr_model_config, make_batch, h2o_atoms
    ):
        """Plain eager call with no pre-set requires_grad (today's
        default calling convention) must still produce forces via
        autograd exactly as before the guard was added."""
        model = _make_so3lr_model(so3lr_model_config)
        batch = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)

        assert not batch.positions.requires_grad
        out = model(batch.to_dict(), compute_stress=False)

        assert out["forces"] is not None
        assert not torch.allclose(
            out["forces"], torch.zeros_like(out["forces"])
        )

    def test_pre_set_requires_grad_matches_default(
        self, so3lr_model_config, make_batch, h2o_atoms
    ):
        """Pre-setting requires_grad_(True) before the (eager) model
        call — the calling convention compiled callers now rely on —
        must give identical results to the default, not-pre-set
        convention."""
        model = _make_so3lr_model(so3lr_model_config)
        batch_default = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
        batch_preset = make_batch(h2o_atoms, r_max=5.0, cutoff_lr=10.0)
        batch_preset.positions.requires_grad_(True)

        out_default = model(batch_default.to_dict(), compute_stress=False)
        out_preset = model(batch_preset.to_dict(), compute_stress=False)

        assert torch.allclose(out_default["energy"], out_preset["energy"])
        assert torch.allclose(out_default["forces"], out_preset["forces"])
