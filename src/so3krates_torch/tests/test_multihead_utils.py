"""Tests for multi-head conversion utilities in tools/multihead_utils.py.

Covers pretrained_to_mh_model (single-head -> multi-head conversion,
which must duplicate the pretrained output head into every new head)
and reduce_mh_model_to_sh (multi-head -> single-head extraction),
verifying the two are exact inverses of each other.
"""

import pytest
import torch

from so3krates_torch.modules.models import SO3LR
from so3krates_torch.tools.multihead_utils import (
    pretrained_to_mh_model,
    reduce_mh_model_to_sh,
)
from so3krates_torch.tools.finetune import setup_finetuning
from so3krates_torch.tools.model_setup import (
    resolve_atomic_energy_shifts,
    set_atomic_energy_shifts_in_model,
)


@pytest.fixture
def pretrained_model(so3lr_model_config):
    """SO3LR model with a non-trivial (2-layer) energy MLP.

    Uses float32 to match pretrained_to_mh_model's current dtype
    handling: it reads "default_dtype" off the architecture settings
    dict (which, as built from config["ARCHITECTURE"] in production,
    never has that key) and otherwise falls back to "float32".
    """
    config = {
        **so3lr_model_config,
        "dtype": torch.float32,
        "final_mlp_layers": 2,
        "energy_regression_dim": so3lr_model_config["num_features"],
    }
    return SO3LR(**config)


@pytest.fixture
def architecture_settings(so3lr_model_config):
    """Dict shaped like config["ARCHITECTURE"], as consumed by
    pretrained_to_mh_model/reduce_mh_model_to_sh."""
    settings = {
        **so3lr_model_config,
        "final_mlp_layers": 2,
        "energy_regression_dim": so3lr_model_config["num_features"],
    }
    del settings["dtype"]
    return settings


class TestPretrainedToMhModelDuplicatesHead:
    def test_duplicates_energy_shifts_and_scales_into_every_head(
        self, pretrained_model, architecture_settings
    ):
        # Give the pretrained head non-default shifts/scales so the
        # test can't pass by coincidence (both classes default to
        # zeros/ones at init).
        sh_block = pretrained_model.atomic_energy_output_block
        with torch.no_grad():
            sh_block.energy_shifts.data = torch.arange(
                118, dtype=torch.float32
            )
            sh_block.energy_scales.weight.data = (
                torch.arange(118, dtype=torch.float32).unsqueeze(0) + 1.0
            )

        settings = {**architecture_settings, "num_output_heads": 3}
        mh_model = pretrained_to_mh_model(
            settings, pretrained_model, device_name="cpu"
        )
        mh_block = mh_model.atomic_energy_output_block

        assert torch.allclose(mh_block.energy_shifts, sh_block.energy_shifts)
        assert torch.allclose(
            mh_block.energy_scales, sh_block.energy_scales.weight.T
        )

    def test_duplicates_mlp_layers_into_every_head(
        self, pretrained_model, architecture_settings
    ):
        num_heads = 3
        settings = {**architecture_settings, "num_output_heads": num_heads}
        mh_model = pretrained_to_mh_model(
            settings, pretrained_model, device_name="cpu"
        )
        sh_block = pretrained_model.atomic_energy_output_block
        mh_block = mh_model.atomic_energy_output_block

        assert len(sh_block.layers) > 0, "fixture must exercise >=2 layers"
        for i, sh_layer in enumerate(sh_block.layers):
            for h in range(num_heads):
                assert torch.allclose(
                    mh_block.layers_weights[i][h], sh_layer.weight.T
                )
                assert torch.allclose(
                    mh_block.layers_bias[i][h], sh_layer.bias
                )

        for h in range(num_heads):
            assert torch.allclose(
                mh_block.final_layer_weights[h],
                sh_block.final_layer.weight.T,
            )
            assert torch.allclose(
                mh_block.final_layer_bias[h], sh_block.final_layer.bias
            )

    def test_every_head_reproduces_pretrained_forward_pass(
        self, pretrained_model, architecture_settings, make_batch, h2o_atoms
    ):
        """Immediately after conversion, every head must reproduce the
        original single-head model's predictions exactly."""
        num_heads = 3
        settings = {**architecture_settings, "num_output_heads": num_heads}
        pretrained_model.eval()
        batch = make_batch(
            h2o_atoms, r_max=5.0, cutoff_lr=10.0, dtype=torch.float32
        )
        expected = pretrained_model(batch.to_dict(), compute_stress=False)

        mh_model = pretrained_to_mh_model(
            settings, pretrained_model, device_name="cpu"
        )
        mh_model.eval()
        out = mh_model(batch.to_dict(), compute_stress=False)

        for h in range(num_heads):
            assert torch.allclose(
                out["energy"][h], expected["energy"], atol=1e-10
            )
            assert torch.allclose(
                out["forces"][h], expected["forces"], atol=1e-8
            )


class TestPretrainedToMhModelToleratesExtraArchitectureConfigFields:
    def test_tolerates_compute_avg_num_neighbors(
        self, pretrained_model, architecture_settings
    ):
        """config["ARCHITECTURE"] from a real pydantic model_dump()
        always includes every schema field, including
        compute_avg_num_neighbors, which is not a model constructor
        kwarg (create_model() in model_setup.py never forwards it
        either). pretrained_to_mh_model must tolerate it instead of
        blowing up with a TypeError."""
        settings = {
            **architecture_settings,
            "num_output_heads": 2,
            "compute_avg_num_neighbors": True,
        }
        pretrained_to_mh_model(settings, pretrained_model, device_name="cpu")

    def test_tolerates_atomic_energy_shifts(
        self, pretrained_model, architecture_settings
    ):
        """ARCHITECTURE.atomic_energy_shifts is a documented config key
        (examples/training/train_settings_example.yaml) consumed by
        resolve_atomic_energy_shifts, not by the model constructor
        (which takes atomic_type_shifts instead). It must not reach
        MultiHeadSO3LR.__init__ and blow up with a TypeError."""
        settings = {
            **architecture_settings,
            "num_output_heads": 2,
            "atomic_energy_shifts": {"1": -13.6, "6": -1000.0},
        }
        pretrained_to_mh_model(settings, pretrained_model, device_name="cpu")

    def test_reduce_mh_model_to_sh_tolerates_atomic_energy_shifts(
        self, pretrained_model, architecture_settings
    ):
        num_heads = 2
        settings = {**architecture_settings, "num_output_heads": num_heads}
        mh_model = pretrained_to_mh_model(
            settings, pretrained_model, device_name="cpu"
        )
        settings_with_extra = {
            **architecture_settings,
            "atomic_energy_shifts": {"1": -13.6, "6": -1000.0},
        }
        reduce_mh_model_to_sh(
            mh_model.state_dict(),
            settings_with_extra,
            head_idx=0,
            model_choice="so3lr",
            device="cpu",
            dtype="float32",
        )


class TestModelToMultiheadFromAlreadyMultiHeadSource:
    """convert_to_multihead + finetune_choice without a pretrained_model
    reaches pretrained_to_mh_model with a source model that create_model()
    already built as MultiHeadSO3LR (models.py's convert_to_multihead
    branch). Conversion must not crash on that already-multi-head input."""

    def test_does_not_raise_on_already_multihead_source(
        self, architecture_settings
    ):
        from so3krates_torch.modules.models import MultiHeadSO3LR

        num_heads = 2
        settings = {**architecture_settings, "num_output_heads": num_heads}
        source = MultiHeadSO3LR(
            num_output_heads=num_heads, **architecture_settings
        )

        pretrained_to_mh_model(settings, source, device_name="cpu")

    def test_raises_clear_error_on_head_count_mismatch(
        self, architecture_settings
    ):
        from so3krates_torch.modules.models import MultiHeadSO3LR

        source = MultiHeadSO3LR(num_output_heads=2, **architecture_settings)
        settings = {**architecture_settings, "num_output_heads": 5}

        with pytest.raises(ValueError, match="already multi-head"):
            pretrained_to_mh_model(settings, source, device_name="cpu")

    def test_preserves_existing_head_weights_when_already_multihead(
        self, architecture_settings
    ):
        """Skipping the broadcast for an already-multihead source must
        not corrupt its per-head weights (e.g. via a botched no-op)."""
        from so3krates_torch.modules.models import MultiHeadSO3LR

        num_heads = 2
        settings = {**architecture_settings, "num_output_heads": num_heads}
        source = MultiHeadSO3LR(
            num_output_heads=num_heads, **architecture_settings
        )
        before = (
            source.atomic_energy_output_block.final_layer_weights.detach().clone()
        )

        result = pretrained_to_mh_model(settings, source, device_name="cpu")

        assert torch.allclose(
            result.atomic_energy_output_block.final_layer_weights, before
        )


class TestReduceMhModelToShRoundTrip:
    def test_extracted_head_matches_pretrained_model(
        self, pretrained_model, architecture_settings, make_batch, h2o_atoms
    ):
        """Extracting any head right after conversion must reproduce the
        original pretrained model exactly (duplication + extraction are
        inverses)."""
        num_heads = 3
        settings = {**architecture_settings, "num_output_heads": num_heads}
        pretrained_model.eval()
        batch = make_batch(
            h2o_atoms, r_max=5.0, cutoff_lr=10.0, dtype=torch.float32
        )
        expected = pretrained_model(batch.to_dict(), compute_stress=False)

        mh_model = pretrained_to_mh_model(
            settings, pretrained_model, device_name="cpu"
        )

        for head_idx in range(num_heads):
            sh_model = reduce_mh_model_to_sh(
                mh_model.state_dict(),
                architecture_settings,
                head_idx,
                model_choice="so3lr",
                device="cpu",
                dtype="float32",
            )
            sh_model.eval()
            out = sh_model(batch.to_dict(), compute_stress=False)
            assert torch.allclose(
                out["energy"], expected["energy"], atol=1e-10
            )
            assert torch.allclose(out["forces"], expected["forces"], atol=1e-8)


class TestManualAtomicEnergyShiftsApplyToAllHeads:
    """A user-provided ARCHITECTURE.atomic_energy_shifts must survive
    the pretrained -> multi-head conversion and end up applied
    identically to every head's energy prediction (there is only one
    shared energy_shifts tensor across heads by design)."""

    def test_manual_shifts_survive_conversion_and_apply_to_every_head(
        self, pretrained_model, architecture_settings, make_batch
    ):
        from ase.build import molecule

        num_heads = 3
        manual_shifts = {"1": -13.6, "6": -1000.0}
        settings = {
            **architecture_settings,
            "num_output_heads": num_heads,
            "atomic_energy_shifts": manual_shifts,
        }

        # Mirrors run_train.py's actual call order: resolve BEFORE
        # conversion (from the still-single-head pretrained model),
        # then apply AFTER conversion.
        resolved = resolve_atomic_energy_shifts(
            {"ARCHITECTURE": settings, "TRAINING": {}},
            pretrained_model,
            warm_start=True,
            average_atomic_energy_shifts=None,
        )

        mh_model = setup_finetuning(
            model=pretrained_model,
            finetune_choice="naive",
            device_name="cpu",
            convert_to_multihead=True,
            architecture_settings=settings,
        )
        set_atomic_energy_shifts_in_model(mh_model, resolved)

        shifts = mh_model.atomic_energy_output_block.energy_shifts
        assert shifts.shape == (118,)
        assert shifts[0].item() == pytest.approx(-13.6)  # H, z=1
        assert shifts[5].item() == pytest.approx(-1000.0)  # C, z=6

        mh_model.eval()
        batch = make_batch(
            molecule("CH4"), r_max=5.0, cutoff_lr=10.0, dtype=torch.float32
        )
        out = mh_model(batch.to_dict(), compute_stress=False)
        carbon_node_energy_per_head = out["node_energy"][0, :, 0]
        assert carbon_node_energy_per_head.shape == (num_heads,)
        assert torch.allclose(
            carbon_node_energy_per_head,
            carbon_node_energy_per_head[0].expand(num_heads),
        )
