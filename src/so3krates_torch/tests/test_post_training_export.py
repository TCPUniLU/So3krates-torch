"""Tests for the post-training model export logic in cli/run_train.py.

Covers TRAINING.post_training_export: "multihead" (default, saves the
trained model as-is), "multihead+singlehead" (also extracts a
standalone single-head model per head), and "singlehead" (per-head
models only, no combined multi-head file).
"""

import pytest
import torch

from so3krates_torch.modules.models import SO3LR
from so3krates_torch.tools.model_setup import set_avg_num_neighbors_in_model
from so3krates_torch.tools.multihead_utils import pretrained_to_mh_model
from so3krates_torch.cli.run_train import _export_trained_model


TRAINED_AVG_NUM_NEIGHBORS = 42.0


@pytest.fixture
def base_config(so3lr_model_config):
    return {
        "GENERAL": {"name_exp": "myrun"},
        "ARCHITECTURE": {
            **so3lr_model_config,
            "final_mlp_layers": 2,
            "energy_regression_dim": so3lr_model_config["num_features"],
        },
        "TRAINING": {},
    }


@pytest.fixture
def singlehead_model(so3lr_model_config):
    config = {
        **so3lr_model_config,
        "dtype": torch.float32,
        "final_mlp_layers": 2,
        "energy_regression_dim": so3lr_model_config["num_features"],
    }
    model = SO3LR(**config)
    # A value that differs from both the class default (1.0) and the
    # conftest fixture's own default (10.0), so a test asserting on it
    # can't pass by fixture coincidence.
    set_avg_num_neighbors_in_model(model, TRAINED_AVG_NUM_NEIGHBORS)
    return model


@pytest.fixture
def multihead_model(singlehead_model, base_config):
    settings = {**base_config["ARCHITECTURE"], "num_output_heads": 2}
    model = pretrained_to_mh_model(
        settings, singlehead_model, device_name="cpu"
    )
    # Mirrors run_train.py's real order: set_avg_num_neighbors_in_model
    # runs AFTER convert_to_multihead, applying the retained/computed
    # training value on top of whatever the fresh multi-head
    # construction defaulted to.
    set_avg_num_neighbors_in_model(model, TRAINED_AVG_NUM_NEIGHBORS)
    return model


class TestMultiheadExport:
    """Default export mode: unchanged existing behavior."""

    def test_writes_pth_and_model_files(
        self, singlehead_model, base_config, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        base_config["TRAINING"]["post_training_export"] = "multihead"
        _export_trained_model(singlehead_model, base_config, "cpu", "float32")
        assert (tmp_path / "myrun.pth").exists()
        assert (tmp_path / "myrun.model").exists()

    def test_no_per_head_files_written(
        self, multihead_model, base_config, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        base_config["TRAINING"]["post_training_export"] = "multihead"
        _export_trained_model(
            multihead_model,
            base_config,
            "cpu",
            "float32",
            head_names=["a", "b"],
        )
        assert not (tmp_path / "myrun_a.model").exists()
        assert not (tmp_path / "myrun_b.model").exists()


class TestMultiheadPlusSingleheadExport:
    def test_writes_combined_and_per_head_files(
        self, multihead_model, base_config, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        base_config["TRAINING"][
            "post_training_export"
        ] = "multihead+singlehead"
        _export_trained_model(
            multihead_model,
            base_config,
            "cpu",
            "float32",
            head_names=["a", "b"],
        )

        assert (tmp_path / "myrun.pth").exists()
        assert (tmp_path / "myrun.model").exists()
        assert (tmp_path / "myrun_a.model").exists()
        assert (tmp_path / "myrun_b.model").exists()

    def test_per_head_files_load_as_plain_so3lr(
        self, multihead_model, base_config, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        base_config["TRAINING"][
            "post_training_export"
        ] = "multihead+singlehead"
        _export_trained_model(
            multihead_model,
            base_config,
            "cpu",
            "float32",
            head_names=["a", "b"],
        )

        loaded = torch.load(tmp_path / "myrun_a.model", weights_only=False)
        assert isinstance(loaded, SO3LR)
        assert not hasattr(loaded, "num_output_heads")

    def test_per_head_models_retain_trained_avg_num_neighbors(
        self, multihead_model, base_config, tmp_path, monkeypatch
    ):
        """reduce_mh_model_to_sh rebuilds the single-head model from
        config["ARCHITECTURE"], which has no avg_num_neighbors field,
        so the extracted model must have it (and the per-layer
        att_norm_inv/att_norm_ev it drives) restored explicitly rather
        than silently defaulting to 1.0."""
        monkeypatch.chdir(tmp_path)
        base_config["TRAINING"][
            "post_training_export"
        ] = "multihead+singlehead"
        _export_trained_model(
            multihead_model,
            base_config,
            "cpu",
            "float32",
            head_names=["a", "b"],
        )

        loaded = torch.load(tmp_path / "myrun_a.model", weights_only=False)
        assert loaded.avg_num_neighbors == pytest.approx(
            TRAINED_AVG_NUM_NEIGHBORS
        )
        for layer in loaded.euclidean_transformers:
            block = layer.euclidean_attention_block
            assert block.att_norm_inv == pytest.approx(
                TRAINED_AVG_NUM_NEIGHBORS
            )
            assert block.att_norm_ev == pytest.approx(
                TRAINED_AVG_NUM_NEIGHBORS
            )

    def test_per_head_models_get_heads_attribute(
        self, multihead_model, base_config, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        base_config["TRAINING"][
            "post_training_export"
        ] = "multihead+singlehead"
        _export_trained_model(
            multihead_model,
            base_config,
            "cpu",
            "float32",
            head_names=["a", "b"],
        )

        loaded_a = torch.load(tmp_path / "myrun_a.model", weights_only=False)
        loaded_b = torch.load(tmp_path / "myrun_b.model", weights_only=False)
        assert loaded_a.heads == ["a"]
        assert loaded_b.heads == ["b"]

    def test_combined_model_gets_heads_attribute(
        self, multihead_model, base_config, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        base_config["TRAINING"][
            "post_training_export"
        ] = "multihead+singlehead"
        _export_trained_model(
            multihead_model,
            base_config,
            "cpu",
            "float32",
            head_names=["a", "b"],
        )

        loaded = torch.load(tmp_path / "myrun.model", weights_only=False)
        assert loaded.heads == ["a", "b"]


class TestPerHeadExportFidelity:
    """The guarantee that actually matters: each exported per-head model
    must reproduce the multi-head model's prediction *for its own head*.

    The round-trip tests in test_multihead_utils.py cannot catch a
    head-index mixup, because right after conversion every head is an
    exact duplicate of the pretrained head. Here the heads are made to
    differ first.
    """

    @pytest.fixture
    def divergent_multihead_model(self, multihead_model):
        block = multihead_model.atomic_energy_output_block
        with torch.no_grad():
            block.final_layer_weights[1] += 0.5
            block.final_layer_bias[1] += 0.25
            block.layers_weights[0][1] += 0.1
        multihead_model.eval()
        return multihead_model

    def test_each_head_model_reproduces_its_own_head(
        self,
        divergent_multihead_model,
        base_config,
        tmp_path,
        monkeypatch,
        make_batch,
        h2o_atoms,
    ):
        monkeypatch.chdir(tmp_path)
        base_config["TRAINING"]["post_training_export"] = "singlehead"
        head_names = ["a", "b"]
        batch = make_batch(
            h2o_atoms, r_max=5.0, cutoff_lr=10.0, dtype=torch.float32
        )
        expected = divergent_multihead_model(
            batch.to_dict(), compute_stress=False
        )
        # Guard against a no-op perturbation making this test vacuous.
        assert not torch.allclose(
            expected["energy"][0], expected["energy"][1]
        ), "heads must differ for this test to be meaningful"

        _export_trained_model(
            divergent_multihead_model,
            base_config,
            "cpu",
            "float32",
            head_names=head_names,
        )

        for head_idx, head_name in enumerate(head_names):
            loaded = torch.load(
                tmp_path / f"myrun_{head_name}.model", weights_only=False
            )
            loaded.eval()
            out = loaded(batch.to_dict(), compute_stress=False)
            assert torch.allclose(
                out["energy"], expected["energy"][head_idx], atol=1e-10
            ), f"head '{head_name}' (idx {head_idx}) mismatch"
            assert torch.allclose(
                out["forces"], expected["forces"][head_idx], atol=1e-8
            )


class TestSingleheadOnlyExport:
    def test_writes_only_per_head_files(
        self, multihead_model, base_config, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        base_config["TRAINING"]["post_training_export"] = "singlehead"
        _export_trained_model(
            multihead_model,
            base_config,
            "cpu",
            "float32",
            head_names=["a", "b"],
        )

        assert not (tmp_path / "myrun.pth").exists()
        assert not (tmp_path / "myrun.model").exists()
        assert (tmp_path / "myrun_a.model").exists()
        assert (tmp_path / "myrun_b.model").exists()


class TestExportValidation:
    def test_singlehead_export_without_heads_raises(
        self, singlehead_model, base_config, tmp_path, monkeypatch
    ):
        monkeypatch.chdir(tmp_path)
        base_config["TRAINING"]["post_training_export"] = "singlehead"
        with pytest.raises(ValueError, match="post_training_export"):
            _export_trained_model(
                singlehead_model, base_config, "cpu", "float32"
            )
