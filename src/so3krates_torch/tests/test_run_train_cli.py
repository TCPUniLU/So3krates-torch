"""`torchkrates-train --dry-run` and the shipped example configs."""

import logging
from pathlib import Path

import pytest
import torch
import yaml

from so3krates_torch.cli.run_train import (
    run_dry_run,
    setup_config_from_yaml,
)

REPO = Path(__file__).resolve().parents[3]
EXAMPLES = REPO / "examples/training"
ANCHORING_EXAMPLE = EXAMPLES / "anchoring/finetune_anchoring.yaml"


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(dtype)


def _redirect_outputs(config, tmp_path):
    for key in ("checkpoints_dir", "model_dir", "log_dir"):
        config["GENERAL"][key] = str(tmp_path / key)
    return config


def _dry_run(config):
    run_dry_run(config)
    # run_dry_run installs its own log handlers, so read its log file.
    log = "".join(
        p.read_text() for p in Path(config["GENERAL"]["log_dir"]).glob("*.log")
    )
    assert "Dry-run completed successfully" in log


def test_dry_run_computes_forces(tmp_path):
    raw = yaml.safe_load(
        (EXAMPLES / "train_settings_example.yaml").read_text()
    )
    raw["TRAINING"]["path_to_train_data"] = str(
        EXAMPLES / "md17_ethanol_small.xyz"
    )
    raw["ARCHITECTURE"].update(num_features=16, degrees=[1], num_layers=1)
    config_path = tmp_path / "config.yaml"
    config_path.write_text(yaml.safe_dump(raw))
    config = _redirect_outputs(
        setup_config_from_yaml(str(config_path)), tmp_path
    )

    _dry_run(config)


def test_anchoring_example_config_is_valid_and_runs(tmp_path, monkeypatch):
    # The example's paths are relative to the repository root.
    monkeypatch.chdir(REPO)
    config = setup_config_from_yaml(str(ANCHORING_EXAMPLE))
    training = config["TRAINING"]
    assert Path(training["path_to_train_data"]).is_file()
    assert Path(training["pretrained_weights"]).is_file()
    assert training["long_range_anchor_weights"]

    _dry_run(_redirect_outputs(config, tmp_path))


def _att_norms(model):
    return [
        (
            layer.euclidean_attention_block.att_norm_inv,
            layer.euclidean_attention_block.att_norm_ev,
        )
        for layer in model.euclidean_transformers
    ]


def test_pretrained_weights_keep_avg_num_neighbors(monkeypatch):
    """The state dict has no avg_num_neighbors; warm-starting from it must
    use ARCHITECTURE.avg_num_neighbors (copied from the checkpoint's
    settings), as the bundled model does."""
    from so3krates_torch.calculator.so3 import load_pretrained_so3lr
    from so3krates_torch.tools.model_setup import _setup_model_for_training

    monkeypatch.chdir(REPO)
    config = setup_config_from_yaml(str(ANCHORING_EXAMPLE))
    model, warm_start = _setup_model_for_training(config, torch.device("cpu"))
    reference = load_pretrained_so3lr("v1")

    assert warm_start
    assert model.avg_num_neighbors == pytest.approx(13.168995780096482)
    assert model.avg_num_neighbors == reference.avg_num_neighbors
    assert _att_norms(model) == _att_norms(reference)


def test_pretrained_weights_without_avg_num_neighbors_warns(
    monkeypatch, caplog
):
    from so3krates_torch.tools.model_setup import _setup_model_for_training

    monkeypatch.chdir(REPO)
    config = setup_config_from_yaml(str(ANCHORING_EXAMPLE))
    del config["ARCHITECTURE"]["avg_num_neighbors"]
    with caplog.at_level(logging.WARNING):
        model, _ = _setup_model_for_training(config, torch.device("cpu"))

    assert model.avg_num_neighbors == 1.0
    assert "ARCHITECTURE.avg_num_neighbors is not set" in caplog.text
