"""End-to-end smoke test for multi-head replay fine-tuning.

Exercises the full user-facing workflow: start from a pretrained
single-head model, convert it to multi-head (duplicating the
pretrained head into every new head), fine-tune with the normal
dataset routed to one head and a replay dataset routed to another,
and export according to TRAINING.post_training_export.
"""

import numpy as np
import pytest
import torch
from ase.build import molecule
import ase.io

from so3krates_torch.config import TrainConfig
from so3krates_torch.modules.models import SO3LR
from so3krates_torch.cli.run_train import run_training


def _make_xyz(path, n_structures, seed):
    rng = np.random.default_rng(seed)
    mols = ["H2O", "NH3", "CH4"]
    atoms_list = []
    for i in range(n_structures):
        atoms = molecule(mols[i % len(mols)])
        atoms.info["REF_energy"] = -10.0 * len(atoms)
        atoms.arrays["REF_forces"] = rng.normal(size=(len(atoms), 3)) * 0.1
        atoms_list.append(atoms)
    ase.io.write(path, atoms_list)


def _small_so3lr_kwargs():
    return dict(
        r_max=5.0,
        r_max_lr=None,
        num_radial_basis_fn=4,
        degrees=[1],
        num_features=8,
        num_heads=1,
        num_layers=1,
        num_elements=118,
        energy_regression_dim=8,
        final_mlp_layers=1,
        zbl_repulsion_bool=False,
        electrostatic_energy_bool=False,
        dispersion_energy_bool=False,
        dtype=torch.float32,
        seed=1,
    )


def _base_config(tmp_path, finetune_path, replay_path, pretrained_path):
    return {
        "GENERAL": {
            "name_exp": str(tmp_path / "e2e_run"),
            "checkpoints_dir": str(tmp_path / "ckpt"),
            "log_dir": str(tmp_path / "logs"),
            "default_dtype": "float32",
            "seed": 1,
        },
        "ARCHITECTURE": {
            "degrees": [1],
            "r_max": 5.0,
            "r_max_lr": None,
            "num_features": 8,
            "num_heads": 1,
            "num_layers": 1,
            "num_radial_basis_fn": 4,
            "energy_regression_dim": 8,
            "final_mlp_layers": 1,
            "zbl_repulsion_bool": False,
            "electrostatic_energy_bool": False,
            "dispersion_energy_bool": False,
            "convert_to_multihead": True,
            "use_multihead": True,
            "num_output_heads": 2,
        },
        "TRAINING": {
            "batch_size": 2,
            "valid_batch_size": 2,
            "lr": 1e-3,
            "num_epochs": 1,
            "patience": 1,
            "path_to_train_data": str(finetune_path),
            "pretrained_model": str(pretrained_path),
            "finetune_choice": "naive",
            "replay_as_heads": True,
            "replay_datasets": [str(replay_path)],
            "replay_fractions": [1.0],
            "replay_total": 10,
            "post_training_export": "multihead+singlehead",
        },
        "MISC": {
            "device": "cpu",
            "no_checkpoint": True,
            "restart_latest": False,
        },
    }


@pytest.fixture
def workflow_paths(tmp_path):
    finetune_path = tmp_path / "finetune.xyz"
    replay_path = tmp_path / "replay.xyz"
    pretrained_path = tmp_path / "pretrained.model"
    _make_xyz(finetune_path, 20, seed=0)
    _make_xyz(replay_path, 20, seed=1)
    torch.set_default_dtype(torch.float32)
    pretrained_model = SO3LR(**_small_so3lr_kwargs())
    torch.save(pretrained_model, pretrained_path)
    return finetune_path, replay_path, pretrained_path


@pytest.mark.parametrize(
    "export_mode,expect_combined,expect_per_head",
    [
        ("multihead", True, False),
        ("multihead+singlehead", True, True),
        ("singlehead", False, True),
    ],
)
def test_run_training_exports_per_post_training_export_choice(
    tmp_path, workflow_paths, export_mode, expect_combined, expect_per_head
):
    finetune_path, replay_path, pretrained_path = workflow_paths
    raw_config = _base_config(
        tmp_path, finetune_path, replay_path, pretrained_path
    )
    raw_config["TRAINING"]["post_training_export"] = export_mode
    raw_config["GENERAL"]["name_exp"] = str(tmp_path / f"run_{export_mode}")
    config = TrainConfig.model_validate(raw_config).model_dump()

    run_training(config)

    name_exp = config["GENERAL"]["name_exp"]
    combined_model_path = f"{name_exp}.model"
    finetune_head_path = f"{name_exp}_finetune.model"
    replay_head_path = f"{name_exp}_replay_0.model"

    import os

    assert os.path.exists(combined_model_path) == expect_combined
    assert os.path.exists(finetune_head_path) == expect_per_head
    assert os.path.exists(replay_head_path) == expect_per_head

    if expect_per_head:
        loaded = torch.load(finetune_head_path, weights_only=False)
        assert isinstance(loaded, SO3LR)
        assert not hasattr(loaded, "num_output_heads")
        # Loadable and runnable like any normal single-head model.
        loaded.eval()


def test_replay_as_heads_checkpoints_load_with_default_weights_only(
    tmp_path, workflow_paths
):
    """The synthesized heads dict (with its live ASE Atoms) must never
    reach config["TRAINING"], since CheckpointBuilder pickles config
    into every checkpoint and torch >=2.6 defaults torch.load to
    weights_only=True — which raises on unpickling ASE Atoms/numpy
    objects. A checkpoint written mid-run must load back with that
    default, exactly as CheckpointIO.load calls it."""
    import glob

    finetune_path, replay_path, pretrained_path = workflow_paths
    raw_config = _base_config(
        tmp_path, finetune_path, replay_path, pretrained_path
    )
    # Actually exercise checkpoint writing (the other tests disable it).
    raw_config["MISC"]["no_checkpoint"] = False
    config = TrainConfig.model_validate(raw_config).model_dump()

    run_training(config)

    # name_exp is an absolute path (tmp_path / "e2e_run"), and
    # CheckpointIO's filename is "<tag>_epoch-N.pt" — since that
    # filename is itself absolute, os.path.join(checkpoints_dir,
    # filename) discards checkpoints_dir and resolves next to name_exp.
    name_exp = config["GENERAL"]["name_exp"]
    ckpt_files = glob.glob(f"{name_exp}_epoch-*.pt")
    assert ckpt_files, "expected at least one checkpoint to be written"
    for ckpt_path in ckpt_files:
        # This is torch's own default (no weights_only kwarg) — the
        # same call CheckpointIO.load makes.
        torch.load(ckpt_path, map_location="cpu")
