"""Regression tests for the model-loading logic in ``cli/run_eval.py``
and ``cli/run_metric.py`` (via ``tools.utils.ensemble_from_folder``).

Covers the original pickled-``.model``/directory-of-``.model``
behavior (unchanged) plus the new additive paths: a pretrained
keyword (``"v1"``/``"v2-s"``/``"v2-m"``/``"v2-l"``) and a bare
``.pt`` state_dict with a sibling ``<stem>_settings.yaml``.
"""

import yaml
import torch
import pytest

from so3krates_torch.cli.run_eval import run_evaluation
from so3krates_torch.modules.models import SO3LR
from so3krates_torch.tools.utils import ensemble_from_folder


def _write_pt_checkpoint(tmp_path, so3lr_model_config, stem="my_so3lr"):
    """Save a small SO3LR model as a ``<stem>.pt`` state_dict plus a
    sibling ``<stem>_settings.yaml``, matching the bundled-checkpoint
    naming convention consumed by ``settings_path_for_checkpoint``."""
    model = SO3LR(**so3lr_model_config)
    pt_path = tmp_path / f"{stem}.pt"
    torch.save(model.state_dict(), pt_path)

    settings_path = tmp_path / f"{stem}_settings.yaml"
    architecture = dict(so3lr_model_config)
    architecture["dtype"] = "float64"
    with open(settings_path, "w") as f:
        yaml.safe_dump({"ARCHITECTURE": architecture}, f)

    return model, pt_path, settings_path


class TestRunEvaluationModelLoading:
    """``run_evaluation``'s ``model_path`` resolution."""

    def test_loads_single_dot_model_file(
        self, so3lr_model_config, example_xyz_with_data, tmp_path
    ):
        """Unchanged behavior: a single pickled ``.model`` file."""
        model = SO3LR(**so3lr_model_config)
        model_path = tmp_path / "model.model"
        torch.save(model, model_path)

        result, is_ensemble = run_evaluation(
            model_path=str(model_path),
            data_path=example_xyz_with_data,
            device="cpu",
            dtype="float64",
            multispecies=True,
        )
        assert is_ensemble is False
        assert "energies" in result

    def test_loads_directory_of_dot_model_files(
        self, so3lr_model_config, example_xyz_with_data, tmp_path
    ):
        """Unchanged behavior: a directory containing ``*.model``
        files is globbed and loaded as an ensemble."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        for i in range(2):
            model = SO3LR(**so3lr_model_config)
            torch.save(model, model_dir / f"model_{i}.model")

        result, is_ensemble = run_evaluation(
            model_path=str(model_dir),
            data_path=example_xyz_with_data,
            device="cpu",
            dtype="float64",
            multispecies=True,
        )
        assert is_ensemble is True
        assert "energies" in result

    def test_loads_pretrained_keyword(self, example_xyz_with_data):
        """New: ``model_path`` matching one of the 4 pretrained
        keywords loads via ``load_pretrained_so3lr`` directly.

        Uses ``"v1"`` rather than a v2 checkpoint: v2 checkpoints have
        ``num_theory_levels > 1`` and require a ``theory_level`` to be
        threaded through the batch (via ``evaluate_model``'s
        ``theory_level_override``, not currently exposed by
        ``run_evaluation``/this CLI) -- an orthogonal, pre-existing gap
        unrelated to the model-loading logic under test here.
        """
        result, is_ensemble = run_evaluation(
            model_path="v1",
            data_path=example_xyz_with_data,
            device="cpu",
            dtype="float32",
            multispecies=True,
            # v1's saved ARCHITECTURE.r_max_lr is null (SO3LRCalculator
            # normally overrides this attribute post-construction); the
            # CLI's --r_max_lr does the equivalent job here.
            r_max_lr=12.0,
            dispersion_energy_cutoff_lr_damping=2.0,
        )
        assert is_ensemble is False
        assert "energies" in result
        assert len(result["energies"]) == 3  # 3 structures in fixture

    def test_loads_pt_with_sibling_settings_yaml(
        self, so3lr_model_config, example_xyz_with_data, tmp_path
    ):
        """New: ``model_path`` ending in ``.pt`` looks up a sibling
        ``<stem>_settings.yaml`` and constructs+loads via the shared
        helper (same path as ``load_pretrained_so3lr``)."""
        _, pt_path, _ = _write_pt_checkpoint(tmp_path, so3lr_model_config)

        result, is_ensemble = run_evaluation(
            model_path=str(pt_path),
            data_path=example_xyz_with_data,
            device="cpu",
            dtype="float64",
            multispecies=True,
        )
        assert is_ensemble is False
        assert "energies" in result

    def test_pt_without_sibling_settings_yaml_raises(
        self, so3lr_model_config, example_xyz_with_data, tmp_path
    ):
        """A ``.pt`` file with no matching ``_settings.yaml`` sibling
        must fail loudly rather than silently misbehaving."""
        model = SO3LR(**so3lr_model_config)
        pt_path = tmp_path / "orphan.pt"
        torch.save(model.state_dict(), pt_path)

        with pytest.raises(FileNotFoundError):
            run_evaluation(
                model_path=str(pt_path),
                data_path=example_xyz_with_data,
                device="cpu",
                dtype="float64",
            )


class TestEnsembleFromFolderModelLoading:
    """``tools.utils.ensemble_from_folder``'s ``path_to_models``
    resolution, used by ``cli/run_metric.py``."""

    def test_loads_directory_of_dot_model_files(
        self, so3lr_model_config, tmp_path
    ):
        """Unchanged behavior: directory of pickled ``.model`` files."""
        model_dir = tmp_path / "models"
        model_dir.mkdir()
        for i in range(2):
            model = SO3LR(**so3lr_model_config)
            torch.save(model, model_dir / f"model_{i}.model")

        ensemble = ensemble_from_folder(
            path_to_models=str(model_dir), device="cpu", dtype=torch.float64
        )
        assert len(ensemble) == 2
        assert all(isinstance(m, torch.nn.Module) for m in ensemble.values())

    def test_loads_pretrained_keyword(self):
        """New: a pretrained keyword loads via ``load_pretrained_so3lr``
        directly instead of requiring a directory."""
        ensemble = ensemble_from_folder(
            path_to_models="v1", device="cpu", dtype=torch.float32
        )
        assert list(ensemble.keys()) == ["v1"]
        assert isinstance(ensemble["v1"], torch.nn.Module)

    def test_loads_pt_with_sibling_settings_yaml(
        self, so3lr_model_config, tmp_path
    ):
        """New: a bare ``.pt`` with a sibling ``_settings.yaml`` is
        constructed+loaded via the shared helper."""
        _, pt_path, _ = _write_pt_checkpoint(
            tmp_path, so3lr_model_config, stem="my_so3lr"
        )

        ensemble = ensemble_from_folder(
            path_to_models=str(pt_path), device="cpu", dtype=torch.float64
        )
        assert list(ensemble.keys()) == ["my_so3lr"]
        assert isinstance(ensemble["my_so3lr"], torch.nn.Module)
