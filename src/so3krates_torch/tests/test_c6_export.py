"""Exporting C6-ratio predictions from evaluation (SO3LR-tauri)."""

import pytest
from ase.build import molecule


@pytest.fixture
def tauri_like_model(so3lr_model_config):
    from so3krates_torch.modules.models import SO3LR

    model = SO3LR(
        **{
            **so3lr_model_config,
            "c6_ratios_bool": True,
            "use_simple_hirshfeld": True,
        }
    )
    return model.eval()


def test_evaluate_model_returns_c6_ratios(tauri_like_model, device):
    from so3krates_torch.tools.eval import evaluate_model

    atoms_list = [molecule("H2O"), molecule("NH3"), molecule("CH4")]
    result = evaluate_model(
        atoms_list=atoms_list,
        model=tauri_like_model,
        batch_size=2,
        device=str(device),
        model_type="so3lr",
        multi_species=True,
        compute_c6_ratios=True,
    )
    c6 = result["c6_ratios"]
    assert [len(c) for c in c6] == [3, 4, 5]
    assert all((c > 0).all() for c in c6)
    assert (
        evaluate_model(
            atoms_list=atoms_list,
            model=tauri_like_model,
            batch_size=2,
            device=str(device),
            model_type="so3lr",
            multi_species=True,
        )["c6_ratios"]
        is None
    )


def test_ensemble_prediction_stacks_c6_ratios(tauri_like_model, device):
    from so3krates_torch.tools.eval import ensemble_prediction

    result = ensemble_prediction(
        models=[tauri_like_model, tauri_like_model],
        atoms_list=[molecule("H2O"), molecule("H2O")],
        device=str(device),
        model_type="so3lr",
        batch_size=2,
        compute_c6_ratios=True,
    )
    # same layout as hirshfeld_ratios: (models, structures, atoms)
    assert result["c6_ratios"].shape == (2, 2, 3)


def test_compute_c6_ratios_without_head_raises(so3lr_model_config, device):
    from so3krates_torch.modules.models import SO3LR
    from so3krates_torch.tools.eval import evaluate_model

    with pytest.raises(ValueError, match="c6_ratios_bool"):
        evaluate_model(
            atoms_list=[molecule("H2O")],
            model=SO3LR(**so3lr_model_config).eval(),
            batch_size=1,
            device=str(device),
            model_type="so3lr",
            compute_c6_ratios=True,
        )


def test_run_metric_accepts_c6_ratios_output_arg():
    from so3krates_torch.cli.run_metric import parse_output_args

    args = parse_output_args(["energy", "C6_ratios"])
    assert args["c6_ratios"] is True
    assert args["hirshfeld_ratios"] is False
    with pytest.raises(ValueError):
        parse_output_args(["nonsense"])


def test_eval_cli_args_have_c6_fields():
    from so3krates_torch.config.cli_models import EvalArgs

    fields = EvalArgs.model_fields
    assert fields["compute_c6_ratios"].default is False
    assert fields["c6_key"].default == "REF_c6_ratios"
