"""Models as a state dict (``<stem>.pt``/``<stem>.pth``) plus
``<stem>_settings.yaml``: single-head, multi-theory-level and multi-head
models load wherever a pickled ``.model`` does."""

import pytest
import torch
import yaml
from ase.build import molecule

from so3krates_torch.calculator.so3 import load_so3lr_model
from so3krates_torch.modules.models import MultiHeadSO3LR, SO3LR
from so3krates_torch.tests.test_multihead_conversion import _trained_like
from so3krates_torch.tools.model_setup import set_avg_num_neighbors_in_model


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    dtype = torch.get_default_dtype()
    yield
    torch.set_default_dtype(dtype)


def _settings(config, **extra):
    settings = {**config, **extra}
    settings["dtype"] = str(settings.get("dtype", "float64")).rsplit(".")[-1]
    return settings


def _save_pair(model, tmp_path, settings, suffix=".pt", stem="model"):
    path = tmp_path / f"{stem}{suffix}"
    torch.save(model.state_dict(), path)
    with open(tmp_path / f"{stem}_settings.yaml", "w") as f:
        yaml.safe_dump({"ARCHITECTURE": settings}, f)
    return path


def _predict(model, make_batch, theory_level=None):
    atoms = molecule("CH3CH2OH")
    atoms.rattle(0.02, seed=3)
    if theory_level is not None:
        atoms.info["theory_level"] = theory_level
    batch = make_batch(atoms, r_max=5.0, cutoff_lr=10.0)
    model = model.double().eval()
    out = model(batch.to_dict(), compute_stress=False)
    return out["energy"].detach(), out["forces"].detach()


def _assert_same_predictions(a, b, make_batch, **kwargs):
    for x, y in zip(
        _predict(a, make_batch, **kwargs), _predict(b, make_batch, **kwargs)
    ):
        torch.testing.assert_close(x, y)


@pytest.fixture
def multihead(multihead_model_config):
    torch.manual_seed(0)
    model = MultiHeadSO3LR(**multihead_model_config).double()
    head = model.atomic_energy_output_block
    with torch.no_grad():  # as a trained model: observable shifts/scales
        head.energy_shifts.normal_()
        head.energy_scales.uniform_(0.5, 2.0)
    return model


@pytest.mark.parametrize("suffix", [".pt", ".pth"])
def test_multihead_pair_round_trips(
    multihead, multihead_model_config, tmp_path, make_batch, suffix
):
    path = _save_pair(
        multihead, tmp_path, _settings(multihead_model_config), suffix
    )

    loaded = load_so3lr_model(str(path))

    assert type(loaded) is MultiHeadSO3LR
    assert loaded.num_output_heads == multihead.num_output_heads
    assert loaded.select_heads == multihead.select_heads
    assert loaded.return_mean == multihead.return_mean
    _assert_same_predictions(loaded, multihead, make_batch)


@pytest.mark.parametrize("levels", [1, 3])
def test_single_head_pairs_round_trip(
    so3lr_model_config, tmp_path, make_batch, levels
):
    torch.manual_seed(0)
    model = _trained_like(
        SO3LR(**so3lr_model_config, num_theory_levels=levels)
    ).double()
    path = _save_pair(
        model,
        tmp_path,
        _settings(so3lr_model_config, num_theory_levels=levels),
        ".pth",
    )

    loaded = load_so3lr_model(str(path))

    assert type(loaded) is SO3LR
    theory_level = 2 if levels > 1 else None
    _assert_same_predictions(
        loaded, model, make_batch, theory_level=theory_level
    )


def test_checkpoint_dict_with_settings_is_unwrapped(
    multihead, multihead_model_config, tmp_path, make_batch
):
    path = tmp_path / "run_epoch-3.pt"
    torch.save({"model": multihead.state_dict(), "epochs": 3}, path)
    with open(tmp_path / "run_epoch-3_settings.yaml", "w") as f:
        yaml.safe_dump({"ARCHITECTURE": _settings(multihead_model_config)}, f)

    loaded = load_so3lr_model(str(path))

    assert type(loaded) is MultiHeadSO3LR
    _assert_same_predictions(loaded, multihead, make_batch)


def test_settings_and_weights_head_count_mismatch_raises(
    multihead, multihead_model_config, tmp_path
):
    settings = _settings(multihead_model_config, num_output_heads=2)
    path = _save_pair(multihead, tmp_path, settings)
    with pytest.raises(ValueError, match="2.*4|4.*2"):
        load_so3lr_model(str(path))


def test_multihead_settings_with_single_head_weights_raises(
    so3lr_model_config, multihead_model_config, tmp_path
):
    model = SO3LR(**so3lr_model_config)
    path = _save_pair(model, tmp_path, _settings(multihead_model_config))
    with pytest.raises(ValueError, match="num_output_heads"):
        load_so3lr_model(str(path))


def test_load_so3lr_model_pickled_and_bundled(multihead, tmp_path):
    path = tmp_path / "trained.model"
    torch.save(multihead, path)

    assert type(load_so3lr_model(str(path))) is MultiHeadSO3LR
    assert type(load_so3lr_model("v1")) is SO3LR


def test_overrides_on_pickled_model_raise(multihead, tmp_path):
    path = tmp_path / "trained.model"
    torch.save(multihead, path)
    with pytest.raises(ValueError, match="use_pme"):
        load_so3lr_model(str(path), use_pme=True)


# --- settings written from a model ----------------------------------------


def _round_trip(model, tmp_path, suffix=".pth"):
    from so3krates_torch.calculator.so3 import model_settings

    settings = model_settings(model)
    path = _save_pair(model, tmp_path, settings, suffix)  # YAML-safe
    return load_so3lr_model(str(path))


@pytest.mark.parametrize(
    "kind",
    [
        "single",
        "theory_levels",
        "multihead",
        "no_final_bias",
        "pme",
        "nhl_c6_simple_hirshfeld",
    ],
)
def test_model_settings_rebuild_the_model(
    kind, so3lr_model_config, multihead, tmp_path, make_batch
):
    torch.manual_seed(1)
    extra = {
        "single": {},
        "theory_levels": {"num_theory_levels": 3},
        "no_final_bias": {"final_layer_bias": False},
        "pme": {"use_pme": True, "pme_smearing": 1.1},
        "nhl_c6_simple_hirshfeld": {
            "nhl_repulsion_bool": True,
            "zbl_repulsion_bool": False,
            "c6_ratios_bool": True,
            "use_simple_hirshfeld": True,
        },
    }
    if kind == "multihead":
        model = multihead
    else:
        model = _trained_like(SO3LR(**{**so3lr_model_config, **extra[kind]}))
        model = model.double()
        # changed after construction, as training does
        set_avg_num_neighbors_in_model(model, 7.5)
        model.r_max_lr = 9.0

    loaded = _round_trip(model, tmp_path)

    assert type(loaded) is type(model)
    assert loaded.avg_num_neighbors == model.avg_num_neighbors
    assert loaded.r_max_lr == model.r_max_lr
    level = 2 if kind == "theory_levels" else None
    _assert_same_predictions(loaded, model, make_batch, theory_level=level)


@pytest.mark.parametrize("name", ["v1", "tauri-s-2606"])
def test_model_settings_of_bundled_models(name, tmp_path, make_batch):
    from so3krates_torch.calculator.so3 import load_pretrained_so3lr

    model = load_pretrained_so3lr(
        name, r_max_lr=10.0, dispersion_energy_cutoff_lr_damping=2.0
    )
    loaded = _round_trip(model, tmp_path)
    level = 5 if name.startswith("tauri") else None
    _assert_same_predictions(loaded, model, make_batch, theory_level=level)


# --- training writes <name>_settings.yaml ---------------------------------


def _pretrained(config, tmp_path, **arch):
    from so3krates_torch.tools.model_setup import create_model

    cfg = {**config, "ARCHITECTURE": {**config["ARCHITECTURE"], **arch}}
    path = tmp_path / "pretrained.model"
    torch.save(create_model(cfg, "cpu"), path)
    return str(path)


@pytest.mark.parametrize(
    "variant", ["plain", "multihead", "one_theory_level", "lora"]
)
def test_training_writes_settings_that_rebuild_the_model(
    variant, tmp_path, monkeypatch, make_batch
):
    from so3krates_torch.cli.run_train import run_training
    from so3krates_torch.tests.test_run_train_cli import _small_config

    config = _small_config(tmp_path, num_epochs=1)
    arch, training = config["ARCHITECTURE"], config["TRAINING"]
    if variant == "multihead":
        training["pretrained_model"] = _pretrained(config, tmp_path)
        arch.update(convert_to_multihead=True, num_output_heads=2)
        training["finetune_choice"] = "naive"
    elif variant == "one_theory_level":
        training["pretrained_model"] = _pretrained(
            config, tmp_path, num_theory_levels=3
        )
        arch["convert_theory_level"] = 1
        training["finetune_choice"] = "naive"
    elif variant == "lora":
        training["pretrained_model"] = _pretrained(config, tmp_path)
        training["finetune_choice"] = "lora"
    monkeypatch.chdir(tmp_path)

    run_training(config)

    model_dir = tmp_path / "model_dir"
    name = config["GENERAL"]["name_exp"]
    assert (model_dir / f"{name}_settings.yaml").is_file()
    pair = load_so3lr_model(str(model_dir / f"{name}.pth"))
    pickled = torch.load(model_dir / f"{name}.model", weights_only=False)
    assert type(pair) is type(pickled)
    _assert_same_predictions(pair, pickled, make_batch)


# --- every tool accepts a .pt/.pth + settings pair --------------------------


@pytest.fixture
def mh_files(multihead, tmp_path):
    """A multi-head model as training saves it: .model, .pth, settings."""
    from so3krates_torch.calculator.so3 import model_settings

    pair = _save_pair(
        multihead, tmp_path, model_settings(multihead), ".pth", "mh"
    )
    pickled = tmp_path / "mh.model"
    torch.save(multihead, pickled)
    return str(pair), str(pickled)


def _energy(calc):
    atoms = molecule("CH3CH2OH")
    atoms.calc = calc
    return atoms.get_potential_energy()


def test_eval_multihead_pair(mh_files, example_xyz_with_data):
    from so3krates_torch.cli.run_eval import run_evaluation

    results = [
        run_evaluation(
            model_path=path,
            data_path=example_xyz_with_data,
            device="cpu",
            dtype="float64",
            multispecies=True,
            multihead_model=True,
        )[0]
        for path in mh_files
    ]
    for a, b in zip(results[0]["energies"], results[1]["energies"]):
        torch.testing.assert_close(torch.as_tensor(a), torch.as_tensor(b))


def test_metric_loads_multihead_pair(mh_files):
    from so3krates_torch.tools.utils import ensemble_from_folder

    ensemble = ensemble_from_folder(mh_files[0], "cpu", torch.float64)
    assert [type(m) for m in ensemble.values()] == [MultiHeadSO3LR]


def test_md_loads_multihead_pair(mh_files, make_batch):
    from so3krates_torch.cli.run_md_nvalchemi import _load_raw_model

    pair, pickled = (_load_raw_model(p, "cpu") for p in mh_files)
    assert type(pair) is MultiHeadSO3LR
    _assert_same_predictions(pair, pickled, make_batch)


def test_calculators_accept_pairs(mh_files, so3lr_model_config, tmp_path):
    from so3krates_torch.calculator.so3 import (
        MultiHeadSO3LRCalculator,
        TorchkratesCalculator,
        model_settings,
    )

    energies = [
        _energy(
            MultiHeadSO3LRCalculator(
                model_path=p, device="cpu", default_dtype="float64"
            )
        )
        for p in mh_files
    ]
    assert energies[0] == pytest.approx(energies[1], abs=1e-8)

    single = SO3LR(**so3lr_model_config).double()
    pair = _save_pair(single, tmp_path, model_settings(single), ".pt", "sh")
    pickled = tmp_path / "sh.model"
    torch.save(single, pickled)
    energies = [
        _energy(
            TorchkratesCalculator(model_paths=str(p), default_dtype="float64")
        )
        for p in (pair, pickled)
    ]
    assert energies[0] == pytest.approx(energies[1], abs=1e-8)


def test_lammps_export_of_pair(mh_files, monkeypatch):
    import sys

    from so3krates_torch.cli import create_lammps_model

    monkeypatch.setattr(
        sys,
        "argv",
        ["x", mh_files[0], "--elements", "C", "H", "O"],
    )
    create_lammps_model.main()
    exported = torch.load(mh_files[0] + "-mliap_lammps.pt", weights_only=False)
    assert type(exported.model.model) is MultiHeadSO3LR


def test_convert_heads_select_head_from_pair(mh_files, tmp_path):
    from so3krates_torch.cli.convert_heads import main as convert_heads

    out = tmp_path / "head1.model"
    convert_heads(
        ["--model", mh_files[0], "--select-head", "1", "--output", str(out)]
    )
    assert type(torch.load(out, weights_only=False)) is SO3LR
