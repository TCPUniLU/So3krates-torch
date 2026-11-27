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
