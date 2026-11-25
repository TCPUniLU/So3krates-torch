"""Tests for the ``torchkrates-convert-heads`` CLI."""

import pytest
import torch

from so3krates_torch.cli.convert_heads import main
from so3krates_torch.modules.models import MultiHeadSO3LR, SO3LR
from so3krates_torch.tools.multihead_utils import to_multihead_model


@pytest.fixture(autouse=True)
def _restore_default_dtype():
    before = torch.get_default_dtype()
    yield
    torch.set_default_dtype(before)


@pytest.fixture
def sh_file(tmp_path, so3lr_model_config):
    path = tmp_path / "sh.model"
    torch.save(SO3LR(**so3lr_model_config), path)
    return str(path)


@pytest.fixture
def mh_file(tmp_path, so3lr_model_config):
    path = tmp_path / "mh.model"
    torch.save(to_multihead_model(SO3LR(**so3lr_model_config), 3), path)
    return str(path)


def _load(path):
    return torch.load(path, weights_only=False)


def test_to_multihead(tmp_path, sh_file):
    out = str(tmp_path / "out.model")
    main(["--model", sh_file, "--output", out, "--to-multihead", "3"])
    model = _load(out)
    assert type(model) is MultiHeadSO3LR
    assert model.num_output_heads == 3
    assert model.r_max_lr == 10.0


def test_select_head(tmp_path, mh_file):
    out = str(tmp_path / "out.model")
    main(["--model", mh_file, "--output", out, "--select-head", "2"])
    assert type(_load(out)) is SO3LR


def test_seed_reproducible(tmp_path, sh_file):
    paths = [str(tmp_path / f"o{i}.model") for i in range(2)]
    for path in paths:
        main(
            ["--model", sh_file, "--output", path]
            + ["--to-multihead", "2", "--seed", "5"]
        )
    a, b = (_load(p).state_dict() for p in paths)
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_bundled_v1_keyword(tmp_path):
    out = str(tmp_path / "out.model")
    main(["--model", "v1", "--output", out, "--to-multihead", "2"])
    assert type(_load(out)) is MultiHeadSO3LR


def test_tauri_defaults_to_theory_level_5(tmp_path):
    argv = ["--model", "tauri-s-2602", "--to-multihead", "2"]
    default, explicit = (str(tmp_path / f"{n}.model") for n in "de")
    main(argv + ["--output", default])
    main(argv + ["--output", explicit, "--theory-level", "5"])
    a, b = _load(default).state_dict(), _load(explicit).state_dict()
    assert all(torch.equal(a[k], b[k]) for k in a)


def test_tauri_other_theory_level(tmp_path):
    out = str(tmp_path / "out.model")
    main(
        ["--model", "tauri-s-2602", "--output", out]
        + ["--to-multihead", "2", "--theory-level", "0"]
    )
    assert type(_load(out)) is MultiHeadSO3LR


@pytest.mark.parametrize(
    "mode",
    [["--to-multihead", "2", "--select-head", "0"], []],
    ids=["both", "neither"],
)
def test_mode_flags_exclusive(tmp_path, sh_file, mode):
    out = str(tmp_path / "out.model")
    with pytest.raises(SystemExit):
        main(["--model", sh_file, "--output", out] + mode)


def test_refuses_existing_output_without_force(tmp_path, sh_file):
    out = tmp_path / "out.model"
    out.write_text("keep me")
    argv = ["--model", sh_file, "--output", str(out), "--to-multihead", "2"]
    with pytest.raises(SystemExit):
        main(argv)
    assert out.read_text() == "keep me"
    main(argv + ["--force"])
    assert type(_load(out)) is MultiHeadSO3LR


def test_refuses_output_equal_to_input(sh_file):
    with pytest.raises(SystemExit):
        main(
            ["--model", sh_file, "--output", sh_file]
            + ["--to-multihead", "2", "--force"]
        )
    assert type(_load(sh_file)) is SO3LR


def test_state_dict_input_gives_clear_error(
    tmp_path, so3lr_model_config, capsys
):
    path = tmp_path / "sd.pt"
    torch.save(SO3LR(**so3lr_model_config).state_dict(), path)
    with pytest.raises(SystemExit):
        main(
            ["--model", str(path), "--output", str(tmp_path / "o.model")]
            + ["--to-multihead", "2"]
        )
    assert "state_dict" in capsys.readouterr().err


def test_missing_model_file(tmp_path, capsys):
    with pytest.raises(SystemExit):
        main(
            ["--model", str(tmp_path / "nope.model")]
            + ["--output", str(tmp_path / "o.model"), "--to-multihead", "2"]
        )
    assert "nope.model" in capsys.readouterr().err


def test_select_head_on_single_head_model(tmp_path, sh_file, capsys):
    with pytest.raises(SystemExit):
        main(
            ["--model", sh_file, "--output", str(tmp_path / "o.model")]
            + ["--select-head", "0"]
        )
    assert "MultiHeadSO3LR" in capsys.readouterr().err


def test_select_head_with_theory_level_rejected(tmp_path, mh_file):
    with pytest.raises(SystemExit):
        main(
            ["--model", mh_file, "--output", str(tmp_path / "o.model")]
            + ["--select-head", "0", "--theory-level", "1"]
        )
