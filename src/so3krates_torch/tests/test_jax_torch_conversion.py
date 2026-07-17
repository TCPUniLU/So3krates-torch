import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("mlff")

import numpy as np
import torch

# Captured *before* importing `v1_stagewise_parity` below, since that
# import already triggers its module-level
# `torch.set_default_dtype(torch.float64)` side effect. Capturing the
# default dtype inside the `_restore_default_dtype` fixture body would
# run after collection/import and would therefore observe the
# already-mutated float64 value, making the fixture a no-op.
_PRE_IMPORT_DEFAULT_DTYPE = torch.get_default_dtype()

from so3krates_torch.scripts.v1_stagewise_parity import (
    JAX_V1_CONFIG,
    V1_TORCH_SETTINGS,
    build_jax_v1,
    build_jax_v1_config_from_settings,
    build_torch_v1,
    jax_to_torch,
    run_weight_roundtrip_check,
    torch_to_jax,
)
from so3krates_torch.tools.jax_torch_conversion import (
    flatten_params,
    get_flax_to_torch_mapping,
    get_model_settings_flax_to_torch,
    get_model_settings_torch_to_flax,
    get_torch_to_flax_mapping,
)
from so3krates_torch.tools.model_parity import check_model_parity

# Regression tests for `so3krates_torch.tools.jax_torch_conversion`.
#
# Importing `v1_stagewise_parity` (a standalone JAX<->Torch parity dev
# script, not itself a pytest test -- see its module docstring) has a
# module-level side effect of `torch.set_default_dtype(torch.float64)`,
# needed there for the parity math. The `_restore_default_dtype`
# fixture below guards against that leaking into other test modules
# that run later in the same pytest session, using the default dtype
# captured in `_PRE_IMPORT_DEFAULT_DTYPE` above -- before the import ran.
#
# These tests exercise the shared conversion primitives
# (`jax_to_torch`/`torch_to_jax`/`get_model_settings_flax_to_torch`/
# `get_torch_to_flax_mapping`) via the small "v1" model built by
# `v1_stagewise_parity`, reusing its fixtures rather than reimplementing
# them.


@pytest.fixture(scope="module", autouse=True)
def _restore_default_dtype():
    yield
    torch.set_default_dtype(_PRE_IMPORT_DEFAULT_DTYPE)


def test_weight_roundtrip_passes():
    """General regression net for the shared conversion primitives:
    jax -> torch -> jax and torch -> jax -> torch must round-trip.
    """
    assert run_weight_roundtrip_check() is True


def test_final_layer_bias_detected_from_checkpoint():
    """Regression test for bug 1: `final_layer_bias` must be detected
    from `flat_params` instead of hardcoded `False`.

    flax `nn.Dense` defaults to `use_bias=True`, so `energy_dense_final`
    has a bias -- the common real-checkpoint case. Before Task 1's fix,
    `jax_to_torch` raised `KeyError` for this case (the torch model was
    built with `final_layer_bias=False`, so its state dict had no
    `atomic_energy_output_block.final_layer.bias` key to load into).
    """
    model_jax, flax_params, cfg = build_jax_v1(JAX_V1_CONFIG)
    flat_params = flatten_params(flax_params)
    assert "params/observables_0/energy_dense_final/bias" in flat_params

    torch_model = jax_to_torch(cfg, flax_params, V1_TORCH_SETTINGS)

    bias = torch_model.atomic_energy_output_block.final_layer.bias
    assert bias is not None

    expected = flat_params["params/observables_0/energy_dense_final/bias"]
    np.testing.assert_allclose(
        bias.detach().cpu().numpy(),
        np.asarray(expected),
        atol=1e-10,
        rtol=1e-8,
    )


def test_energy_offset_scales_not_injected_when_not_learned():
    """Regression test for bug 2: `energy_offset`/`atomic_scales` must
    only be mapped torch->flax when
    `energy_learn_atomic_type_shifts`/`scales` is True (symmetric with
    the flax->torch direction).

    `V1_TORCH_SETTINGS` has both flags False. Before Task 1's fix, a
    torch->jax export always produced these two keys regardless of the
    flags.
    """
    assert V1_TORCH_SETTINGS["energy_learn_atomic_type_shifts"] is False
    assert V1_TORCH_SETTINGS["energy_learn_atomic_type_scales"] is False

    torch_model = build_torch_v1(V1_TORCH_SETTINGS)
    _cfg, flax_params = torch_to_jax(torch_model, V1_TORCH_SETTINGS)
    flat_params = flatten_params(flax_params)

    assert "params/observables_0/energy_offset" not in flat_params
    assert "params/observables_0/atomic_scales" not in flat_params


def test_model_parity_matches_for_v1_pair():
    """check_model_parity must report PASS for the v1 JAX/torch pair,
    which is independently known to match (see the parity script)."""
    model_jax, flax_params, cfg = build_jax_v1(JAX_V1_CONFIG)
    torch_model = jax_to_torch(cfg, flax_params, V1_TORCH_SETTINGS)
    assert check_model_parity(
        cfg,
        flax_params,
        torch_model,
        r_max=V1_TORCH_SETTINGS["r_max"],
        r_max_lr=V1_TORCH_SETTINGS["r_max_lr"],
    )


# ---------------------------------------------------------------------
# End-to-end CLI tests for `--check_parity` wiring (torchkrates-jax2torch
# / torchkrates-torch2jax). These exercise the real CLI `main()`
# entrypoints (argv -> argparse -> Pydantic validation -> conversion ->
# parity check), not the underlying conversion/parity primitives, which
# are already covered above.
# ---------------------------------------------------------------------

import json
import pickle
import sys

import yaml

from so3krates_torch.cli.jax_to_torch import main as jax_to_torch_main
from so3krates_torch.cli.torch_to_jax import main as torch_to_jax_main


def test_jax2torch_cli_check_parity_passes_by_default(tmp_path, monkeypatch):
    """Round-trip a real v1 JAX model through `torchkrates-jax2torch`
    and confirm the CLI wiring for `--check_parity` (default True)
    actually runs the check and still reports overall success.
    """
    _model_jax, flax_params, cfg = build_jax_v1(JAX_V1_CONFIG)

    params_path = tmp_path / "params.pkl"
    with open(params_path, "wb") as f:
        pickle.dump(flax_params, f)

    hyperparams_path = tmp_path / "hyperparameters.json"
    with open(hyperparams_path, "w") as f:
        json.dump(cfg.to_dict(), f)

    save_model_path = tmp_path / "so3lr_torch.model"

    argv = [
        "torchkrates-jax2torch",
        "--path_to_params",
        str(params_path),
        "--path_to_hyperparams",
        str(hyperparams_path),
        "--save_model_path",
        str(save_model_path),
        "--dtype",
        "float64",
    ]
    if V1_TORCH_SETTINGS["trainable_rbf"]:
        argv.append("--trainable_rbf")

    monkeypatch.setattr(sys, "argv", argv)
    assert jax_to_torch_main() == 0
    assert save_model_path.exists()


def test_jax2torch_cli_check_parity_flag_can_be_disabled(
    tmp_path, monkeypatch
):
    """Same setup as above, but with `--no-check_parity` -- proves the
    CLI accepts the disabling flag and completes without running (or
    being blocked by) the parity check.
    """
    _model_jax, flax_params, cfg = build_jax_v1(JAX_V1_CONFIG)

    params_path = tmp_path / "params.pkl"
    with open(params_path, "wb") as f:
        pickle.dump(flax_params, f)

    hyperparams_path = tmp_path / "hyperparameters.json"
    with open(hyperparams_path, "w") as f:
        json.dump(cfg.to_dict(), f)

    save_model_path = tmp_path / "so3lr_torch.model"

    argv = [
        "torchkrates-jax2torch",
        "--path_to_params",
        str(params_path),
        "--path_to_hyperparams",
        str(hyperparams_path),
        "--save_model_path",
        str(save_model_path),
        "--dtype",
        "float64",
        "--no-check_parity",
    ]
    if V1_TORCH_SETTINGS["trainable_rbf"]:
        argv.append("--trainable_rbf")

    monkeypatch.setattr(sys, "argv", argv)
    assert jax_to_torch_main() == 0
    assert save_model_path.exists()


def test_torch2jax_cli_check_parity_passes_by_default(tmp_path, monkeypatch):
    """Reverse direction: round-trip a real v1 torch model through
    `torchkrates-torch2jax` and confirm the CLI wiring for
    `--check_parity` (default True) actually runs the check and still
    reports overall success.
    """
    torch_model = build_torch_v1(V1_TORCH_SETTINGS)

    state_dict_path = tmp_path / "checkpoint.pt"
    torch.save(torch_model.state_dict(), state_dict_path)

    # `V1_TORCH_SETTINGS["dtype"]` is a `torch.dtype` object (e.g.
    # `torch.float64`), which does not YAML-serialize cleanly via plain
    # `yaml.dump` -- convert to the bare string form the CLI's own
    # `getattr(torch, args.dtype)` expects.
    serializable_settings = dict(V1_TORCH_SETTINGS)
    serializable_settings["dtype"] = str(V1_TORCH_SETTINGS["dtype"]).rsplit(
        ".", 1
    )[-1]

    hyperparams_path = tmp_path / "config.yaml"
    with open(hyperparams_path, "w") as f:
        yaml.dump(
            {"ARCHITECTURE": serializable_settings},
            f,
            default_flow_style=False,
        )

    save_settings_path = tmp_path / "jax_settings"
    save_params_path = tmp_path / "jax_params"

    argv = [
        "torchkrates-torch2jax",
        "--path_to_state_dict",
        str(state_dict_path),
        "--path_to_hyperparams",
        str(hyperparams_path),
        "--save_settings_path",
        str(save_settings_path),
        "--save_params_path",
        str(save_params_path),
        "--dtype",
        "float64",
    ]
    if V1_TORCH_SETTINGS["trainable_rbf"]:
        argv.append("--trainable_rbf")

    monkeypatch.setattr(sys, "argv", argv)
    assert torch_to_jax_main() == 0
    assert (save_settings_path / "hyperparameters.json").exists()
    assert (save_params_path / "params.pkl").exists()


# ---------------------------------------------------------------------
# Task 2: v2 config keys (use_rms_norm/qk_norm/use_residual_scalars) and
# `legacy_dispersion_bool`'s v1/v2 default-ambiguity fix.
#
# v1.0-lrs-gems ("v1") and so3lr_dev ("v2") JAX configs/checkpoints both
# omit `legacy_so3lr_bool`/`legacy_dispersion_bool` when unset, but mean
# opposite things by the omission (v1: legacy=True; v2, including all
# three real shipped so3lr-s/-m/-l checkpoints: legacy=False). The tests
# below cover both conversion directions: the v1-shaped default, the
# v2-shaped default, and that an explicit value always wins regardless
# of shape.
# ---------------------------------------------------------------------

# Minimal flat_params stand-in for `get_model_settings_flax_to_torch`'s
# only genuinely-required lookup in these tests (num_theory_levels
# detection via the energy_dense_final kernel's shape) -- these tests
# only exercise config-translation logic, not real weight arrays.
_MINIMAL_FLAT_PARAMS = {
    "params/observables_0/energy_dense_final/kernel": np.zeros(
        (V1_TORCH_SETTINGS["energy_regression_dim"], 1)
    )
}


def _v1_shaped_cfg():
    """A v1.0-lrs-gems-shaped cfg: no use_rms_norm/qk_norm/
    use_residual_scalars/legacy_so3lr_bool keys at all."""
    return build_jax_v1_config_from_settings(V1_TORCH_SETTINGS)


def _v2_shaped_cfg():
    """Same base cfg, with a v2 marker key set (use_rms_norm=True) but
    still no explicit legacy_so3lr_bool -- mirrors a real so3lr_dev
    config/checkpoint that omits the key."""
    cfg = build_jax_v1_config_from_settings(V1_TORCH_SETTINGS)
    cfg.model.use_rms_norm = True
    return cfg


def test_flax_to_torch_v1_config_defaults_legacy_dispersion_true():
    """v1-shaped cfg (no legacy_so3lr_bool, no v2 markers) must default
    `legacy_dispersion_bool` to True -- this repo's deliberately-chosen
    v1 default."""
    settings = get_model_settings_flax_to_torch(
        cfg=_v1_shaped_cfg(),
        device="cpu",
        use_defined_shifts=False,
        flat_params=_MINIMAL_FLAT_PARAMS,
    )
    assert settings["legacy_dispersion_bool"] is True
    assert settings["use_rms_norm"] is False
    assert settings["qk_norm"] is False
    assert settings["use_residual_scalars"] is False


def test_flax_to_torch_v2_config_defaults_legacy_dispersion_false():
    """v2-shaped cfg (has a v2 marker key, still no legacy_so3lr_bool)
    must default `legacy_dispersion_bool` to False -- matches
    `so3lr_dev`'s own `model_config.get('legacy_so3lr_bool', False)` and
    all three real shipped so3lr-s/-m/-l checkpoints, which omit the
    key."""
    settings = get_model_settings_flax_to_torch(
        cfg=_v2_shaped_cfg(),
        device="cpu",
        use_defined_shifts=False,
        flat_params=_MINIMAL_FLAT_PARAMS,
    )
    assert settings["legacy_dispersion_bool"] is False
    assert settings["use_rms_norm"] is True


@pytest.mark.parametrize("cfg_builder", [_v1_shaped_cfg, _v2_shaped_cfg])
@pytest.mark.parametrize("explicit_value", [True, False])
def test_flax_to_torch_explicit_legacy_so3lr_bool_always_wins(
    cfg_builder, explicit_value
):
    """An explicitly-set `legacy_so3lr_bool` in the config must never be
    overridden by the v1/v2 disambiguation, regardless of config
    shape."""
    cfg = cfg_builder()
    cfg.model.legacy_so3lr_bool = explicit_value
    settings = get_model_settings_flax_to_torch(
        cfg=cfg,
        device="cpu",
        use_defined_shifts=False,
        flat_params=_MINIMAL_FLAT_PARAMS,
    )
    assert settings["legacy_dispersion_bool"] is explicit_value


def test_torch_to_flax_v1_settings_default_legacy_so3lr_bool_true():
    """v1-shaped torch settings dict (no use_rms_norm/qk_norm/
    use_residual_scalars/legacy_dispersion_bool keys) must default
    `legacy_so3lr_bool` to True."""
    cfg = get_model_settings_torch_to_flax(V1_TORCH_SETTINGS)
    assert cfg.model.legacy_so3lr_bool is True
    assert cfg.model.use_rms_norm is False
    assert cfg.model.qk_norm is False
    assert cfg.model.use_residual_scalars is False


def test_torch_to_flax_v2_settings_default_legacy_so3lr_bool_false():
    """v2-shaped torch settings dict (has a v2 marker key, no explicit
    legacy_dispersion_bool) must default `legacy_so3lr_bool` to
    False."""
    v2_settings = dict(V1_TORCH_SETTINGS)
    v2_settings["use_rms_norm"] = True
    cfg = get_model_settings_torch_to_flax(v2_settings)
    assert cfg.model.legacy_so3lr_bool is False
    assert cfg.model.use_rms_norm is True


@pytest.mark.parametrize("is_v2", [False, True])
@pytest.mark.parametrize("explicit_value", [True, False])
def test_torch_to_flax_explicit_legacy_dispersion_bool_always_wins(
    is_v2, explicit_value
):
    """An explicitly-set `legacy_dispersion_bool` in the torch settings
    must never be overridden by the v1/v2 disambiguation, regardless of
    settings shape."""
    settings = dict(V1_TORCH_SETTINGS)
    if is_v2:
        settings["use_rms_norm"] = True
    settings["legacy_dispersion_bool"] = explicit_value
    cfg = get_model_settings_torch_to_flax(settings)
    assert cfg.model.legacy_so3lr_bool is explicit_value


# ---------------------------------------------------------------------
# Task 2: weight-mapping side of the same three v2 flags --
# `use_rms_norm` must suppress the layer-norm scale/bias mapping (JAX's
# nn.RMSNorm(use_scale=False, ...) has no learnable parameters), and
# `use_residual_scalars` must add the top-level resid_lambdas/x0_lambdas
# entries, in both conversion directions.
# ---------------------------------------------------------------------


def test_flax_to_torch_mapping_suppresses_layer_norm_weights_under_rms_norm():
    cfg = _v1_shaped_cfg()
    cfg.model.layer_normalization_1 = True
    cfg.model.layer_normalization_2 = True

    mapping_without_rms = get_flax_to_torch_mapping(cfg, trainable_rbf=True)
    assert (
        "params/layers_0/layer_normalization_1/scale"
        in mapping_without_rms
    )
    assert (
        "params/layers_0/layer_normalization_2/scale"
        in mapping_without_rms
    )

    cfg.model.use_rms_norm = True
    mapping_with_rms = get_flax_to_torch_mapping(cfg, trainable_rbf=True)
    assert (
        "params/layers_0/layer_normalization_1/scale"
        not in mapping_with_rms
    )
    assert (
        "params/layers_0/layer_normalization_1/bias" not in mapping_with_rms
    )
    assert (
        "params/layers_0/layer_normalization_2/scale"
        not in mapping_with_rms
    )
    assert (
        "params/layers_0/layer_normalization_2/bias" not in mapping_with_rms
    )


def test_torch_to_flax_mapping_suppresses_layer_norm_weights_under_rms_norm():
    cfg = _v1_shaped_cfg()
    cfg.model.layer_normalization_1 = True
    cfg.model.layer_normalization_2 = True

    mapping_without_rms = get_torch_to_flax_mapping(cfg, trainable_rbf=True)
    assert (
        "euclidean_transformers.0.layer_norm_inv_1.weight"
        in mapping_without_rms
    )
    assert (
        "euclidean_transformers.0.layer_norm_inv_2.weight"
        in mapping_without_rms
    )

    cfg.model.use_rms_norm = True
    mapping_with_rms = get_torch_to_flax_mapping(cfg, trainable_rbf=True)
    assert (
        "euclidean_transformers.0.layer_norm_inv_1.weight"
        not in mapping_with_rms
    )
    assert (
        "euclidean_transformers.0.layer_norm_inv_1.bias"
        not in mapping_with_rms
    )
    assert (
        "euclidean_transformers.0.layer_norm_inv_2.weight"
        not in mapping_with_rms
    )
    assert (
        "euclidean_transformers.0.layer_norm_inv_2.bias"
        not in mapping_with_rms
    )


def test_flax_to_torch_mapping_adds_resid_lambdas_only_when_flagged():
    cfg = _v1_shaped_cfg()
    mapping_off = get_flax_to_torch_mapping(cfg, trainable_rbf=True)
    assert "params/resid_lambdas" not in mapping_off
    assert "params/x0_lambdas" not in mapping_off

    cfg.model.use_residual_scalars = True
    mapping_on = get_flax_to_torch_mapping(cfg, trainable_rbf=True)
    assert mapping_on["params/resid_lambdas"] == "resid_lambdas"
    assert mapping_on["params/x0_lambdas"] == "x0_lambdas"


def test_torch_to_flax_mapping_adds_resid_lambdas_only_when_flagged():
    cfg = _v1_shaped_cfg()
    mapping_off = get_torch_to_flax_mapping(cfg, trainable_rbf=True)
    assert "resid_lambdas" not in mapping_off
    assert "x0_lambdas" not in mapping_off

    cfg.model.use_residual_scalars = True
    mapping_on = get_torch_to_flax_mapping(cfg, trainable_rbf=True)
    assert mapping_on["resid_lambdas"] == "params/resid_lambdas"
    assert mapping_on["x0_lambdas"] == "params/x0_lambdas"
