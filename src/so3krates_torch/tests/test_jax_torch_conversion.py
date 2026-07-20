import copy

import pytest

pytest.importorskip("jax")
pytest.importorskip("flax")
pytest.importorskip("so3lr")

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
from so3krates_torch.tools.model_parity import (
    _build_jax_inputs,
    _match_theory_levels,
    check_model_parity,
)

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
#
# Task 3 amendment: this file used to require old (v1) `mlff` at module
# level (`pytest.importorskip("mlff")`) for *every* test here, even ones
# that never touched it. `model_parity.py`'s JAX backend is now
# `so3lr_dev` (`import so3lr`) exclusively -- see
# `.superpowers/sdd/v2arch-task-3-amend-report.md` for the full
# rationale. `_build_jax_v1_so3lr_dev` below is the so3lr_dev-based
# replacement for the old, now-deleted `build_jax_v1` used by the 4
# tests that used to call it directly. `test_weight_roundtrip_passes`
# is the sole, deliberate exception that still needs real old `mlff`
# (see its own skip conditions below).


@pytest.fixture(scope="module", autouse=True)
def _restore_default_dtype():
    yield
    torch.set_default_dtype(_PRE_IMPORT_DEFAULT_DTYPE)


def _build_jax_v1_so3lr_dev(config):
    """so3lr_dev counterpart of the now-deleted, old-``mlff``-based
    ``v1_stagewise_parity.build_jax_v1``.

    Mirrors its structure exactly (build model, build example inputs,
    ``model.init``, upcast to float64) but uses ``so3lr_dev`` (this
    repo's sole default JAX backend after the Task 3 amendment) via
    ``model_parity.py``'s own ``_build_jax_inputs`` instead of old
    ``mlff``'s incompatible graph pipeline.

    Two deliberate overrides on top of a plain copy of ``config``, both
    explained in full in the Task 3 amendment report:

    - ``legacy_so3lr_bool = True``: required -- ``so3lr_dev``'s own
      factory defaults this to ``False``, which would silently build
      the *wrong* architecture (non-legacy ``NLHRepulsionSparse``
      instead of legacy ``ZBLRepulsionSparse``, an extra C6 head, etc.
      -- see ``so3lr_dev/so3lr/mlff/nn/representation/
      so3krates_sparse.py``).
    - ``use_final_bias_bool = True``: matches
      ``V1_TORCH_SETTINGS["final_layer_bias"]``. Needed because
      ``so3lr_dev/so3lr/mlff/config/from_config.py`` defaults this to
      ``False`` when unset, and (real, reported, out-of-scope-to-fix-
      here finding) ``jax_torch_conversion.py``'s torch->flax direction
      never translates ``final_layer_bias`` into this key at all --
      irrelevant for *this* helper since we set it explicitly, but see
      the amendment report for where it does matter
      (``test_torch2jax_cli_check_parity_passes_by_default``).

    ``model.init``'s example inputs are built via ``_build_jax_inputs``
    then forced down to a genuine single theory level via
    ``_match_theory_levels`` -- so3lr_dev's own ``ASE_to_jraph``
    hardcodes a 16-level ``theory_mask`` for *every* structure
    (unrelated to ``cfg``/``legacy_so3lr_bool``), which would otherwise
    make this "v1" model come out with a 16-wide (not 1-wide, matching
    the real v1 architecture's implicit single theory level)
    ``energy_dense_final`` -- see the amendment report for the full
    explanation. ``check_model_parity`` does this same reconciliation
    automatically for its own JAX-side inputs (detected from
    ``flax_params``), so callers of it don't need to repeat this, but
    building this module's own initial ``params`` still requires it
    directly.
    """
    import jax
    import jax.tree_util as jtu
    from so3lr.mlff.config import make_so3krates_sparse_from_config

    cfg = copy.deepcopy(config)
    cfg.model.legacy_so3lr_bool = True
    cfg.model.use_final_bias_bool = V1_TORCH_SETTINGS["final_layer_bias"]

    example_inputs = _match_theory_levels(
        _build_jax_inputs(cfg, None, 0), num_theory_levels=1
    )
    model = make_so3krates_sparse_from_config(cfg)
    params = model.init(jax.random.PRNGKey(0), example_inputs)

    # so3lr_dev hard-codes param_dtype=jnp.float32 for several Dense/
    # Embed modules regardless of jax_enable_x64 -- same rationale as
    # the now-deleted old-mlff `build_jax_v1`'s identical cast (lossless
    # float32 -> float64 widening, not a precision-losing one).
    params = jtu.tree_map(lambda x: x.astype("float64"), params)
    return model, params, cfg


def _mlff_jraph_compatible() -> bool:
    """True iff the installed ``jraph`` has old (v1) ``mlff``'s required
    extra ``GraphsTuple`` fields (``idx_i_lr``/``idx_j_lr``/``n_pairs``,
    from the ``kabylda/jraph`` fork).

    A plain ``pytest.importorskip("mlff")`` is not enough to skip
    ``test_weight_roundtrip_passes`` on this machine: old ``mlff``
    itself imports fine (it's installed), but its own
    ``ASE_to_jraph`` crashes with a ``TypeError`` (not an
    ``ImportError``) once the environment's ``jraph`` is vanilla --
    which it now is, since ``so3lr_dev`` (this repo's sole default JAX
    backend after the Task 3 amendment) requires vanilla ``jraph`` and
    the two packages cannot both be satisfied by one top-level ``jraph``
    install at once. See the Task 3 amendment report for the empirical
    trace of this exact failure.
    """
    import jraph

    return "idx_i_lr" in jraph.GraphsTuple._fields


@pytest.mark.skipif(
    not _mlff_jraph_compatible(),
    reason=(
        "old mlff's ASE_to_jraph needs the kabylda/jraph fork's extra "
        "GraphsTuple fields (idx_i_lr/idx_j_lr/n_pairs); the installed "
        "jraph is vanilla, so3lr_dev's own requirement as this repo's "
        "sole default JAX backend -- see the Task 3 amendment report"
    ),
)
def test_weight_roundtrip_passes():
    """General regression net for the shared conversion primitives:
    jax -> torch -> jax and torch -> jax -> torch must round-trip.

    Deliberate, narrow exception (Task 3 amendment): this is the one
    remaining test still exercising old ``mlff`` directly, via
    ``run_weight_roundtrip_check()`` (``v1_stagewise_parity.py``, out of
    scope to edit), which hard-codes a call to the now-deleted
    ``build_jax_v1`` with no way to inject a different model-builder.
    Duplicating its ~30-line round-trip logic against
    ``_build_jax_v1_so3lr_dev`` instead was considered and rejected as
    real, unwarranted duplication for one test -- see the amendment
    report.
    """
    pytest.importorskip("mlff")
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
    model_jax, flax_params, cfg = _build_jax_v1_so3lr_dev(JAX_V1_CONFIG)
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
    model_jax, flax_params, cfg = _build_jax_v1_so3lr_dev(JAX_V1_CONFIG)
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
    _model_jax, flax_params, cfg = _build_jax_v1_so3lr_dev(JAX_V1_CONFIG)

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
    _model_jax, flax_params, cfg = _build_jax_v1_so3lr_dev(JAX_V1_CONFIG)

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

    Uses the shared `V1_TORCH_SETTINGS` (`final_layer_bias=True`)
    directly -- `check_model_parity`'s own `_with_final_bias_bool`
    defensive fix (model_parity.py) fills in `cfg.model.
    use_final_bias_bool` from the torch model's own bias state
    whenever `jax_torch_conversion.py`'s `get_model_settings_torch_to_
    flax` (out of scope, already-approved Task 2 code) leaves it unset,
    so this exact scenario -- a bias-enabled torch model round-tripped
    through the CLI's torch->flax direction -- now round-trips its
    final-layer bias correctly again, end to end.
    """
    settings = dict(V1_TORCH_SETTINGS)
    torch_model = build_torch_v1(settings)

    state_dict_path = tmp_path / "checkpoint.pt"
    torch.save(torch_model.state_dict(), state_dict_path)

    # `V1_TORCH_SETTINGS["dtype"]` is a `torch.dtype` object (e.g.
    # `torch.float64`), which does not YAML-serialize cleanly via plain
    # `yaml.dump` -- convert to the bare string form the CLI's own
    # `getattr(torch, args.dtype)` expects.
    serializable_settings = dict(settings)
    serializable_settings["dtype"] = str(settings["dtype"]).rsplit(".", 1)[-1]

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
    if settings["trainable_rbf"]:
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
    assert "params/layers_0/layer_normalization_1/scale" in mapping_without_rms
    assert "params/layers_0/layer_normalization_2/scale" in mapping_without_rms

    cfg.model.use_rms_norm = True
    mapping_with_rms = get_flax_to_torch_mapping(cfg, trainable_rbf=True)
    assert (
        "params/layers_0/layer_normalization_1/scale" not in mapping_with_rms
    )
    assert "params/layers_0/layer_normalization_1/bias" not in mapping_with_rms
    assert (
        "params/layers_0/layer_normalization_2/scale" not in mapping_with_rms
    )
    assert "params/layers_0/layer_normalization_2/bias" not in mapping_with_rms


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


# ---------------------------------------------------------------------
# v2arch-task-4-fix6: `use_simple_hirshfeld` is not a real `so3lr_dev`
# config key either (same invented-key gotcha already fixed for
# `nhl_repulsion_bool`/`c6_ratios_bool` above) -- the real discriminator
# is `legacy_so3lr_bool` (`so3krates_sparse.py`:
# `HirshfeldSparse(..., legacy=legacy_so3lr_bool, ...)`). This test
# exercises `get_flax_to_torch_mapping` with `flat_params=None` so only
# the config/legacy_so3lr_bool-derived primary source is in play, not
# the `flat_params` structural fallback that every real call site
# happens to also provide today and which previously masked this bug.
# ---------------------------------------------------------------------


def test_flax_to_torch_mapping_derives_simple_hirshfeld_from_legacy_bool():
    """A non-legacy-shaped cfg (has a v2 marker key, omits both
    `use_simple_hirshfeld` and `legacy_so3lr_bool`) must derive
    `use_simple_hirshfeld=True` (non-legacy -> new/simple Hirshfeld
    arch) from the `legacy_so3lr_bool`-derived default, not the old,
    wrong, always-`False` `getattr(..., False)` default -- called with
    `flat_params=None` (the default) so the structural
    `flat_params`-based fallback can't paper over a regression here."""
    cfg = _v2_shaped_cfg()
    mapping = get_flax_to_torch_mapping(cfg, trainable_rbf=True)
    # Simple (single-embedding) Hirshfeld arch has no Embed_1 mapping --
    # see `get_flax_to_torch_mapping`'s use_simple_hirshfeld branch.
    assert "params/observables_2/Embed_1/embedding" not in mapping
    assert (
        mapping["params/observables_2/Embed_0/embedding"]
        == "hirshfeld_output_block.element_embedding.weight"
    )


def test_flax_to_torch_mapping_derives_simple_hirshfeld_false_for_v1():
    """A v1.0-lrs-gems-shaped cfg (no v2 markers, no legacy_so3lr_bool)
    must derive `use_simple_hirshfeld=False` (legacy -> old/attention
    Hirshfeld arch) -- the counterpart of the test above, confirming
    the v1 default is unchanged by this fix."""
    cfg = _v1_shaped_cfg()
    mapping = get_flax_to_torch_mapping(cfg, trainable_rbf=True)
    assert (
        mapping["params/observables_2/Embed_1/embedding"]
        == "hirshfeld_output_block.q_embedding.weight"
    )


# ---------------------------------------------------------------------
# Task 4: `--so3lr_dev_checkpoint {s,m,l}` on `torchkrates-jax2torch`.
#
# Lets callers point at one of the three real, bundled `so3lr_dev`
# checkpoints (so3lr-s/-m/-l) by name instead of spelling out
# `--path_to_params`/`--path_to_hyperparams` -- resolved via
# `importlib.resources.files("so3lr")`. `Jax2TorchArgs` itself gets an
# independent `@model_validator` enforcing the same mutual-exclusivity/
# at-least-one-form invariant the CLI's own `argparser.error(...)`
# calls already enforce, since `Jax2TorchArgs` can be constructed
# directly (as these tests do) without going through the CLI.
# ---------------------------------------------------------------------

from pydantic import ValidationError

from so3krates_torch.config import Jax2TorchArgs


def test_jax2torch_args_accepts_so3lr_dev_checkpoint():
    args = Jax2TorchArgs(
        so3lr_dev_checkpoint="s",
        save_model_path="/tmp/does_not_need_to_exist.model",
    )
    assert args.so3lr_dev_checkpoint == "s"
    assert args.path_to_params is None
    assert args.path_to_hyperparams is None


def test_jax2torch_args_rejects_checkpoint_and_path_together():
    with pytest.raises(ValidationError):
        Jax2TorchArgs(
            so3lr_dev_checkpoint="s",
            path_to_params="params.pkl",
            save_model_path="/tmp/does_not_need_to_exist.model",
        )


def test_jax2torch_args_rejects_neither_given():
    with pytest.raises(ValidationError):
        Jax2TorchArgs(save_model_path="/tmp/does_not_need_to_exist.model")


def test_so3lr_dev_checkpoint_resolves_to_real_existing_files():
    """Confirms `importlib.resources.files("so3lr")` genuinely resolves
    each bundled so3lr-s/-m/-l checkpoint to real, existing
    `params.pkl`/`hyperparameters.json` files in this environment --
    the core claim `--so3lr_dev_checkpoint` relies on -- independent of
    whether `convert_flax_to_torch`/`SO3LR.__init__` can currently
    build a torch model from their *contents* (see
    `test_jax2torch_cli_so3lr_dev_checkpoint_end_to_end` below for that,
    currently-failing, separate concern).
    """
    from importlib.resources import files

    for size in ("s", "m", "l"):
        checkpoint_dir = files("so3lr") / "models" / f"so3lr-{size}"
        params_path = checkpoint_dir / "params.pkl"
        hyperparams_path = checkpoint_dir / "hyperparameters.json"
        assert params_path.is_file()
        assert hyperparams_path.is_file()


def test_jax2torch_cli_so3lr_dev_checkpoint_end_to_end(tmp_path, monkeypatch):
    """Real end-to-end run of `torchkrates-jax2torch
    --so3lr_dev_checkpoint s`, exercising the full CLI wiring (argv ->
    --so3lr_dev_checkpoint resolution -> Jax2TorchArgs validation ->
    conversion) against a genuine bundled checkpoint.

    `--no-check_parity`: this test's purpose is confirming the
    checkpoint-resolution wiring, not re-exercising the already-covered
    `--check_parity` machinery (see
    `test_jax2torch_cli_check_parity_passes_by_default` above) -- kept
    light so it doesn't also pay for a full JAX-side parity build on a
    real (larger-than-the-synthetic-v1-test-model) bundled checkpoint.

    Previously xfailed on three bugs, now all fixed (see
    v2arch-task-4-fix2-report.md and v2arch-task-4-fix3-report.md): (1)
    `convert_flax_to_torch_params` transposing `energy_offset`/
    `atomic_scales` before stripping the JAX padding row; (2)
    `get_flax_to_torch_mapping`/`get_model_settings_flax_to_torch` reading
    the non-existent `nhl_repulsion_bool` JAX config key instead of
    `legacy_so3lr_bool`; (3) `get_model_settings_flax_to_torch` not
    auto-detecting `use_simple_hirshfeld` from `flat_params` the same way
    `get_flax_to_torch_mapping` does, and an unconditional `energy_shifts`
    overwrite in `convert_flax_to_torch` clobbering the genuinely-learned
    `energy_offset`-derived value. `--check_parity=True` against the real
    so3lr-s/-m/-l checkpoints now passes for all three (see
    v2arch-task-4-fix3-report.md for exact numbers).
    """
    save_model_path = tmp_path / "so3lr_s_torch.model"

    argv = [
        "torchkrates-jax2torch",
        "--so3lr_dev_checkpoint",
        "s",
        "--save_model_path",
        str(save_model_path),
        "--no-check_parity",
    ]

    monkeypatch.setattr(sys, "argv", argv)
    assert jax_to_torch_main() == 0
    assert save_model_path.exists()


def test_so3lr_dev_checkpoint_s_parity_at_machine_precision_ceiling():
    """Tight-tolerance regression test for the `c6_ratios_bool`
    conversion bug (v2arch-task-4-fix4): before that fix,
    `get_flax_to_torch_mapping`/`get_model_settings_flax_to_torch` gated
    the torch-side `c6_ratios_output_block` on a non-existent
    `cfg.model.c6_ratios_bool` JAX config key (always `False`), so the
    checkpoint's real, trained `c6_ratios` head was silently dropped and
    `DispersionInteraction` fell back to `so3lr_dev`'s legacy
    (`C6 ~ hirshfeld_ratio**2`) formula instead -- a real, ~1e-3-eV-level
    physics bug, not float noise (see
    v2arch-residual-diff-investigation-report.md and
    v2arch-task-4-fix4-report.md for the full root-cause trail).

    `check_model_parity`'s own default tolerance (`atol=1e-3`,
    `rtol=1e-2`) is loose enough that this bug still silently "passed"
    it -- exactly why it went unnoticed until a dedicated investigation.
    This test uses `rtol=0.0` (so only the absolute diff matters -- some
    force components are near zero, which would otherwise inflate a
    relative-diff-based check into a false failure, as already noted for
    the diagnostic numbers in v2arch-task-4-fix2-report.md) and a much
    tighter `atol=1e-6`, comfortably above the ~1e-7-level noise floor
    every other term already sits at (per the investigation's per-term
    breakdown) but far below the ~1e-3 the c6_ratios bug used to produce
    -- so a future regression of this exact bug would fail loudly
    instead of silently passing a loose tolerance again.

    Only `so3lr-s` (cheapest of the three real checkpoints) is checked
    here, deliberately, to keep this fast enough for the default test
    suite; `so3lr-m`/`-l` are checked at the same tight tolerance in
    v2arch-task-4-fix4-report.md's manual verification.
    """
    import json
    import pickle
    from importlib.resources import files

    from ml_collections import config_dict

    from so3krates_torch.tools.jax_torch_conversion import (
        convert_flax_to_torch,
    )
    from so3krates_torch.tools.model_parity import check_model_parity

    checkpoint_dir = files("so3lr") / "models" / "so3lr-s"
    params_path = str(checkpoint_dir / "params.pkl")
    hyperparams_path = str(checkpoint_dir / "hyperparameters.json")

    torch_model = convert_flax_to_torch(
        path_to_flax_params=params_path,
        path_to_flax_hyperparams=hyperparams_path,
        dtype=torch.float64,
    )

    with open(params_path, "rb") as f:
        flax_params = pickle.load(f)
    with open(hyperparams_path, "r") as f:
        cfg = config_dict.ConfigDict(json.load(f))

    assert check_model_parity(
        cfg=cfg,
        flax_params=flax_params,
        torch_model=torch_model,
        r_max=cfg.model.cutoff,
        r_max_lr=cfg.model.cutoff_lr,
        atol=1e-6,
        rtol=0.0,
    )
