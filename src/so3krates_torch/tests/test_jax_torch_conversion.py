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
    build_torch_v1,
    jax_to_torch,
    run_weight_roundtrip_check,
    torch_to_jax,
)
from so3krates_torch.tools.jax_torch_conversion import flatten_params
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
