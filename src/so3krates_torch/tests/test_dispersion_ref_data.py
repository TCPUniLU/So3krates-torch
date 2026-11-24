"""Reference tables for dispersion and NLH repulsion.

Every table is pinned bit for bit (SHA-256 of its float64 bytes) to the
official SO3LR repository, github.com/general-molecular-simulations/so3lr,
``so3lr/mlff/nn/observable/dispersion_ref_data.py`` and ``nlh_ref.npz`` at
commit a533140 (2026-11-23).
"""

import hashlib
import importlib.resources

import numpy as np
import pytest
import torch

from so3krates_torch.blocks import dispersion_ref_data
from so3krates_torch.blocks.physical_potentials import _dispersion_tables

_OFFICIAL_SHA256 = {
    "alphas_legacy": (
        "360482dd6e363f923dda1a519c58787b96aeb3ca1551aa2098ef24ca34d52e57"
    ),
    "C6_coef_legacy": (
        "a92fd62b85724b564d07d14d3bd69957aa1adef8b3a41fb63cd14e6b36af4dec"
    ),
    "alphas": (
        "62a7b3338937f86139d9e87cb1101ffe67dacbe48bb895f2655701a03343cb00"
    ),
    "C6_coef": (
        "fec39fb68cf9e568fe99476ff4743c2b3bc08a73095b891d25dca8c2cf4fbf74"
    ),
    "NHL_AA": (  # official name: NLH_AA
        "2bd12c497a658a919e9be4550aa19a60b0cef0ddb89f6f9768ef25fd7cd54ac3"
    ),
    "NHL_BB": (  # official name: NLH_BB
        "bd3a34984de615f99b53a82266bd29f42eb53599da36fa515370dc866e57a24a"
    ),
}
_SHAPES = {
    "alphas_legacy": (102,),
    "C6_coef_legacy": (102,),
    "alphas": (86,),
    "C6_coef": (86,),
    "NHL_AA": (92, 92, 3),
    "NHL_BB": (92, 92, 3),
}


def _sha256(array) -> str:
    a = np.ascontiguousarray(np.asarray(array), dtype=np.float64)
    return hashlib.sha256(a.tobytes()).hexdigest()


def test_nhl_tables_ship_as_npz():
    path = importlib.resources.files("so3krates_torch.blocks").joinpath(
        "nhl_tables.npz"
    )
    assert path.is_file()


@pytest.mark.parametrize("name", sorted(_OFFICIAL_SHA256))
def test_table_matches_official_so3lr(name):
    table = np.asarray(getattr(dispersion_ref_data, name))
    assert table.shape == _SHAPES[name]
    assert table.dtype == np.float64
    assert _sha256(table) == _OFFICIAL_SHA256[name]


@pytest.mark.parametrize("name", ["NHL_AA", "NHL_BB"])
def test_nhl_table_is_pair_symmetric(name):
    table = getattr(dispersion_ref_data, name)
    assert np.array_equal(table, table.transpose(1, 0, 2))


def test_legacy_dispersion_tables_used_are_official_float64():
    """mixing_rules(legacy_dispersion_bool=True) uses exactly these values,
    without float32 rounding."""
    alphas, c6 = _dispersion_tables(legacy_dispersion_bool=True)
    assert alphas.dtype == torch.float64
    assert c6.dtype == torch.float64
    assert _sha256(alphas.numpy()) == _OFFICIAL_SHA256["alphas_legacy"]
    assert _sha256(c6.numpy()) == _OFFICIAL_SHA256["C6_coef_legacy"]
    # Pa (Z=91): the old mlff table had 2876.77; the official value is
    # 2367.42.
    assert c6[90].item() == 2367.42
