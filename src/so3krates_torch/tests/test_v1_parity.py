"""Permanent checkpoint-level JAX<->torch parity suite for the real,
bundled *v1* ``so3lr_dev`` checkpoint (``so3lr_dev``'s
``models/so3lr/`` -- the single legacy/v1 SO3LR model, distinct from the
three multi-theory-level v2 ``so3lr-s``/``-m``/``-l`` checkpoints that
``test_v2_parity.py`` already covers).

This mirrors ``test_v2_parity.py``'s structure but for the v1
architecture (single theory level, learnable ZBL repulsion, legacy
real-space dispersion, ``legacy_so3lr_bool=True``). Getting the v1 side
to build in JAX at all was blocked by a real, previously-unfixed bug:
the v1 checkpoint's ``params.pkl`` stores
``observables_0/energy_offset`` and ``observables_0/atomic_scales`` as
1-D ``(zmax + 1,)`` arrays (an older ``EnergySparse`` layout), while the
currently-installed ``so3lr_dev``'s ``EnergySparse.__call__``
(``mlff/nn/observable/observable_sparse.py`` lines 240-258) now declares
both with shape ``(zmax + 1, num_theory_levels)`` -- so binding the 1-D
checkpoint leaf raised ``flax.errors.ScopeParamShapeError`` ((119,) vs
(119, 1)) at ``model.apply`` time. This is a pure params-shape version
drift (unrelated to any cfg flag -- ``legacy_so3lr_bool`` /
``energy_learn_atomic_type_shifts`` / ``use_final_bias_bool`` are all
correctly set in the v1 hyperparameters). Since a v1 model has exactly
one theory level, reshaping ``(zmax + 1,)`` -> ``(zmax + 1, 1)`` is
byte-for-byte value-preserving; the fix
(``model_parity._reshape_legacy_energy_head_params``) does exactly that
on the JAX side, a no-op for the already-2-D v2 checkpoints. See
``.superpowers/sdd/task-2-report.md`` for the full root-cause writeup.

Two cfg overrides are needed to run either test; both are pure test-
setup choices (like ``test_v2_parity.py``'s own PME ``cutoff_lr``
override), not model/architecture changes -- both sides use the same
values, so the model-vs-model comparison is unaffected by their absolute
magnitude:

* ``cutoff_lr``: the v1 checkpoint ships ``cfg.model.cutoff_lr = null``
  (its long-range neighbor list used a separate, data-side
  ``neighbors_lr_cutoff``), but ``so3lr_dev``'s ``ASE_to_jraph`` requires
  a concrete ``cutoff_lr`` to build the long-range neighbor list at all.
  Overridden to 12.0 (the ``SO3LRCalculator``'s own default v1 long-range
  cutoff) for the no-PME test, and 10.0 for the PME test (matching
  ``test_v2_parity.py``'s own periodic-cell neighbor-list-explosion
  precaution).
* ``dispersion_energy_cutoff_lr_damping``: the v1 checkpoint ships this
  as ``null``, but ``so3lr_dev``'s legacy dispersion path
  (``observable_sparse.py`` line 2121) *requires* a damping window
  whenever ``cutoff_lr`` is set. Overridden to 2.0 (the value
  ``scripts/v1_stagewise_parity.py``'s own ``V1_TORCH_SETTINGS`` uses)
  for the no-PME test; not needed for the PME test, which disables
  dispersion entirely.
"""

import copy
import json
import pickle
from importlib.resources import files

import pytest

pytest.importorskip("so3lr")
pytest.importorskip("jax")
pytest.importorskip("flax")

import torch
from ml_collections import config_dict

from so3krates_torch.tools.jax_torch_conversion import convert_flax_to_torch
from so3krates_torch.tools.model_parity import check_model_parity


def _load_v1_checkpoint():
    """Resolve + load the real bundled v1 ``so3lr`` checkpoint's
    ``params.pkl``/``hyperparameters.json`` (the single legacy model
    under ``so3lr_dev``'s ``models/so3lr/``, no size suffix), mirroring
    ``test_v2_parity._load_checkpoint``."""
    checkpoint_dir = files("so3lr") / "models" / "so3lr"
    params_path = str(checkpoint_dir / "params.pkl")
    hyperparams_path = str(checkpoint_dir / "hyperparameters.json")

    with open(params_path, "rb") as f:
        flax_params = pickle.load(f)
    with open(hyperparams_path, "r") as f:
        raw_hyperparams = json.load(f)

    return params_path, flax_params, raw_hyperparams


def _v1_hyperparams(raw_hyperparams: dict, **model_overrides) -> dict:
    """Deep-copy the v1 checkpoint's raw ``hyperparameters.json`` dict and
    apply per-test ``model`` overrides (see the module docstring for why
    ``cutoff_lr``/``dispersion_energy_cutoff_lr_damping`` must be set)."""
    modified = copy.deepcopy(raw_hyperparams)
    modified["model"].update(model_overrides)
    return modified


def _check_v1_parity(
    tmp_path, label: str, structure_path=None, **model_overrides
) -> bool:
    """Build the torch model for the real v1 checkpoint with the given
    ``model`` overrides applied to its hyperparameters, and check parity
    against the JAX reference on ``structure_path`` (the bundled
    ``aqm_small.xyz`` methane frame when ``None``), at
    ``check_model_parity``'s own tightened default tolerance
    (``atol=1e-5, rtol=1e-4``) -- see the calling tests' docstrings for
    the real observed numbers that justify relying on the default."""
    params_path, flax_params, raw_hyperparams = _load_v1_checkpoint()

    modified_hyperparams = _v1_hyperparams(raw_hyperparams, **model_overrides)
    modified_hyperparams_path = tmp_path / f"hyperparameters_v1_{label}.json"
    with open(modified_hyperparams_path, "w") as f:
        json.dump(modified_hyperparams, f)

    torch_model = convert_flax_to_torch(
        path_to_flax_params=params_path,
        path_to_flax_hyperparams=str(modified_hyperparams_path),
        dtype=torch.float64,
    )
    cfg = config_dict.ConfigDict(modified_hyperparams)

    return check_model_parity(
        cfg=cfg,
        flax_params=flax_params,
        torch_model=torch_model,
        r_max=cfg.model.cutoff,
        r_max_lr=cfg.model.cutoff_lr,
        structure_path=structure_path,
    )


def test_so3lr_v1_checkpoint_parity(tmp_path):
    """No-PME parity for the real v1 checkpoint on the bundled
    ``aqm_small.xyz`` methane frame (5 atoms, non-periodic), with the
    full v1 physics stack active (learnable ZBL, real-space
    electrostatics, legacy real-space dispersion).

    ``cutoff_lr=12.0`` (the ``SO3LRCalculator``'s own default v1
    long-range cutoff) and ``dispersion_energy_cutoff_lr_damping=2.0``
    override the checkpoint's shipped ``null`` values -- both required by
    ``so3lr_dev`` to build the long-range neighbor list / damp legacy
    dispersion at all (see module docstring); pure test-setup choices,
    applied identically to both sides.

    Freshly observed, via ``check_model_parity``'s printed report:
    energy max_abs_diff=4.051e-08 eV (max_rel_diff=4.579e-09), forces
    max_abs_diff=1.556e-07 eV/Angstrom (max_rel_diff=4.684e-07) --
    comfortably inside ``check_model_parity``'s own default tolerance
    (``atol=1e-5, rtol=1e-4``, ~247x margin on energy, ~64x on forces),
    at the float64 noise floor. No custom ``atol``/``rtol`` override
    needed.
    """
    assert _check_v1_parity(
        tmp_path,
        "nopme",
        cutoff_lr=12.0,
        dispersion_energy_cutoff_lr_damping=2.0,
    )


def test_so3lr_v1_checkpoint_pme_electrostatics_only_charged_parity(tmp_path):
    """PME-electrostatics-only parity for the real v1 checkpoint on a
    charged rattled 2x2x2 NaCl rock-salt supercell (16 atoms, formal
    +1/-1 ionic charges) -- the exact charged-system convention
    ``test_v2_parity.py`` uses (``a=5.6402``, ``repeat((2, 2, 2))``,
    ``rattle(stdev=0.05, seed=7)`` to avoid the periodic-boundary-aligned
    PME mesh-interpolation differentiability edge case that test
    documents at length).

    Electrostatics-only is v1's supported PME mode: v1 keeps legacy
    real-space dispersion, so PME is isolated to electrostatics here by
    setting ``dispersion_energy_bool=False`` (exactly as
    ``test_v2_parity.test_so3lr_checkpoint_s_pme_electrostatics_only_
    charged_parity`` does) -- with dispersion off, the long-range
    neighbor list is populated solely by electrostatics' own gate, and no
    ``dispersion_energy_cutoff_lr_damping`` override is needed.
    ``cutoff_lr=10.0`` overrides the checkpoint's ``null`` for the same
    periodic-cell neighbor-list-explosion reason ``test_v2_parity.py``
    documents (the k-space part captures the true long-range tail
    separately regardless of this real-space residual cutoff).

    Freshly observed, via ``check_model_parity``'s printed report:
    energy max_abs_diff=9.197e-07 eV (max_rel_diff=7.248e-07), forces
    max_abs_diff=6.936e-08 eV/Angstrom (max_rel_diff=1.453e-04) --
    passing ``check_model_parity``'s default tolerance (``atol=1e-5,
    rtol=1e-4``): energy has ~11x margin, and forces pass on the
    absolute term (max_abs_diff=6.936e-08 << atol=1e-5, ~144x margin) --
    the ~1.5e-04 max_rel_diff is an artifact of dividing tiny near-zero
    force components by their own near-zero magnitude, not a real
    disagreement (``np.allclose``'s ``atol + rtol*|b|`` criterion is
    dominated by ``atol`` there). At the same machine-precision-ceiling
    order of magnitude as the no-PME test above and
    ``test_v2_parity.py``'s charged-NaCl PME tests. No custom
    ``atol``/``rtol`` override needed.
    """
    from ase.build import bulk
    from ase.io import write as ase_write

    nacl = bulk("NaCl", crystalstructure="rocksalt", a=5.6402).repeat(
        (2, 2, 2)
    )
    nacl.rattle(stdev=0.05, seed=7)
    structure_path = tmp_path / "nacl_v1_pme.xyz"
    ase_write(str(structure_path), nacl)

    assert _check_v1_parity(
        tmp_path,
        "pme_electro_only",
        structure_path=str(structure_path),
        kspace_electrostatics="pme",
        kspace_smearing=0.5,
        kspace_spacing=0.25,
        cutoff_lr=10.0,
        dispersion_energy_bool=False,
    )
