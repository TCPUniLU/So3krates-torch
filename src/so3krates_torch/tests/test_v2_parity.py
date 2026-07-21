"""Permanent checkpoint-level JAX<->torch parity suite for the real,
bundled ``so3lr_dev`` checkpoints (``so3lr-s``/``-m``/``-l``), plus one
PME-enabled numerical-consistency test.

This is the final task of the 5-task plan
(``.plans/plans/2026-07-17-so3lr-v2-rmsnorm-qknorm-residual-scalars.md``)
porting ``use_rms_norm``/``qk_norm``/``use_residual_scalars`` and
extending JAX<->torch parity tooling to ``so3lr_dev``. Task 4 found and
fixed six real, previously-latent conversion bugs while getting these
three real checkpoints to actually convert and agree with the JAX
reference at the genuine float64 noise floor (~1e-7-1e-8) -- this file
locks that hard-won result in as a permanent regression suite (Part A),
plus closes the one remaining, never-yet-tested gap: PME-enabled
numerics against real checkpoint weights (Part B).

Part A follows the exact same checkpoint-resolution/model-building
pattern as the existing, narrower
``test_so3lr_dev_checkpoint_s_parity_at_machine_precision_ceiling``
regression test in ``test_jax_torch_conversion.py`` (added by Task 4's
fix #4, tied to one specific historical bug -- ``c6_ratios_bool``) --
see that test's own docstring for the full bug history. This file's
three tests are the general, permanent "does checkpoint-level parity
work" suite for all three real checkpoints, deliberately relying on
``check_model_parity``'s own tightened default tolerance
(``atol=1e-5, rtol=1e-4``, calibrated by Task 4's fix #5 against this
exact real-checkpoint noise floor) rather than duplicating that other
test's even-tighter, bug-specific ``atol=1e-6, rtol=0.0`` choice --
relying on the default is what real users actually get via the CLI's
own ``--check_parity`` default-on path, which is the more meaningful
thing for a general-purpose permanent suite to guard.

Sequenced cheapest-first (``s`` -> ``m`` -> ``l``, matching the plan's
own 2-layer/128-feature -> 3/128 -> 3/256-with-doubled-degrees cost
ordering): if ``s`` fails, debugging there is far cheaper than ``l``,
and plain top-to-bottom pytest file-order collection already gives this
ordering for free without needing explicit ordering markers.
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


def _load_checkpoint(size: str):
    """Resolve + load a real bundled ``so3lr-{size}`` checkpoint's
    ``params.pkl``/``hyperparameters.json``, exactly as
    ``test_so3lr_dev_checkpoint_s_parity_at_machine_precision_ceiling``
    (``test_jax_torch_conversion.py``) does."""
    checkpoint_dir = files("so3lr") / "models" / f"so3lr-{size}"
    params_path = str(checkpoint_dir / "params.pkl")
    hyperparams_path = str(checkpoint_dir / "hyperparameters.json")

    with open(params_path, "rb") as f:
        flax_params = pickle.load(f)
    with open(hyperparams_path, "r") as f:
        cfg = config_dict.ConfigDict(json.load(f))

    return params_path, hyperparams_path, flax_params, cfg


def _check_checkpoint_parity(size: str) -> bool:
    """Build the torch model for real checkpoint ``size`` and check its
    parity against the JAX reference, at ``check_model_parity``'s own
    (tightened, Task-4-fix-#5-calibrated) default tolerance.

    Deliberately uses ``dtype=torch.float64``: matches the template
    test's own convention, and JAX always runs at float64 internally
    regardless (``check_model_parity``'s own docstring) -- so this keeps
    both sides at the same precision and avoids any float32-vs-float64
    ambiguity (already investigated and ruled out as an explanation for
    anything in this plan's history -- see
    ``.superpowers/sdd/v2arch-task-4-fix-report.md``).
    """
    params_path, hyperparams_path, flax_params, cfg = _load_checkpoint(size)

    torch_model = convert_flax_to_torch(
        path_to_flax_params=params_path,
        path_to_flax_hyperparams=hyperparams_path,
        dtype=torch.float64,
    )

    return check_model_parity(
        cfg=cfg,
        flax_params=flax_params,
        torch_model=torch_model,
        r_max=cfg.model.cutoff,
        r_max_lr=cfg.model.cutoff_lr,
    )


def test_so3lr_checkpoint_s_parity():
    """Cheapest real checkpoint (2 layers, 128 features)."""
    assert _check_checkpoint_parity("s")


def test_so3lr_checkpoint_m_parity():
    """Mid-size real checkpoint (3 layers, 128 features)."""
    assert _check_checkpoint_parity("m")


def test_so3lr_checkpoint_l_parity():
    """Largest real checkpoint (3 layers, 256 features, doubled
    degrees) -- most expensive, run last."""
    assert _check_checkpoint_parity("l")


# ---------------------------------------------------------------------
# Part B: PME-enabled numerical-consistency test (so3lr-s only).
#
# No existing test in this codebase exercises PME against real
# checkpoint weights -- test_pme_electrostatics.py/test_pme_dispersion.py
# both use synthetic references (hand-built weights/structures), and all
# three Part-A tests above use the shipped checkpoints' actual
# kspace-disabled default. This test closes that gap: same real
# so3lr-s weights, PME/Ewald forced on for both sides via a modified
# copy of cfg.
#
# Critical gotcha (this is exactly the kind of mistake that caused a
# real OOM crash earlier in this plan's history -- see
# .superpowers/sdd/progress.md's Task-6 OOM incident from the
# PME-dispersion plan): the real so3lr-s checkpoint's own
# cfg.model.cutoff_lr is 1000.0 (Angstrom) -- only sane when PME/k-space
# is handling the true long-range tail and the real-space neighbor list
# is effectively unbounded/irrelevant. For a periodic test structure
# with a small unit cell, naively building a real-space long-range
# neighbor list out to 1000.0 Angstrom would enumerate an enormous
# number of periodic images and either hang or exhaust memory.
# cutoff_lr is therefore overridden down to 10.0 (matching the
# already-proven-working convention in test_pme_dispersion.py's own PME
# tests) in the modified cfg copy below -- this only changes which pairs
# feed the real-space *residual* dispersion/electrostatics sum (the
# k-space part, computed separately via the smearing/mesh grid, captures
# the true long-range tail regardless), not any learned parameter or
# model architecture -- safe to override for this test.
#
# Second gotcha, discovered while writing this exact test (see the test
# function's own docstring below for the full root-cause diagnosis): the
# brief's literal suggested structure -- a bare, un-rattled
# bulk("Ar", "fcc", a=5.26).repeat((3, 3, 3)) -- places 6 of its 27
# atoms exactly on a periodic-cell-boundary fractional coordinate,
# which triggers an unrelated PME mesh-interpolation differentiability
# edge case (independently present in both torch-pme and so3lr_dev's
# jaxpme backend) at exactly those atoms, producing large, misleading
# force "disagreements" that are not actually a JAX<->torch model
# discrepancy (confirmed via independent finite-difference checks on
# each side). A small deterministic ``atoms.rattle(...)`` avoids this.
# ---------------------------------------------------------------------


def test_so3lr_checkpoint_s_pme_enabled_parity(tmp_path):
    """PME/Ewald forced on (both electrostatics and dispersion) for the
    real so3lr-s checkpoint, against a small self-contained periodic
    structure.

    PME introduces zero new learnable parameters on either side (already
    established and relied upon throughout this plan's PME-dispersion
    work) -- so params.pkl is left completely unmodified; only the
    hyperparams (kspace_electrostatics/kspace_smearing/kspace_spacing/
    cutoff_lr) are overridden, in a deep-copied dict written out to a
    fresh temp hyperparameters.json (convert_flax_to_torch takes a file
    path, not a ConfigDict object, so plain dict/json manipulation is
    simplest here).

    Setting cfg.model.kspace_electrostatics="pme" alone is enough to
    turn on BOTH torch's use_pme and use_pme_dispersion:
    get_model_settings_flax_to_torch derives both from that same single
    JAX config key (see the comment at ~line 626 in
    jax_torch_conversion.py) -- there is no separate JAX flag for "PME
    dispersion only" vs. "PME electrostatics only" -- so no extra
    plumbing is needed here.

    Structure: ase.build.bulk("Ar", "fcc", a=5.26).repeat((3, 3, 3)) (27
    atoms) -- the exact, already-proven convention from
    test_pme_dispersion.py -- with one deliberate, investigated,
    documented addition: a small ``atoms.rattle(stdev=0.05, seed=42)``.
    Kept otherwise self-contained (not so3lr_dev's own bundled
    water/other fixtures), matching this plan's explicit guidance to
    keep this file's tests self-contained.

    Why the rattle (a real finding, not cosmetic): the un-rattled
    lattice places 6 of its 27 atoms exactly on a periodic-cell-boundary
    fractional coordinate (frac_z == 0.0 exactly -- an artifact of
    ``bulk("Ar", "fcc", ...)``'s 1-atom primitive cell being repeated by
    plain integer tiling, so every tile's corner atom sits exactly on
    the boundary). At those 6 atoms specifically, BOTH sides'
    forces (autodiff gradients) disagreed sharply from each other
    (observed: per-atom force diff up to ~0.236 eV/Angstrom, energy
    still agreeing to ~5e-8 relative) while the other 21 (generic,
    off-boundary) atoms agreed to ~1e-8. Root-cause isolation (finite
    differences taken independently on each side, not just
    JAX-vs-torch): each side's own analytic force at those atoms
    disagreed with *that same side's own* central-difference numerical
    gradient of its own energy function too -- e.g. one atom's JAX
    analytic force was [-0.0283, 0.1646, -0.1081] against a JAX finite-
    difference force of [-0.0283, 0.0283, 0.0283], and torch's analytic
    force at the same atom was [-0.1646, 0.0283, 0.0283] against a torch
    finite-difference force of [-0.0283, 0.0283, 0.0283] -- i.e. the two
    finite-difference values (computed completely independently, one in
    JAX and one in torch) agree with each other almost exactly, while
    each side's own *analytic* gradient is wrong relative to its own
    energy function, in a different way per side. This means the
    mismatch is not a JAX<->torch model/conversion discrepancy at all
    (the true, FD-verified physics agrees) -- it is a differentiability
    edge case in mesh-based PME charge assignment (present in both
    torch-pme, used by this repo's own PMEElectrostaticInteraction/
    PMEDispersionInteraction, and so3lr_dev's jaxpme backend,
    independently) specifically at atoms sitting exactly on a mesh/cell
    boundary -- a known pitfall of testing PME forces on an
    artificially-perfect, boundary-aligned lattice, not a bug in either
    framework's real-world (non-degenerate-lattice) behavior, and not a
    conversion bug this repo could fix. This is exactly why
    test_pme_dispersion.py's own gradcheck test (test 2) uses a random
    4-atom structure rather than this exact symmetric lattice, and why
    its energy-only tests (1/3/4) that DO use this lattice never check
    forces. `atoms.rattle(stdev=0.05, seed=42)` breaks the exact
    boundary alignment (perturbing every atom by ~1% of the lattice
    spacing, deterministically) without changing the cell, atom count,
    or overall self-contained/synthetic nature of the structure -- after
    rattling, per-atom force diffs are small and uniformly distributed
    (~5e-5-7e-5 across all 27 atoms, no outlier), confirming genuine
    mesh-discretization-level agreement rather than this boundary
    artifact. This finding, including the full FD-based diagnosis, is
    written up in v2arch-task-5-report.md.
    """
    from ase.build import bulk
    from ase.io import write as ase_write

    params_path, _hyperparams_path, flax_params, _cfg_unused = (
        _load_checkpoint("s")
    )

    with open(params_path.replace("params.pkl", "hyperparameters.json")) as f:
        raw_hyperparams = json.load(f)

    modified_hyperparams = copy.deepcopy(raw_hyperparams)
    modified_hyperparams["model"]["kspace_electrostatics"] = "pme"
    modified_hyperparams["model"]["kspace_smearing"] = 0.5
    modified_hyperparams["model"]["kspace_spacing"] = 0.25
    # See the module-level gotcha comment above: the real checkpoint's
    # own cutoff_lr=1000.0 would enumerate an enormous number of
    # periodic images for this small periodic cell and either hang or
    # exhaust memory. 10.0 matches test_pme_dispersion.py's own
    # already-proven-working convention for the real-space residual
    # sum's neighbor list; the k-space part captures the true long-range
    # tail separately regardless of this value.
    modified_hyperparams["model"]["cutoff_lr"] = 10.0

    modified_hyperparams_path = tmp_path / "hyperparameters_pme.json"
    with open(modified_hyperparams_path, "w") as f:
        json.dump(modified_hyperparams, f)

    torch_model = convert_flax_to_torch(
        path_to_flax_params=params_path,
        path_to_flax_hyperparams=str(modified_hyperparams_path),
        dtype=torch.float64,
    )

    cfg = config_dict.ConfigDict(modified_hyperparams)

    ar = bulk("Ar", "fcc", a=5.26).repeat((3, 3, 3))
    # See the docstring above: breaks exact periodic-boundary-aligned
    # fractional coordinates that trigger an unrelated PME mesh-
    # interpolation differentiability edge case (present in both
    # torch-pme and so3lr_dev's jaxpme backend independently), so this
    # test measures genuine PME numerical agreement instead.
    ar.rattle(stdev=0.05, seed=42)
    structure_path = tmp_path / "ar_pme.xyz"
    ase_write(str(structure_path), ar)

    # PME/Ewald mesh discretization has its own, separately documented,
    # legitimately-larger error floor in this exact codebase than the
    # kspace-disabled machine-precision-ceiling case above (see
    # test_pme_dispersion.py's Test 1/Test 4 -- meV/atom-level
    # residual-split agreement was the documented target there, not
    # eV-level machine-precision agreement; also flagged forward
    # explicitly by Task 4's fix #5 report as something this task should
    # account for). Observed on this run (post-rattle, see docstring
    # above): energy max_abs_diff=2.611e-05 eV, forces
    # max_abs_diff=5.824e-05 eV/Angstrom, both smoothly distributed
    # across all 27 atoms (no outlier) -- atol=5e-4 gives a comfortable
    # ~8.5x margin above the worst observed value without being an
    # arbitrarily huge, unjustified tolerance. rtol=0.0 for the same
    # reason as the tight-tolerance Part-A template test in
    # test_jax_torch_conversion.py: some force components are near
    # zero, which would otherwise inflate a relative-diff-based check
    # into a false failure.
    assert check_model_parity(
        cfg=cfg,
        flax_params=flax_params,
        torch_model=torch_model,
        r_max=cfg.model.cutoff,
        r_max_lr=cfg.model.cutoff_lr,
        structure_path=str(structure_path),
        atol=5e-4,
        rtol=0.0,
    )
