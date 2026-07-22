"""Permanent checkpoint-level JAX<->torch parity suite for the real,
bundled ``so3lr_dev`` checkpoints (``so3lr-s``/``-m``/``-l``), plus
PME-enabled numerical-consistency tests.

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

Part C (added by Task 3 of the later, independent
``.plans/plans/2026-07-21-pme-electrostatics-parity-fix.md`` plan) adds
charged-system (rattled NaCl rock-salt) PME parity tests -- Parts A/B
never exercise a genuinely charged periodic structure under PME, which
is exactly the case most sensitive to that plan's Task 1 fix (a missing
``electrostatic_energy_scale`` erf-damping term and a spurious extra
0.5 factor in ``PMEElectrostaticInteraction``'s k-space energy, both
affecting every PME-electrostatics run, not just charged ones). Part
B's existing Ar-based test was also re-measured and its tolerance
tightened as part of that same task, now that the underlying bug it was
unknowingly compensating for is fixed. See Part C's own module-level
comment further down for details.

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
# bulk("Ar", "fcc", a=5.26).repeat((3, 3, 3)) -- has 9 atoms sitting
# exactly on a periodic-cell-boundary fractional coordinate, 6 of which
# trigger an unrelated PME mesh-interpolation differentiability edge
# case (independently present in both torch-pme and so3lr_dev's jaxpme
# backend) -- boundary-alignment alone is necessary but not sufficient
# to predict exactly which atoms misbehave (an independent review traced
# the trigger further, to bare torch-pme mis-differentiating at
# symmetric/degenerate charge configurations) -- producing large,
# misleading force "disagreements" at those atoms that are not actually
# a JAX<->torch model discrepancy (confirmed via independent
# finite-difference checks on each side). A small deterministic
# ``atoms.rattle(...)`` avoids this regardless of the exact trigger.
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
    lattice has 9 of its 27 atoms sitting exactly on a
    periodic-cell-boundary fractional coordinate (frac_z == 0.0 exactly
    -- an artifact of ``bulk("Ar", "fcc", ...)``'s 1-atom primitive cell
    being repeated by plain integer tiling, so every tile's corner atom
    sits exactly on the boundary), but only 6 of those 9 actually
    misbehave -- an independent review of this test confirmed
    boundary-alignment alone is necessary but not sufficient to predict
    which atoms are affected, and traced the trigger further (using bare
    ``torchpme.PMECalculator`` with no SO3LR model code at all) to
    mis-differentiation at symmetric/degenerate per-atom charge
    configurations, which is what a uniform-element Ar lattice produces.
    At those 6 atoms specifically, BOTH sides'
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

    (
        params_path,
        _hyperparams_path,
        flax_params,
        _cfg_unused,
    ) = _load_checkpoint("s")

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

    # NOTE (re-validated as part of the PME-electrostatics-parity-fix
    # plan's Task 3, .superpowers/sdd/pme-elec-task-3-brief.md): the
    # numbers/tolerance below were re-measured fresh *after* that plan's
    # Task 1 fix (.superpowers/sdd/pme-elec-task-1-report.md) landed --
    # that fix removed a spurious extra 0.5 factor from
    # PMEElectrostaticInteraction's k-space energy (present for *every*
    # PME-electrostatics case, not just charged systems) and added the
    # model's own electrostatic_energy_scale erf-damping to the
    # real-space residual term, both of which change this Ar test's
    # k-space contribution too even though Ar is near-neutral. The
    # previously-documented atol=5e-4 (calibrated against a *pre-fix*
    # observation of energy max_abs_diff=2.611e-05 eV, forces
    # max_abs_diff=5.824e-05 eV/Angstrom) is now far looser than
    # necessary: freshly re-measured post-fix on this exact structure,
    # energy max_abs_diff=1.310e-06 eV (max_rel_diff=5.312e-08), forces
    # max_abs_diff=6.534e-09 eV/Angstrom (max_rel_diff=1.122e-06) -- both
    # roughly an order of magnitude tighter than before, now landing at
    # the same machine-precision-ceiling order of magnitude as Part A's
    # kspace-disabled tests and the new charged-NaCl PME tests below.
    # Switched to check_model_parity's own tightened default tolerance
    # (atol=1e-5, rtol=1e-4, see the module docstring's Part-A
    # rationale) instead of a bespoke looser one -- it comfortably covers
    # the fresh numbers (~7.6x margin on energy, ~1500x on forces)
    # without carrying forward a tolerance sized for a bug that no longer
    # exists.
    assert check_model_parity(
        cfg=cfg,
        flax_params=flax_params,
        torch_model=torch_model,
        r_max=cfg.model.cutoff,
        r_max_lr=cfg.model.cutoff_lr,
        structure_path=str(structure_path),
    )


# ---------------------------------------------------------------------
# Part C: charged-system (NaCl) PME parity tests.
#
# Parts A/B above never exercise a genuinely charged periodic structure
# under PME: Part A uses the shipped checkpoints' kspace-disabled
# default, and Part B's Ar lattice is a single, near-neutral element.
# The bug this whole plan fixed (see
# .superpowers/sdd/pme-elec-task-1-report.md/-task-2-report.md) --
# ``PMEElectrostaticInteraction`` silently missing the model's own
# ``electrostatic_energy_scale`` erf-damping entirely, plus a spurious
# extra 0.5 factor halving the k-space electrostatics energy -- affected
# every PME-electrostatics run, not just charged ones, but a charged
# rock-salt lattice is exactly the case where getting the long-range
# electrostatics tail wrong would matter most in practice. This section
# closes that specific coverage gap with a rattled 2x2x2 NaCl rock-salt
# supercell (16 atoms, formal +1/-1 ionic charges) -- the exact
# construction proven during this plan's diagnosis phase (see Task 1's
# report) and reused (2-atom primitive cell only) by
# ``test_pme_electrostatics.py``'s own Madelung test.
# ---------------------------------------------------------------------


def _build_rattled_nacl_structure(tmp_path, seed=7):
    """2x2x2 NaCl rock-salt supercell (16 atoms), rattled to avoid the
    same periodic-boundary-aligned PME mesh-interpolation edge case
    documented in Part B's test docstring above (this exact convention
    -- ``a=5.6402``, ``repeat((2, 2, 2))``,
    ``rattle(stdev=0.05, seed=7)`` -- is the setup used during this
    plan's diagnosis phase; see
    ``.superpowers/sdd/pme-elec-task-1-report.md``)."""
    from ase.build import bulk
    from ase.io import write as ase_write

    nacl = bulk("NaCl", crystalstructure="rocksalt", a=5.6402).repeat(
        (2, 2, 2)
    )
    nacl.rattle(stdev=0.05, seed=seed)
    structure_path = tmp_path / "nacl_pme.xyz"
    ase_write(str(structure_path), nacl)
    return str(structure_path)


def _pme_hyperparams(raw_hyperparams: dict, **model_overrides) -> dict:
    """Deep-copy a checkpoint's raw ``hyperparameters.json`` dict, force
    PME kspace electrostatics (and, per ``get_model_settings_flax_to_
    torch``'s single shared ``kspace_electrostatics`` flag, PME
    dispersion too) on -- same convention as Part B's test above --
    override ``cutoff_lr`` down to 10.0 for the same real-space
    neighbor-list-explosion reason documented in Part B's module-level
    comment, and apply any additional per-test ``model`` overrides (e.g.
    ``electrostatic_energy_bool=False`` to isolate dispersion)."""
    modified = copy.deepcopy(raw_hyperparams)
    modified["model"]["kspace_electrostatics"] = "pme"
    modified["model"]["kspace_smearing"] = 0.5
    modified["model"]["kspace_spacing"] = 0.25
    modified["model"]["cutoff_lr"] = 10.0
    modified["model"].update(model_overrides)
    return modified


def _check_checkpoint_pme_parity_on_structure(
    size: str,
    structure_path: str,
    tmp_path,
    label: str,
    **model_overrides,
) -> bool:
    """Build the torch model for real checkpoint ``size`` with PME
    kspace forced on (plus any ``model_overrides``, e.g. to isolate
    dispersion) and check parity against JAX on ``structure_path``, at
    ``check_model_parity``'s own tightened default tolerance
    (``atol=1e-5, rtol=1e-4``) -- see the calling test functions'
    docstrings below for the real observed numbers that justify relying
    on the default here rather than a custom looser one."""
    params_path, _hp_path, flax_params, _cfg_unused = _load_checkpoint(size)
    with open(params_path.replace("params.pkl", "hyperparameters.json")) as f:
        raw_hyperparams = json.load(f)

    modified_hyperparams = _pme_hyperparams(raw_hyperparams, **model_overrides)
    modified_hyperparams_path = (
        tmp_path / f"hyperparameters_pme_{size}_{label}.json"
    )
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


def test_so3lr_checkpoint_s_pme_electrostatics_charged_parity(tmp_path):
    """Charged-system PME electrostatics(+dispersion) parity, cheapest
    real checkpoint (2 layers, 128 features), on the rattled 2x2x2 NaCl
    supercell.

    Both ``electrostatic_energy_bool`` and ``dispersion_energy_bool``
    are left at the checkpoint's own (both ``True``) default -- unlike
    the isolated-dispersion regression test below, this is a combined
    check, which is fine: it is electrostatics specifically that this
    plan's Task 1 fix changed, and this rattled charged lattice is
    exactly the structure most sensitive to that fix's k-space erf-
    damping/0.5-factor corrections.

    Freshly observed (post-fix), via ``check_model_parity``'s printed
    report: energy max_abs_diff=1.280e-06 eV (max_rel_diff=2.585e-08),
    forces max_abs_diff=2.877e-08 eV/Angstrom
    (max_rel_diff=5.769e-06) -- comfortably inside
    ``check_model_parity``'s own default tolerance (``atol=1e-5,
    rtol=1e-4``, ~7.8x margin on energy, ~350x on forces), at the same
    order of magnitude as this file's kspace-disabled Part-A tests and
    Part B's re-validated Ar test. No custom ``atol``/``rtol`` override
    needed.
    """
    structure_path = _build_rattled_nacl_structure(tmp_path)
    assert _check_checkpoint_pme_parity_on_structure(
        "s", structure_path, tmp_path, "electro"
    )


def test_so3lr_checkpoint_m_pme_electrostatics_charged_parity(tmp_path):
    """Same as the ``-s`` test above, mid-size checkpoint (3 layers,
    128 features).

    Freshly observed: energy max_abs_diff=7.269e-07 eV
    (max_rel_diff=1.442e-08), forces max_abs_diff=8.521e-08
    eV/Angstrom (max_rel_diff=9.416e-06) -- again comfortably inside
    the default ``atol=1e-5, rtol=1e-4`` tolerance (~14x margin on
    energy, ~117x on forces).
    """
    structure_path = _build_rattled_nacl_structure(tmp_path)
    assert _check_checkpoint_pme_parity_on_structure(
        "m", structure_path, tmp_path, "electro"
    )


def test_so3lr_checkpoint_l_pme_electrostatics_charged_parity(tmp_path):
    """Same as the ``-s``/``-m`` tests above, largest checkpoint (3
    layers, 256 features, doubled degrees).

    Freshly observed: energy max_abs_diff=4.059e-07 eV
    (max_rel_diff=8.307e-09), forces max_abs_diff=9.448e-08
    eV/Angstrom (max_rel_diff=8.175e-06) -- comfortably inside the
    default ``atol=1e-5, rtol=1e-4`` tolerance (~25x margin on energy,
    ~106x on forces). Runtime is not materially higher than ``-s``/
    ``-m`` for this small 16-atom structure (~20s per test, dominated
    by JAX JIT compilation, not model size) -- so unlike some other
    per-checkpoint-size cost tradeoffs in this codebase, there was no
    reason to skip ``-m``/``-l`` here.
    """
    structure_path = _build_rattled_nacl_structure(tmp_path)
    assert _check_checkpoint_pme_parity_on_structure(
        "l", structure_path, tmp_path, "electro"
    )


def test_so3lr_checkpoint_s_pme_dispersion_charged_regression(tmp_path):
    """PME-dispersion-only regression check on the same charged rattled
    NaCl supercell (cheapest checkpoint only -- this closes a coverage
    gap, it is not re-validating a fix: dispersion's PME real-space
    residual formula is already believed correct, per this plan's own
    empirical finding recorded in
    ``.superpowers/sdd/pme-elec-task-1-report.md``/plan docs; no code
    change was needed or made to the dispersion path).

    Isolates dispersion by setting ``electrostatic_energy_bool=False``
    in the hyperparameters override (rather than keeping both terms on
    and treating it as a combined check, as the electrostatics tests
    above do) -- ``get_model_settings_flax_to_torch`` reads
    ``cfg.model.electrostatic_energy_bool``/``dispersion_energy_bool``
    directly from the (here, modified) hyperparameters dict on both the
    JAX-model-construction side (``make_so3krates_sparse_from_config``)
    and the torch side, so this cleanly removes electrostatics from
    both models' energy rather than merely occluding it -- a strictly
    stronger isolation than "keep both on".

    Freshly observed: energy max_abs_diff=1.302e-06 eV
    (max_rel_diff=3.302e-08), forces max_abs_diff=1.727e-08
    eV/Angstrom (max_rel_diff=7.474e-07) -- comfortably inside the
    default ``atol=1e-5, rtol=1e-4`` tolerance (~7.7x margin on energy,
    ~580x on forces), confirming the already-correct PME-dispersion
    real-space/k-space split holds up on a genuinely charged structure
    too (charge only matters here insofar as it changes which atoms'
    Hirshfeld ratios/C6 coefficients get evaluated at which positions,
    not the dispersion formula itself).
    """
    structure_path = _build_rattled_nacl_structure(tmp_path)
    assert _check_checkpoint_pme_parity_on_structure(
        "s",
        structure_path,
        tmp_path,
        "disp_only",
        electrostatic_energy_bool=False,
    )


def test_so3lr_checkpoint_s_pme_electrostatics_only_charged_parity(
    tmp_path,
):
    """PME-electrostatics-only (``dispersion_energy_bool=False``)
    numeric JAX parity check on the charged rattled NaCl supercell --
    closes a whole-branch-review-flagged gap left by Task 3's other new
    tests.

    Every other new charged-system test in this file (the three
    ``_electrostatics_charged_parity`` tests above, and the
    ``_dispersion_charged_regression`` test) leaves
    ``dispersion_energy_bool=True``, which independently forces
    ``self.use_lr=True`` in ``models.py`` via dispersion's own gate --
    so none of them can distinguish "the electrostatics ``use_lr`` gate
    fix genuinely works" from "the long-range neighbor list happened to
    be populated anyway because dispersion needed it too". Task 2's
    synthetic ``test_pme_electrostatics_only_no_dispersion``
    (``test_pme_electrostatics.py``) exercises exactly this
    ``use_pme=True, dispersion_energy_bool=False`` configuration but
    only asserts finite/non-NaN output, not a numeric reference. This
    test closes that gap with a real JAX cross-check: same rattled NaCl
    structure and PME settings as the other charged-system tests, same
    real ``so3lr-s`` weights, but ``dispersion_energy_bool=False`` so
    the long-range neighbor list is populated *solely* by electrostatics'
    own (fixed) gate.

    Freshly observed: energy max_abs_diff=1.233e-06 eV
    (max_rel_diff=2.519e-08), forces max_abs_diff=3.383e-08
    eV/Angstrom (max_rel_diff=5.538e-06) -- comfortably inside the
    default ``atol=1e-5, rtol=1e-4`` tolerance (~8.1x margin on energy,
    ~296x on forces), at the same order of magnitude as every other
    charged-system test in this file -- confirms the ``use_lr`` gating
    fix produces numerically correct results in exactly the
    configuration it changed, not merely that it avoids a crash.
    """
    structure_path = _build_rattled_nacl_structure(tmp_path)
    assert _check_checkpoint_pme_parity_on_structure(
        "s",
        structure_path,
        tmp_path,
        "electro_only",
        dispersion_energy_bool=False,
    )
