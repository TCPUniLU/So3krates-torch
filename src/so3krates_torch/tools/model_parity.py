"""JAX <-> Torch model output (energy/forces) parity check.

Standalone numerical-core module: given a JAX SO3krates/SO3LR model
(``cfg`` + ``flax_params``) and a loaded torch counterpart, run both on
one real structure and check that predicted energy and forces agree
within tolerance. This catches conversion bugs (e.g. a correctly-shaped
but wrong-valued converted checkpoint) that a pure weight/shape
comparison cannot.

Closely modeled on the already-verified patterns in
``so3krates_torch.scripts.v1_stagewise_parity`` (graph building,
masking) -- but written fresh here rather than imported, since that
script has module-level global side effects (forcing float64 torch
default dtype, disabling jax jit) that are fine for a one-off dev
script but wrong to impose on a library function used as part of a
larger CLI command.
"""

import copy
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch

_AQM_SMALL = (
    Path(__file__).resolve().parent.parent / "tests" / "data" / "aqm_small.xyz"
)


def _atoms_with_dummy_calc(atoms):
    """Copy of `atoms` with a dummy SinglePointCalculator attached.

    Mirrors ``v1_stagewise_parity._atoms_with_dummy_calc``, which works
    around a real crash in old (v1) ``mlff``'s ``ASE_to_jraph`` --
    ``AttributeError: 'float' object has no attribute 'reshape'`` when
    ``atoms.calc is None``. ``so3lr_dev``'s own ``ASE_to_jraph`` does
    not have that bug (confirmed: it falls back to a NaN-valued energy
    placeholder instead of crashing), so this is no longer strictly
    required for correctness -- but attaching a dummy calculator is
    harmless (we don't need real reference energies for a
    model-vs-model parity check) and keeps this module's fixture
    loading parallel to the reference script's.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms = atoms.copy()
    atoms.calc = SinglePointCalculator(
        atoms, energy=0.0, forces=np.zeros((len(atoms), 3))
    )
    return atoms


def _build_jax_inputs(cfg, structure_path, index: int) -> dict:
    """Single-structure ``inputs`` dict for ``so3lr_dev``'s ``model.apply``.

    ``so3lr_dev``'s graph pipeline is genuinely different from old
    ``mlff`` v1's, not just a renamed import: its ``ASE_to_jraph``
    returns a ``(graph, lr_data)`` tuple (long-range neighbor indices are
    produced alongside the main graph rather than embedded inside it),
    batching goes through ``dynamically_batch_with_lr`` (not plain
    ``jraph.dynamically_batch``, which has no notion of the long-range
    pair budget), and ``graph_to_batch_fn`` takes the batched
    ``(main_graph, lr_batch)`` pair as its first two positional args plus
    an optional ``kspace_constants`` dict. Padding follows the same
    ``+1`` convention as ``so3lr_dev``'s own ``so3lr/cli/so3lr_eval.py``
    batching call (one spare node/edge/graph/pair slot for jraph's
    mandatory padding graph), specialized here to a single structure.

    When ``cfg.model.kspace_electrostatics`` is set (PME/Ewald k-space
    electrostatics enabled), the k-space grid/smearing are computed for
    this structure's cell via ``so3lr_dev``'s own
    ``kspace_utils.setup_kspace_grid`` and passed through as
    ``kspace_constants``, mirroring ``so3lr_dev``'s own
    ``from_config.py::_compute_kspace_constants`` (electrostatics and
    dispersion share the same grid/smearing). Left as ``None`` -- a
    clean no-op -- when kspace is disabled (the shipped ``so3lr-s/-m/-l``
    checkpoints' default).
    """
    import jax.numpy as jnp
    import jax.tree_util as jtu
    from ase.io import read
    from so3lr.mlff.data.dataloader_sparse_ase import ASE_to_jraph
    from so3lr.mlff.utils.jraph_utils import (
        dynamically_batch_with_lr,
        graph_to_batch_fn,
    )
    from so3lr.mlff.utils.kspace_utils import setup_kspace_grid

    xyz_path = structure_path if structure_path is not None else _AQM_SMALL
    atoms = _atoms_with_dummy_calc(read(str(xyz_path), index=index))
    cutoff, cutoff_lr = cfg.model.cutoff, cfg.model.cutoff_lr

    graph, lr_data = ASE_to_jraph(
        atoms,
        cutoff=cutoff,
        calculate_neighbors_lr=True,
        cutoff_lr=cutoff_lr,
    )
    graph_batch = next(
        iter(
            dynamically_batch_with_lr(
                [(graph, lr_data)],
                n_node=int(graph.n_node[0]) + 1,
                n_edge=int(graph.n_edge[0]) + 1,
                n_graph=2,
                n_pairs=int(lr_data["n_pairs"][0]) + 1,
            )
        )
    )

    kspace_electrostatics = cfg.model.get("kspace_electrostatics", None)
    kspace_constants = None
    # Truthy check, not `is not None`: real so3lr_dev configs use `None`
    # for "disabled" and a method string ("pme"/"ewald") for "enabled"
    # (see `setup_kspace_grid`'s own docstring), but a cfg produced by
    # this repo's own `get_model_settings_torch_to_flax`
    # (jax_torch_conversion.py) can set this key to an explicit Python
    # `False` when converting torch's boolean `use_pme`/
    # `use_pme_dispersion` flags back to JAX's single shared flag (real,
    # reported finding -- out of scope to fix here, see the Task 3
    # amendment report). `False` is never a valid *enabled* kspace
    # method, so treating it the same as `None` is correct for both
    # sources and avoids crashing on a non-periodic structure when
    # kspace is actually meant to be off.
    if kspace_electrostatics:
        box = jnp.asarray(np.array(atoms.get_cell()))
        k_grid, k_smearing = setup_kspace_grid(
            kspace_electrostatics=kspace_electrostatics,
            kspace_smearing=cfg.model.get("kspace_smearing", 2.0),
            kspace_spacing=cfg.model.get("kspace_spacing", 1.0),
            box=box,
        )
        kspace_constants = {
            "k_grid": k_grid,
            "k_smearing": k_smearing,
            "k_grid_disp": k_grid,
            "k_smearing_disp": k_smearing,
        }

    # graph_to_batch_fn returns plain numpy -- jax.vmap indexing inside
    # GeometryEmbedSparse raises TracerArrayConversionError on raw
    # numpy, so convert to jax arrays first.
    inputs = jtu.tree_map(
        jnp.asarray,
        graph_to_batch_fn(*graph_batch, kspace_constants=kspace_constants),
    )
    return jtu.tree_map(
        lambda x: (
            x.astype(jnp.float64)
            if jnp.issubdtype(x.dtype, jnp.floating)
            else x
        ),
        inputs,
    )


# so3lr_dev's own ``ASE_to_jraph`` (``mlff/data/dataloader_sparse_ase.py``)
# hardcodes a 16-column ``theory_mask`` with column 5 always active, for
# *every* structure it builds -- there is no ``cfg`` knob for this (it is
# a pure input/runtime shape, not a model-architecture setting; so3lr_dev's
# own flax observable head infers ``num_theory_levels`` from
# ``theory_mask.shape[-1]`` at ``model.init``/``model.apply`` time). That
# hardcoding is correct for the real shipped ``so3lr-s/-m/-l`` checkpoints
# (all genuinely 16-theory-level models), but wrong whenever the actual
# ``flax_params`` being tested describe a model with a *different*
# ``num_theory_levels`` -- e.g. a single-theory-level "v1" model produced
# by this repo's own conversion utilities. ``check_model_parity`` detects
# the real value from ``flax_params`` itself (see
# ``_detect_num_theory_levels``) and reconciles both sides via
# ``_match_theory_levels``/``_theory_level_index`` below, so callers never
# need to know or pass any of this explicitly.
_SO3LR_DEV_MAX_THEORY_LEVELS = 16
_SO3LR_DEV_THEORY_LEVEL = 5


def _detect_num_theory_levels(
    flax_params: dict, default: int = _SO3LR_DEV_MAX_THEORY_LEVELS
) -> int:
    """``num_theory_levels`` implied by ``flax_params``'s own
    ``energy_dense_final`` kernel width (``(regression_dim, T)``),
    falling back to ``default`` (so3lr_dev's own hardcoded value) if the
    expected param path isn't found (e.g. a differently-shaped params
    tree)."""
    params = flax_params.get("params", flax_params)
    try:
        kernel = params["observables_0"]["energy_dense_final"]["kernel"]
    except (KeyError, TypeError):
        return default
    return int(kernel.shape[-1])


def _theory_level_index(num_theory_levels: int) -> int:
    """Which one-hot ``theory_mask``/``theory_level`` column represents
    "the active theory level" for a model with ``num_theory_levels``
    columns. Column 5 (so3lr_dev's own hardcoded choice) only exists
    when ``num_theory_levels`` actually is so3lr_dev's own hardcoded 16;
    for any other value, column 0 is used instead -- any valid column
    works here, since nothing in this repo's real checkpoints/tests
    attaches meaning to a *specific* index beyond "populated by real
    weights"."""
    return (
        _SO3LR_DEV_THEORY_LEVEL
        if num_theory_levels == _SO3LR_DEV_MAX_THEORY_LEVELS
        else 0
    )


def _match_theory_levels(inputs_jax: dict, num_theory_levels: int) -> dict:
    """Override ``inputs_jax``'s ``theory_mask``/``theory_level`` to
    have exactly ``num_theory_levels`` columns (one-hot at
    ``_theory_level_index(num_theory_levels)``) when they don't already
    match -- see the module-level comment above
    ``_SO3LR_DEV_MAX_THEORY_LEVELS`` for why this is needed. A clean
    no-op (returns ``inputs_jax`` unchanged) when they already match,
    which is always true for the real shipped so3lr-s/-m/-l checkpoints.
    """
    if inputs_jax["theory_mask"].shape[-1] == num_theory_levels:
        return inputs_jax
    import jax.numpy as jnp

    n_graphs = inputs_jax["theory_mask"].shape[0]
    idx = _theory_level_index(num_theory_levels)
    theory_mask = (
        jnp.zeros((n_graphs, num_theory_levels), dtype=bool)
        .at[:, idx]
        .set(True)
    )
    inputs_jax = dict(inputs_jax)
    inputs_jax["theory_mask"] = theory_mask
    inputs_jax["theory_level"] = jnp.full(
        (n_graphs,), idx, dtype=inputs_jax["theory_level"].dtype
    )
    return inputs_jax


# Analogous gap to the theory-levels one above, for a different cfg key:
# ``get_model_settings_torch_to_flax`` (jax_torch_conversion.py, out of
# scope, already-approved Task 2 code) never sets
# ``cfg.model.use_final_bias_bool`` on the torch->flax cfg-building path,
# and so3lr_dev's own factory (``from_config.py``) defaults that key to
# ``False`` when absent -- so a bias-enabled torch model converted via
# that path would otherwise silently lose its final-layer bias on the
# JAX side of this parity check (a real, constant, zero-gradient per-atom
# energy offset). ``check_model_parity`` derives the real value directly
# from ``torch_model`` itself (see ``_detect_final_layer_bias``) and
# fills it into a *copy* of ``cfg`` only when genuinely absent (see
# ``_with_final_bias_bool``) -- mirroring the exact "explicit value
# always wins, only fill in the default when absent" convention
# ``jax_torch_conversion.py``'s own ``is_v2_config``/``legacy_so3lr_bool``
# handling already established. This is a clean no-op for all real
# shipped so3lr/-s/-m/-l checkpoints, whose ``hyperparameters.json``
# always sets ``use_final_bias_bool`` explicitly.
def _detect_final_layer_bias(torch_model: torch.nn.Module) -> Optional[bool]:
    """Whether ``torch_model``'s own final energy-output bias is
    present, read directly off its already-built module.

    ``atomic_energy_output_block.final_layer`` is a plain
    ``torch.nn.Linear`` (constructed with ``bias=final_layer_bias``, see
    ``AtomicEnergyOutputHead.__init__``) for the base ``So3krates``/
    ``SO3LR`` torch models -- ``.bias`` is ``None`` iff the model was
    built with ``final_layer_bias=False``, a real ``Parameter``
    otherwise. Returns ``None`` (meaning "can't tell, don't touch cfg")
    if ``torch_model`` doesn't have this exact shape -- e.g.
    ``MultiHeadSO3LR``'s ``MultiAtomicEnergyOutputHead``, whose
    equivalent bias is a raw ``Parameter`` named ``final_layer_bias``
    (never ``None``, always present, a different shape entirely), not a
    ``final_layer.bias``. Confirmed: this repo's current CLI/test suite
    never actually calls ``check_model_parity`` with such a model (only
    plain ``So3krates``/``SO3LR`` instances are ever built by
    ``torch_to_jax.py``/``jax_to_torch.py``), but this function stays
    defensive here regardless, skipping cleanly rather than guessing or
    crashing if that ever changes.
    """
    output_block = getattr(torch_model, "atomic_energy_output_block", None)
    final_layer = getattr(output_block, "final_layer", None)
    if not isinstance(final_layer, torch.nn.Linear):
        return None
    return final_layer.bias is not None


def _with_final_bias_bool(cfg, torch_model: torch.nn.Module):
    """Copy of ``cfg`` with ``cfg.model.use_final_bias_bool`` filled in
    from ``torch_model``'s own bias state (``_detect_final_layer_bias``)
    -- see the module comment above for why this is needed.

    A clean no-op (returns ``cfg`` unchanged, no copy made) whenever
    ``cfg.model.use_final_bias_bool`` is already explicitly set
    (including an explicit ``False`` -- that value always wins
    unchanged) or when ``torch_model``'s bias state can't be reliably
    determined (``_detect_final_layer_bias`` returns ``None``). Uses
    ``hasattr`` rather than ``cfg.model.get(...)`` to check presence
    without triggering a ``ConfigDict`` default, matching the same
    idiom ``jax_torch_conversion.py``'s own ``is_v2_config`` check
    already uses (``hasattr(cfg.model, k)``).
    """
    if hasattr(cfg.model, "use_final_bias_bool"):
        return cfg
    use_final_bias_bool = _detect_final_layer_bias(torch_model)
    if use_final_bias_bool is None:
        return cfg
    cfg = copy.deepcopy(cfg)
    cfg.model.use_final_bias_bool = use_final_bias_bool
    return cfg


def _build_torch_batch(
    r_max: float,
    r_max_lr,
    structure_path,
    index: int,
    dtype: torch.dtype,
    theory_level: Optional[int] = None,
):
    """Torch-side counterpart of ``_build_jax_inputs``.

    Mirrors ``v1_stagewise_parity.build_example_torch_batch``: ASE atoms
    -> Configuration -> AtomicData -> vendored PyG DataLoader. Floating
    tensors are cast to ``dtype`` (matching the torch model's own
    dtype -- see the dtype-handling notes in ``check_model_parity``);
    integer/bool tensors (indices, masks, ...) are left untouched.

    ``theory_level``, if given, is written onto the built
    ``Configuration`` before batching (harmless no-op for a
    single-theory-level torch model -- ``AtomicEnergyOutputHead`` only
    reads it when its own ``num_theory_levels > 1``), so this batch's
    active theory column can be kept consistent with whatever
    ``_match_theory_levels`` picked for the JAX side.
    """
    from ase.io import read

    from so3krates_torch.data.atomic_data import AtomicData
    from so3krates_torch.data.utils import (
        KeySpecification,
        config_from_atoms,
    )
    from so3krates_torch.tools import torch_geometric
    from so3krates_torch.tools.utils import AtomicNumberTable

    xyz_path = structure_path if structure_path is not None else _AQM_SMALL
    atoms = read(str(xyz_path), index=index)
    z_table = AtomicNumberTable([int(z) for z in range(1, 119)])
    config = config_from_atoms(atoms, key_specification=KeySpecification())
    if theory_level is not None:
        config.theory_level = theory_level
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            AtomicData.from_config(
                config,
                z_table=z_table,
                cutoff=r_max,
                cutoff_lr=r_max_lr,
            )
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    batch = next(iter(data_loader))
    return batch.apply(
        lambda x: x.to(dtype) if torch.is_floating_point(x) else x
    )


def _jax_energy_forces(
    cfg,
    flax_params: dict,
    inputs: dict,
    model_factory: Optional[Callable] = None,
    jax_input_builder: Optional[Callable] = None,
):
    """Energy (scalar) + forces (n_real_atoms, 3), real atoms/graph only.

    Follows the ``jax.value_and_grad``-of-total-energy recipe from
    mlff's own ASE calculator (``mlff/md/calculator_sparse.py``,
    ``calculate_fn``), adapted to this flat-dict ``inputs`` convention.
    ``out["energy"]`` has one entry per graph (real + jraph's mandatory
    padding graph) -- masked with ``inputs["graph_mask"]`` before
    summing to a scalar. The gradient comes out with one row per node
    (real + padding); masked with ``inputs["node_mask"]`` to real atoms.

    ``model_factory``, if given, replaces the default ``so3lr_dev``
    ``make_so3krates_sparse_from_config`` import for building the JAX
    model from ``cfg``.

    ``jax_input_builder`` is accepted for signature symmetry with
    ``check_model_parity`` (which resolves it and builds ``inputs``
    before calling this function) but is otherwise unused here --
    ``inputs`` has already been built by the time this function runs.
    """
    import jax

    if model_factory is None:
        from so3lr.mlff.config import (
            make_so3krates_sparse_from_config as model_factory,
        )

    model = model_factory(cfg)
    graph_mask = inputs["graph_mask"]
    node_mask = np.asarray(inputs["node_mask"])

    def energy_fn(positions):
        inputs_r = dict(inputs, positions=positions)
        out = model.apply(flax_params, inputs_r)
        return (out["energy"] * graph_mask).sum()

    energy, grad = jax.value_and_grad(energy_fn)(inputs["positions"])
    forces = -grad

    energy_np = float(np.asarray(energy))
    forces_np = np.asarray(forces)[node_mask]
    return energy_np, forces_np


def _torch_energy_forces(torch_model: torch.nn.Module, batch):
    """Energy (scalar) + forces (n_atoms, 3) from the torch model.

    Mirrors ``v1_stagewise_parity._run_with_grad``: the model computes
    ``out["energy"]``/``out["forces"]`` internally given
    ``positions.requires_grad_(True)``.
    """
    batch = batch.clone()
    batch.positions.requires_grad_(True)
    out = torch_model(batch.to_dict(), compute_stress=False)
    energy_np = float(out["energy"].detach().cpu().reshape(-1)[0])
    forces_np = out["forces"].detach().cpu().numpy()
    return energy_np, forces_np


def _max_abs_rel_diff(a: np.ndarray, b: np.ndarray):
    """Diagnostic ``(max_abs_diff, max_rel_diff)`` of ``a`` vs. ``b``."""
    abs_diff = np.abs(a - b)
    max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0
    rel_diff = abs_diff / (np.abs(b) + 1e-30)
    max_rel = float(np.max(rel_diff)) if rel_diff.size else 0.0
    return max_abs, max_rel


def check_model_parity(
    cfg,
    flax_params: dict,
    torch_model: torch.nn.Module,
    r_max: float,
    r_max_lr: Optional[float] = None,
    structure_path: Optional[str] = None,
    index: int = 0,
    atol: float = 1e-3,
    rtol: float = 1e-2,
    model_factory: Optional[Callable] = None,
    jax_input_builder: Optional[Callable] = None,
) -> bool:
    """Run both models on one real structure and compare energy+forces.

    ``cfg`` is the JAX ``ConfigDict`` describing the model architecture,
    ``flax_params`` the corresponding ``{"params": {...}}`` tree.
    ``torch_model`` is a loaded torch SO3LR/So3krates ``nn.Module``
    instance. ``r_max``/``r_max_lr`` are the short-/long-range cutoffs
    used to build the torch-side neighbor lists (the JAX side uses
    ``cfg.model.cutoff``/``cfg.model.cutoff_lr``, which must describe
    the same architecture).

    If ``structure_path`` is ``None``, the repo's bundled
    ``tests/data/aqm_small.xyz`` fixture is used; otherwise any
    ASE-readable file is accepted (``index`` selects which frame).

    Dtype handling: JAX is run with ``jax_enable_x64`` enabled and the
    JAX-side inputs are cast to float64 explicitly (this only widens
    precision; ``flax_params`` itself is left at whatever precision it
    already is, so the real converted weights are what gets tested).
    The torch side is run at ``torch_model``'s own native dtype, on a
    *deep copy* of ``torch_model`` -- the caller's model is never
    mutated by this read-only check.

    Prints a small PASS/FAIL report table with the actual max abs/rel
    diff for energy and forces. Returns True iff both agree within
    ``atol``/``rtol`` (``np.allclose`` semantics:
    ``|a - b| <= atol + rtol * |b|``).

    ``model_factory``, if given, is forwarded to ``_jax_energy_forces``
    to build the JAX model from ``cfg`` (defaults to the ``so3lr_dev``
    factory, so existing callers are unaffected).

    ``jax_input_builder``, if given, replaces the default
    ``_build_jax_inputs`` for building the JAX-side ``inputs`` dict from
    ``cfg``/``structure_path``/``index``. Independent of
    ``model_factory``.

    The JAX-side ``theory_mask``/``theory_level`` (and the matching
    torch-side ``theory_level``) are reconciled automatically against
    ``flax_params``'s own ``num_theory_levels`` (see
    ``_detect_num_theory_levels``/``_match_theory_levels`` above) -- so
    a single-theory-level model and a real 16-theory-level so3lr_dev
    checkpoint both "just work" without the caller needing to know
    about this at all.

    ``cfg.model.use_final_bias_bool`` is likewise filled in from
    ``torch_model``'s own final-layer bias state when genuinely absent
    from ``cfg.model`` (see ``_with_final_bias_bool``/
    ``_detect_final_layer_bias`` above) -- a no-op for cfgs that already
    set it explicitly (true for all real shipped so3lr_dev checkpoints).
    """
    import jax

    jax.config.update("jax_enable_x64", True)

    if jax_input_builder is None:
        jax_input_builder = _build_jax_inputs

    torch_model = copy.deepcopy(torch_model).eval()
    model_dtype = next(torch_model.parameters()).dtype

    cfg = _with_final_bias_bool(cfg, torch_model)

    num_theory_levels = _detect_num_theory_levels(flax_params)
    inputs_jax = _match_theory_levels(
        jax_input_builder(cfg, structure_path, index), num_theory_levels
    )
    batch_torch = _build_torch_batch(
        r_max,
        r_max_lr,
        structure_path,
        index,
        model_dtype,
        theory_level=_theory_level_index(num_theory_levels),
    )

    energy_jax, forces_jax = _jax_energy_forces(
        cfg, flax_params, inputs_jax, model_factory=model_factory
    )
    energy_torch, forces_torch = _torch_energy_forces(torch_model, batch_torch)

    energy_jax_arr = np.asarray(energy_jax)
    energy_torch_arr = np.asarray(energy_torch)
    energy_ok = bool(
        np.allclose(energy_jax_arr, energy_torch_arr, atol=atol, rtol=rtol)
    )
    forces_ok = bool(
        np.allclose(forces_jax, forces_torch, atol=atol, rtol=rtol)
    )

    energy_abs, energy_rel = _max_abs_rel_diff(
        energy_jax_arr, energy_torch_arr
    )
    forces_abs, forces_rel = _max_abs_rel_diff(forces_jax, forces_torch)

    print("\n--- JAX <-> Torch model output parity ---")
    print(
        f"{'quantity':<10} {'status':<6} {'max_abs_diff':>14} "
        f"{'max_rel_diff':>14}"
    )
    print(
        f"{'energy':<10} {'PASS' if energy_ok else 'FAIL':<6} "
        f"{energy_abs:>14.3e} {energy_rel:>14.3e}"
    )
    print(
        f"{'forces':<10} {'PASS' if forces_ok else 'FAIL':<6} "
        f"{forces_abs:>14.3e} {forces_rel:>14.3e}"
    )

    return energy_ok and forces_ok
