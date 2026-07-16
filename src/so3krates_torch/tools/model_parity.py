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

    Works around a bug in the mlff v1.0 checkout's ``ASE_to_jraph``: it
    crashes with ``AttributeError: 'float' object has no attribute
    'reshape'`` when ``atoms.calc is None`` (see
    ``v1_stagewise_parity._atoms_with_dummy_calc``). We don't need real
    reference energies for a model-vs-model parity check, so a dummy
    calculator is enough to route around the crash.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms = atoms.copy()
    atoms.calc = SinglePointCalculator(
        atoms, energy=0.0, forces=np.zeros((len(atoms), 3))
    )
    return atoms


def _build_jax_inputs(cfg, structure_path, index: int) -> dict:
    """Single-structure ``inputs`` dict for ``model.apply``, float64.

    Mirrors ``v1_stagewise_parity.build_example_jax_inputs``: mlff's
    own graph pipeline (``ASE_to_jraph`` + ``jraph.dynamically_batch``
    + ``graph_to_batch_fn``), padded by exactly 1 node/edge/pair for the
    mandatory jraph padding graph. Floating leaves are cast to float64
    explicitly (independent of ``flax_params``'s own dtype -- see the
    module-level dtype-handling notes in ``check_model_parity``).
    """
    import jax.numpy as jnp
    import jax.tree_util as jtu
    import jraph
    from ase.io import read
    from mlff.data.dataloader_sparse_ase import ASE_to_jraph
    from mlff.utils.jraph_utils import graph_to_batch_fn

    xyz_path = structure_path if structure_path is not None else _AQM_SMALL
    atoms = _atoms_with_dummy_calc(read(str(xyz_path), index=index))
    cutoff, cutoff_lr = cfg.model.cutoff, cfg.model.cutoff_lr

    graph = ASE_to_jraph(
        atoms,
        cutoff=cutoff,
        calculate_neighbors_lr=True,
        cutoff_lr=cutoff_lr,
    )
    batched = next(
        iter(
            jraph.dynamically_batch(
                [graph],
                n_node=int(graph.n_node[0]) + 1,
                n_edge=int(graph.n_edge[0]) + 1,
                n_graph=2,
                n_pairs=len(graph.idx_i_lr) + 1,
            )
        )
    )
    # graph_to_batch_fn returns plain numpy -- jax.vmap indexing inside
    # GeometryEmbedSparse raises TracerArrayConversionError on raw
    # numpy, so convert to jax arrays first.
    inputs = jtu.tree_map(jnp.asarray, graph_to_batch_fn(batched))
    return jtu.tree_map(
        lambda x: (
            x.astype(jnp.float64)
            if jnp.issubdtype(x.dtype, jnp.floating)
            else x
        ),
        inputs,
    )


def _build_torch_batch(
    r_max: float,
    r_max_lr,
    structure_path,
    index: int,
    dtype: torch.dtype,
):
    """Torch-side counterpart of ``_build_jax_inputs``.

    Mirrors ``v1_stagewise_parity.build_example_torch_batch``: ASE atoms
    -> Configuration -> AtomicData -> vendored PyG DataLoader. Floating
    tensors are cast to ``dtype`` (matching the torch model's own
    dtype -- see the dtype-handling notes in ``check_model_parity``);
    integer/bool tensors (indices, masks, ...) are left untouched.
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
):
    """Energy (scalar) + forces (n_real_atoms, 3), real atoms/graph only.

    Follows the ``jax.value_and_grad``-of-total-energy recipe from
    mlff's own ASE calculator (``mlff/md/calculator_sparse.py``,
    ``calculate_fn``), adapted to this flat-dict ``inputs`` convention.
    ``out["energy"]`` has one entry per graph (real + jraph's mandatory
    padding graph) -- masked with ``inputs["graph_mask"]`` before
    summing to a scalar. The gradient comes out with one row per node
    (real + padding); masked with ``inputs["node_mask"]`` to real atoms.

    ``model_factory``, if given, replaces the default v1 ``mlff``
    ``make_so3krates_sparse_from_config`` import for building the JAX
    model from ``cfg`` (e.g. a v2/``so3lr_dev`` equivalent).
    """
    import jax

    if model_factory is None:
        from mlff.config import (
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
    to build the JAX model from ``cfg`` (defaults to the v1 ``mlff``
    factory, so existing callers are unaffected).
    """
    import jax

    jax.config.update("jax_enable_x64", True)

    torch_model = copy.deepcopy(torch_model).eval()
    model_dtype = next(torch_model.parameters()).dtype

    inputs_jax = _build_jax_inputs(cfg, structure_path, index)
    batch_torch = _build_torch_batch(
        r_max, r_max_lr, structure_path, index, model_dtype
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
