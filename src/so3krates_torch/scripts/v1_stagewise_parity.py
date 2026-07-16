# --- float64 + eager JAX must be configured before anything else ever
# touches a jax or torch array. Do not reorder these imports/calls. ---
import jax

jax.config.update("jax_enable_x64", True)
jax.config.update("jax_disable_jit", True)

import torch

torch.set_default_dtype(torch.float64)

"""V1 stage-wise JAX <-> Torch numerical parity check.

Standalone dev tool (NOT part of the pytest suite -- see CLAUDE.md) for
the "legacy SO3LR v1" architecture (single theory level, ZBL repulsion,
legacy dispersion, real-space electrostatics/dispersion -- no
multitheory, NHL repulsion, c6 head, or PME):

1. Builds a small but architecturally faithful "v1" SO3LR model in both
   JAX (``mlff``, branch ``v1.0``) and PyTorch (this repo), using random
   float64 weights.
2. Wraps the existing JAX<->Torch weight-conversion utilities in
   ``so3krates_torch.tools.jax_torch_conversion`` as pure in-memory
   functions (no pickle/JSON round trip through disk) and runs a weight
   round-trip fidelity check (jax -> torch -> jax and torch -> jax ->
   torch), isolating conversion-layer bugs from model-math bugs before
   any activation-level comparison.
3. Captures stage-wise activations (embedding -> per-layer
   representation -> output heads -> energy/forces) on both sides via
   torch forward hooks and ``jax.apply(..., capture_intermediates=True)``
   and compares them stage by stage, including a diagnostic
   equivariant-feature (spherical-harmonic convention) realignment
   search.
4. Optionally (``--compile``) checks that ``torch.compile`` reproduces
   the eager energy/forces for this model.

Run with the ``so3`` mamba env active (JAX/flax/mlff v1.0 are only
available there)::

    eval "$(mamba shell hook --shell bash)" && mamba activate so3
    python src/so3krates_torch/scripts/v1_stagewise_parity.py
"""

import argparse
import itertools
from pathlib import Path

import numpy as np
import jax.numpy as jnp
import jax.tree_util as jtu
import jraph
from ml_collections import config_dict
from ase.io import read

from mlff.config import make_so3krates_sparse_from_config

from so3krates_torch.modules.models import SO3LR
from so3krates_torch.tools.jax_torch_conversion import (
    flatten_params,
    get_flax_to_torch_mapping,
    get_model_settings_flax_to_torch,
    convert_flax_to_torch_params,
    convert_torch_to_flax,
)

_AQM_SMALL = (
    Path(__file__).resolve().parent.parent / "tests" / "data" / "aqm_small.xyz"
)


# ---------------------------------------------------------------------
# Step 1.1: matched v1 settings (torch) + config_dict (jax)
# ---------------------------------------------------------------------

# A single, complete kwargs dict for `SO3LR(**V1_TORCH_SETTINGS)`. Kept
# intentionally small (num_features=16, num_layers=1) for fast
# iteration -- this is a parity/round-trip smoke test, not a realistic
# production-sized SO3LR.
V1_TORCH_SETTINGS = dict(
    r_max=4.5,
    r_max_lr=12.0,
    num_radial_basis_fn=8,
    degrees=[1, 2],
    num_features=16,
    num_heads=2,
    num_layers=1,
    num_features_head=8,  # unused by torch; kept for jax-side parity
    num_elements=118,
    avg_num_neighbors=4.0,
    message_normalization="avg_num_neighbors",
    cutoff_fn="cosine",
    radial_basis_fn="gaussian",
    trainable_rbf=True,
    energy_regression_dim=16,
    energy_activation_fn="silu",
    activation_fn="silu",
    qk_non_linearity="identity",
    energy_learn_atomic_type_scales=False,
    energy_learn_atomic_type_shifts=False,
    layer_normalization_1=False,
    layer_normalization_2=False,
    residual_mlp_1=False,
    residual_mlp_2=False,
    use_charge_embed=False,
    use_spin_embed=False,
    input_convention="positions",
    layers_behave_like_identity_fn_at_init=False,
    output_is_zero_at_init=False,
    # v1/legacy SO3LR physics -- see the v2-flag guard below.
    zbl_repulsion_bool=True,
    electrostatic_energy_bool=True,
    electrostatic_energy_scale=4.0,
    dispersion_energy_bool=True,
    dispersion_energy_cutoff_lr_damping=2.0,
    dispersion_energy_scale=1.2,
    c6_ratios_bool=False,
    use_simple_hirshfeld=False,
    # This v1.0 mlff checkout's `energy_dense_final` Dense layer does NOT
    # set `use_bias=False`, so the matching torch model must keep a bias
    # here too (get_model_settings_flax_to_torch now detects this
    # correctly from the checkpoint instead of assuming either way).
    final_layer_bias=True,
    device="cpu",
    dtype=torch.float64,
)

# v2-only flags that must never sneak into a "v1" settings dict. Kept as
# a guard because Tasks 2-4 (and future edits) extend this same module.
_V2_ONLY_FLAGS = ("c6_ratios_bool", "nhl_repulsion_bool", "use_pme")


def _assert_v1_only_settings(settings: dict) -> None:
    for flag in _V2_ONLY_FLAGS:
        assert not settings.get(flag, False), (
            f"{flag!r} is a v2-only flag and must be False/absent in "
            "V1_TORCH_SETTINGS -- this script targets the v1/legacy "
            "SO3LR architecture only."
        )
    num_theory_levels = settings.get("num_theory_levels", 1)
    assert num_theory_levels == 1, (
        "V1_TORCH_SETTINGS must use a single theory level "
        f"(got num_theory_levels={num_theory_levels}); v1 SO3LR has no "
        "multi-theory-level / multi-fidelity support."
    )


_assert_v1_only_settings(V1_TORCH_SETTINGS)


def build_jax_v1_config_from_settings(
    settings: dict,
) -> config_dict.ConfigDict:
    """Build the JAX ``ConfigDict`` matching a ``V1_TORCH_SETTINGS``-shaped
    torch settings dict (see ``build_jax_v1_config`` for the default).

    NOTE on the brief's suggested ``legacy_so3lr_bool`` /
    ``kspace_electrostatics`` config keys: neither exists anywhere in
    this local ``mlff`` v1.0 checkout (confirmed by grepping the
    package), so they are intentionally omitted here. This checkout's
    ``make_so3krates_sparse_from_config`` only ever builds a single
    theory level and has no k-space/PME electrostatics path, so "single
    theory" and "legacy" are the only options available -- there is
    nothing to toggle.
    """
    cfg = config_dict.ConfigDict()
    cfg.model = config_dict.ConfigDict()
    cfg.model.num_layers = settings["num_layers"]
    cfg.model.num_features = settings["num_features"]
    cfg.model.num_heads = settings["num_heads"]
    cfg.model.num_features_head = settings["num_features_head"]
    # JAX resolves this string via getattr(utils.radial_basis_fn, name),
    # so it must be the exact (capitalised) class name -- unlike the
    # torch side, which lower()s whatever string it is given.
    cfg.model.radial_basis_fn = "Gaussian"
    cfg.model.num_radial_basis_fn = settings["num_radial_basis_fn"]
    cfg.model.cutoff_fn = settings["cutoff_fn"]
    cfg.model.cutoff = settings["r_max"]
    cfg.model.cutoff_lr = settings["r_max_lr"]
    cfg.model.degrees = list(settings["degrees"])
    cfg.model.residual_mlp_1 = settings["residual_mlp_1"]
    cfg.model.residual_mlp_2 = settings["residual_mlp_2"]
    cfg.model.layer_normalization_1 = settings["layer_normalization_1"]
    cfg.model.layer_normalization_2 = settings["layer_normalization_2"]
    cfg.model.message_normalization = settings["message_normalization"]
    cfg.model.qk_non_linearity = settings["qk_non_linearity"]
    cfg.model.activation_fn = settings["activation_fn"]
    cfg.model.layers_behave_like_identity_fn_at_init = settings[
        "layers_behave_like_identity_fn_at_init"
    ]
    cfg.model.output_is_zero_at_init = settings["output_is_zero_at_init"]
    cfg.model.input_convention = settings["input_convention"]
    cfg.model.use_charge_embed = settings["use_charge_embed"]
    cfg.model.use_spin_embed = settings["use_spin_embed"]
    cfg.model.energy_regression_dim = settings["energy_regression_dim"]
    cfg.model.energy_activation_fn = settings["energy_activation_fn"]
    cfg.model.energy_learn_atomic_type_scales = settings[
        "energy_learn_atomic_type_scales"
    ]
    cfg.model.energy_learn_atomic_type_shifts = settings[
        "energy_learn_atomic_type_shifts"
    ]
    cfg.model.electrostatic_energy_bool = settings["electrostatic_energy_bool"]
    cfg.model.electrostatic_energy_scale = settings[
        "electrostatic_energy_scale"
    ]
    cfg.model.dispersion_energy_bool = settings["dispersion_energy_bool"]
    cfg.model.dispersion_energy_cutoff_lr_damping = settings[
        "dispersion_energy_cutoff_lr_damping"
    ]
    cfg.model.dispersion_energy_scale = settings["dispersion_energy_scale"]
    cfg.model.zbl_repulsion_bool = settings["zbl_repulsion_bool"]

    cfg.data = config_dict.ConfigDict()
    cfg.data.avg_num_neighbors = settings["avg_num_neighbors"]

    # Training is always sparse-format for long-range blocks in this
    # mlff checkout (see mlff/config/from_config.py:run_training).
    cfg.neighborlist_format_lr = "sparse"
    # `make_so3krates_sparse_from_config` (mlff/config/from_config.py)
    # reads this top-level field unconditionally -- added upstream after
    # this script was first written (mlff v1.0-lrs-gems branch, PR #39
    # "new_pass_obs"); `None` matches `SO3kratesSparse`'s own default and
    # this script never needs sown intermediate quantities.
    cfg.output_intermediate_quantities = None
    return cfg


def build_jax_v1_config() -> config_dict.ConfigDict:
    return build_jax_v1_config_from_settings(V1_TORCH_SETTINGS)


JAX_V1_CONFIG = build_jax_v1_config()


# ---------------------------------------------------------------------
# Step 1.2: build JAX v1 model + random float64 params
# ---------------------------------------------------------------------


def _atoms_with_dummy_calc(atoms):
    """Copy of `atoms` with a dummy SinglePointCalculator attached.

    Works around a bug in this mlff v1.0 checkout's ``ASE_to_jraph``
    (``mlff/data/dataloader_sparse_ase.py``): when ``atoms.calc is
    None``, its "no calculator" fallback sets ``energy = np.nan`` (a
    bare Python float) but the code a few lines later unconditionally
    calls ``energy.reshape(-1)`` on it, crashing with
    ``AttributeError: 'float' object has no attribute 'reshape'``.
    ``aqm_small.xyz`` stores its reference energy in ``atoms.info``,
    not via an attached calculator, so it hits this path. We don't need
    real reference energies for a model-vs-model parity check, so a
    dummy calculator (energy=0, zero forces) is enough to route around
    the crash.
    """
    from ase.calculators.singlepoint import SinglePointCalculator

    atoms = atoms.copy()
    atoms.calc = SinglePointCalculator(
        atoms, energy=0.0, forces=np.zeros((len(atoms), 3))
    )
    return atoms


def build_example_jax_inputs(
    cfg: config_dict.ConfigDict, xyz_path=None, index: int = 0
) -> dict:
    """Single-molecule ``inputs`` dict for ``model.init``/``apply``/capture.

    Loads one molecule (default: the first, methane, 5 atoms) from
    ``xyz_path`` (default: the repo's existing ``aqm_small.xyz`` test
    fixture) and builds short-range/long-range neighbor lists with
    mlff's own graph pipeline (``ASE_to_jraph`` + ``jraph.
    dynamically_batch`` + ``graph_to_batch_fn``) -- the same pipeline
    ``run_training``/``run_evaluation`` use, so this exercises exactly
    the input convention the model expects in production, not a
    hand-rolled approximation of it.

    ``jraph.dynamically_batch`` always pads to at least 2 graphs (the
    real one plus one padding graph), which introduces one bookkeeping
    self-loop edge (padding-node -> padding-node) into the raw
    ``idx_i``/``idx_j``/``idx_i_lr``/``idx_j_lr`` arrays. That padding
    edge is real batching machinery, not a bug -- ``node_mask`` (also
    returned here) marks which nodes are real vs padding, and must be
    used to filter it out of any edge-level comparison (see
    ``check_edge_sets`` in Task 2).
    """
    from mlff.data.dataloader_sparse_ase import ASE_to_jraph
    from mlff.utils.jraph_utils import graph_to_batch_fn

    xyz_path = xyz_path if xyz_path is not None else _AQM_SMALL
    atoms = _atoms_with_dummy_calc(read(str(xyz_path), index=index))
    cutoff, cutoff_lr = cfg.model.cutoff, cfg.model.cutoff_lr

    graph = ASE_to_jraph(
        atoms, cutoff=cutoff, calculate_neighbors_lr=True, cutoff_lr=cutoff_lr
    )
    # Pad by exactly 1 node/edge/pair for the mandatory padding graph.
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
    # graph_to_batch_fn (mlff.utils.jraph_utils) returns plain numpy
    # arrays -- training_utils.fit() explicitly converts with
    # jax.tree.map(jnp.array, ...) before ever calling the model; skip
    # that step and jax.vmap indexing inside GeometryEmbedSparse raises
    # TracerArrayConversionError (numpy fancy-indexing with a tracer).
    inputs = jtu.tree_map(jnp.asarray, graph_to_batch_fn(batched))
    return jtu.tree_map(
        lambda x: x.astype(jnp.float64)
        if jnp.issubdtype(x.dtype, jnp.floating)
        else x,
        inputs,
    )


def build_jax_v1(config: config_dict.ConfigDict):
    """Build the JAX v1 SO3krates/SO3LR model with random float64 params.

    Returns ``(model, params, cfg)``. ``params`` is the *full* variables
    dict as returned by ``model.init(...)`` (i.e. ``{"params": {...}}``),
    not the stripped-down inner tree -- this matches what
    ``jax_torch_conversion.py``'s converters expect (they were written
    against pickled ``{"params": ...}`` checkpoints).

    mlffCalculatorSparse checkpoint-loading fallback (mentioned as an
    option in the task brief) is intentionally not implemented: in this
    environment ``jax==0.8.2`` removed ``jax.experimental.host_callback``,
    which ``mlff.md`` (transitively imported by ``mlffCalculatorSparse``)
    requires, and no real v1 checkpoint directory exists on this machine
    anyway. The random-init path below is the only available path and is
    sufficient to exercise every conversion stage.
    """
    model = make_so3krates_sparse_from_config(config)
    example_inputs = build_example_jax_inputs(config)
    params = model.init(jax.random.PRNGKey(0), example_inputs)

    # This v1.0 mlff checkout hard-codes `param_dtype=jnp.float32` for
    # several Dense/Embed modules regardless of `jax_enable_x64` (e.g.
    # embeddings, most Dense filters) -- confirmed empirically: roughly
    # half the leaves come out float32 even with x64 enabled globally.
    # Cast explicitly to honor the float64-everywhere global constraint;
    # this is a lossless widening cast (float32 -> float64), not a
    # precision-losing one.
    params = jtu.tree_map(lambda x: x.astype(jnp.float64), params)
    return model, params, config


# ---------------------------------------------------------------------
# Step 1.3: build torch v1 model + set ev-init-to-zeros
# ---------------------------------------------------------------------


def build_torch_v1(settings: dict = None) -> SO3LR:
    settings = dict(V1_TORCH_SETTINGS if settings is None else settings)
    model_torch = SO3LR(**settings).double().eval()
    model_torch.initialize_ev_to_zeros = True
    model_torch.ev_embedding.initialization_to_zeros = True
    return model_torch


# ---------------------------------------------------------------------
# Step 1.4: in-memory jax<->torch converters (no pickle/JSON files)
# ---------------------------------------------------------------------


def jax_to_torch(
    cfg: config_dict.ConfigDict, flax_params: dict, settings: dict = None
) -> SO3LR:
    """Convert JAX params to a loaded torch ``SO3LR`` model, in memory."""
    settings = settings or {}
    trainable_rbf = settings.get("trainable_rbf", True)
    dtype = settings.get("dtype", torch.float64)
    flat_params = flatten_params(flax_params)

    torch_settings = get_model_settings_flax_to_torch(
        cfg=cfg,
        device=settings.get("device", "cpu"),
        use_defined_shifts=settings.get("use_defined_shifts", False),
        flat_params=flat_params,
        num_elements=settings.get("num_elements", 118),
        trainable_rbf=trainable_rbf,
        dtype=dtype,
    )
    torch_model = SO3LR(**torch_settings)

    mapping = get_flax_to_torch_mapping(
        cfg=cfg, trainable_rbf=trainable_rbf, flat_params=flat_params
    )
    state_dict = convert_flax_to_torch_params(
        torch_state_dict=torch_model.state_dict(),
        flax_params=flax_params,
        mapping=mapping,
        dtype=dtype,
    )
    torch_model.load_state_dict(state_dict)
    return torch_model.double().eval()


def torch_to_jax(torch_model: SO3LR, settings: dict = None):
    """Convert a torch ``SO3LR`` model to ``(cfg, flax_params)`` in memory."""
    settings = settings if settings is not None else V1_TORCH_SETTINGS
    cfg, flax_params = convert_torch_to_flax(
        torch_state_dict=torch_model.state_dict(),
        torch_settings=settings,
        trainable_rbf=settings.get("trainable_rbf", True),
        dtype="float64",
    )
    return cfg, flax_params


# ---------------------------------------------------------------------
# Step 1.5: weight round-trip fidelity check
# ---------------------------------------------------------------------

# Pre-existing, *documented* asymmetries in the shared conversion
# utilities (src/so3krates_torch/tools/jax_torch_conversion.py) that
# make a byte-exact round trip impossible for a couple of leaves. These
# are real findings, not bugs in this script -- see the comments below
# and the task-1 report for detail. Keying by the JAX-side flattened
# leaf name (as produced by `flatten_params`).
_PADDING_ROW_LEAVES = {
    # Row 0 of this (num_elements + 1, num_features) embedding table is
    # the "atomic number 0" padding slot. convert_flax_to_torch_params
    # drops it (`torched[:, 1:]`, torch has no padding row) and
    # convert_torch_to_flax_params re-synthesizes it as an all-zero row
    # on the way back. A freshly random-initialized JAX model gives this
    # row a real (nonzero) value, so it can never round-trip byte-exact
    # -- but it is also never read by the model (atomic number 0 never
    # occurs in real inputs), so row 0 is excluded from the comparison
    # below rather than failing the whole leaf.
    "params/feature_embeddings_0/Embed_0/embedding",
}


def _leaf_diff_table(reference: dict, roundtripped: dict, atol: float):
    """Compare two flattened (str -> array-like) dicts leaf by leaf.

    Returns ``(rows, all_ok)``. Each row is
    ``(key, status, max_abs_diff, note)``. ``all_ok`` is True iff every
    row either PASS-ed or was a *documented* exception.
    """
    ref_keys, rt_keys = set(reference), set(roundtripped)
    rows = []
    all_ok = True

    for key in sorted(ref_keys - rt_keys):
        rows.append((key, "MISSING", float("nan"), ""))
        all_ok = False

    for key in sorted(rt_keys - ref_keys):
        rows.append((key, "EXTRA", float("nan"), ""))
        all_ok = False

    for key in sorted(ref_keys & rt_keys):
        a = np.asarray(reference[key])
        b = np.asarray(roundtripped[key])
        if a.shape != b.shape:
            rows.append((key, "SHAPE", float("nan"), "shape mismatch"))
            all_ok = False
            continue

        note = ""
        if key in _PADDING_ROW_LEAVES and a.shape[0] > 1:
            a, b = a[1:], b[1:]
            note = "row 0 (padding) excluded, see comment"
        diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        passed = diff <= atol
        rows.append((key, "PASS" if passed else "FAIL", diff, note))
        all_ok = all_ok and passed

    return rows, all_ok


def _print_table(title: str, rows) -> None:
    print(f"\n--- {title} ---")
    print(f"{'key':<70} {'status':<8} {'max_abs_diff':>14}  note")
    for key, status, diff, note in rows:
        diff_str = "" if diff != diff else f"{diff:.3e}"  # nan check
        print(f"{key:<70} {status:<8} {diff_str:>14}  {note}")


def run_weight_roundtrip_check(atol: float = 1e-12) -> bool:
    """Step 1.5: jax -> torch -> jax and torch -> jax -> torch."""
    print("=" * 78)
    print("Weight round-trip fidelity check (Task 1, Step 1.5)")
    print("=" * 78)

    model_jax, params_jax, cfg = build_jax_v1(JAX_V1_CONFIG)

    torch_model = jax_to_torch(cfg, params_jax, V1_TORCH_SETTINGS)
    cfg_back, params_back = torch_to_jax(torch_model, V1_TORCH_SETTINGS)
    rows_j2t2j, ok_j2t2j = _leaf_diff_table(
        flatten_params(params_jax),
        flatten_params(params_back),
        atol=atol,
    )
    _print_table("jax -> torch -> jax", rows_j2t2j)

    model_torch = build_torch_v1(V1_TORCH_SETTINGS)
    sd0 = {
        k: v.detach().cpu().numpy()
        for k, v in model_torch.state_dict().items()
    }
    cfg_t, params_t = torch_to_jax(model_torch, V1_TORCH_SETTINGS)
    model_torch_back = jax_to_torch(cfg_t, params_t, V1_TORCH_SETTINGS)
    sd1 = {
        k: v.detach().cpu().numpy()
        for k, v in model_torch_back.state_dict().items()
    }
    rows_t2j2t, ok_t2j2t = _leaf_diff_table(sd0, sd1, atol=atol)
    _print_table("torch -> jax -> torch", rows_t2j2t)

    overall_ok = ok_j2t2j and ok_t2j2t
    print("\n" + "=" * 78)
    print(
        "OVERALL: "
        + ("PASS" if overall_ok else "FAIL")
        + " (atol="
        + f"{atol:.0e}"
        + ", '*' rows are documented pre-existing exceptions, see source)"
    )
    print("=" * 78)
    return overall_ok


# ---------------------------------------------------------------------
# Task 2, Step 2.3: build both inputs from the same molecule
# ---------------------------------------------------------------------


def build_example_torch_batch(
    settings: dict = None, xyz_path=None, index: int = 0
):
    """Torch-side counterpart of ``build_example_jax_inputs``.

    Same recipe as the ``make_batch`` fixture in
    ``src/so3krates_torch/tests/conftest.py`` (ASE atoms -> Configuration
    -> AtomicData -> vendored PyG DataLoader), so this exercises the
    exact torch-side batch convention the model is tested against.
    """
    from so3krates_torch.data.atomic_data import AtomicData
    from so3krates_torch.data.utils import KeySpecification, config_from_atoms
    from so3krates_torch.tools import torch_geometric
    from so3krates_torch.tools.utils import AtomicNumberTable

    settings = settings or V1_TORCH_SETTINGS
    xyz_path = xyz_path if xyz_path is not None else _AQM_SMALL
    atoms = read(str(xyz_path), index=index)
    z_table = AtomicNumberTable([int(z) for z in range(1, 119)])
    config = config_from_atoms(atoms, key_specification=KeySpecification())
    data_loader = torch_geometric.dataloader.DataLoader(
        dataset=[
            AtomicData.from_config(
                config,
                z_table=z_table,
                cutoff=settings["r_max"],
                cutoff_lr=settings.get("r_max_lr"),
            )
        ],
        batch_size=1,
        shuffle=False,
        drop_last=False,
    )
    return next(iter(data_loader))


def check_edge_sets(batch_torch, inputs_jax) -> bool:
    """Sanity gate: torch and JAX must see the *same* neighbor lists.

    Compares short-range (``idx_i``/``idx_j``) and long-range
    (``idx_i_lr``/``idx_j_lr``) edges as sets of unordered... actually
    *directed* ``(sender, receiver)`` pairs (both pipelines are already
    symmetric, so directionality is preserved and comparable directly).

    ``jraph.dynamically_batch`` (used by ``build_example_jax_inputs``)
    always pads to at least 2 graphs, which introduces one bookkeeping
    self-loop edge (padding-node -> padding-node) per neighbor list.
    That is legitimate padding machinery, not a real edge, so it is
    filtered out here via ``node_mask`` before comparing -- confirmed
    empirically: without this filter the sets differ by exactly one
    edge, always ``(pad_idx, pad_idx)``.
    """
    bd = batch_torch.to_dict()
    torch_sr = set(
        zip(bd["edge_index"][0].tolist(), bd["edge_index"][1].tolist())
    )
    torch_lr = set(
        zip(bd["edge_index_lr"][0].tolist(), bd["edge_index_lr"][1].tolist())
    )

    node_mask = np.asarray(inputs_jax["node_mask"])
    idx_i, idx_j = np.asarray(inputs_jax["idx_i"]), np.asarray(
        inputs_jax["idx_j"]
    )
    valid = node_mask[idx_i] & node_mask[idx_j]
    jax_sr = set(zip(idx_j[valid].tolist(), idx_i[valid].tolist()))

    idx_i_lr = np.asarray(inputs_jax["idx_i_lr"])
    idx_j_lr = np.asarray(inputs_jax["idx_j_lr"])
    valid_lr = node_mask[idx_i_lr] & node_mask[idx_j_lr]
    jax_lr = set(zip(idx_j_lr[valid_lr].tolist(), idx_i_lr[valid_lr].tolist()))

    sr_ok = torch_sr == jax_sr
    lr_ok = torch_lr == jax_lr
    print(
        f"edge-set sanity gate: short-range {'PASS' if sr_ok else 'FAIL'} "
        f"(torch={len(torch_sr)}, jax={len(jax_sr)}), "
        f"long-range {'PASS' if lr_ok else 'FAIL'} "
        f"(torch={len(torch_lr)}, jax={len(jax_lr)})"
    )
    if not sr_ok:
        print(
            f"  short-range symmetric diff: "
            f"{torch_sr.symmetric_difference(jax_sr)}"
        )
    if not lr_ok:
        print(
            f"  long-range symmetric diff: "
            f"{torch_lr.symmetric_difference(jax_lr)}"
        )
    return sr_ok and lr_ok


# ---------------------------------------------------------------------
# Task 2, Step 2.1: torch stage-wise activation capture (forward hooks)
# ---------------------------------------------------------------------


def capture_torch(model: SO3LR, batch) -> dict:
    """Register forward hooks and run one forward pass to collect
    per-stage torch activations as float64 numpy arrays.

    ``euclidean_transformers[k]`` returns a ``(inv, ev)`` tuple -- split
    into ``layer{k}_inv``/``layer{k}_ev``. Every other hooked module
    returns a plain tensor.
    """
    stages: dict = {}
    handles = []

    def to_numpy(t):
        # `.detach()` and `.numpy()` alias the underlying storage when
        # the tensor is already CPU float64 (no dtype/device conversion
        # to force a copy) -- `.clone()` is required, or a later
        # in-place op elsewhere in the model (e.g. `_combine_energies`'s
        # `atomic_energies +=`) silently mutates an already-captured
        # snapshot. Confirmed empirically: without `.clone()`,
        # `nn_energy` silently became `nn_energy + zbl + electrostatic +
        # dispersion` because `atomic_energy_output_block`'s returned
        # tensor is exactly the one `_combine_energies` later adds onto
        # in place.
        return t.detach().clone().cpu().double().numpy()

    def make_hook(name):
        def hook(_module, _inputs, output):
            if isinstance(output, tuple):
                inv, ev = output
                stages[f"{name}_inv"] = to_numpy(inv)
                stages[f"{name}_ev"] = to_numpy(ev)
            else:
                stages[name] = to_numpy(output)

        return hook

    handles.append(
        model.inv_feature_embedding.register_forward_hook(make_hook("emb_inv"))
    )
    handles.append(
        model.ev_embedding.register_forward_hook(make_hook("emb_ev"))
    )
    handles.append(
        model.radial_embedding.register_forward_hook(make_hook("rbf"))
    )
    for k, layer in enumerate(model.euclidean_transformers):
        handles.append(layer.register_forward_hook(make_hook(f"layer{k}")))
    handles.append(
        model.atomic_energy_output_block.register_forward_hook(
            make_hook("nn_energy")
        )
    )
    handles.append(
        model.partial_charges_output_block.register_forward_hook(
            make_hook("charges")
        )
    )
    handles.append(
        model.hirshfeld_output_block.register_forward_hook(
            make_hook("hirshfeld")
        )
    )
    repulsion_module = (
        model.nhl_repulsion
        if getattr(model, "nhl_repulsion_bool", False)
        else model.zbl_repulsion
    )
    handles.append(repulsion_module.register_forward_hook(make_hook("zbl")))
    handles.append(
        model.electrostatic_potential.register_forward_hook(
            make_hook("electrostatic")
        )
    )
    handles.append(
        model.dispersion_potential.register_forward_hook(
            make_hook("dispersion")
        )
    )

    try:
        out = model(batch.to_dict(), compute_stress=False)
    finally:
        for h in handles:
            h.remove()

    stages["energy"] = to_numpy(out["energy"])
    stages["forces"] = to_numpy(out["forces"])
    return stages


# ---------------------------------------------------------------------
# Task 2, Step 2.2: JAX stage-wise activation capture
# ---------------------------------------------------------------------

# NOTE on the brief's suggested `sow('record', ...)` capture: the sow
# calls it describes (`chi_in`/`chi_out`/`alpha`/`alpha_r`/`alpha_s`)
# live in `mlff/nn/layer/so3krates_layer.py`'s `So3kratesLayer` -- the
# *dense/legacy* layer. The *sparse* v1 layer this script actually uses
# (`SO3kratesLayerSparse`, `mlff/nn/layer/so3krates_layer_sparse.py`)
# has no `sow` calls at all (confirmed by grepping the file). So the
# 'record' mutable collection is always empty for this model and
# `capture_intermediates=True` (the brief's own "backstop for un-sown
# stages") is not a backstop here -- it is the *only* available
# mechanism. Confirmed empirically via
# `jax.tree_util.tree_map(lambda x: x.shape, state['intermediates'])`
# that every stage below is reachable through it.
_ENERGY_HEAD_HAS_LEARNED_SCALE_SHIFT = False  # matches V1 settings


def capture_jax(model, params: dict, inputs: dict, num_layers: int) -> dict:
    """Run one ``model.apply`` with ``capture_intermediates=True`` and
    collect the same canonical stage names ``capture_torch`` uses, as
    float64 numpy arrays (squeezing the batch/pair axes where torch's
    hooks capture per-node/per-edge tensors without them).
    """
    out, state = model.apply(
        params,
        inputs,
        mutable=["intermediates"],
        capture_intermediates=True,
    )
    inter = state["intermediates"]

    def arr(x):
        return np.asarray(x, dtype=np.float64)

    stages = {}
    stages["emb_inv"] = arr(inter["feature_embeddings_0"]["__call__"][0])

    geom = inter["geometry_embeddings_0"]["__call__"][0]
    stages["emb_ev"] = arr(geom["ev"])
    stages["rbf"] = arr(geom["rbf_ij"])

    for k in range(num_layers):
        layer_out = inter[f"layers_{k}"]["__call__"][0]
        stages[f"layer{k}_inv"] = arr(layer_out["x"])
        stages[f"layer{k}_ev"] = arr(layer_out["ev"])

    obs0 = inter["observables_0"]
    # `energy_dense_final` is the raw per-atom Dense output; the v1
    # config always has energy_learn_atomic_type_scales/shifts=False
    # (see `_ENERGY_HEAD_HAS_LEARNED_SCALE_SHIFT`), so
    # scale=1/offset=0 and this *is* the final atomic-energy-head
    # output -- i.e. directly comparable to torch's
    # `atomic_energy_output_block` output. If that assumption ever
    # changes, this must be recomputed as
    # `energy_dense_final * atomic_scales + energy_offset` instead.
    assert not _ENERGY_HEAD_HAS_LEARNED_SCALE_SHIFT
    stages["nn_energy"] = arr(
        obs0["energy_dense_final"]["__call__"][0]
    ).squeeze(-1)
    stages["zbl"] = arr(obs0["zbl_repulsion"]["__call__"][0]["zbl_repulsion"])
    stages["electrostatic"] = arr(
        obs0["electrostatic_energy"]["__call__"][0]["electrostatic_energy"]
    )
    stages["dispersion"] = arr(
        obs0["dispersion_energy"]["__call__"][0]["dispersion_energy"]
    )
    # `partial_charges`/`hirshfeld_ratios` are internal helper modules
    # (not top-level observables in the JAX model, unlike torch's
    # standalone output heads) -- called once each here since
    # `electrostatic_energy_bool`/`dispersion_energy_bool` are both
    # True and neither calls the other's helper module.
    stages["charges"] = arr(
        obs0["electrostatic_energy"]["partial_charges"]["__call__"][0][
            "partial_charges"
        ]
    )
    stages["hirshfeld"] = arr(
        inter["observables_2"]["__call__"][0]["hirshfeld_ratios"]
    )
    stages["energy"] = arr(out["energy"])

    # Padding masks (underscore-prefixed: `compare()` uses these to trim
    # jraph's mandatory padding node/graph/edges before comparing against
    # torch, which never has padding; they are not themselves a "stage").
    node_mask = np.asarray(inputs["node_mask"])
    idx_i, idx_j = np.asarray(inputs["idx_i"]), np.asarray(inputs["idx_j"])
    stages["_node_mask"] = node_mask
    stages["_edge_mask"] = node_mask[idx_i] & node_mask[idx_j]
    stages["_graph_mask"] = np.asarray(inputs["graph_mask"])
    return stages


# ---------------------------------------------------------------------
# Task 3, Step 3.1: comparison driver
# ---------------------------------------------------------------------

# Pipeline order for the report; also determines how each stage's jax
# side is trimmed of jraph padding before comparison ("node"/"edge"/
# "graph" -> which `_*_mask` from `capture_jax` applies, None -> no
# padding to trim, e.g. torch-only stages skip the jax side entirely).
_STAGE_ORDER = [
    ("emb_inv", "node"),
    ("emb_ev", "node"),
    ("rbf", "edge"),
    ("nn_energy", "node"),
    ("charges", "node"),
    ("hirshfeld", "node"),
    ("zbl", "node"),
    ("electrostatic", "node"),
    ("dispersion", "node"),
    ("energy", "graph"),
    ("forces", None),
]


def _stage_sort_key(name: str) -> tuple:
    for i, (stage, _kind) in enumerate(_STAGE_ORDER):
        if name == stage:
            return (i, 0)
        if name.startswith("layer") and name.endswith(f"_{stage}"):
            # layer{k}_inv / layer{k}_ev: sort by layer index, keep
            # inv before ev within a layer.
            k = int(name[len("layer") :].split("_", 1)[0])
            return (2, k, 0 if name.endswith("_inv") else 1)
    return (len(_STAGE_ORDER), 0)


def _stage_kind(name: str) -> str:
    if name.startswith("layer") and (
        name.endswith("_inv") or name.endswith("_ev")
    ):
        return "node"
    for stage, kind in _STAGE_ORDER:
        if name == stage:
            return kind
    return None


def _squeeze_trailing_unit_dims(a: np.ndarray) -> np.ndarray:
    while a.ndim > 1 and a.shape[-1] == 1:
        a = a.squeeze(-1)
    return a


def _align_jax_to_torch(
    name: str, torch_val: np.ndarray, jax_val: np.ndarray, jax_stages: dict
) -> np.ndarray:
    """Trim jraph's padding node/edge/graph out of a jax-side stage
    array so its shape lines up with torch's (padding-free) array.
    """
    kind = _stage_kind(name)
    mask_key = {
        "node": "_node_mask",
        "edge": "_edge_mask",
        "graph": "_graph_mask",
    }.get(kind)
    if mask_key is not None and mask_key in jax_stages:
        jax_val = jax_val[jax_stages[mask_key]]
    return jax_val


# ---------------------------------------------------------------------
# Task 3, Step 3.2: equivariant-feature alignment (SH convention)
# ---------------------------------------------------------------------


def _degree_blocks(degrees) -> list:
    """(start, end) column slices for each degree ``l`` in ``degrees``,
    given the SO3krates convention of concatenating per-degree
    components of size ``2*l + 1`` in ``degrees`` order.
    """
    blocks = []
    start = 0
    for l in degrees:
        size = 2 * l + 1
        blocks.append((start, start + size))
        start += size
    return blocks


def _best_sign_permutation(a: np.ndarray, b: np.ndarray) -> tuple:
    """Search permutation x sign-flip of ``b``'s last axis that
    minimizes max-abs-diff against ``a`` (same map applied to every
    row -- a real SH-convention mismatch is a fixed relabeling, not a
    per-atom one). Exhaustive: fine for the block sizes SO3krates
    actually uses (``2l+1`` for l up to ~4, i.e. <= 9 columns).

    Returns ``(max_abs_diff, permutation, signs)`` for the best map
    found; ``permutation``/``signs`` are ``None`` if the block has 0 or
    1 columns (nothing to search).
    """
    n = a.shape[-1]
    if n <= 1:
        diff = float(np.max(np.abs(a - b))) if a.size else 0.0
        return diff, None, None

    best = (float("inf"), None, None)
    for perm in itertools.permutations(range(n)):
        b_perm = b[..., list(perm)]
        for signs in itertools.product((1.0, -1.0), repeat=n):
            diff = float(np.max(np.abs(a - b_perm * np.array(signs))))
            if diff < best[0]:
                best = (diff, perm, signs)
    return best


def align_equivariant_stage(
    name: str, a: np.ndarray, b: np.ndarray, degrees, atol: float
) -> tuple:
    """Diagnostic realignment for an `_ev`-suffixed stage.

    Tries, independently per degree block, a fixed permutation/sign
    map of ``b`` (jax) onto ``a`` (torch). Returns
    ``(max_abs_diff, status, note)`` where ``status`` is "PASS" (raw
    diff already within tolerance), "PASS*" (passes only after a
    reported remap -- a real SH-convention difference, not a bug), or
    "FAIL" (no consistent per-degree map found -- a genuine
    divergence).
    """
    raw_diff = float(np.max(np.abs(a - b))) if a.size else 0.0
    if raw_diff <= atol:
        return raw_diff, "PASS", ""

    blocks = _degree_blocks(degrees)
    if blocks and blocks[-1][1] != a.shape[-1]:
        # `degrees` doesn't account for the full width (e.g. called on
        # a non-`_ev` stage by mistake) -- fall back to the raw result
        # rather than silently mis-slicing.
        return raw_diff, "FAIL", "degrees do not match feature width"

    notes = []
    worst_after_remap = 0.0
    for l, (start, end) in zip(degrees, blocks):
        diff, perm, signs = _best_sign_permutation(
            a[..., start:end], b[..., start:end]
        )
        worst_after_remap = max(worst_after_remap, diff)
        if perm is not None and (
            list(perm) != list(range(end - start)) or any(s < 0 for s in signs)
        ):
            notes.append(f"l={l}: perm={perm}, signs={signs}")
    status = "PASS*" if worst_after_remap <= atol else "FAIL"
    note = "; ".join(notes) if notes else ""
    if status == "FAIL":
        note = (note + "; " if note else "") + (
            f"no consistent per-degree map found "
            f"(best remaining diff={worst_after_remap:.3e})"
        )
    return worst_after_remap, status, note


# ---------------------------------------------------------------------
# Task 3, Step 3.1: comparison driver
# ---------------------------------------------------------------------


def compare(
    stages_torch: dict,
    stages_jax: dict,
    atol_default: float = 1e-10,
    atol_aggregate: float = 1e-8,
    degrees=None,
) -> tuple:
    """Compare every stage shared by both capture dicts.

    Returns ``(rows, all_ok)``; each row is
    ``(stage, shape, max_abs_diff, max_rel_diff, status)``. Per-stage
    tolerance: `energy`/`forces` (aggregate, float64 summation-order
    drift) use ``atol_aggregate``; every other (per-node/per-edge)
    stage uses the tighter ``atol_default``. `_ev`-suffixed stages
    (``emb_ev``, ``layer{k}_ev``) get the Step 3.2 SH-convention
    realignment diagnostic instead of a plain diff when the raw
    comparison fails -- see ``align_equivariant_stage``. A "PASS*"
    status there is not a bug, it means a real (and now
    characterized) spherical-harmonic convention difference, not a
    numerical divergence -- see the printed remap note.
    """
    degrees = degrees if degrees is not None else V1_TORCH_SETTINGS["degrees"]
    shared = sorted(
        (set(stages_torch) & set(stages_jax))
        - {"_node_mask", "_edge_mask", "_graph_mask"},
        key=_stage_sort_key,
    )
    rows = []
    all_ok = True
    for name in shared:
        a = _squeeze_trailing_unit_dims(np.asarray(stages_torch[name]))
        b = _squeeze_trailing_unit_dims(
            _align_jax_to_torch(
                name, a, np.asarray(stages_jax[name]), stages_jax
            )
        )
        atol = atol_aggregate if name in ("energy", "forces") else atol_default

        if a.shape != b.shape:
            rows.append(
                (
                    name,
                    f"{a.shape} vs {b.shape}",
                    float("nan"),
                    float("nan"),
                    "SHAPE",
                )
            )
            all_ok = False
            continue

        if name.endswith("_ev"):
            max_abs, status, note = align_equivariant_stage(
                name, a, b, degrees, atol
            )
            passed = status != "FAIL"
            rows.append((name, str(a.shape), max_abs, note, status))
            all_ok = all_ok and passed
            continue

        abs_diff = np.abs(a - b)
        max_abs = float(np.max(abs_diff)) if abs_diff.size else 0.0
        denom = np.maximum(np.abs(a), np.abs(b))
        rel_diff = np.where(
            denom > 0, abs_diff / np.maximum(denom, 1e-300), 0.0
        )
        max_rel = float(np.max(rel_diff)) if rel_diff.size else 0.0
        passed = max_abs <= atol
        rows.append(
            (
                name,
                str(a.shape),
                max_abs,
                max_rel,
                "PASS" if passed else "FAIL",
            )
        )
        all_ok = all_ok and passed

    return rows, all_ok


def _print_stage_table(title: str, rows) -> None:
    print(f"\n--- {title} ---")
    print(
        f"{'stage':<16} {'shape':<16} {'max_abs_diff':>14} "
        f"{'max_rel_diff / SH remap note':<14}  status"
    )
    for name, shape, max_abs, fourth, status in rows:
        abs_str = "" if max_abs != max_abs else f"{max_abs:.3e}"
        # `fourth` is max_rel_diff (float) for ordinary stages, or the
        # Step 3.2 SH-remap note (str) for `_ev` stages.
        if isinstance(fourth, str):
            fourth_str = fourth
        else:
            fourth_str = "" if fourth != fourth else f"{fourth:.3e}"
        print(
            f"{name:<16} {shape:<16} {abs_str:>14} {fourth_str:<14}  {status}"
        )


# ---------------------------------------------------------------------
# Task 4: compiled-torch parity check (energy/forces only)
# ---------------------------------------------------------------------

# This intentionally does *not* reuse capture_torch's hook-based
# per-stage capture -- Inductor fuses/removes module boundaries under
# torch.compile, so hooks are not a reliable capture point through a
# compiled graph. Only the top-level (energy, forces) output dict is
# compared, mirroring `TestSO3LRCompile` in
# `src/so3krates_torch/tests/test_compile.py`, which already validates
# this exact tolerance for the full ZBL+electrostatics+dispersion path.
_COMPILE_ATOL = 1e-8
_COMPILE_RTOL = 1e-6


def _run_with_grad(model, batch):
    batch = batch.clone()
    batch.positions.requires_grad_(True)
    return model(batch.to_dict(), compute_stress=False)


def _energy_forces_close(out_a: dict, out_b: dict) -> bool:
    return torch.allclose(
        out_a["energy"],
        out_b["energy"],
        atol=_COMPILE_ATOL,
        rtol=_COMPILE_RTOL,
    ) and torch.allclose(
        out_a["forces"],
        out_b["forces"],
        atol=_COMPILE_ATOL,
        rtol=_COMPILE_RTOL,
    )


def _try_compiled_run(model, batch, fullgraph: bool, out_eager: dict) -> tuple:
    """Returns ``(status, note)``. ``status`` is one of PASS/FAIL/CRASHED.

    CRASHED (distinct from FAIL, a numeric mismatch) means
    ``torch.compile``/Inductor itself raised while compiling or
    running -- e.g. the documented CPU Inductor crash on the
    ZBL/electrostatics/dispersion scatter-sums. That crash is normally
    only seen with ``fullgraph=True`` on CPU (see
    `TestSO3LRCompile`/`calculator/so3.py`'s ``compile=True`` guard),
    but empirically also reproduces with ``fullgraph=False`` for this
    script's deliberately tiny V1_TORCH_SETTINGS -- worth surfacing
    plainly rather than silently swallowing or crashing the whole run.
    """
    try:
        compiled = torch.compile(model, dynamic=True, fullgraph=fullgraph)
        out_compiled = _run_with_grad(compiled, batch)
    except Exception as exc:  # noqa: BLE001 -- Inductor raises many types
        return "CRASHED", f"{type(exc).__name__}: {exc}".splitlines()[0]
    if _energy_forces_close(out_eager, out_compiled):
        return "PASS", ""
    return "FAIL", ""


def compiled_parity(model_torch: SO3LR, batch) -> dict:
    """Eager-vs-``torch.compile`` energy/forces parity for one model.

    Returns ``{"fullgraph=False": (status, note), "fullgraph=True":
    (status, note)}`` with ``status`` in PASS/FAIL/CRASHED/SKIPPED.
    ``fullgraph=True`` is SKIPPED (not attempted) when CUDA is
    unavailable -- this v1 config always has ZBL/electrostatics/
    dispersion enabled, and that combination is only confirmed working
    with ``fullgraph=True`` on CUDA (see `calculator/so3.py`'s
    ``compile=True`` guard and
    `TestSO3LRCompile.test_fullgraph_true_parity_gpu`'s skip reason).
    """
    results = {}

    model_cpu = model_torch.to("cpu").eval()
    batch_cpu = batch.to("cpu")
    out_eager = _run_with_grad(model_cpu, batch_cpu)
    results["fullgraph=False"] = _try_compiled_run(
        model_cpu, batch_cpu, fullgraph=False, out_eager=out_eager
    )

    if torch.cuda.is_available():
        model_cuda = model_torch.to("cuda").eval()
        batch_cuda = batch.to("cuda")
        out_eager_cuda = _run_with_grad(model_cuda, batch_cuda)
        results["fullgraph=True"] = _try_compiled_run(
            model_cuda, batch_cuda, fullgraph=True, out_eager=out_eager_cuda
        )
        model_torch.to("cpu")
    else:
        results["fullgraph=True"] = ("SKIPPED", "CUDA not available")

    return results


def _print_compile_results(title: str, results: dict) -> bool:
    print(f"\n--- {title} ---")
    ok = True
    for key, (status, note) in results.items():
        print(f"  {key:<14} {status:<8} {note}")
        if status not in ("PASS", "SKIPPED"):
            ok = False
    return ok


def _pass_stage_check(
    label: str,
    model_torch: SO3LR,
    model_jax,
    params_jax,
    inputs_jax,
    batch_torch,
    num_layers: int,
    atol: float,
    run_compile: bool,
) -> bool:
    stages_torch = capture_torch(model_torch, batch_torch)
    stages_jax = capture_jax(
        model_jax, params_jax, inputs_jax, num_layers=num_layers
    )
    rows, ok = compare(stages_torch, stages_jax, atol_default=atol)
    _print_stage_table(label, rows)

    if run_compile:
        compile_results = compiled_parity(model_torch, batch_torch)
        compile_ok = _print_compile_results(
            f"{label} -- compiled-vs-eager (Task 4)", compile_results
        )
        ok = ok and compile_ok
    return ok


def main() -> bool:
    parser = argparse.ArgumentParser(
        description="V1 stage-wise JAX<->Torch numerical parity check."
    )
    parser.add_argument("--xyz", default=str(_AQM_SMALL))
    parser.add_argument("--index", type=int, default=0)
    parser.add_argument(
        "--num-layers", type=int, default=V1_TORCH_SETTINGS["num_layers"]
    )
    parser.add_argument(
        "--num-features", type=int, default=V1_TORCH_SETTINGS["num_features"]
    )
    parser.add_argument("--atol", type=float, default=1e-10)
    parser.add_argument(
        "--compile",
        action="store_true",
        help="Also run Task 4 (compiled-vs-eager energy/forces parity).",
    )
    args = parser.parse_args()

    settings = dict(V1_TORCH_SETTINGS)
    settings["num_layers"] = args.num_layers
    settings["num_features"] = args.num_features
    _assert_v1_only_settings(settings)
    cfg = build_jax_v1_config_from_settings(settings)

    ok_roundtrip = run_weight_roundtrip_check(atol=1e-12)

    model_jax, params_jax, cfg = build_jax_v1(cfg)
    inputs_jax = build_example_jax_inputs(
        cfg, xyz_path=args.xyz, index=args.index
    )
    batch_torch = build_example_torch_batch(
        settings, xyz_path=args.xyz, index=args.index
    )

    print("\n" + "=" * 78)
    print("Stage-wise activation parity (Task 2/3/4)")
    print("=" * 78)
    edge_sets_ok = check_edge_sets(batch_torch, inputs_jax)

    # Pass A: weights originate on the JAX side, converted to torch.
    model_torch_a = jax_to_torch(cfg, params_jax, settings)
    ok_a = _pass_stage_check(
        "Pass A: JAX weights -> torch",
        model_torch_a,
        model_jax,
        params_jax,
        inputs_jax,
        batch_torch,
        settings["num_layers"],
        args.atol,
        args.compile,
    )

    # Pass B: weights originate on the torch side, converted to JAX.
    model_torch_b = build_torch_v1(settings)
    cfg_b, params_jax_b = torch_to_jax(model_torch_b, settings)
    ok_b = _pass_stage_check(
        "Pass B: torch weights -> JAX",
        model_torch_b,
        model_jax,
        params_jax_b,
        inputs_jax,
        batch_torch,
        settings["num_layers"],
        args.atol,
        args.compile,
    )

    overall_ok = ok_roundtrip and edge_sets_ok and ok_a and ok_b
    print("\n" + "=" * 78)
    print("FINAL VERDICT: " + ("PASS" if overall_ok else "FAIL"))
    print("=" * 78)
    return overall_ok


if __name__ == "__main__":
    raise SystemExit(0 if main() else 1)
