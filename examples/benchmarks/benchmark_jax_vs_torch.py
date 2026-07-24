"""Benchmark JAX (so3lr_dev, after JIT) vs PyTorch (eager and
torch.compile) inference speed for a chosen SO3LR model, across
increasing structure sizes (e.g. water boxes up to ~10k atoms).

Only the forward (energy + forces) pass is timed -- graph/neighbor-list
construction happens once per structure, outside every timed loop.

Usage
-----
  python examples/benchmarks/benchmark_jax_vs_torch.py \\
      --model v2-s \\
      --data_dir path/to/water_boxes \\
      --device cuda \\
      --output results/benchmark_v2-s_cuda.npz
"""

import argparse
import copy
import glob
import json
import os
import pickle
import time
from typing import Dict, List, Optional, Tuple

import numpy as np
import torch
from ase.io import read as ase_read

from so3krates_torch.calculator.so3 import load_pretrained_so3lr
from so3krates_torch.tools.model_parity import _build_torch_batch
from so3krates_torch.tools.model_parity import (
    _build_jax_inputs,
    _detect_num_theory_levels,
    _match_theory_levels,
    _reshape_legacy_energy_head_params,
    _theory_level_index,
    _with_final_bias_bool,
)

_CHECKPOINT_DIRS = {
    "v1": "so3lr",
    "v2-s": "so3lr-s",
    "v2-m": "so3lr-m",
    "v2-l": "so3lr-l",
}

BACKENDS = ("torch-eager", "torch-compiled", "jax-jit")


def _discover_structures(data_dir: str) -> List[Tuple[str, int]]:
    """Every ASE-readable file directly under ``data_dir``, as
    ``(path, n_atoms)`` pairs sorted by ascending atom count. Files ASE
    can't read are skipped with a printed warning rather than aborting.
    """
    paths = sorted(
        p for p in glob.glob(os.path.join(data_dir, "*")) if os.path.isfile(p)
    )
    structures = []
    for path in paths:
        try:
            atoms = ase_read(path, index=0)
        except Exception as exc:
            print(f"WARNING: skipping unreadable file {path}: {exc}")
            continue
        structures.append((path, len(atoms)))
    return sorted(structures, key=lambda item: item[1])


def _summarize(times_s: np.ndarray, n_atoms: int) -> dict:
    """Derived timing stats for one (structure, backend) combination."""
    mean_s = float(np.mean(times_s))
    return {
        "times_s": times_s,
        "mean_s": mean_s,
        "std_s": float(np.std(times_s)),
        "min_s": float(np.min(times_s)),
        "max_s": float(np.max(times_s)),
        "atoms_per_s": n_atoms / mean_s if mean_s > 0 else 0.0,
        "ms_per_atom_step": (
            mean_s * 1000.0 / n_atoms if n_atoms > 0 else 0.0
        ),
    }


def _make_record(
    path: str,
    n_atoms: int,
    backend: str,
    status: str,
    times: Optional[np.ndarray] = None,
    error: Optional[str] = None,
) -> dict:
    record = {
        "n_atoms": n_atoms,
        "structure_file": os.path.basename(path),
        "backend": backend,
        "status": status,
    }
    if status == "ok":
        record.update(_summarize(times, n_atoms))
    else:
        record["error"] = error
    return record


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark JAX (so3lr_dev, JIT) vs torch (eager / "
            "torch.compile) inference speed for a chosen SO3LR model."
        )
    )
    parser.add_argument(
        "--model", required=True, choices=sorted(_CHECKPOINT_DIRS)
    )
    parser.add_argument("--data_dir", required=True)
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda"])
    parser.add_argument(
        "--dtype", default="float64", choices=["float32", "float64"]
    )
    parser.add_argument("--r_max_lr", type=float, default=12.0)
    parser.add_argument(
        "--dispersion_energy_cutoff_lr_damping", type=float, default=2.0
    )
    parser.add_argument("--use_pme", action="store_true")
    parser.add_argument("--pme_smearing", type=float, default=0.5)
    parser.add_argument("--pme_mesh_spacing", type=float, default=0.25)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--repeats", type=int, default=50)
    parser.add_argument("--output", default=None)
    return parser


def _build_torch_model(args: argparse.Namespace) -> torch.nn.Module:
    overrides = dict(
        r_max_lr=args.r_max_lr,
        dispersion_energy_cutoff_lr_damping=(
            args.dispersion_energy_cutoff_lr_damping
        ),
    )
    if args.use_pme:
        overrides.update(
            use_pme=True,
            pme_smearing=args.pme_smearing,
            pme_mesh_spacing=args.pme_mesh_spacing,
        )
    model = load_pretrained_so3lr(args.model, device=args.device, **overrides)
    dtype = getattr(torch, args.dtype)
    return model.to(dtype).to(args.device).eval()


def _time_torch_eager(
    model: torch.nn.Module, batch, warmup: int, repeats: int, device: str
) -> np.ndarray:
    is_cuda = str(device).startswith("cuda")
    batch = batch.clone()
    batch.positions.requires_grad_(True)
    batch_dict = batch.to_dict()

    def _call():
        return model(batch_dict, compute_stress=False)

    for _ in range(warmup):
        _call()
    if is_cuda:
        torch.cuda.synchronize()

    times = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        if is_cuda:
            torch.cuda.synchronize()
        t0 = time.perf_counter()
        _call()
        if is_cuda:
            torch.cuda.synchronize()
        times[i] = time.perf_counter() - t0
    return times


def _time_torch_compiled(
    model: torch.nn.Module, batch, warmup: int, repeats: int, device: str
) -> np.ndarray:
    compiled = torch.compile(model, dynamic=True, fullgraph=True)
    return _time_torch_eager(compiled, batch, warmup, repeats, device)


def _configure_jax_environment(device: str) -> None:
    """Must run before any ``jax`` import (this script's own JAX imports
    are all lazy/local for exactly this reason)."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    if device == "cpu":
        os.environ["JAX_PLATFORMS"] = "cpu"


def _configure_jax_precision(dtype: str) -> None:
    import jax

    jax.config.update("jax_enable_x64", dtype == "float64")


def _load_jax_checkpoint(
    model_key: str,
    r_max_lr: float,
    damping: float,
    use_pme: bool,
    pme_smearing: float,
    pme_mesh_spacing: float,
):
    """Mutates the raw ``hyperparameters.json`` dict *before* wrapping it
    in a ``ConfigDict`` (not after) -- matching
    ``tests/test_v1_parity.py``'s/``tests/test_v2_parity.py``'s
    ``_v1_hyperparams``/``_pme_hyperparams`` convention exactly, so a key
    like ``kspace_electrostatics`` that may not exist at all in a
    checkpoint shipped with PME off can still be set without relying on
    ``ConfigDict``'s new-attribute/type-locking semantics.
    """
    from importlib.resources import files

    from ml_collections import config_dict

    checkpoint_dir = files("so3lr") / "models" / _CHECKPOINT_DIRS[model_key]
    with open(str(checkpoint_dir / "params.pkl"), "rb") as f:
        flax_params = pickle.load(f)
    with open(str(checkpoint_dir / "hyperparameters.json")) as f:
        raw_hyperparams = json.load(f)

    raw_hyperparams["model"]["cutoff_lr"] = r_max_lr
    raw_hyperparams["model"]["dispersion_energy_cutoff_lr_damping"] = damping
    if use_pme:
        raw_hyperparams["model"]["kspace_electrostatics"] = "pme"
        raw_hyperparams["model"]["kspace_smearing"] = pme_smearing
        raw_hyperparams["model"]["kspace_spacing"] = pme_mesh_spacing

    cfg = config_dict.ConfigDict(raw_hyperparams)
    return cfg, flax_params


def _cast_pytree_floats(tree, dtype: np.dtype):
    import jax
    import jax.numpy as jnp

    def _cast(x):
        if hasattr(x, "dtype") and jnp.issubdtype(x.dtype, jnp.floating):
            return x.astype(dtype)
        return x

    return jax.tree_util.tree_map(_cast, tree)


def _time_jax_jit(
    cfg,
    flax_params: dict,
    jax_inputs: dict,
    model_factory,
    warmup: int,
    repeats: int,
) -> np.ndarray:
    import jax

    model = model_factory(cfg)
    graph_mask = jax_inputs["graph_mask"]

    def energy_fn(positions):
        inputs_r = dict(jax_inputs, positions=positions)
        out = model.apply(flax_params, inputs_r)
        return (out["energy"] * graph_mask).sum()

    value_and_grad_fn = jax.jit(jax.value_and_grad(energy_fn))
    positions = jax_inputs["positions"]

    for _ in range(warmup):
        energy, grad = value_and_grad_fn(positions)
        energy.block_until_ready()
        grad.block_until_ready()

    times = np.empty(repeats, dtype=np.float64)
    for i in range(repeats):
        t0 = time.perf_counter()
        energy, grad = value_and_grad_fn(positions)
        energy.block_until_ready()
        grad.block_until_ready()
        times[i] = time.perf_counter() - t0
    return times


def run_benchmark(args: argparse.Namespace) -> dict:
    _configure_jax_environment(args.device)
    _configure_jax_precision(args.dtype)

    from so3lr.mlff.config import make_so3krates_sparse_from_config

    torch_model = _build_torch_model(args)
    cfg, flax_params = _load_jax_checkpoint(
        args.model,
        args.r_max_lr,
        args.dispersion_energy_cutoff_lr_damping,
        args.use_pme,
        args.pme_smearing,
        args.pme_mesh_spacing,
    )
    cfg = _with_final_bias_bool(cfg, torch_model)
    num_theory_levels = _detect_num_theory_levels(flax_params)
    flax_params = _reshape_legacy_energy_head_params(
        flax_params, num_theory_levels
    )
    dtype_np = np.dtype(args.dtype)
    flax_params = _cast_pytree_floats(flax_params, dtype_np)
    torch_dtype = getattr(torch, args.dtype)

    structures = _discover_structures(args.data_dir)
    if not structures:
        raise ValueError(f"No readable structures found in {args.data_dir}")

    records = []
    for path, n_atoms in structures:
        print(f"\n=== {os.path.basename(path)} ({n_atoms} atoms) ===")

        batch = None
        try:
            batch = _build_torch_batch(
                r_max=torch_model.r_max,
                r_max_lr=torch_model.r_max_lr,
                structure_path=path,
                index=0,
                dtype=torch_dtype,
                theory_level=_theory_level_index(num_theory_levels),
            ).to(args.device)
        except Exception as exc:
            print(f"WARNING: failed to build torch batch for {path}: {exc}")
            records.append(
                _make_record(
                    path, n_atoms, "torch-eager", "failed", error=str(exc)
                )
            )
            if args.device != "cpu":
                records.append(
                    _make_record(
                        path,
                        n_atoms,
                        "torch-compiled",
                        "failed",
                        error=str(exc),
                    )
                )

        if batch is not None:
            try:
                eager_model = copy.deepcopy(torch_model)
                times = _time_torch_eager(
                    eager_model,
                    batch,
                    args.warmup,
                    args.repeats,
                    args.device,
                )
                records.append(
                    _make_record(path, n_atoms, "torch-eager", "ok", times)
                )
            except Exception as exc:
                records.append(
                    _make_record(
                        path,
                        n_atoms,
                        "torch-eager",
                        "failed",
                        error=str(exc),
                    )
                )

            if args.device == "cpu":
                print(
                    "WARNING: skipping torch-compiled on CPU -- "
                    "torch.compile is unsupported on CPU for long-range "
                    "SO3LR models (Inductor crashes on the ZBL/"
                    "electrostatics/dispersion scatter-reductions). Run "
                    "on --device cuda to benchmark torch-compiled."
                )
            else:
                try:
                    compiled_model = copy.deepcopy(torch_model)
                    times = _time_torch_compiled(
                        compiled_model,
                        batch,
                        args.warmup,
                        args.repeats,
                        args.device,
                    )
                    records.append(
                        _make_record(
                            path, n_atoms, "torch-compiled", "ok", times
                        )
                    )
                except Exception as exc:
                    records.append(
                        _make_record(
                            path,
                            n_atoms,
                            "torch-compiled",
                            "failed",
                            error=str(exc),
                        )
                    )

        try:
            jax_inputs = _match_theory_levels(
                _build_jax_inputs(cfg, path, 0), num_theory_levels
            )
            jax_inputs = _cast_pytree_floats(jax_inputs, dtype_np)
            times = _time_jax_jit(
                cfg,
                flax_params,
                jax_inputs,
                make_so3krates_sparse_from_config,
                args.warmup,
                args.repeats,
            )
            records.append(_make_record(path, n_atoms, "jax-jit", "ok", times))
        except Exception as exc:
            records.append(
                _make_record(
                    path, n_atoms, "jax-jit", "failed", error=str(exc)
                )
            )

    return {"config": vars(args), "records": records}


def print_table(results: dict) -> None:
    config = results["config"]
    pme_state = "on" if config["use_pme"] else "off"
    print(
        f"\nModel: {config['model']}   Device: {config['device']}   "
        f"dtype: {config['dtype']}   PME: {pme_state}"
    )
    header = (
        f"{'n_atoms':>8} {'backend':<16} {'mean(ms)':>10} {'std(ms)':>9} "
        f"{'atoms/s':>10} {'ms/(atom*step)':>15} {'speedup':>9}"
    )
    sep = "=" * len(header)
    print(sep)
    print(header)
    print("-" * len(header))

    by_size: Dict[int, Dict[str, dict]] = {}
    for r in results["records"]:
        by_size.setdefault(r["n_atoms"], {})[r["backend"]] = r

    for n_atoms in sorted(by_size):
        baseline = by_size[n_atoms].get("torch-eager")
        baseline_mean = (
            baseline["mean_s"]
            if baseline and baseline["status"] == "ok"
            else None
        )
        for backend in BACKENDS:
            r = by_size[n_atoms].get(backend)
            if r is None:
                continue
            if r["status"] != "ok":
                print(
                    f"{n_atoms:>8} {backend:<16} "
                    f"{'FAILED':>10} {r['error']}"
                )
                continue
            speedup = (
                f"{baseline_mean / r['mean_s']:.2f}x"
                if baseline_mean and r["mean_s"] > 0
                else "n/a"
            )
            print(
                f"{n_atoms:>8} {backend:<16} {r['mean_s'] * 1000:>10.2f} "
                f"{r['std_s'] * 1000:>9.2f} {r['atoms_per_s']:>10.0f} "
                f"{r['ms_per_atom_step']:>15.4f} {speedup:>9}"
            )
    print(sep)


def save_results(results: dict, output_path: str) -> None:
    np.savez(output_path, results=results)
    print(f"\nSaved results to {output_path}")


def main() -> None:
    parser = build_arg_parser()
    args = parser.parse_args()
    if args.output is None:
        args.output = f"benchmark_{args.model}_{args.device}.npz"

    results = run_benchmark(args)
    print_table(results)
    save_results(results, args.output)


if __name__ == "__main__":
    main()
