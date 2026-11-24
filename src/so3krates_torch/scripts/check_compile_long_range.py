"""
Diagnose torch.compile behavior for SO3LR's long-range potentials
(ZBL repulsion, electrostatics, dispersion) on the current device.

Context: on CPU, `torch.compile(model, dynamic=True, fullgraph=True)`
fails immediately on an unsupported `Tensor.requires_grad_()` call inside
`prepare_graph` (a Dynamo tracing limitation, backend-independent).
With `fullgraph=False` that call is tolerated as a graph break, but
compilation then crashes later with an `AssertionError` deep inside
Inductor's C++ (CPU) vectorized codegen while generating code for the
long-range scatter-sum reductions (ZBL / electrostatics / dispersion each
scatter_add per-edge energies onto per-node buffers with dynamic shapes).

That crash traceback lives entirely under `torch/_inductor/codegen/cpp.py`,
which is the CPU-only vectorized backend. The CUDA backend uses a
different codegen path (Triton), which has mature native support for
atomic-add/scatter-reduce patterns. This script checks whether the crash
is CPU-backend-specific by running the same forward pass on whatever
device is available (pass --device cuda on a GPU machine).

Usage examples:
    python check_compile_long_range.py
    python check_compile_long_range.py --device cuda
    python check_compile_long_range.py --device cuda --compute-stress
    python check_compile_long_range.py --molecule CH3CH2OH --dtype float32
"""

import argparse
import logging
import sys
import traceback

import torch
from ase.build import molecule

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def build_model(dtype: torch.dtype, seed: int = 42):
    from so3krates_torch.modules.models import SO3LR

    config = {
        "r_max": 5.0,
        "r_max_lr": 10.0,
        "num_radial_basis_fn": 8,
        "degrees": [1, 2],
        "num_features": 16,
        "num_heads": 2,
        "num_layers": 1,
        "num_elements": 118,
        "avg_num_neighbors": 10.0,
        "final_mlp_layers": 1,
        "dtype": dtype,
        "seed": seed,
        "zbl_repulsion_bool": True,
        "electrostatic_energy_bool": True,
        "dispersion_energy_bool": True,
        "dispersion_energy_cutoff_lr_damping": 2.0,
    }
    model = SO3LR(**config)
    model.eval()
    return model


def build_batch(atoms, r_max: float, r_max_lr: float, dtype, device):
    from so3krates_torch.data.utils import KeySpecification, config_from_atoms
    from so3krates_torch.data.atomic_data import AtomicData
    from so3krates_torch.tools import torch_geometric
    from so3krates_torch.tools.utils import AtomicNumberTable

    torch.set_default_dtype(dtype)
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
    return next(iter(data_loader)).to(device)


def run_forward(model, batch, compute_stress: bool):
    out = model(batch.to_dict(), compute_stress=compute_stress)
    return out


def try_variant(
    label: str,
    model,
    batch,
    compute_stress: bool,
    fullgraph: bool,
    dynamic: bool,
):
    logger.info("--- %s ---", label)
    torch._dynamo.reset()
    try:
        compiled = torch.compile(model, dynamic=dynamic, fullgraph=fullgraph)
        out = run_forward(compiled, batch, compute_stress)
        energy = out["energy"].detach()
        forces = out["forces"].detach()
        logger.info(
            "%s: OK  (energy=%s, |forces| mean=%.6g)",
            label,
            energy.tolist(),
            forces.abs().mean().item(),
        )
        return {"label": label, "ok": True, "energy": energy, "forces": forces}
    except Exception as exc:  # noqa: BLE001 - diagnostic script
        tb = traceback.format_exc()
        first_line = str(exc).strip().splitlines()[0] if str(exc) else ""
        logger.error(
            "%s: FAILED  (%s: %s)", label, type(exc).__name__, first_line
        )
        return {
            "label": label,
            "ok": False,
            "exc_type": type(exc).__name__,
            "message": first_line,
            "traceback": tb,
        }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to run on (default: cuda if available, else cpu).",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float32",
    )
    parser.add_argument(
        "--molecule",
        type=str,
        default="H2O",
        help="ASE molecule name from ase.build.molecule (default: H2O).",
    )
    parser.add_argument("--r-max", type=float, default=5.0)
    parser.add_argument("--r-max-lr", type=float, default=10.0)
    parser.add_argument(
        "--compute-stress",
        action="store_true",
        default=False,
        help="Also request stress, exercising the displacement "
        "requires_grad_ path in get_symmetric_displacement.",
    )
    parser.add_argument(
        "--dump-traceback",
        type=str,
        default=None,
        help="If set, write full tracebacks of any failures to this file.",
    )
    args = parser.parse_args()

    device = torch.device(args.device)
    dtype = torch.float32 if args.dtype == "float32" else torch.float64

    logger.info("torch version: %s", torch.__version__)
    logger.info("device: %s", device)
    if device.type == "cuda":
        logger.info("GPU: %s", torch.cuda.get_device_name(device))
    logger.info("dtype: %s", dtype)

    model = build_model(dtype=dtype).to(device)
    atoms = molecule(args.molecule)
    batch = build_batch(
        atoms,
        r_max=args.r_max,
        r_max_lr=args.r_max_lr,
        dtype=dtype,
        device=device,
    )

    results = []

    # Eager baseline — sanity check before touching torch.compile at all.
    logger.info("--- eager (baseline) ---")
    try:
        batch_eager = batch.clone()
        out_eager = run_forward(model, batch_eager, args.compute_stress)
        energy_eager = out_eager["energy"].detach()
        forces_eager = out_eager["forces"].detach()
        logger.info(
            "eager: OK  (energy=%s, |forces| mean=%.6g)",
            energy_eager.tolist(),
            forces_eager.abs().mean().item(),
        )
    except Exception:
        logger.error("Eager baseline failed — fix this before compiling.")
        traceback.print_exc()
        sys.exit(1)

    # fullgraph=False: known to tolerate the requires_grad_ break as a
    # graph break, but this is where the CPU Inductor scatter crash was
    # observed. This is the primary test for the CPU-vs-GPU hypothesis.
    results.append(
        try_variant(
            "compile(dynamic=True, fullgraph=False)",
            model,
            batch.clone(),
            args.compute_stress,
            fullgraph=False,
            dynamic=True,
        )
    )

    # fullgraph=True: expected to fail on requires_grad_ regardless of
    # device/backend (a Dynamo tracing limitation, not an Inductor one).
    results.append(
        try_variant(
            "compile(dynamic=True, fullgraph=True)",
            model,
            batch.clone(),
            args.compute_stress,
            fullgraph=True,
            dynamic=True,
        )
    )

    print("\n===== SUMMARY =====")
    print(
        f"device: {device}  dtype: {dtype}  compute_stress: {args.compute_stress}"
    )
    for r in results:
        status = (
            "OK" if r["ok"] else f"FAILED ({r['exc_type']}: {r['message']})"
        )
        print(f"  {r['label']}: {status}")

    if args.dump_traceback:
        with open(args.dump_traceback, "w") as f:
            for r in results:
                if not r["ok"]:
                    f.write(f"===== {r['label']} =====\n")
                    f.write(r["traceback"])
                    f.write("\n\n")
        print(
            f"\nFull tracebacks of failures written to {args.dump_traceback}"
        )

    any_scatter_crash = any(
        (not r["ok"]) and "AssertionError" in r.get("exc_type", "")
        for r in results
    )
    if device.type == "cuda" and any_scatter_crash:
        print(
            "\nThe Inductor AssertionError reproduced on GPU too — it is "
            "NOT CPU-specific."
        )
    elif device.type == "cuda":
        print(
            "\nNo Inductor AssertionError on GPU — consistent with the "
            "hypothesis that the CPU vectorized codegen crash is "
            "backend-specific and won't affect GPU inference."
        )


if __name__ == "__main__":
    main()
