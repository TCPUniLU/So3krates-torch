import numpy as np
from ase.md.langevin import Langevin
from ase import units
from ase.io import read
from so3krates_torch.calculator.so3 import (
    TorchkratesCalculator,
    SO3LRCalculator,
)
import argparse
from ase.md import MDLogger
from ase.md.velocitydistribution import MaxwellBoltzmannDistribution
from ase.io.trajectory import TrajectoryWriter
import os


def failed_geo(bond_matrix, mol, get_geo=False, threshold=2.0):
    dist = bond_matrix * mol.get_all_distances(mic=True)
    dist_fail = dist > threshold
    return np.any(dist_fail)


def get_bond_matrix(path):
    m = np.load(path)
    np.fill_diagonal(m, 0)
    return m


argparser = argparse.ArgumentParser()
argparser.add_argument("--start_path", type=str, default="./geometry.in")
argparser.add_argument("--model_path", type=str, default=None)
argparser.add_argument("--r_max_lr", type=float, default=12.0)
argparser.add_argument("--compute_stress", action="store_true", default=False)
argparser.add_argument(
    "--dispersion_energy_cutoff_lr_damping", type=float, default=2.0
)
argparser.add_argument("--device", type=str, default="cuda")
argparser.add_argument("--interval", type=int, default=100)
argparser.add_argument("--steps", type=int, default=1e6)
argparser.add_argument("--dt", type=float, default=1)
argparser.add_argument("--T", type=float, default=300)
argparser.add_argument("--friction", type=float, default=0.01)
argparser.add_argument("--bond_matrix", type=str, default=None)
argparser.add_argument("--save_path", type=str, default=None)
argparser.add_argument("--save_path_traj", type=str, default=None)
argparser.add_argument("--logger_off", action="store_false", default=True)
argparser.add_argument("--so3lr", action="store_true", default=False)
argparser.add_argument(
    "--so3lr_model",
    type=str,
    default="v2-s",
    choices=["v1", "v2-s", "v2-m", "v2-l"],
    help="Which bundled pretrained SO3LR checkpoint to use with --so3lr.",
)
argparser.add_argument(
    "--use_pme",
    action="store_true",
    default=False,
    help="Use PME (k-space) electrostatics instead of the default "
    "real-space Coulomb sum. Requires torch-pme and a periodic "
    "--start_path structure (a cell). Only applies with --so3lr.",
)
argparser.add_argument("--dtype", type=str, default="float32")
args = argparser.parse_args()

if args.save_path is not None:
    if not os.path.exists(os.path.dirname(args.save_path)):
        os.makedirs(os.path.dirname(args.save_path))

if args.model_path is None and not args.so3lr:
    raise ValueError("Must provide either model_path or use SO3LR")


if os.path.exists(args.start_path):
    mol = read(args.start_path)
else:
    # No --start_path given (or it doesn't exist): fall back to a small,
    # bundled, self-contained periodic structure so this example can be
    # run out of the box, e.g. to smoke-test --use_pme (PME requires a
    # periodic cell) across all 4 --so3lr_model checkpoints.
    from ase.build import bulk

    print(
        f"--start_path {args.start_path!r} not found; falling back to a "
        "small bundled NaCl bulk structure (2x2x2 rocksalt supercell)."
    )
    mol = bulk("NaCl", "rocksalt", a=5.64) * (2, 2, 2)

if args.so3lr:
    print(f"Using SO3LR ({args.so3lr_model}), use_pme={args.use_pme}")
    mol.calc = SO3LRCalculator(
        model=args.so3lr_model,
        r_max_lr=args.r_max_lr,
        compute_stress=args.compute_stress,
        dispersion_energy_cutoff_lr_damping=args.dispersion_energy_cutoff_lr_damping,
        device=args.device,
        default_dtype=args.dtype,
        use_pme=args.use_pme,
    )

else:
    print(f"Using model: {args.model_path}")
    mol.calc = TorchkratesCalculator(
        model_paths=args.model_path,
        r_max_lr=args.r_max_lr,
        compute_stress=args.compute_stress,
        dispersion_energy_cutoff_lr_damping=args.dispersion_energy_cutoff_lr_damping,
        device=args.device,
        default_dtype=args.dtype,
    )

if args.save_path_traj is not None:
    writer = TrajectoryWriter(args.save_path_traj, atoms=mol, mode="a")

print(f"Running at T={args.T} K")

dyn = Langevin(
    mol,
    timestep=args.dt * units.fs,
    temperature_K=args.T,  # temperature in K
    friction=args.friction / units.fs,
    logfile="./md.log",
    append_trajectory=True,
)

if args.bond_matrix is not None:
    bond_matrix = get_bond_matrix(args.bond_matrix)
else:
    bond_matrix = None

if args.save_path is not None:

    def write_xzy():
        dyn.atoms.write(args.save_path, append=True)


md_logger = MDLogger(
    dyn=dyn, atoms=mol, logfile="md.log", header=True, peratom=True, mode="a"
)

MaxwellBoltzmannDistribution(atoms=mol, temperature_K=args.T)


for step in range(args.steps):
    dyn.step()
    dyn.nsteps = step + 1
    if step % args.interval == 0 or step == args.steps - 1:
        print(
            f"Step: {step+1} / {args.steps}"
            f" -- Energy: {mol.get_potential_energy()}"
            f" -- Temperature: {mol.get_temperature()} K"
        )
        if args.save_path is not None:
            write_xzy()

        if args.save_path_traj is not None:
            writer.write()
        if not args.logger_off:
            md_logger()

        if bond_matrix is not None:
            failed = failed_geo(bond_matrix, mol)
            if failed:
                print(f"Geometry failed at step {step}!")
                break
