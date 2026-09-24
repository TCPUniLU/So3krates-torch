"""Run NVT-equilibration + NVE-production MD on a periodic system with
SO3LR, logging an unwrapped trajectory and the system dipole moment at
regular intervals (for a later, separate IR-spectrum post-processing step
based on the dipole-dipole autocorrelation function).

Mirrors the scope of a typical LAMMPS workflow:
    fix nvt ...      # equilibrate at target temperature
    fix nve ...      # production
    compute dip all dipole
    fix ave/time ... file dipole.dat

On the dipole: the model's own dipole output is a plain sum q_i * r_i over
absolute positions. The cell as a whole is neutral, but individual
molecules are not (typically |q_mol| ~ 0.03 e), so that sum changes when a
molecule is translated by a lattice vector -- a physically identical
configuration. With unwrapped coordinates those per-molecule offsets
random-walk as molecules diffuse, adding a spurious drift to the dipole
time series. This script therefore also logs a drift-free dipole, summing
each molecule's dipole about its own center of mass, which is invariant
under lattice translations. Both are written to the dipole file.
"""

import argparse
import os
import shutil
import subprocess
import tempfile
from collections import deque

import numpy as np
from ase import units
from ase.data import atomic_numbers, covalent_radii
from ase.geometry import find_mic
from ase.io import read, write
from ase.io.lammpsdata import read_lammps_data
from ase.io.trajectory import Trajectory
from ase.md.nose_hoover_chain import NoseHooverChainNVT
from ase.md.velocitydistribution import (
    MaxwellBoltzmannDistribution,
    Stationary,
)
from ase.md.verlet import VelocityVerlet
from ase.neighborlist import neighbor_list

from so3krates_torch.calculator.so3 import (
    SO3LRCalculator,
    TorchkratesCalculator,
)


def get_bond_matrix(atoms, bond_matrix_path=None, cutoff_scale=1.2):
    """Return a boolean (n_atoms, n_atoms) bond-adjacency matrix, either
    loaded from disk or auto-derived from covalent radii (MIC-aware)."""
    if bond_matrix_path is not None:
        bonds = np.load(bond_matrix_path).astype(bool)
        if bonds.shape != (len(atoms), len(atoms)):
            raise ValueError(
                f"Bond matrix has shape {bonds.shape}, expected "
                f"{(len(atoms), len(atoms))} for this structure."
            )
        np.fill_diagonal(bonds, False)
        return bonds

    numbers = atoms.get_atomic_numbers()
    radii = covalent_radii[numbers]
    max_cutoff = 2.0 * radii.max() * cutoff_scale
    i, j, d = neighbor_list("ijd", atoms, cutoff=max_cutoff)

    n = len(atoms)
    bonds = np.zeros((n, n), dtype=bool)
    threshold = cutoff_scale * (radii[i] + radii[j])
    mask = d < threshold
    bonds[i[mask], j[mask]] = True
    bonds[j[mask], i[mask]] = True
    return bonds


def find_molecules(bonds):
    """Label each atom with the index of the connected component (molecule)
    it belongs to."""
    n = len(bonds)
    mol_index = np.full(n, -1, dtype=int)
    n_molecules = 0
    for root in range(n):
        if mol_index[root] >= 0:
            continue
        mol_index[root] = n_molecules
        queue = deque([root])
        while queue:
            i = queue.popleft()
            for j in np.nonzero(bonds[i])[0]:
                if mol_index[j] < 0:
                    mol_index[j] = n_molecules
                    queue.append(j)
        n_molecules += 1
    return mol_index, n_molecules


def unwrap_molecules(atoms, bonds):
    """Rebuild a single continuous (unwrapped) coordinate branch for every
    bonded molecule in ``atoms``, using the bond graph and the minimum-image
    convention. Runs once, before MD starts; atoms with no bonds (e.g. lone
    ions) are left untouched.

    Returns True if any molecule is bonded to its own periodic image
    (a percolating/infinite network), in which case the unwrapped branch --
    and any sum q_i * r_i dipole built from it -- is arbitrary.
    """
    positions = atoms.get_positions()
    unwrapped = positions.copy()
    cell = atoms.get_cell()
    pbc = atoms.get_pbc()

    n = len(atoms)
    visited = np.zeros(n, dtype=bool)
    percolating = False
    for root in range(n):
        if visited[root]:
            continue
        visited[root] = True
        queue = deque([root])
        while queue:
            i = queue.popleft()
            for j in np.nonzero(bonds[i])[0]:
                displacement = positions[j] - unwrapped[i]
                mic_displacement, _ = find_mic(displacement, cell, pbc)
                if visited[j]:
                    closure = unwrapped[j] - unwrapped[i]
                    if not np.allclose(closure, mic_displacement, atol=1e-6):
                        percolating = True
                    continue
                unwrapped[j] = unwrapped[i] + mic_displacement
                visited[j] = True
                queue.append(j)

    atoms.set_positions(unwrapped)
    return percolating


def molecular_dipole(positions, charges, masses, mol_index, mol_mass):
    """Total dipole as the sum of per-molecule dipoles, each taken about
    that molecule's own center of mass.

    Unlike a plain sum q_i * r_i this is invariant under translating any
    molecule by a lattice vector, so it carries no diffusion-driven drift.
    """
    weighted = masses[:, None] * positions
    com = (
        np.stack(
            [np.bincount(mol_index, weights=weighted[:, k]) for k in range(3)],
            axis=1,
        )
        / mol_mass[:, None]
    )
    relative = positions - com[mol_index]
    return (charges[:, None] * relative).sum(axis=0)


def make_thermo_printer(atoms, dyn, density, log_pressure):
    def thermo():
        epot = atoms.get_potential_energy()
        ekin = atoms.get_kinetic_energy()
        temp = atoms.get_temperature()
        line = (
            f"step={dyn.nsteps:8d}  "
            f"time={dyn.nsteps * dyn.dt / units.fs:10.2f} fs  "
            f"T={temp:8.2f} K  "
            f"Etot={epot + ekin:14.6f} eV  "
            f"density={density:.4f} amu/A^3"
        )
        if log_pressure:
            # include_ideal_gas adds the kinetic term, matching LAMMPS'
            # `press` rather than reporting the virial part alone.
            stress = atoms.get_stress(voigt=False, include_ideal_gas=True)
            pressure = -np.trace(stress) / 3.0
            line += f"  P={pressure:12.6f} eV/A^3"
        print(line, flush=True)

    return thermo


def make_dipole_logger(atoms, dyn, dipole_path, mol_index):
    masses = atoms.get_masses()
    mol_mass = np.bincount(mol_index, weights=masses)

    def log_dipole():
        raw = atoms.get_dipole_moment()
        charges = atoms.calc.results["partial_charges"]
        mu = molecular_dipole(
            atoms.get_positions(), charges, masses, mol_index, mol_mass
        )
        time_fs = dyn.nsteps * dyn.dt / units.fs
        with open(dipole_path, "a") as f:
            f.write(
                f"{dyn.nsteps} {time_fs:.6f} "
                f"{mu[0]:.8f} {mu[1]:.8f} {mu[2]:.8f} "
                f"{raw[0]:.8f} {raw[1]:.8f} {raw[2]:.8f}\n"
            )

    return log_dipole


def build_calculator(args):
    if args.so3lr:
        return SO3LRCalculator(
            r_max_lr=args.r_max_lr,
            compute_stress=args.log_pressure,
            dispersion_energy_cutoff_lr_damping=(
                args.dispersion_energy_cutoff_lr_damping
            ),
            device=args.device,
            default_dtype=args.dtype,
        )
    return TorchkratesCalculator(
        model_paths=args.model_path,
        r_max_lr=args.r_max_lr,
        compute_stress=args.log_pressure,
        dispersion_energy_cutoff_lr_damping=(
            args.dispersion_energy_cutoff_lr_damping
        ),
        device=args.device,
        default_dtype=args.dtype,
        model_type="so3lr",
    )


def convert_lammps_restart_to_data(restart_path, lammps_executable):
    """Convert a binary LAMMPS restart file to a LAMMPS data file by
    shelling out to the LAMMPS executable itself (read_restart +
    write_data). Returns (tmpdir, data_path); caller must clean up tmpdir."""
    tmpdir = tempfile.mkdtemp(prefix="lammps_restart_")
    data_path = os.path.join(tmpdir, "restart.data")
    script_path = os.path.join(tmpdir, "convert.in")
    with open(script_path, "w") as f:
        f.write(
            f"read_restart {os.path.abspath(restart_path)}\n"
            f"write_data {data_path} nocoeff\n"
        )
    result = subprocess.run(
        [lammps_executable, "-in", script_path, "-log", "none"],
        cwd=tmpdir,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not os.path.exists(data_path):
        shutil.rmtree(tmpdir, ignore_errors=True)
        raise RuntimeError(
            f"Failed to convert LAMMPS restart file '{restart_path}' via "
            f"'{lammps_executable}'.\n--- stdout ---\n{result.stdout}\n"
            f"--- stderr ---\n{result.stderr}"
        )
    return tmpdir, data_path


def load_lammps_restart(args):
    """Build an ase.Atoms (with velocities, if present) from a LAMMPS
    restart file, using the user-supplied atom-type -> element mapping."""
    z_of_type = {
        i + 1: atomic_numbers[symbol.strip()]
        for i, symbol in enumerate(args.lammps_type_map.split(","))
    }
    tmpdir, data_path = convert_lammps_restart_to_data(
        args.lammps_restart, args.lammps_executable
    )
    try:
        atoms = read_lammps_data(
            data_path,
            Z_of_type=z_of_type,
            units=args.lammps_units,
            atom_style=args.lammps_atom_style,
        )
    finally:
        shutil.rmtree(tmpdir, ignore_errors=True)
    return atoms


def parse_args():
    parser = argparse.ArgumentParser()
    start_group = parser.add_mutually_exclusive_group(required=True)
    start_group.add_argument("--start_path", type=str, default=None)
    start_group.add_argument("--lammps_restart", type=str, default=None)
    parser.add_argument("--lammps_type_map", type=str, default=None)
    parser.add_argument("--lammps_units", type=str, default=None)
    parser.add_argument("--lammps_atom_style", type=str, default=None)
    parser.add_argument("--lammps_executable", type=str, default="lmp")
    parser.add_argument(
        "--resample_velocities", action="store_true", default=False
    )
    parser.add_argument("--model_path", type=str, default=None)
    parser.add_argument("--so3lr", action="store_true", default=False)
    parser.add_argument("--r_max_lr", type=float, default=12.0)
    parser.add_argument(
        "--dispersion_energy_cutoff_lr_damping", type=float, default=2.0
    )
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--dtype", type=str, default="float32")

    parser.add_argument("--dt", type=float, default=0.5)
    parser.add_argument("--equil_steps", type=int, default=200000)
    parser.add_argument("--equil_T", type=float, default=300.0)
    parser.add_argument("--tdamp", type=float, default=100.0)
    parser.add_argument("--tchain", type=int, default=3)
    parser.add_argument("--prod_steps", type=int, default=400000)

    parser.add_argument("--thermo_interval", type=int, default=4000)
    parser.add_argument("--dipole_interval", type=int, default=4)
    parser.add_argument("--traj_interval", type=int, default=None)

    parser.add_argument("--traj_path", type=str, default="production.traj")
    parser.add_argument("--dipole_path", type=str, default="dipole.dat")
    parser.add_argument("--log_path", type=str, default="md.log")
    parser.add_argument("--equil_out_path", type=str, default=None)

    parser.add_argument("--bond_matrix", type=str, default=None)
    parser.add_argument("--bond_cutoff_scale", type=float, default=1.2)
    parser.add_argument("--log_pressure", action="store_true", default=False)
    parser.add_argument("--seed", type=int, default=None)

    args = parser.parse_args()

    if (args.model_path is None) == (not args.so3lr):
        raise ValueError("Must provide either --model_path or --so3lr")

    if args.lammps_restart is not None:
        if args.lammps_type_map is None:
            raise ValueError(
                "--lammps_type_map is required when using " "--lammps_restart"
            )
        if args.lammps_units is None:
            raise ValueError(
                "--lammps_units is required when using --lammps_restart "
                "(must match the `units` command of the original LAMMPS "
                "run, e.g. real/metal/si)"
            )

    if args.traj_interval is None:
        args.traj_interval = args.thermo_interval

    return args


def main():
    args = parse_args()

    for path in (
        args.traj_path,
        args.dipole_path,
        args.log_path,
        args.equil_out_path,
    ):
        if path is not None and os.path.dirname(path):
            os.makedirs(os.path.dirname(path), exist_ok=True)

    if args.lammps_restart is not None:
        atoms = load_lammps_restart(args)
    else:
        atoms = read(args.start_path)
    if not np.all(atoms.get_pbc()) or atoms.cell.rank < 3:
        raise ValueError(
            "The starting structure must be fully periodic (pbc=True in all "
            "three directions with a non-degenerate cell) for this script."
        )

    atoms.calc = build_calculator(args)

    bonds = get_bond_matrix(
        atoms,
        bond_matrix_path=args.bond_matrix,
        cutoff_scale=args.bond_cutoff_scale,
    )
    percolating = unwrap_molecules(atoms, bonds)
    mol_index, n_molecules = find_molecules(bonds)
    print(f"Found {n_molecules} molecules in {len(atoms)} atoms")

    if percolating:
        print(
            "WARNING: at least one molecule is bonded to its own periodic "
            "image (an infinite/percolating network). Unwrapping such a "
            "structure picks an arbitrary branch, and a dipole built from "
            "atomic positions is not well defined for it. Treat the logged "
            "dipole with care."
        )

    density = atoms.get_masses().sum() / atoms.get_volume()

    if os.path.exists(args.dipole_path):
        os.remove(args.dipole_path)
    with open(args.dipole_path, "a") as f:
        f.write(
            "# step time[fs] mu_x mu_y mu_z mu_raw_x mu_raw_y mu_raw_z "
            "[e*Ang]\n"
            "# mu     = sum of per-molecule dipoles about each molecule's "
            "own center of mass (drift-free)\n"
            "# mu_raw = model output, sum q_i * r_i over absolute positions "
            "(drifts as molecules diffuse)\n"
        )

    rng = np.random.default_rng(args.seed)
    if atoms.has("momenta") and not args.resample_velocities:
        print("Using velocities loaded from the starting structure/restart.")
    else:
        MaxwellBoltzmannDistribution(
            atoms, temperature_K=args.equil_T, rng=rng
        )
        Stationary(atoms)

    print(f"Equilibrating at T={args.equil_T} K for {args.equil_steps} steps")
    dyn_eq = NoseHooverChainNVT(
        atoms,
        timestep=args.dt * units.fs,
        temperature_K=args.equil_T,
        tdamp=args.tdamp * units.fs,
        tchain=args.tchain,
        logfile=args.log_path,
        loginterval=args.thermo_interval,
    )
    dyn_eq.attach(
        make_thermo_printer(atoms, dyn_eq, density, args.log_pressure),
        interval=args.thermo_interval,
    )
    dyn_eq.run(args.equil_steps)

    if args.equil_out_path is not None:
        write(args.equil_out_path, atoms)

    print(f"Running NVE production for {args.prod_steps} steps")
    dyn_prod = VelocityVerlet(
        atoms,
        timestep=args.dt * units.fs,
        logfile=args.log_path,
        loginterval=args.thermo_interval,
    )
    dyn_prod.attach(
        make_thermo_printer(atoms, dyn_prod, density, args.log_pressure),
        interval=args.thermo_interval,
    )
    dyn_prod.attach(
        make_dipole_logger(atoms, dyn_prod, args.dipole_path, mol_index),
        interval=args.dipole_interval,
    )
    traj = Trajectory(args.traj_path, "w", atoms)
    dyn_prod.attach(traj.write, interval=args.traj_interval)
    dyn_prod.run(args.prod_steps)
    traj.close()


if __name__ == "__main__":
    main()
