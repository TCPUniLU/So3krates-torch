"""CLI script: torchkrates-tune-pme

Runs torchpme.tuning.tune_pme() on a sample of training structures to find
optimal PME parameters (smearing, mesh_spacing) for a given r_max cutoff.
"""
import argparse
import logging
import torch
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)


def _build_sr_neighbor_list(positions, cell, cutoff, device, dtype):
    """Build a simple SR neighbor list via matscipy for a single structure."""
    import numpy as np_
    from matscipy.neighbours import neighbour_list

    pos_np = positions.cpu().numpy()
    cell_np = cell.cpu().numpy()
    i, j, d = neighbour_list(
        "ijd",
        a=None,
        cutoff=cutoff,
        positions=pos_np,
        cell=cell_np,
        pbc=[True, True, True],
    )
    if len(i) == 0:
        return (
            torch.zeros((0, 2), dtype=torch.long, device=device),
            torch.zeros(0, dtype=dtype, device=device),
        )
    ni = torch.stack(
        [
            torch.tensor(i, dtype=torch.long, device=device),
            torch.tensor(j, dtype=torch.long, device=device),
        ],
        dim=1,
    )  # (E, 2)
    nd = torch.tensor(d, dtype=dtype, device=device)
    return ni, nd


def tune_pme_params(
    data_path: str,
    r_max: float,
    n_samples: int = 50,
    accuracy: float = 1e-3,
    charges_key: str = None,
    device: str = "cpu",
    dtype_str: str = "float64",
):
    """Run tune_pme on up to n_samples periodic structures from data_path.

    Returns:
        smearing (float), mesh_spacing (float) — medians over all structures.
    """
    from torchpme.tuning import tune_pme as _tune_pme

    dtype = torch.float64 if dtype_str == "float64" else torch.float32
    dev = torch.device(device)

    # Load structures
    data_path = Path(data_path)
    if data_path.suffix in (".h5", ".hdf5"):
        from so3krates_torch.data.hdf5_utils import load_atoms_from_hdf5

        all_atoms = load_atoms_from_hdf5(str(data_path))
    else:
        from ase.io import read

        all_atoms = read(str(data_path), index=":")

    # Filter to periodic structures and subsample
    periodic = [a for a in all_atoms if all(a.pbc)]
    if not periodic:
        raise ValueError("No periodic structures found in dataset.")
    sample = periodic[:n_samples]
    logging.info(
        f"Tuning PME on {len(sample)} structures (r_max={r_max} Å, "
        f"accuracy={accuracy})"
    )

    smearings, mesh_spacings = [], []

    for atoms in sample:
        positions = torch.tensor(
            atoms.get_positions(), dtype=dtype, device=dev
        )
        cell = torch.tensor(
            np.array(atoms.get_cell()), dtype=dtype, device=dev
        )
        N = len(atoms)

        if charges_key and charges_key in atoms.arrays:
            q = torch.tensor(
                atoms.arrays[charges_key], dtype=dtype, device=dev
            ).unsqueeze(
                1
            )  # (N, 1)
        else:
            q = torch.ones(N, 1, dtype=dtype, device=dev)

        ni, nd = _build_sr_neighbor_list(positions, cell, r_max, dev, dtype)
        if ni.shape[0] == 0:
            continue

        try:
            smearing, mesh_params, _ = _tune_pme(
                charges=q,
                cell=cell,
                positions=positions,
                cutoff=r_max,
                neighbor_indices=ni,
                neighbor_distances=nd,
                accuracy=accuracy,
                exponent=1,
            )
            smearings.append(smearing)
            mesh_spacings.append(mesh_params["mesh_spacing"])
        except Exception as e:
            logging.warning(f"tune_pme failed for a structure: {e}")
            continue

    if not smearings:
        raise RuntimeError("tune_pme failed for all structures.")

    med_smearing = float(np.median(smearings))
    med_mesh_spacing = float(np.median(mesh_spacings))
    return med_smearing, med_mesh_spacing


def tune_dispersion_params(
    data_path: str,
    r_max: float,
    n_samples: int = 50,
    accuracy: float = 1e-3,
    device: str = "cpu",
    dtype_str: str = "float64",
):
    """Scan PME dispersion parameters by comparing against a tight PME
    reference on training structures with unit Hirshfeld ratios.

    The reference uses smearing = r_max / 10 and
    mesh_spacing = smearing_ref / 5 — expensive but run only once per
    structure.

    Returns:
        smearing (float), mesh_spacing (float) — coarsest params where
        median error vs reference < accuracy (eV/atom).
    """
    import torchpme
    from itertools import product as iproduct

    dtype = torch.float64 if dtype_str == "float64" else torch.float32
    dev = torch.device(device)

    # Load structures (reuse same logic as tune_pme_params)
    data_path_obj = Path(data_path)
    if data_path_obj.suffix in (".h5", ".hdf5"):
        from so3krates_torch.data.hdf5_utils import load_atoms_from_hdf5

        all_atoms = load_atoms_from_hdf5(str(data_path_obj))
    else:
        from ase.io import read

        all_atoms = read(str(data_path_obj), index=":")

    periodic = [a for a in all_atoms if all(a.pbc)]
    if not periodic:
        raise ValueError("No periodic structures found in dataset.")
    sample = periodic[:n_samples]
    logging.info(
        "Tuning PME dispersion on %d structures (r_max=%.2f Å, "
        "accuracy=%.1e eV/atom)",
        len(sample),
        r_max,
        accuracy,
    )

    # C6 coefficient table from physical_potentials
    from so3krates_torch.blocks.physical_potentials import (
        C6_COEF,
        BOHR,
    )

    c6_coef = C6_COEF.to(device=dev, dtype=dtype)

    # Reference PME parameters (tight)
    smearing_ref = r_max / 10.0
    mesh_spacing_ref = smearing_ref / 5.0

    # Scan grid
    smearing_factors = [1 / 8, 1 / 6, 1 / 5, 1 / 4, 1 / 3]
    mesh_fractions = [1 / 4, 1 / 3, 1 / 2]
    grid = list(iproduct(smearing_factors, mesh_fractions))

    # Per-grid-point median errors across structures
    errors_per_point = {pt: [] for pt in grid}

    for atoms in sample:
        positions = torch.tensor(
            atoms.get_positions(), dtype=dtype, device=dev
        )
        cell = torch.tensor(
            np.array(atoms.get_cell()), dtype=dtype, device=dev
        )
        N = len(atoms)
        Z = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.long, device=dev
        )

        # Unit Hirshfeld C6: c_i = sqrt(C6_COEF[Z_i-1] * BOHR^6)
        C6_i = c6_coef[Z - 1] * (BOHR ** 6)
        c_charges = torch.sqrt(C6_i.clamp(min=1e-30)).unsqueeze(1)

        ni, nd = _build_sr_neighbor_list(
            positions, cell, r_max, dev, dtype
        )
        if ni.shape[0] == 0:
            continue

        # Compute reference energy
        try:
            ref_pot = torchpme.InversePowerLawPotential(
                exponent=6, smearing=smearing_ref
            )
            ref_calc = torchpme.PMECalculator(
                potential=ref_pot,
                mesh_spacing=mesh_spacing_ref,
                interpolation_nodes=4,
            )
            phi_ref = ref_calc.forward(
                charges=c_charges,
                cell=cell,
                positions=positions,
                neighbor_indices=ni,
                neighbor_distances=nd,
            )
            E_ref = float(
                (-0.5 * c_charges * phi_ref).sum().item()
            )
        except Exception as exc:
            logging.warning(
                "Reference PME failed for a structure: %s", exc
            )
            continue

        # Evaluate each grid point
        for sf, mf in grid:
            smearing_g = r_max * sf
            mesh_spacing_g = smearing_g * mf
            try:
                pot_g = torchpme.InversePowerLawPotential(
                    exponent=6, smearing=smearing_g
                )
                calc_g = torchpme.PMECalculator(
                    potential=pot_g,
                    mesh_spacing=mesh_spacing_g,
                    interpolation_nodes=4,
                )
                phi_g = calc_g.forward(
                    charges=c_charges,
                    cell=cell,
                    positions=positions,
                    neighbor_indices=ni,
                    neighbor_distances=nd,
                )
                E_g = float(
                    (-0.5 * c_charges * phi_g).sum().item()
                )
                err = abs(E_g - E_ref) / N
                errors_per_point[(sf, mf)].append(err)
            except Exception as exc:
                logging.warning(
                    "PME failed for smearing=%.3f mesh=%.3f: %s",
                    smearing_g,
                    mesh_spacing_g,
                    exc,
                )

    # Pick coarsest (largest smearing, largest mesh_spacing) within
    # accuracy. Grid is ordered from finest to coarsest — reverse to
    # get coarsest first.
    chosen = None
    chosen_err = float("inf")

    # Iterate from coarsest to finest; keep first that meets accuracy
    for sf, mf in reversed(grid):
        errs = errors_per_point[(sf, mf)]
        if not errs:
            continue
        med_err = float(np.median(errs))
        if med_err < accuracy:
            chosen = (sf * r_max, (sf * r_max) * mf)
            chosen_err = med_err
            break  # coarsest acceptable found

    if chosen is None:
        # Fall back to finest grid point
        sf, mf = grid[0]
        errs = errors_per_point[(sf, mf)]
        chosen = (sf * r_max, (sf * r_max) * mf)
        chosen_err = (
            float(np.median(errs)) if errs else float("inf")
        )
        logging.warning(
            "No grid point met accuracy=%.1e; using finest params "
            "smearing=%.4f mesh=%.4f (err=%.4f eV/atom)",
            accuracy,
            chosen[0],
            chosen[1],
            chosen_err,
        )

    return float(chosen[0]), float(chosen[1]), float(chosen_err)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Tune PME parameters (smearing, mesh_spacing) for "
            "torchkrates PME electrostatics."
        )
    )
    parser.add_argument(
        "--data_path",
        required=True,
        help="Path to training data (.xyz, .extxyz, .h5)",
    )
    parser.add_argument(
        "--r_max",
        type=float,
        required=True,
        help="SR cutoff radius in Angstrom",
    )
    parser.add_argument(
        "--n_samples",
        type=int,
        default=50,
        help="Max structures to use for tuning",
    )
    parser.add_argument(
        "--accuracy",
        type=float,
        default=1e-3,
        help="Target accuracy for PME error bound",
    )
    parser.add_argument(
        "--charges_key",
        default=None,
        help="Key in atoms.arrays for partial charges "
        "(default: unit charges)",
    )
    parser.add_argument(
        "--device",
        default="cpu",
        help="Device for torch tensors (default: cpu)",
    )
    parser.add_argument(
        "--dtype",
        default="float64",
        choices=["float32", "float64"],
        help="Tensor dtype (default: float64)",
    )
    parser.add_argument(
        "--update_config",
        default=None,
        help="If given, write PME params to this YAML config",
    )
    parser.add_argument(
        "--tune_dispersion",
        action="store_true",
        help=(
            "Scan PME dispersion parameters (smearing, mesh_spacing) "
            "by comparing against a tight PME reference on training "
            "structures with unit Hirshfeld ratios."
        ),
    )
    parser.add_argument(
        "--dispersion_accuracy",
        type=float,
        default=1e-3,
        help=(
            "Target accuracy for dispersion scan in eV/atom "
            "(default: 1e-3). Picks the coarsest parameters "
            "within this threshold vs the tight-parameter reference."
        ),
    )
    args = parser.parse_args()

    smearing, mesh_spacing = tune_pme_params(
        data_path=args.data_path,
        r_max=args.r_max,
        n_samples=args.n_samples,
        accuracy=args.accuracy,
        charges_key=args.charges_key,
        device=args.device,
        dtype_str=args.dtype,
    )

    print(
        f"\nPME tuning results (median over structures):\n"
        f"  Electrostatics:\n"
        f"    pme_smearing:     {smearing:.4f} Å\n"
        f"    pme_mesh_spacing: {mesh_spacing:.4f} Å\n"
    )

    if args.tune_dispersion:
        d_smearing, d_mesh_spacing, d_err = tune_dispersion_params(
            data_path=args.data_path,
            r_max=args.r_max,
            n_samples=args.n_samples,
            accuracy=args.dispersion_accuracy,
            device=args.device,
            dtype_str=args.dtype,
        )
        print(
            f"  Dispersion (C6, scanned vs tight-PME reference):\n"
            f"    pme_smearing_dispersion:     "
            f"{d_smearing:.4f} Å\n"
            f"    pme_mesh_spacing_dispersion: "
            f"{d_mesh_spacing:.4f} Å\n"
            f"    median error vs reference:   "
            f"{d_err * 1000:.2f} meV/atom"
        )
        if args.update_config:
            import yaml

            config_path = Path(args.update_config)
            with open(config_path) as f:
                config = yaml.safe_load(f)
            config.setdefault("ARCHITECTURE", {})
            config["ARCHITECTURE"][
                "pme_smearing_dispersion"
            ] = round(d_smearing, 6)
            config["ARCHITECTURE"][
                "pme_mesh_spacing_dispersion"
            ] = round(d_mesh_spacing, 6)
            with open(config_path, "w") as f:
                yaml.dump(config, f, default_flow_style=False)
            logging.info(
                "Updated %s with PME dispersion parameters.",
                config_path,
            )

    if args.update_config:
        import yaml

        config_path = Path(args.update_config)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        config.setdefault("ARCHITECTURE", {})
        config["ARCHITECTURE"]["pme_smearing"] = round(smearing, 6)
        config["ARCHITECTURE"]["pme_mesh_spacing"] = round(mesh_spacing, 6)
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Updated {config_path} with PME parameters.")


if __name__ == "__main__":
    main()
