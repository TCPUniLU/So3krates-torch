"""CLI script: torchkrates-tune-pme

Runs torchpme.tuning.tune_pme() on a sample of training structures to find
optimal PME parameters (smearing, mesh_spacing) for a given r_max cutoff.
"""

import argparse
import logging
import time
import torch
import numpy as np
from pathlib import Path

logging.basicConfig(level=logging.INFO)

# Reference tight-PME setup for the dispersion convergence scan (expensive,
# run once per structure, only during tuning).
SMEARING_REF_FACTOR = 1 / 10  # smearing_ref = r_max_lr / 10
MESH_REF_FACTOR = 1 / 5  # mesh_spacing_ref = smearing_ref / 5

# Coarser candidate grid, scanned against the tight reference above.
SMEARING_FACTORS = [1 / 8, 1 / 6, 1 / 5, 1 / 4, 1 / 3]  # * r_max_lr
MESH_FRACTIONS = [1 / 4, 1 / 3, 1 / 2]  # * smearing


def _load_periodic_structures(data_path, n_samples):
    """Load up to n_samples periodic structures from data_path.

    Shared loading/filtering logic used by both the electrostatics and
    dispersion tuners: .h5/.hdf5 via load_atoms_from_hdf5, else ase.io.read;
    filter to structures where all(a.pbc) holds.
    """
    data_path = Path(data_path)
    if data_path.suffix in (".h5", ".hdf5"):
        from so3krates_torch.data.hdf5_utils import load_atoms_from_hdf5

        all_atoms = load_atoms_from_hdf5(str(data_path))
    else:
        from ase.io import read

        all_atoms = read(str(data_path), index=":")

    periodic = [a for a in all_atoms if all(a.pbc)]
    if not periodic:
        raise ValueError("No periodic structures found in dataset.")
    return periodic[:n_samples]


def _build_sr_neighbor_list(positions, cell, cutoff, device, dtype):
    """Build a simple SR neighbor list via matscipy for a single structure."""
    import numpy as np_
    from matscipy.neighbours import neighbour_list

    pos_np = positions.cpu().numpy()
    cell_np = cell.cpu().numpy()
    i, j, d = neighbour_list(
        "ijd",
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
    sample = _load_periodic_structures(data_path, n_samples)
    logging.info(
        f"Tuning PME on {len(sample)} structures (r_max={r_max} Å, "
        f"accuracy={accuracy})"
    )

    smearings, mesh_spacings = [], []
    t0 = time.time()

    for idx, atoms in enumerate(sample, 1):
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

        elapsed = time.time() - t0
        avg = elapsed / idx
        eta = avg * (len(sample) - idx)
        logging.info(
            "[electrostatics %d/%d] elapsed %.1fs  ETA %.0fs",
            idx,
            len(sample),
            elapsed,
            eta,
        )

    if not smearings:
        raise RuntimeError("tune_pme failed for all structures.")

    med_smearing = float(np.median(smearings))
    med_mesh_spacing = float(np.median(mesh_spacings))

    if med_mesh_spacing > med_smearing:
        logging.warning(
            "Electrostatics: mesh_spacing (%.4f Å) > smearing (%.4f Å) "
            "— the FFT mesh is coarser than the Gaussian width. "
            "This can cause B-spline interpolation aliasing not captured "
            "by the analytical error bound. "
            "Consider tightening --accuracy or setting "
            "pme_mesh_spacing = smearing / 2 (%.4f Å) manually.",
            med_mesh_spacing,
            med_smearing,
            med_smearing / 2,
        )

    return med_smearing, med_mesh_spacing


def tune_dispersion_params(
    data_path: str,
    r_max_lr: float,
    n_samples: int = 50,
    accuracy: float = 1e-3,
    device: str = "cpu",
    dtype_str: str = "float64",
):
    """Scan PME dispersion parameters (smearing, mesh_spacing) via a
    convergence scan against a tight-PME reference, on up to n_samples
    periodic structures from data_path.

    torchpme.tuning.tune_pme only supports exponent=1 (Coulomb electro-
    statics), so it cannot be used for exponent=6 (C6 dispersion). Instead,
    for each structure we compute a reference PME-dispersion energy with a
    tight (well-converged) (smearing, mesh_spacing) pair, then scan a grid
    of coarser candidates and record the per-atom energy error against that
    reference. Pseudo-charges use unit Hirshfeld ratios (real Hirshfeld
    ratios are model outputs, not available at tuning time) — box geometry,
    which drives the optimal mesh, is preserved by using the real training
    structures, and unit ratios keep the reference self-consistent since
    both scan and reference use the same (constant) ratios.

    Returns:
        smearing (float), mesh_spacing (float) — the coarsest grid point
        (largest smearing, then largest mesh_spacing) whose median error
        over all structures is below `accuracy` (eV/atom).
    """
    from so3krates_torch.blocks.physical_potentials import (
        atomic_c6_pseudo_charges,
        PMEDispersionInteraction,
    )

    dtype = torch.float64 if dtype_str == "float64" else torch.float32
    dev = torch.device(device)

    sample = _load_periodic_structures(data_path, n_samples)
    logging.info(
        f"Tuning PME dispersion on {len(sample)} structures "
        f"(r_max_lr={r_max_lr} Å, accuracy={accuracy})"
    )

    # Precompute per-structure positions/cell/pseudo-charges once — these
    # don't depend on the (smearing, mesh_spacing) being tested.
    structures = []
    for atoms in sample:
        positions = torch.tensor(
            atoms.get_positions(), dtype=dtype, device=dev
        )
        cell = torch.tensor(
            np.array(atoms.get_cell()), dtype=dtype, device=dev
        )
        atomic_numbers = torch.tensor(
            atoms.get_atomic_numbers(), dtype=torch.long, device=dev
        )
        N = len(atoms)
        hirshfeld_ratios = torch.ones(N, dtype=dtype, device=dev)
        C6_i = atomic_c6_pseudo_charges(
            atomic_numbers, hirshfeld_ratios, legacy_dispersion_bool=True
        )
        q = torch.sqrt(torch.clamp(C6_i, min=0.0))
        batch_segments = torch.zeros(N, dtype=torch.long, device=dev)
        structures.append((positions, cell, q, N, batch_segments))

    # Tight reference PME setup.
    smearing_ref = r_max_lr * SMEARING_REF_FACTOR
    mesh_spacing_ref = smearing_ref * MESH_REF_FACTOR
    logging.info(
        "Dispersion reference PME: smearing_ref=%.4f Å, "
        "mesh_spacing_ref=%.4f Å",
        smearing_ref,
        mesh_spacing_ref,
    )

    ref_calc = PMEDispersionInteraction(
        smearing=smearing_ref, mesh_spacing=mesh_spacing_ref
    ).to(dev)

    ref_energies = []
    t0 = time.time()
    for idx, (positions, cell, q, N, batch_segments) in enumerate(
        structures, 1
    ):
        with torch.no_grad():
            e_ref = ref_calc(
                c6_pseudo_charges=q,
                positions=positions,
                cell=cell,
                batch_segments=batch_segments,
                num_graphs=1,
                num_nodes=N,
            )
        ref_energies.append(e_ref.sum().item())

        elapsed = time.time() - t0
        avg = elapsed / idx
        eta = avg * (len(structures) - idx)
        logging.info(
            "[dispersion reference %d/%d] elapsed %.1fs  ETA %.0fs",
            idx,
            len(structures),
            elapsed,
            eta,
        )

    # Coarser candidate grid, scanned against the tight reference above.
    grid = [
        (r_max_lr * sf, r_max_lr * sf * mf)
        for sf in SMEARING_FACTORS
        for mf in MESH_FRACTIONS
    ]

    results = []  # (smearing, mesh_spacing, median_err)
    t0 = time.time()
    for gi, (smearing, mesh_spacing) in enumerate(grid, 1):
        calc = PMEDispersionInteraction(
            smearing=smearing, mesh_spacing=mesh_spacing
        ).to(dev)

        errs = []
        for (positions, cell, q, N, batch_segments), e_ref in zip(
            structures, ref_energies
        ):
            with torch.no_grad():
                e_scan = calc(
                    c6_pseudo_charges=q,
                    positions=positions,
                    cell=cell,
                    batch_segments=batch_segments,
                    num_graphs=1,
                    num_nodes=N,
                )
            errs.append(abs(e_scan.sum().item() - e_ref) / N)

        med_err = float(np.median(errs))
        results.append((smearing, mesh_spacing, med_err))

        elapsed = time.time() - t0
        avg = elapsed / gi
        eta = avg * (len(grid) - gi)
        logging.info(
            "[dispersion scan %d/%d] smearing=%.4f Å mesh_spacing=%.4f Å "
            "median_err=%.3e eV/atom  elapsed %.1fs  ETA %.0fs",
            gi,
            len(grid),
            smearing,
            mesh_spacing,
            med_err,
            elapsed,
            eta,
        )

    passing = [r for r in results if r[2] < accuracy]
    if not passing:
        best = min(results, key=lambda r: r[2])
        raise RuntimeError(
            "No dispersion scan grid point met the target accuracy "
            f"({accuracy:.1e} eV/atom). Best achieved: smearing="
            f"{best[0]:.4f} Å, mesh_spacing={best[1]:.4f} Å, "
            f"median_err={best[2]:.3e} eV/atom. Consider loosening "
            "--dispersion_accuracy or extending the scan grid "
            "(SMEARING_FACTORS/MESH_FRACTIONS in run_tune_pme.py)."
        )

    # Coarsest passing grid point: largest smearing, then (among ties)
    # largest mesh_spacing.
    passing.sort(key=lambda r: (r[0], r[1]))
    best_smearing, best_mesh_spacing, best_err = passing[-1]

    logging.info(
        "Dispersion tuning selected smearing=%.4f Å, mesh_spacing=%.4f Å "
        "(median_err=%.3e eV/atom, target %.1e)",
        best_smearing,
        best_mesh_spacing,
        best_err,
        accuracy,
    )

    return best_smearing, best_mesh_spacing


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
        "--r_max_lr",
        type=float,
        default=None,
        help="Long-range cutoff radius in Angstrom, required with "
        "--tune_dispersion",
    )
    parser.add_argument(
        "--tune_dispersion",
        action="store_true",
        help=(
            "Scan PME dispersion parameters (smearing, mesh_spacing) by "
            "comparing against a tight-PME reference on training structures "
            "with unit Hirshfeld ratios."
        ),
    )
    parser.add_argument(
        "--dispersion_accuracy",
        type=float,
        default=1e-3,
        help=(
            "Target accuracy for the dispersion scan in eV/atom "
            "(default: 1e-3). Picks the coarsest parameters within "
            "this threshold vs. the tight-parameter reference."
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

    disp_smearing = None
    disp_mesh_spacing = None
    if args.tune_dispersion:
        if args.r_max_lr is None:
            parser.error("--tune_dispersion requires --r_max_lr")
        disp_smearing, disp_mesh_spacing = tune_dispersion_params(
            data_path=args.data_path,
            r_max_lr=args.r_max_lr,
            n_samples=args.n_samples,
            accuracy=args.dispersion_accuracy,
            device=args.device,
            dtype_str=args.dtype,
        )
        print(
            f"\n  Dispersion (C6, scanned vs tight-PME reference):\n"
            f"    pme_dispersion_smearing:     {disp_smearing:.4f} Å\n"
            f"    pme_dispersion_mesh_spacing: {disp_mesh_spacing:.4f} Å\n"
        )

    if args.update_config:
        import yaml

        config_path = Path(args.update_config)
        with open(config_path) as f:
            config = yaml.safe_load(f)
        config.setdefault("ARCHITECTURE", {})
        config["ARCHITECTURE"]["pme_smearing"] = round(smearing, 6)
        config["ARCHITECTURE"]["pme_mesh_spacing"] = round(mesh_spacing, 6)
        if args.tune_dispersion:
            config["ARCHITECTURE"]["pme_dispersion_smearing"] = round(
                disp_smearing, 6
            )
            config["ARCHITECTURE"]["pme_dispersion_mesh_spacing"] = round(
                disp_mesh_spacing, 6
            )
        with open(config_path, "w") as f:
            yaml.dump(config, f, default_flow_style=False)
        logging.info(f"Updated {config_path} with PME parameters.")


if __name__ == "__main__":
    main()
