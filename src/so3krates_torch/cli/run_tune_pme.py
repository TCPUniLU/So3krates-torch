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
