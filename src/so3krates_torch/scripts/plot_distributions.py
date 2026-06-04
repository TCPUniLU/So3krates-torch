"""Plot property distributions from a raw or preprocessed HDF5 dataset."""

import argparse
import sys
from pathlib import Path

import numpy as np


def collect_from_raw(hdf5_path: str) -> dict:
    import json

    import h5py

    from so3krates_torch.data.hdf5_utils import iter_atoms_from_hdf5

    with h5py.File(hdf5_path, "r") as f:
        keyspec_info = json.loads(f.attrs.get("keyspec_info", "{}"))
        keyspec_arrays = json.loads(f.attrs.get("keyspec_arrays", "{}"))

    energy_key = keyspec_info.get("energy", "energy")
    forces_key = keyspec_arrays.get("forces", "forces")
    charges_key = keyspec_arrays.get("charges", "charges")
    hirshfeld_key = keyspec_arrays.get("hirshfeld_ratios", "hirshfeld_ratios")
    stress_key = keyspec_info.get("stress", "stress")

    energies_per_atom = []
    force_components = []
    charges = []
    hirshfeld = []
    stress_components = []

    for atoms in iter_atoms_from_hdf5(hdf5_path, batch_size=10_000):
        n = len(atoms)

        if energy_key in atoms.info:
            energies_per_atom.append(atoms.info[energy_key] / n)

        if forces_key in atoms.arrays:
            force_components.append(atoms.arrays[forces_key].ravel())

        if charges_key in atoms.arrays:
            charges.append(atoms.arrays[charges_key].ravel())

        if hirshfeld_key in atoms.arrays:
            hirshfeld.append(atoms.arrays[hirshfeld_key].ravel())

        if stress_key in atoms.info:
            s = np.asarray(atoms.info[stress_key]).ravel()
            stress_components.append(s)

    data = {}
    if energies_per_atom:
        data["energy_per_atom"] = np.array(energies_per_atom)
    if force_components:
        data["force_components"] = np.concatenate(force_components)
    if charges:
        data["charges"] = np.concatenate(charges)
    if hirshfeld:
        data["hirshfeld_ratios"] = np.concatenate(hirshfeld)
    if stress_components:
        data["stress_components"] = np.concatenate(stress_components)

    return data


def collect_from_preprocessed(hdf5_path: str) -> dict:
    from so3krates_torch.data.hdf5_utils import PreprocessedHDF5Dataset

    dataset = PreprocessedHDF5Dataset(hdf5_path, validate_cutoffs=False)

    energies_per_atom = []
    force_components = []
    charges = []
    hirshfeld = []
    stress_components = []

    for i in range(len(dataset)):
        d = dataset[i]
        n = d.positions.shape[0]

        if d.energy is not None:
            energies_per_atom.append(d.energy.item() / n)

        if d.forces is not None:
            force_components.append(d.forces.numpy().ravel())

        if d.charges is not None:
            charges.append(d.charges.numpy().ravel())

        if d.hirshfeld_ratios is not None:
            hirshfeld.append(d.hirshfeld_ratios.numpy().ravel())

        if d.stress is not None:
            stress_components.append(d.stress.numpy().ravel())

    data = {}
    if energies_per_atom:
        data["energy_per_atom"] = np.array(energies_per_atom)
    if force_components:
        data["force_components"] = np.concatenate(force_components)
    if charges:
        data["charges"] = np.concatenate(charges)
    if hirshfeld:
        data["hirshfeld_ratios"] = np.concatenate(hirshfeld)
    if stress_components:
        data["stress_components"] = np.concatenate(stress_components)

    return data


PANEL_CONFIG = {
    "energy_per_atom": {
        "label": "Energy per atom (eV/atom)",
        "title": "Energy per atom",
    },
    "force_components": {
        "label": "Force component (eV/Å)",
        "title": "Force components",
    },
    "charges": {
        "label": "Partial charge (e)",
        "title": "Partial charges",
    },
    "hirshfeld_ratios": {
        "label": "Hirshfeld ratio",
        "title": "Hirshfeld ratios",
    },
    "stress_components": {
        "label": "Stress component (eV/Å³)",
        "title": "Stress components",
    },
}


def plot_distributions(data: dict, output: str | None, bins: int = 80):
    import matplotlib.pyplot as plt

    keys = [k for k in PANEL_CONFIG if k in data]
    if not keys:
        print("No plottable properties found in file.")
        return

    ncols = min(3, len(keys))
    nrows = (len(keys) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5 * ncols, 4 * nrows), squeeze=False
    )
    fig.suptitle("Property distributions", fontsize=14)

    for ax_idx, key in enumerate(keys):
        ax = axes[ax_idx // ncols][ax_idx % ncols]
        vals = data[key]
        cfg = PANEL_CONFIG[key]

        ax.hist(vals, bins=bins, density=True, color="steelblue", alpha=0.8)
        ax.set_xlabel(cfg["label"])
        ax.set_ylabel("Density")
        ax.set_title(cfg["title"])

        mean, std = vals.mean(), vals.std()
        ax.axvline(mean, color="tomato", lw=1.5, label=f"mean={mean:.3g}")
        ax.axvline(mean - std, color="tomato", lw=1, ls="--")
        ax.axvline(
            mean + std, color="tomato", lw=1, ls="--", label=f"σ={std:.3g}"
        )
        ax.legend(fontsize=8)

    # hide unused axes
    for ax_idx in range(len(keys), nrows * ncols):
        axes[ax_idx // ncols][ax_idx % ncols].set_visible(False)

    plt.tight_layout()

    if output:
        plt.savefig(output, dpi=150)
        print(f"Saved to {output}")
    else:
        plt.show()


def report_hirshfeld_outliers(vals: np.ndarray, k: float = 1.5) -> None:
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = q3 - q1
    lower = q1 - k * iqr
    upper = q3 + k * iqr

    mask = (vals < lower) | (vals > upper)
    outliers = np.sort(vals[mask])

    print(f"\nHirshfeld ratio outliers (IQR method, k={k}):")
    print(f"  Q1={q1:.4f}  Q3={q3:.4f}  IQR={iqr:.4f}")
    print(f"  Fence: [{lower:.4f}, {upper:.4f}]")
    print(
        f"  Count: {len(outliers)} / {len(vals)}"
        f" ({100 * len(outliers) / len(vals):.2f}%)"
    )
    if len(outliers) == 0:
        return

    print(f"  Min outlier: {outliers[0]:.6f}")
    print(f"  Max outlier: {outliers[-1]:.6f}")

    n_show = 20
    below = np.sort(vals[vals < lower])
    above = np.sort(vals[vals > upper])[::-1]

    if len(below):
        shown = below[:n_show]
        print(f"  Lowest  (up to {n_show}): {shown}")
        if len(below) > n_show:
            print(f"    ... and {len(below) - n_show} more below fence")

    if len(above):
        shown = above[:n_show]
        print(f"  Highest (up to {n_show}): {shown}")
        if len(above) > n_show:
            print(f"    ... and {len(above) - n_show} more above fence")


def main():
    parser = argparse.ArgumentParser(
        description="Plot property distributions from an HDF5 dataset."
    )
    parser.add_argument("hdf5", help="Path to raw or preprocessed HDF5 file")
    parser.add_argument(
        "-o",
        "--output",
        default=None,
        help="Save figure to this path instead of displaying it",
    )
    parser.add_argument(
        "--bins", type=int, default=80, help="Number of histogram bins"
    )
    parser.add_argument(
        "--outlier-k",
        type=float,
        default=1.5,
        help="IQR multiplier for Hirshfeld outlier fence (default 1.5)",
    )
    args = parser.parse_args()

    from so3krates_torch.data.hdf5_utils import detect_file_format

    fmt = detect_file_format(args.hdf5)
    print(f"Detected format: {fmt}")

    if fmt == "hdf5_raw":
        data = collect_from_raw(args.hdf5)
    elif fmt == "hdf5_preprocessed":
        data = collect_from_preprocessed(args.hdf5)
    else:
        print(f"Unsupported format '{fmt}'. Only HDF5 files are supported.")
        sys.exit(1)

    print("Loaded properties:")
    for k, v in data.items():
        print(f"  {k}: {v.shape} — mean={v.mean():.4g}, std={v.std():.4g}")

    if "hirshfeld_ratios" in data:
        report_hirshfeld_outliers(data["hirshfeld_ratios"], k=args.outlier_k)

    plot_distributions(data, args.output, bins=args.bins)


if __name__ == "__main__":
    main()
