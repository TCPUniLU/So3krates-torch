"""
Stamp config_weight values onto structures in a training data file.

Supports XYZ and raw HDF5 formats. For HDF5, only the weight column is
rewritten — atomic positions are never loaded, so this is efficient on
large datasets.

Usage examples:
    python set_weights.py --input data.xyz --output data_w.xyz --weight 2.0
    python set_weights.py --input data.xyz --output data_w.xyz \
        --config-type-weights Default:1.0 dissociation:2.0
    python set_weights.py --input data.h5 --output data_w.h5 \
        --weight 0.5 --config-type-weights special:3.0
"""

import argparse
import logging
import shutil
import sys

import ase.io
import h5py
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def parse_config_type_weights(values):
    """Parse ['Default:1.0', 'dissociation:2.0'] into a dict."""
    result = {}
    for entry in values:
        parts = entry.split(":")
        if len(parts) != 2:
            raise argparse.ArgumentTypeError(
                f"Invalid config-type-weight entry '{entry}'. "
                f"Expected format TYPE:VALUE, e.g. dissociation:2.0"
            )
        type_name, weight_str = parts
        try:
            result[type_name] = float(weight_str)
        except ValueError:
            raise argparse.ArgumentTypeError(
                f"Weight value '{weight_str}' in '{entry}' "
                f"is not a valid float."
            )
    return result


def compute_weight(config_type, uniform_weight, config_type_map):
    """Compute per-structure weight from uniform and per-type factors."""
    return uniform_weight * config_type_map.get(config_type, 1.0)


def process_xyz(args, uniform_weight, config_type_map):
    logging.info(f"Loading XYZ: {args.input}")
    atoms_list = ase.io.read(args.input, index=":")
    logging.info(f"Loaded {len(atoms_list)} structures")

    for atoms in atoms_list:
        config_type = atoms.info.get("config_type", "Default")
        atoms.info["config_weight"] = compute_weight(
            config_type, uniform_weight, config_type_map
        )

    ase.io.write(args.output, atoms_list)
    logging.info(f"Saved {len(atoms_list)} structures to {args.output}")


def process_hdf5(args, uniform_weight, config_type_map):
    from so3krates_torch.data.hdf5_utils import detect_file_format

    file_format = detect_file_format(args.input)
    if file_format == "hdf5_preprocessed":
        logging.error(
            "Input file is a preprocessed HDF5 (contains pre-built neighbor "
            "lists). set_weights.py only supports raw HDF5. Convert the file "
            "with torchkrates-preprocess first, or use the raw HDF5 source."
        )
        sys.exit(1)

    logging.info(f"Copying {args.input} → {args.output}")
    shutil.copy2(args.input, args.output)

    with h5py.File(args.output, "r+") as f:
        n_configs = int(f.attrs["num_configs"])
        logging.info(f"Processing {n_configs} structures")

        if config_type_map:
            raw_types = f["config_metadata/config_type"][:]
            config_types = [
                t.decode() if isinstance(t, bytes) else t for t in raw_types
            ]
        else:
            config_types = ["Default"] * n_configs

        weights = np.array(
            [
                compute_weight(ct, uniform_weight, config_type_map)
                for ct in config_types
            ],
            dtype=np.float64,
        )
        f["config_metadata/weight"][:] = weights

    logging.info(f"Saved weighted HDF5 to {args.output}")


def main():
    parser = argparse.ArgumentParser(
        description="Stamp config_weight values onto structures in a "
        "training data file (XYZ or raw HDF5)."
    )
    parser.add_argument(
        "--input", required=True, help="Input file (.xyz, .extxyz, .h5, .hdf5)"
    )
    parser.add_argument(
        "--output", required=True, help="Output file (same format as input)"
    )
    parser.add_argument(
        "--weight",
        type=float,
        default=None,
        help="Uniform weight applied to all structures.",
    )
    parser.add_argument(
        "--config-type-weights",
        nargs="+",
        metavar="TYPE:VALUE",
        default=None,
        help="Per-config-type weight multipliers, e.g. "
        "Default:1.0 dissociation:2.0. "
        "Structures are matched by atoms.info['config_type'].",
    )
    args = parser.parse_args()

    if args.weight is None and args.config_type_weights is None:
        parser.error(
            "At least one of --weight or --config-type-weights is required."
        )

    uniform_weight = args.weight if args.weight is not None else 1.0
    config_type_map = (
        parse_config_type_weights(args.config_type_weights)
        if args.config_type_weights
        else {}
    )

    if args.input.endswith((".xyz", ".extxyz")):
        process_xyz(args, uniform_weight, config_type_map)
    elif args.input.endswith((".h5", ".hdf5")):
        process_hdf5(args, uniform_weight, config_type_map)
    else:
        parser.error(
            f"Unsupported input format: {args.input}. "
            f"Expected .xyz, .extxyz, .h5, or .hdf5."
        )

    logging.info("Done.")


if __name__ == "__main__":
    main()
