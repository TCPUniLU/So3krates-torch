"""
CLI Script for Preprocessing Atomic Structure Data to HDF5

Supports:
- XYZ → raw HDF5
- XYZ → preprocessed HDF5
- Raw HDF5 → preprocessed HDF5
"""

import argparse
import logging
import sys
import time
from pathlib import Path
from typing import Optional

import ase.io
import torch

from so3krates_torch.data.atomic_data import AtomicData
from so3krates_torch.data.hdf5_utils import (
    compute_and_format_e0s,
    configs_from_hdf5,
    detect_file_format,
    save_atoms_to_hdf5,
    save_preprocessed_hdf5,
    validate_preprocessed_hdf5,
)
from so3krates_torch.data.utils import KeySpecification, update_keyspec_from_kwargs
from so3krates_torch.tools.default_keys import DefaultKeys
from so3krates_torch.tools.utils import (
    AtomicNumberTable,
    create_configs_from_list,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def _log_progress(current, total, start_time, log_interval=100):
    """
    Log progress at regular intervals for HPC-friendly output.

    Args:
        current: Current item number (1-indexed)
        total: Total number of items
        start_time: Start time from time.time()
        log_interval: Log every N items
    """
    if current % log_interval == 0 or current == total:
        elapsed = time.time() - start_time
        rate = current / elapsed if elapsed > 0 else 0
        logging.info(
            f"Progress: {current}/{total} configurations "
            f"({rate:.1f} configs/s)"
        )


def process_xyz_input(args):
    """Load XYZ, convert to raw or preprocessed HDF5."""
    logging.info(f"Loading XYZ file: {args.input}")

    if args.mode == "raw":
        # Stream directly from XYZ → HDF5; no need to load all into RAM
        keyspec = create_keyspec_from_args(args)
        atoms_iter = ase.io.iread(args.input, index=":")
        save_atoms_to_hdf5(
            atoms_iter,
            args.output,
            key_specification=keyspec,
            description=args.description,
            batch_size=args.batch_size,
        )
        logging.info(f"Saved raw HDF5 to: {args.output}")

    elif args.mode == "preprocessed":
        # Load all structures first (needed for E0 computation)
        atoms_list = ase.io.read(args.input, index=":")
        logging.info(f"Loaded {len(atoms_list)} configurations")

        # Convert to Configurations
        keyspec = create_keyspec_from_args(args)
        configs = create_configs_from_list(
            atoms_list, keyspec, head_name="Default"
        )
        logging.info(f"Created {len(configs)} configurations")

        # Create z_table (use all 118 elements for compatibility)
        z_table = AtomicNumberTable([int(z) for z in range(1, 119)])
        logging.info(f"Using full atomic number table (Z=1-118)")

        # Compute E0s from configs
        atomic_energy_shifts = compute_and_format_e0s(configs, z_table)
        logging.info("Computed atomic energy shifts (E0s)")

        # Preprocess sequentially
        logging.info("Preprocessing configurations...")
        data_list = []
        start_time = time.time()
        for idx, config in enumerate(configs, start=1):
            data = AtomicData.from_config(
                config,
                z_table=z_table,
                cutoff=args.r_max,
                cutoff_lr=args.r_max_lr,
            )
            data_list.append(data)
            _log_progress(idx, len(configs), start_time)

        # Save
        logging.info("Saving...")
        save_preprocessed_hdf5(
            data_list,
            args.output,
            r_max=args.r_max,
            r_max_lr=args.r_max_lr,
            z_table=z_table,
            description=args.description,
            atomic_energy_shifts=atomic_energy_shifts,
        )
        logging.info(f"Saved preprocessed HDF5 to: {args.output}")


def process_hdf5_input(args):
    """Load raw HDF5, convert to preprocessed HDF5."""
    logging.info(f"Loading HDF5 file: {args.input}")

    # Detect format
    file_format = detect_file_format(args.input)
    logging.info(f"Detected format: {file_format}")

    if file_format == "hdf5_preprocessed":
        logging.error(
            "Input file is already preprocessed. "
            "Use a raw HDF5 or XYZ file."
        )
        sys.exit(1)

    if args.mode == "raw":
        logging.error(
            "Cannot convert HDF5 → raw HDF5. " "Use XYZ → raw HDF5 instead."
        )
        sys.exit(1)

    # Load configurations from HDF5
    keyspec = create_keyspec_from_args(args)
    configs = configs_from_hdf5(
        args.input, key_specification=keyspec, head_name="Default"
    )
    logging.info(f"Loaded {len(configs)} configurations")

    # Create z_table (use all 118 elements for compatibility)
    z_table = AtomicNumberTable([int(z) for z in range(1, 119)])
    logging.info(f"Using full atomic number table (Z=1-118)")

    # Compute E0s from configs
    atomic_energy_shifts = compute_and_format_e0s(configs, z_table)
    logging.info("Computed atomic energy shifts (E0s)")

    # Preprocess sequentially
    logging.info("Preprocessing configurations...")
    data_list = []
    start_time = time.time()
    for idx, config in enumerate(configs, start=1):
        data = AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=args.r_max,
            cutoff_lr=args.r_max_lr,
        )
        data_list.append(data)
        _log_progress(idx, len(configs), start_time)

    # Save
    save_preprocessed_hdf5(
        data_list,
        args.output,
        r_max=args.r_max,
        r_max_lr=args.r_max_lr,
        z_table=z_table,
        description=args.description,
        atomic_energy_shifts=atomic_energy_shifts,
    )
    logging.info(f"Saved preprocessed HDF5 to: {args.output}")


def create_keyspec_from_args(args) -> KeySpecification:
    """Create KeySpecification from command line arguments."""
    keydict = DefaultKeys.keydict()

    # Update with user-specified keys
    if args.energy_key:
        keydict["energy_key"] = args.energy_key
    if args.forces_key:
        keydict["forces_key"] = args.forces_key
    if args.stress_key:
        keydict["stress_key"] = args.stress_key
    if args.dipole_key:
        keydict["dipole_key"] = args.dipole_key
    if args.charges_key:
        keydict["charges_key"] = args.charges_key

    keyspec = KeySpecification()
    return update_keyspec_from_kwargs(keyspec, keydict)


def main():
    parser = argparse.ArgumentParser(
        description="Preprocess atomic structure data to HDF5 format"
    )
    parser.add_argument(
        "--input", type=str, required=True, help="Input file path"
    )
    parser.add_argument(
        "--output", type=str, required=True, help="Output HDF5 file path"
    )
    parser.add_argument(
        "--mode",
        type=str,
        choices=["raw", "preprocessed"],
        required=True,
        help="Output format: raw (structures only) or preprocessed "
        "(with neighbor lists)",
    )
    parser.add_argument(
        "--r-max",
        type=float,
        default=None,
        help="Short-range cutoff (required for preprocessed mode)",
    )
    parser.add_argument(
        "--r-max-lr",
        type=float,
        default=None,
        help="Long-range cutoff (optional)",
    )
    parser.add_argument(
        "--energy-key",
        type=str,
        default=None,
        help="Key for energy in XYZ (default: REF_energy)",
    )
    parser.add_argument(
        "--forces-key",
        type=str,
        default=None,
        help="Key for forces (default: REF_forces)",
    )
    parser.add_argument(
        "--stress-key",
        type=str,
        default=None,
        help="Key for stress (default: REF_stress)",
    )
    parser.add_argument(
        "--dipole-key",
        type=str,
        default=None,
        help="Key for dipole (default: REF_dipole)",
    )
    parser.add_argument(
        "--charges-key",
        type=str,
        default=None,
        help="Key for charges (default: REF_charges)",
    )
    parser.add_argument(
        "--description", type=str, default=None, help="Dataset description"
    )
    parser.add_argument(
        "--dtype",
        type=str,
        choices=["float32", "float64"],
        default="float64",
        help="Data type (default: float64)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Validate output after creation",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        help="Structures per write batch for raw mode (default: 100000). "
        "Controls peak RAM usage when streaming large XYZ files.",
    )

    args = parser.parse_args()

    # Validation
    if args.mode == "preprocessed" and args.r_max is None:
        parser.error("--r-max required for preprocessed mode")

    # Set PyTorch default dtype
    if args.dtype == "float32":
        torch.set_default_dtype(torch.float32)
    else:
        torch.set_default_dtype(torch.float64)

    # Execute
    if args.input.endswith(".xyz"):
        process_xyz_input(args)
    elif args.input.endswith((".h5", ".hdf5")):
        process_hdf5_input(args)
    else:
        parser.error(f"Unsupported input format: {args.input}")

    # Validation
    if args.validate and args.mode == "preprocessed":
        logging.info("Validating output...")
        try:
            validate_preprocessed_hdf5(
                args.output,
                expected_r_max=args.r_max,
                expected_r_max_lr=args.r_max_lr,
            )
            logging.info("Validation passed!")
        except Exception as e:
            logging.error(f"Validation failed: {e}")
            sys.exit(1)

    logging.info("Done!")


if __name__ == "__main__":
    main()
