"""
CLI Script for Merging Multiple HDF5 Files into One

Supports merging:
- Raw HDF5 v2.0 files (streamed, memory-efficient)
- Preprocessed HDF5 files (group copy with renumbering)

Usage::

    torchkrates-merge --inputs a.hdf5 b.hdf5 --output merged.hdf5
    torchkrates-merge --inputs a.h5 b.h5 c.h5 --output merged.h5 \\
        --description "combined dataset" --batch-size 50000
"""

import argparse
import logging
import sys

from so3krates_torch.config import MergeArgs
from so3krates_torch.data.hdf5_utils import (
    detect_file_format,
    merge_preprocessed_hdf5_files,
    merge_raw_hdf5_files,
)

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Merge multiple HDF5 dataset files into one.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        metavar="FILE",
        help="Two or more input HDF5 files to merge (must be same type).",
    )
    parser.add_argument(
        "--output",
        required=True,
        metavar="FILE",
        help="Output HDF5 file path.",
    )
    parser.add_argument(
        "--description",
        default=None,
        metavar="TEXT",
        help="Optional description stored in merged file metadata "
        "(raw format only).",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=100_000,
        metavar="N",
        help="Structures processed per write batch (raw format only).",
    )
    args = parser.parse_args()
    validated = MergeArgs.model_validate(vars(args))

    # Detect format from first input
    try:
        fmt = detect_file_format(args.inputs[0])
    except ValueError as e:
        parser.error(str(e))

    if fmt not in ("hdf5_raw", "hdf5_preprocessed"):
        parser.error(
            f"Unsupported format '{fmt}' for {args.inputs[0]}. "
            "Only HDF5 files can be merged."
        )

    # Validate all inputs share the same format
    for path in args.inputs[1:]:
        try:
            other_fmt = detect_file_format(path)
        except ValueError as e:
            parser.error(str(e))
        if other_fmt != fmt:
            parser.error(
                f"Format mismatch: {args.inputs[0]} is '{fmt}' "
                f"but {path} is '{other_fmt}'. "
                "All input files must be the same format."
            )

    try:
        if fmt == "hdf5_raw":
            merge_raw_hdf5_files(
                args.inputs,
                args.output,
                batch_size=args.batch_size,
                description=args.description,
            )
        else:
            merge_preprocessed_hdf5_files(args.inputs, args.output)
    except (ValueError, OSError) as e:
        logging.error(str(e))
        sys.exit(1)
