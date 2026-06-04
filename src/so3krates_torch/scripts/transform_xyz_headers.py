"""Rename XYZ header keys in extended XYZ files.

Built-in default mappings (applied unless --no-defaults is given):
  hirsh_ratios        -> REF_hirsh_ratios
  forces              -> REF_forces
  energy              -> REF_energy
  dipole_total        -> REF_dipole_total
  dipole              -> REF_dipole

Custom mappings can be added or used exclusively via --map OLD=NEW.

Examples
--------
# Apply defaults only:
transform_xyz_headers.py train.xyz -o train_fixed.xyz

# Add a custom mapping on top of defaults:
transform_xyz_headers.py train.xyz -o train_fixed.xyz --map REF_hirshfeld_ratios=REF_hirsh_ratios

# Use ONLY a custom mapping (skip defaults):
transform_xyz_headers.py train.xyz -o train_fixed.xyz --no-defaults --map REF_hirshfeld_ratios=REF_hirsh_ratios

# Edit in-place:
transform_xyz_headers.py train.xyz --inplace

# Process a whole directory, writing results alongside originals with a suffix:
transform_xyz_headers.py data/ --suffix _fixed

# Process a whole directory, writing results to a separate directory:
transform_xyz_headers.py data/ -o data_fixed/
"""

import argparse
import os
import re
import sys
from pathlib import Path

DEFAULT_MAPPINGS = {
    "hirsh_ratios": "REF_hirsh_ratios",
    "forces": "REF_forces",
    "energy": "REF_energy",
    "dipole_total": "REF_dipole_total",
    "dipole": "REF_dipole",
}


def apply_mappings(text: str, mappings: dict) -> str:
    for old, new in mappings.items():
        text = re.sub(rf"\b{re.escape(old)}\b", new, text)
    return text


def resolve_inputs(paths: list[str]) -> list[Path]:
    inputs = []
    for p in paths:
        path = Path(p)
        if path.is_dir():
            inputs.extend(sorted(path.glob("*.xyz")))
        elif path.is_file():
            inputs.append(path)
        else:
            print(f"Warning: {p} does not exist, skipping.", file=sys.stderr)
    return inputs


def resolve_output(
    input_path: Path,
    output_arg: str | None,
    inplace: bool,
    suffix: str,
    multi: bool,
) -> Path:
    if inplace:
        return input_path
    if output_arg:
        out = Path(output_arg)
        if multi or out.is_dir():
            out.mkdir(parents=True, exist_ok=True)
            return out / input_path.name
        return out
    # No output specified: write next to input with suffix
    return input_path.with_stem(input_path.stem + suffix)


def main():
    parser = argparse.ArgumentParser(
        description="Rename header keys in extended XYZ files.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "inputs",
        nargs="+",
        metavar="FILE_OR_DIR",
        help="One or more XYZ files or directories of XYZ files.",
    )
    parser.add_argument(
        "-o", "--output",
        default=None,
        metavar="PATH",
        help=(
            "Output file (single input) or directory (multiple inputs). "
            "Defaults to writing alongside the input with --suffix applied."
        ),
    )
    parser.add_argument(
        "--inplace",
        action="store_true",
        help="Edit files in-place (overrides --output and --suffix).",
    )
    parser.add_argument(
        "--suffix",
        default="_fixed",
        metavar="SUFFIX",
        help=(
            "Suffix appended to the stem when writing alongside the input "
            "(default: '_fixed'). Ignored when --output or --inplace is used."
        ),
    )
    parser.add_argument(
        "--map",
        action="append",
        default=[],
        metavar="OLD=NEW",
        help=(
            "Additional mapping in OLD=NEW format. "
            "Can be repeated. Applied after the built-in defaults."
        ),
    )
    parser.add_argument(
        "--no-defaults",
        action="store_true",
        help="Skip the built-in default mappings; use only --map entries.",
    )
    args = parser.parse_args()

    # Parse --map entries
    custom = {}
    for entry in args.map:
        if "=" not in entry:
            parser.error(f"--map value must be OLD=NEW, got: {entry!r}")
        old, new = entry.split("=", 1)
        if not old or not new:
            parser.error(f"--map value must be non-empty OLD=NEW, got: {entry!r}")
        custom[old.strip()] = new.strip()

    mappings = {} if args.no_defaults else dict(DEFAULT_MAPPINGS)
    mappings.update(custom)

    if not mappings:
        parser.error("No mappings to apply. Provide --map or remove --no-defaults.")

    print("Mappings:")
    for old, new in mappings.items():
        print(f"  {old!r} -> {new!r}")

    inputs = resolve_inputs(args.inputs)
    if not inputs:
        print("No XYZ files found.", file=sys.stderr)
        sys.exit(1)

    multi = len(inputs) > 1

    for input_path in inputs:
        output_path = resolve_output(
            input_path, args.output, args.inplace, args.suffix, multi
        )
        text = input_path.read_text()
        new_text = apply_mappings(text, mappings)
        replacements_made = text != new_text
        output_path.write_text(new_text)
        status = "modified" if replacements_made else "unchanged"
        print(f"  {input_path} -> {output_path} ({status})")


if __name__ == "__main__":
    main()
