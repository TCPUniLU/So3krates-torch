# pylint: disable=wrong-import-position
import argparse
import os

os.environ["TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD"] = "1"

import torch
from ase.data import atomic_numbers as ase_atomic_numbers
from ase.data import chemical_symbols

from so3krates_torch.calculator.lammps_mliap_so3 import LAMMPS_MLIAP_SO3
from so3krates_torch.modules.models import MultiHeadSO3LR, SO3LR


def parse_args():
    parser = argparse.ArgumentParser(
        description="Convert a So3krates model for use with LAMMPS MLIAP interface.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "model_path",
        type=str,
        help="Path to the input model (.pt file)",
    )
    parser.add_argument(
        "--elements",
        type=str,
        nargs="+",
        required=True,
        help="Element symbols present in the simulation (e.g. Si O). "
        "Order must match LAMMPS pair_coeff types.",
    )
    parser.add_argument(
        "--head",
        type=str,
        nargs="?",
        help="Head name for multi-head models",
        default=None,
    )
    parser.add_argument(
        "--dtype",
        type=str,
        nargs="?",
        help="Data type for the converted model (float32 or float64)",
        default="float64",
    )
    return parser.parse_args()


def validate_elements(elements):
    """Validate and convert element symbols to atomic numbers."""
    z_list = []
    for elem in elements:
        if elem not in ase_atomic_numbers:
            raise ValueError(
                f"Unknown element symbol: '{elem}'. "
                f"Valid symbols: H, He, Li, Be, ..."
            )
        z_list.append(ase_atomic_numbers[elem])
    return z_list


def validate_model(model):
    """Validate model is compatible with short-range-only LAMMPS."""
    if not isinstance(model, (SO3LR, MultiHeadSO3LR)):
        raise ValueError(
            f"Model must be SO3LR or MultiHeadSO3LR, got {type(model).__name__}. "
            "Only SO3LR-family models are supported for LAMMPS MLIAP."
        )

    if getattr(model, "electrostatic_energy_bool", False):
        raise ValueError(
            "Model has electrostatic_energy_bool=True. "
            "LAMMPS MLIAP only supports short-range interactions (ML + ZBL). "
            "Retrain with electrostatic_energy_bool=False."
        )

    if getattr(model, "dispersion_energy_bool", False):
        raise ValueError(
            "Model has dispersion_energy_bool=True. "
            "LAMMPS MLIAP only supports short-range interactions (ML + ZBL). "
            "Retrain with dispersion_energy_bool=False."
        )

    zbl_status = (
        "enabled"
        if getattr(model, "zbl_repulsion_bool", False)
        else "disabled"
    )
    print(f"Model validation passed. ZBL repulsion: {zbl_status}")


def select_head(model):
    """Interactive head selection for multi-head models."""
    if hasattr(model, "heads"):
        heads = model.heads
    else:
        heads = [None]

    if len(heads) == 1:
        print(
            f"Only one head found in the model: {heads[0]}. Skipping selection."
        )
        return heads[0]

    print("Available heads in the model:")
    for i, head in enumerate(heads):
        print(f"  {i + 1}: {head}")

    selected = input(
        f"Select a head by number (default: {len(heads)}, press Enter to accept): "
    )

    if selected.isdigit() and 1 <= int(selected) <= len(heads):
        return heads[int(selected) - 1]
    if selected == "":
        print("No head selected. Proceeding without specifying a head.")
        return None
    print(f"No valid selection made. Defaulting to the last head: {heads[-1]}")
    return heads[-1]


def main():
    args = parse_args()
    model_path = args.model_path

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    # Validate elements
    atomic_numbers = validate_elements(args.elements)
    print(
        f"Elements: {' '.join(args.elements)} "
        f"(Z = {', '.join(str(z) for z in atomic_numbers)})"
    )

    # Load model
    print(f"Loading model from {model_path}...")
    model = torch.load(model_path, map_location=torch.device("cpu"))

    # Convert dtype
    if args.dtype == "float64":
        model = model.double().to("cpu")
    elif args.dtype == "float32":
        print("Converting model to float32. This may cause loss of precision.")
        model = model.float().to("cpu")
    else:
        raise ValueError(
            f"Invalid dtype: {args.dtype}. Must be 'float32' or 'float64'."
        )

    # Validate model configuration
    validate_model(model)

    # Select head for multi-head models
    if args.head is None:
        head = select_head(model)
    else:
        head = args.head
        if hasattr(model, "heads"):
            if head not in model.heads:
                raise ValueError(
                    f"Head '{head}' not found. Available heads: {model.heads}"
                )
            print(f"Selected head: {head}")

    # Create LAMMPS model
    kwargs = {"head": head} if head is not None else {}
    lammps_model = LAMMPS_MLIAP_SO3(
        model, atomic_numbers=atomic_numbers, **kwargs
    )

    # Save
    output_path = model_path + "-mliap_lammps.pt"
    torch.save(lammps_model, output_path)
    print(f"LAMMPS MLIAP model saved to: {output_path}")
    print(
        f"\nUsage in LAMMPS:\n"
        f"  pair_style mliap unified so3lr {output_path}\n"
        f"  pair_coeff * * {' '.join(args.elements)}"
    )


if __name__ == "__main__":
    main()
