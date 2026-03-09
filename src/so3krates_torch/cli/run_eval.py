import argparse

from so3krates_torch.config import EvalArgs
from so3krates_torch.tools.eval import evaluate_model, ensemble_prediction
from so3krates_torch.tools.utils import save_results_hdf5, save_results_xyz
from ase.io import read
import torch
from so3krates_torch.data.utils import KeySpecification
from pathlib import Path
import os


def run_evaluation(
    model_path: str,
    data_path: str,
    device: str = "cuda",
    batch_size: int = 5,
    model_type: str = "so3lr",
    r_max_lr: float = 12.0,
    multispecies: bool = False,
    multihead_model: bool = False,
    compute_dipole: bool = False,
    compute_stress: bool = False,
    compute_hirshfeld: bool = False,
    compute_partial_charges: bool = False,
    dispersion_energy_cutoff_lr_damping: float = 2.0,
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
    virials_key: str = "REF_virials",
    dipole_key: str = "REF_dipoles",
    charges_key: str = "REF_charges",
    total_charge_key: str = "charge",
    total_spin_key: str = "total_spin",
    hirshfeld_key: str = "REF_hirsh_ratios",
    head_key: str = "head",
    dtype: str = "float32",
    return_att: bool = False,
):
    """Load models from `model_path` (single .model or directory of .model),
    read data from `data_path`, run evaluation or ensemble prediction and
    return (result, is_ensemble).

    This function mirrors the original inline logic but uses explicit
    function arguments instead of relying on an argparse Namespace.
    """

    # set default dtype
    if dtype == "float32":
        torch.set_default_dtype(torch.float32)
    elif dtype == "float64":
        torch.set_default_dtype(torch.float64)
    else:
        raise ValueError(f"Unsupported dtype: {dtype}")

    # check if path ends with .model or is dir
    if model_path.endswith(".model"):
        model_paths = [model_path]
    else:
        model_paths = list(Path(model_path).glob("*.model"))

    models = []
    for mp in model_paths:
        model = torch.load(mp, map_location=device, weights_only=False).to(
            torch.float32 if dtype == "float32" else torch.float64
        )
        model.return_mean = False
        models.append(model)

    data = read(data_path, index=":")
    keyspec = KeySpecification(
        info_keys={
            "energy": energy_key,
            "dipole": dipole_key,
            "total_charge": total_charge_key,
            "total_spin": total_spin_key,
            "head": head_key,
        },
        arrays_keys={
            "forces": forces_key,
            "stress": stress_key,
            "virials": virials_key,
            "hirshfeld_ratios": hirshfeld_key,
            "charges": charges_key,
        },
    )

    if len(models) == 1:
        model = models[0]
        result = evaluate_model(
            atoms_list=data,
            model=model,
            batch_size=batch_size,
            device=device,
            model_type=model_type,
            multihead_model=multihead_model,
            r_max_lr=r_max_lr,
            dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
            dtype=dtype,
            multi_species=multispecies,
            compute_stress=compute_stress,
            compute_dipole=compute_dipole,
            compute_hirshfeld=compute_hirshfeld,
            compute_partial_charges=compute_partial_charges,
            return_att=return_att,
            key_spec=keyspec,
        )
    else:
        result = ensemble_prediction(
            models=models,
            atoms_list=data,
            device=device,
            model_type=model_type,
            dtype=dtype,
            batch_size=batch_size,
            multi_species=multispecies,
            r_max_lr=r_max_lr,
            dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
            compute_stress=compute_stress,
            compute_dipole=compute_dipole,
            compute_hirshfeld=compute_hirshfeld,
            compute_partial_charges=compute_partial_charges,
            key_spec=keyspec,
        )

    return result, len(models) > 1


def main():

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--model_path", type=str, required=True)
    argparser.add_argument("--data_path", type=str, required=True)
    argparser.add_argument("--output_file", type=str, default="results.h5")
    argparser.add_argument("--ensemble_size", type=int, default=1)
    argparser.add_argument("--device", type=str, default="cuda")
    argparser.add_argument("--batch_size", type=int, default=5)
    argparser.add_argument("--model_type", type=str, default="so3lr")
    argparser.add_argument("--r_max_lr", type=float, default=None)
    argparser.add_argument(
        "--multispecies", action="store_true", default=False
    )
    argparser.add_argument(
        "--multihead_model", action="store_true", default=False
    )
    argparser.add_argument(
        "--compute_dipole", action="store_true", default=False
    )
    argparser.add_argument(
        "--compute_stress", action="store_true", default=False
    )
    argparser.add_argument(
        "--compute_hirshfeld", action="store_true", default=False
    )
    argparser.add_argument(
        "--compute_partial_charges", action="store_true", default=False
    )
    argparser.add_argument(
        "--dispersion_energy_cutoff_lr_damping", type=float, default=2.0
    )
    argparser.add_argument(
        "--energy_key", type=str, help="Energy key", default="REF_energy"
    )
    argparser.add_argument(
        "--forces_key", type=str, help="Forces key", default="REF_forces"
    )
    argparser.add_argument(
        "--stress_key", type=str, help="Stress key", default="REF_stress"
    )
    argparser.add_argument(
        "--virials_key", type=str, help="Virials key", default="REF_virials"
    )
    argparser.add_argument(
        "--dipole_key", type=str, help="Dipole key", default="REF_dipoles"
    )
    argparser.add_argument(
        "--charges_key", type=str, help="Charges key", default="REF_charges"
    )
    argparser.add_argument(
        "--total_charge_key",
        type=str,
        help="Total charge key",
        default="charge",
    )
    argparser.add_argument(
        "--total_spin_key",
        type=str,
        help="Total spin key",
        default="total_spin",
    )
    argparser.add_argument(
        "--hirshfeld_key",
        type=str,
        help="Hirshfeld key",
        default="REF_hirsh_ratios",
    )
    argparser.add_argument(
        "--head_key", type=str, help="Head key", default="head"
    )
    argparser.add_argument("--head", type=str, help="Head key", default="head")
    argparser.add_argument("--dtype", type=str, default="float32")
    argparser.add_argument("--return_att", action="store_true")
    args = argparser.parse_args()
    validated = EvalArgs.model_validate(vars(args))
    # turn all args into variables
    model_path = validated.model_path
    data_path = validated.data_path
    output_file = validated.output_file
    ensemble_size = validated.ensemble_size
    device = validated.device
    batch_size = validated.batch_size
    model_type = validated.model_type
    r_max_lr = validated.r_max_lr
    multispecies = validated.multispecies
    multihead_model = validated.multihead_model
    compute_dipole = validated.compute_dipole
    compute_stress = validated.compute_stress
    compute_hirshfeld = validated.compute_hirshfeld
    compute_partial_charges = validated.compute_partial_charges
    dispersion_energy_cutoff_lr_damping = (
        validated.dispersion_energy_cutoff_lr_damping
    )
    energy_key = validated.energy_key
    forces_key = validated.forces_key
    stress_key = validated.stress_key
    virials_key = validated.virials_key
    dipole_key = validated.dipole_key
    charges_key = validated.charges_key
    total_charge_key = validated.total_charge_key
    total_spin_key = validated.total_spin_key
    hirshfeld_key = validated.hirshfeld_key
    head_key = validated.head_key
    dtype = args.dtype
    return_att = args.return_att

    result, is_ensemble = run_evaluation(
        model_path=model_path,
        data_path=data_path,
        device=device,
        batch_size=batch_size,
        model_type=model_type,
        r_max_lr=r_max_lr,
        multispecies=multispecies,
        multihead_model=multihead_model,
        compute_stress=compute_stress,
        compute_dipole=compute_dipole,
        compute_hirshfeld=compute_hirshfeld,
        compute_partial_charges=compute_partial_charges,
        dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
        energy_key=energy_key,
        forces_key=forces_key,
        stress_key=stress_key,
        virials_key=virials_key,
        dipole_key=dipole_key,
        charges_key=charges_key,
        total_charge_key=total_charge_key,
        total_spin_key=total_spin_key,
        hirshfeld_key=hirshfeld_key,
        head_key=head_key,
        dtype=dtype,
        return_att=return_att,
    )
    extension = os.path.splitext(output_file)[1].lower()
    if extension == ".h5" or extension == ".hdf5" or extension == "":
        save_results_hdf5(result, output_file, is_ensemble=is_ensemble)
    elif is_ensemble == False and extension == ".xyz":
        save_results_xyz(data_path, result, output_file)
    else:
        raise ValueError(
            f"Unsupported output file format: {extension} or ensemble results cannot be saved in .xyz"
        )
