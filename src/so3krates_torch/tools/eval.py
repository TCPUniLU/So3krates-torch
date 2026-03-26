import logging
import time

import numpy as np
import torch
from ase.io import read
from typing import Tuple, List, Union, Optional
from so3krates_torch.data.utils import KeySpecification
from so3krates_torch.tools import torch_geometric, torch_tools
from so3krates_torch.tools.torch_tools import to_numpy
from so3krates_torch.tools.utils import (
    MetricsLogger,
    compute_mae,
    compute_q95,
    compute_rel_mae,
    compute_rel_rmse,
    compute_rmse,
)
from so3krates_torch.tools.utils import create_dataloader_from_list
from torchmetrics import Metric


def evaluate_model(
    atoms_list: list,
    model: str,
    batch_size: int,
    device: str,
    model_type: str,
    r_max_lr: float = 12.0,
    multi_species: bool = False,
    multihead_model: bool = False,
    dispersion_energy_cutoff_lr_damping: Optional[float] = None,
    compute_stress: bool = False,
    compute_hirshfeld: bool = False,
    compute_dipole: bool = False,
    compute_partial_charges: bool = False,
    return_att: bool = False,
    dtype: str = "float64",
    key_spec: Optional[KeySpecification] = None,
) -> dict[str, Union[np.ndarray, List[np.ndarray]]]:
    """
    Evaluate a model on a list of ASE atoms objects.

    Args:
        atoms_list (list): List of ASE atoms objects to evaluate on.
        model (torch.nn.Module): Trained model to evaluate.
        batch_size (int): Batch size for evaluation.
        device (str): Device to evaluate the model on (e.g., "cpu", "cuda").
        model_type (str): Type of model ("so3lr", "so3krates").
        r_max_lr (float, optional): Long-range cutoff radius. Defaults to 12.0.
        multi_species (bool, optional): Whether molecules have different
                                       numbers of atoms. Defaults to False.
        dispersion_energy_cutoff_lr_damping (Optional[float], optional):
                                       Dispersion cutoff damping parameter.
                                       Defaults to None.
        compute_stress (bool, optional): Whether to compute stress tensors.
                                        Defaults to False.
        compute_hirshfeld (bool, optional): Whether to compute Hirshfeld
                                           ratios. Defaults to False.
        compute_dipole (bool, optional): Whether to compute dipole moments.
                                        Defaults to False.
        compute_partial_charges (bool, optional): Whether to compute partial
                                                 charges. Defaults to False.
        dtype (str, optional): Data type for model computations.
                              Defaults to "float64".

    Returns:
        dict[str, Union[np.ndarray, List[np.ndarray]]]: Dictionary containing
                                                        evaluation results:
            - "energies": Predicted energies
            - "forces": Predicted forces
            - "stresses": Predicted stress tensors (if compute_stress=True,
                         else None)
            - "dipoles": Predicted dipole vectors (if compute_dipole=True,
                            else None)
            - "hirshfeld_ratios": Predicted Hirshfeld ratios (if
                                 compute_hirshfeld=True, else None)
            - "partial_charges": Predicted partial charges (if
                                compute_partial_charges=True, else None)

            For multi_species=False: Results are numpy arrays with consistent
                                    shapes.
            For multi_species=True: Results are lists of arrays with variable
                                   shapes.
    """

    torch_tools.set_default_dtype(dtype)

    assert model_type.lower() in [
        "so3lr",
        "so3krates",
    ], f"Unknown model type: {model_type}"
    key_spec = KeySpecification() if key_spec is None else key_spec

    data_loader = create_dataloader_from_list(
        atoms_list=atoms_list,
        batch_size=batch_size,
        r_max=model.r_max,
        r_max_lr=r_max_lr,
        shuffle=False,
        drop_last=False,
        key_specification=key_spec,
    )

    # Collect data
    energies_list = []
    forces_list = []

    if compute_stress:
        stresses_list = []
    if compute_dipole:
        dipoles_list = []
    if compute_hirshfeld:
        hirshfeld_ratios_list = []
    if compute_partial_charges:
        partial_charges_list = []
    if return_att:
        att_scores_list = []

    if model_type == "so3lr":
        model.r_max_lr = r_max_lr
        model.dispersion_energy_cutoff_lr_damping = (
            dispersion_energy_cutoff_lr_damping
        )
    model = model.eval()

    for batch in data_loader:
        batch = batch.to(device)
        output = model(
            batch.to_dict(),
            compute_stress=compute_stress,
            return_att=return_att,
        )
        energies = torch_tools.to_numpy(output["energy"])

        energies = [energy for energy in energies]
        energies_list += energies
        if compute_stress:
            stresses = torch_tools.to_numpy(output["stress"])
            stresses = [stress for stress in stresses]
            stresses_list += stresses
        if compute_dipole:
            dipoles = torch_tools.to_numpy(output["dipole"])
            dipoles = [dipole for dipole in dipoles]
            dipoles_list += dipoles
        if compute_hirshfeld:
            hirshfeld = torch_tools.to_numpy(output["hirshfeld_ratios"])
            hirshfeld = np.split(
                hirshfeld,
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )[:-1]
            hirshfeld_temp_list = [h for h in hirshfeld]
            hirshfeld_ratios_list += hirshfeld_temp_list
        if compute_partial_charges:
            partial_charges = torch_tools.to_numpy(output["partial_charges"])
            partial_charges = np.split(
                partial_charges,
                indices_or_sections=batch.ptr[1:],
                axis=0,
            )[:-1]
            partial_charges_temp_list = [charge for charge in partial_charges]
            partial_charges_list += partial_charges_temp_list
        if return_att:
            att_scores = output["att_scores"]
            layers = range(len(att_scores["inv"]))
            # Create batch assignment for edges
            edge_batch = batch.batch[
                batch.edge_index[0]
            ]  # Assign each edge to its source node's batch
            for i in range(len(batch.ptr) - 1):
                new_att_dict = {"ev": {}, "inv": {}}
                # Get mask for edges belonging to graph i
                edge_mask = edge_batch == i
                for layer in layers:
                    new_att_dict["inv"][layer] = att_scores["inv"][layer][
                        edge_mask
                    ]
                    new_att_dict["ev"][layer] = att_scores["ev"][layer][
                        edge_mask
                    ]
                    # add senders and receivers starting from 0
                    senders = batch.edge_index[0][edge_mask] - batch.ptr[i]
                    receivers = batch.edge_index[1][edge_mask] - batch.ptr[i]
                    new_att_dict["senders"] = senders
                    new_att_dict["receivers"] = receivers
                att_scores_list.append(new_att_dict)

        forces = np.split(
            torch_tools.to_numpy(output["forces"]),
            indices_or_sections=batch.ptr[1:],
            axis=0 if not multihead_model else 1,
        )[:-1]
        forces = [force for force in forces]
        forces_list += forces

    if multi_species or multihead_model:
        energies = energies_list
        forces = forces_list
        if compute_stress:
            stresses = stresses_list
        if compute_dipole:
            dipoles = dipoles_list
        if compute_hirshfeld:
            hirshfeld = hirshfeld_ratios_list
        if compute_partial_charges:
            partial_charges = partial_charges_list
    else:
        energies = np.stack(energies_list)
        forces = np.stack(forces_list).reshape(len(energies), -1, 3)
        if compute_stress:
            stresses = np.stack(stresses_list).reshape(len(energies), -1, 3, 3)
        if compute_dipole:
            dipoles = np.stack(dipoles_list).reshape(len(energies), -1, 3)
        if compute_hirshfeld:
            hirshfeld = np.stack(hirshfeld_ratios_list).reshape(
                len(energies), -1, 1
            )
        if compute_partial_charges:
            partial_charges = np.stack(partial_charges_list).reshape(
                len(energies), -1, 1
            )

    results = {
        "energies": energies,
        "forces": forces,
        "stresses": stresses if compute_stress else None,
        "dipoles": dipoles if compute_dipole else None,
        "hirshfeld_ratios": hirshfeld if compute_hirshfeld else None,
        "partial_charges": (
            partial_charges if compute_partial_charges else None
        ),
        "att_scores": (att_scores_list if return_att else None),
    }

    return results


def ensemble_prediction(
    models: list,
    atoms_list: list,
    device: str,
    model_type: str,
    dtype: str = "float64",
    batch_size: int = 1,
    multi_species: bool = False,
    multihead_model: bool = False,
    r_max_lr: float = 12.0,
    dispersion_energy_cutoff_lr_damping: Optional[float] = None,
    compute_stress: bool = False,
    compute_hirshfeld: bool = False,
    compute_dipole: bool = False,
    compute_partial_charges: bool = False,
    key_spec: Optional[KeySpecification] = None,
) -> np.array:
    """
    Generate ensemble predictions for a list of ASE atoms objects using
    multiple models.

    This function evaluates an ensemble of models on the same dataset and
    returns predictions from all models. The results can be used for
    uncertainty quantification or ensemble averaging.

    Args:
        models (list): List of trained models (torch.nn.Module objects).
        atoms_list (list): List of ASE atoms objects to evaluate on.
        device (str): Device to evaluate the models on (e.g., "cpu", "cuda").
        model_type (str): Type of model ("so3lr", "so3krates").
        dtype (str, optional): Data type for model computations.
                              Defaults to "float64".
        batch_size (int, optional): Batch size for evaluation. Defaults to 1.
        multi_species (bool, optional): Whether molecules have different
                                       numbers of atoms. Defaults to False.
        r_max_lr (float, optional): Long-range cutoff radius. Defaults to 12.0.
        dispersion_energy_cutoff_lr_damping (Optional[float], optional):
                                       Dispersion cutoff damping parameter.
                                       Defaults to None.
        compute_stress (bool, optional): Whether to compute stress tensors.
                                        Defaults to False.
        compute_hirshfeld (bool, optional): Whether to compute Hirshfeld
                                           ratios. Defaults to False.
        compute_dipole (bool, optional): Whether to compute dipole moments.
                                        Defaults to False.
        compute_partial_charges (bool, optional): Whether to compute partial
                                                 charges. Defaults to False.

    Returns:
        dict: Dictionary containing ensemble predictions with the following
              keys:
            - 'energies': Energy predictions from all models
            - 'forces': Force predictions from all models
            - 'stresses': Stress predictions (if compute_stress=True,
                         else None)
            - 'hirshfeld_ratios': Hirshfeld ratio predictions (if
                                 compute_hirshfeld=True, else None)
            - 'dipoles': Dipole vector predictions (if compute_dipole=True,
                            else None)
            - 'partial_charges': Partial charge predictions (if
                                compute_partial_charges=True, else None)

        Shape of arrays depends on multi_species flag:
            - If multi_species=False: numpy arrays with shape
              [n_models, n_molecules, ...]
            - If multi_species=True: nested lists with variable shapes
              [n_models][n_molecules] where each molecule can have different
              numbers of atoms
    """
    all_forces = []
    all_energies = []
    if compute_stress:
        all_stresses = []
    if compute_hirshfeld:
        all_hirshfeld = []
    if compute_dipole:
        all_dipoles = []
    if compute_partial_charges:
        all_partial_charges = []
    key_spec = KeySpecification() if key_spec is None else key_spec
    i = 0
    for model in models:
        results = evaluate_model(
            atoms_list=atoms_list,
            model=model,
            batch_size=batch_size,
            device=device,
            model_type=model_type,
            multi_species=multi_species,
            r_max_lr=r_max_lr,
            dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
            compute_stress=compute_stress,
            compute_hirshfeld=compute_hirshfeld,
            compute_dipole=compute_dipole,
            compute_partial_charges=compute_partial_charges,
            dtype=dtype,
            key_spec=key_spec,
        )
        all_forces.append(results["forces"])
        all_energies.append(results["energies"])
        if compute_stress:
            all_stresses.append(results["stresses"])
        if compute_hirshfeld:
            all_hirshfeld.append(results["hirshfeld_ratios"])
        if compute_dipole:
            all_dipoles.append(results["dipoles"])
        if compute_partial_charges:
            all_partial_charges.append(results["partial_charges"])
        i += 1

    if not multi_species and not multihead_model:
        # Stack for fixed-size molecules
        all_forces = np.stack(all_forces).reshape(
            (len(models), len(atoms_list), -1, 3)
        )

        all_energies = np.stack(all_energies).reshape(
            (len(models), len(atoms_list))
        )
        if compute_stress:
            all_stresses = np.stack(all_stresses).reshape(
                (len(models), len(atoms_list), -1, 3, 3)
            )
        if compute_dipole:
            all_dipoles = np.stack(all_dipoles).reshape(
                (len(models), len(atoms_list), -1, 3)
            )
        if compute_hirshfeld:
            all_hirshfeld = np.stack(all_hirshfeld).reshape(
                (len(models), len(atoms_list), -1)
            )
        if compute_partial_charges:
            all_partial_charges = np.stack(all_partial_charges).reshape(
                (len(models), len(atoms_list), -1)
            )

    results = {
        "energies": all_energies,
        "forces": all_forces,
        "stresses": all_stresses if compute_stress else None,
        "hirshfeld_ratios": all_hirshfeld if compute_hirshfeld else None,
        "dipoles": all_dipoles if compute_dipole else None,
        "partial_charges": (
            all_partial_charges if compute_partial_charges else None
        ),
    }

    return results


class ModelEval(Metric):
    def __init__(self, loss_fn: Optional[torch.nn.Module] = None) -> None:
        super().__init__()
        self.loss_fn = loss_fn

        self.add_state(
            "total_loss", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "num_data", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state(
            "E_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_es", default=[], dist_reduce_fx="cat")
        self.add_state("delta_es_per_atom", default=[], dist_reduce_fx="cat")
        self.add_state(
            "Fs_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("fs", default=[], dist_reduce_fx="cat")
        self.add_state("delta_fs", default=[], dist_reduce_fx="cat")
        self.add_state(
            "stress_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_stress", default=[], dist_reduce_fx="cat")
        self.add_state(
            "delta_stress_per_atom", default=[], dist_reduce_fx="cat"
        )
        self.add_state(
            "virials_computed", default=torch.tensor(0.0), dist_reduce_fx="sum"
        )
        self.add_state("delta_virials", default=[], dist_reduce_fx="cat")
        self.add_state(
            "delta_virials_per_atom", default=[], dist_reduce_fx="cat"
        )
        self.add_state(
            "dipoles_computed",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("dipoles", default=[], dist_reduce_fx="cat")
        self.add_state("delta_dipoles", default=[], dist_reduce_fx="cat")

        self.add_state(
            "hirshfeld_ratios_computed",
            default=torch.tensor(0.0),
            dist_reduce_fx="sum",
        )
        self.add_state("hirshfeld_ratios", default=[], dist_reduce_fx="cat")
        self.add_state(
            "delta_hirshfeld_ratios", default=[], dist_reduce_fx="cat"
        )
        self.add_state(
            "delta_hirshfeld_ratios_per_atom", default=[], dist_reduce_fx="cat"
        )

    def update(self, batch, output):  # pylint: disable=arguments-differ
        if self.loss_fn is not None:
            loss = self.loss_fn(pred=output, ref=batch)
            self.total_loss += loss
        self.num_data += batch.num_graphs

        if output.get("energy") is not None and batch.energy is not None:
            self.E_computed += 1.0
            self.delta_es.append(batch.energy - output["energy"].squeeze())
            self.delta_es_per_atom.append(
                (batch.energy - output["energy"])
                / (batch.ptr[1:] - batch.ptr[:-1])
            )
        if output.get("forces") is not None and batch.forces is not None:
            self.Fs_computed += 1.0
            self.fs.append(batch.forces)
            self.delta_fs.append(batch.forces - output["forces"])
        if output.get("stress") is not None and batch.stress is not None:
            self.stress_computed += 1.0
            self.delta_stress.append(batch.stress - output["stress"])
            self.delta_stress_per_atom.append(
                (batch.stress - output["stress"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("virials") is not None and batch.virials is not None:
            self.virials_computed += 1.0
            self.delta_virials.append(batch.virials - output["virials"])
            self.delta_virials_per_atom.append(
                (batch.virials - output["virials"])
                / (batch.ptr[1:] - batch.ptr[:-1]).view(-1, 1, 1)
            )
        if output.get("dipole") is not None and batch.dipole is not None:
            self.dipoles_computed += 1.0
            self.dipoles.append(batch.dipole)
            self.delta_dipoles.append(batch.dipole - output["dipole"])

        if (
            output.get("hirshfeld_ratios") is not None
            and batch.hirshfeld_ratios is not None
        ):
            self.hirshfeld_ratios_computed += 1.0
            self.hirshfeld_ratios.append(batch.hirshfeld_ratios)
            self.delta_hirshfeld_ratios.append(
                batch.hirshfeld_ratios - output["hirshfeld_ratios"]
            )

    def convert(
        self, delta: Union[torch.Tensor, List[torch.Tensor]]
    ) -> np.ndarray:
        if isinstance(delta, list):
            delta = torch.cat(delta)
        return torch_tools.to_numpy(delta)

    def compute(self):
        aux = {}
        if self.loss_fn is not None:
            aux["loss"] = to_numpy(self.total_loss / self.num_data).item()
        if self.E_computed:
            delta_es = self.convert(self.delta_es)
            delta_es_per_atom = self.convert(self.delta_es_per_atom)
            aux["mae_e"] = compute_mae(delta_es)
            aux["mae_e_per_atom"] = compute_mae(delta_es_per_atom)
            aux["rmse_e"] = compute_rmse(delta_es)
            aux["rmse_e_per_atom"] = compute_rmse(delta_es_per_atom)
            aux["q95_e"] = compute_q95(delta_es)

        if self.Fs_computed:
            fs = self.convert(self.fs)
            delta_fs = self.convert(self.delta_fs)
            aux["mae_f"] = compute_mae(delta_fs)
            aux["rel_mae_f"] = compute_rel_mae(delta_fs, fs)
            aux["rmse_f"] = compute_rmse(delta_fs)
            aux["rel_rmse_f"] = compute_rel_rmse(delta_fs, fs)
            aux["q95_f"] = compute_q95(delta_fs)

        if self.stress_computed:
            delta_stress = self.convert(self.delta_stress)
            delta_stress_per_atom = self.convert(self.delta_stress_per_atom)
            aux["mae_stress"] = compute_mae(delta_stress)
            aux["rmse_stress"] = compute_rmse(delta_stress)
            aux["rmse_stress_per_atom"] = compute_rmse(delta_stress_per_atom)
            aux["q95_stress"] = compute_q95(delta_stress)

        if self.virials_computed:
            delta_virials = self.convert(self.delta_virials)
            delta_virials_per_atom = self.convert(self.delta_virials_per_atom)
            aux["mae_virials"] = compute_mae(delta_virials)
            aux["rmse_virials"] = compute_rmse(delta_virials)
            aux["rmse_virials_per_atom"] = compute_rmse(delta_virials_per_atom)
            aux["q95_virials"] = compute_q95(delta_virials)

        if self.dipoles_computed:
            delta_dipoles = self.convert(self.delta_dipoles)
            aux["mae_dipole"] = compute_mae(delta_dipoles)
            aux["rmse_dipole"] = compute_rmse(delta_dipoles)
            aux["q95_dipole"] = compute_q95(delta_dipoles)

        if self.hirshfeld_ratios_computed:
            delta_hirshfeld_ratios = self.convert(self.delta_hirshfeld_ratios)
            aux["mae_hirshfeld_ratios"] = compute_mae(delta_hirshfeld_ratios)
            aux["rmse_hirshfeld_ratios"] = compute_rmse(delta_hirshfeld_ratios)
            aux["q95_hirshfeld_ratios"] = compute_q95(delta_hirshfeld_ratios)

        if self.loss_fn is not None:
            return aux["loss"], aux
        else:
            return aux


def test_model(
    model: torch.nn.Module,
    data_loader: torch_geometric.DataLoader,
    output_args: dict,
    device: str,
    return_predictions: bool = False,
    log: bool = False,
) -> dict:
    """
    Function to test a model on a set of configurations.

    Args:
        model (torch.nn.Module): Model to test.
        data_loader (torch_geometric.DataLoader): DataLoader for the test set.
        output_args (dict): Dictionary of output arguments for the model.
        device (str): Device to run the model on (e.g., "cpu" or "cuda").
        return_predictions (bool, optional): Whether to return the predictions
            made during the test. Defaults to False.

    Returns:
        dict: Dictionary with the computed metrics.
    """
    for param in model.parameters():
        param.requires_grad = False

    metrics = ModelEval().to(device)
    start_time = time.time()
    if return_predictions:
        predictions = {}
        if output_args.get("energy", False):
            predictions["energy"] = []
        if output_args.get("forces", False):
            predictions["forces"] = []
        if output_args.get("stress", False):
            predictions["stress"] = []
        if output_args.get("virials", False):
            predictions["virials"] = []
        if output_args.get("dipole", False):
            predictions["dipole"] = []
        if output_args.get("hirshfeld_ratios", False):
            predictions["hirshfeld_ratios"] = []

    num_batches = len(data_loader)
    if log:
        logging.info(f"Testing model on {num_batches} batches")

    for i, batch in enumerate(data_loader):
        batch = batch.to(device)
        batch_dict = batch.to_dict()
        output = model(
            batch_dict,
            training=False,
            compute_force=output_args["forces"],
            compute_virials=output_args["virials"],
            compute_stress=output_args["stress"],
        )
        if log:
            logging.info(f"Batch no. {i + 1}/{num_batches} - done!")
        aux = metrics(batch, output)

        if return_predictions:
            if output_args.get("energy", False):
                predictions["energy"].append(output["energy"])
            if output_args.get("forces", False):
                predictions["forces"].append(output["forces"])
            if output_args.get("stress", False):
                predictions["stress"].append(output["stress"])
            if output_args.get("virials", False):
                predictions["virials"].append(output["virials"])
            if output_args.get("dipole", False):
                predictions["dipole"].append(output["dipole"])
            if output_args.get("hirshfeld_ratios", False):
                predictions["hirshfeld_ratios"].append(
                    output["hirshfeld_ratios"]
                )

    aux = metrics.compute()
    aux["time"] = time.time() - start_time
    metrics.reset()

    for param in model.parameters():
        param.requires_grad = True
    if return_predictions:
        for key in predictions.keys():
            predictions[key] = (
                torch.cat(predictions[key], dim=0).detach().cpu()
            )
        aux["predictions"] = predictions
    return aux


def test_ensemble(
    ensemble: dict,
    batch_size: int,
    output_args: dict,
    device: str,
    path_to_data: str = None,
    atoms_list: list = None,
    head_name: str = "head",
    logger: MetricsLogger = None,
    log_errors: str = "PerAtomMAE",
    return_predictions: bool = False,
    energy_key: str = "REF_energy",
    forces_key: str = "REF_forces",
    stress_key: str = "REF_stress",
    virials_key: str = "REF_virials",
    dipole_key: str = "REF_dipole",
    hirshfeld_key: str = "REF_hirsh_ratios",
    charges_key: str = "REF_charges",
    total_charge_key: str = "charge",
    total_spin_key: str = "spin",
    head_key: str = "head",
    r_max_lr: float = 12.0,
    log: bool = False,
) -> Tuple[dict, dict]:
    """
    Function to test an ensemble of models on a set of configurations.

    Either `atoms_list` or `path_to_data` must be provided. The data
    is either loaded from file or the atoms list is used directly.

    Args:
        ensemble (dict): Dictionary of models to test.
        batch_size (int): Batch size for testing the models.
        output_args (dict): Dictionary of output arguments for the models.
        device (str): Device to run the models on (e.g., "cpu" or "cuda").
        path_to_data (str, optional): Path to the data file in ASE readable
                                     format. Defaults to None.
        atoms_list (list, optional): List of ASE atoms objects. Defaults to None.
        logger (MetricsLogger, optional): Logger object for eval. Defaults to None.
        log_errors (str, optional): What error to log. Defaults to "PerAtomMAE".
        return_predictions (bool, optional): Whether to return the predictions
                                           made during the test. Defaults to False.
        energy_key (str, optional): How energy is defined in the ase.Atoms.
                                   Defaults to "REF_energy".
        forces_key (str, optional): How forces are defined in the ase.Atoms.
                                   Defaults to "REF_forces".
        stress_key (str, optional): How stress is defined in the ase.Atoms.
                                   Defaults to "REF_stress".
        virials_key (str, optional): How virials are defined in the ase.Atoms.
                                    Defaults to "virials".
        dipole_key (str, optional): How dipoles are defined in the ase.Atoms.
                                   Defaults to "REF_dipole".
        hirshfeld_key (str, optional): How Hirshfeld ratios are defined in the ase.Atoms.
                                      Defaults to "REF_hirsh_ratios".
        charges_key (str, optional): How charges are defined in the ase.Atoms.
                                    Defaults to "charges".
        head_key (str, optional): Which output head to test. Defaults to "head".
        r_max_lr (float, optional): Long-range cutoff radius. Defaults to 12.0.
        log (bool, optional): Whether to log progress. Defaults to False.

    Raises:
        ValueError: Raises an error if neither `atoms_list` nor `path_to_data`
                   is provided.

    Returns:
        Tuple[dict, dict]: Average metrics and ensemble metrics.
    """
    if atoms_list is not None:
        data_to_use = atoms_list
    elif path_to_data is not None:
        data_to_use = read(path_to_data, index=":")
    else:
        raise ValueError("Either atoms_list or path_to_data must be provided")

    # Create KeySpecification from the provided keys
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

    # Get reference model for r_max
    reference_model = ensemble[list(ensemble.keys())[0]]

    # Create dataloader using the create_dataloader function
    data_loader = create_dataloader_from_list(
        atoms_list=data_to_use,
        batch_size=batch_size,
        r_max=reference_model.r_max,
        r_max_lr=r_max_lr,
        key_specification=keyspec,
        shuffle=False,
        drop_last=False,
        head_name=head_name,
    )

    ensemble_metrics = {}
    for tag, model in ensemble.items():
        metrics = test_model(
            model=model,
            data_loader=data_loader,
            output_args=output_args,
            device=device,
            return_predictions=return_predictions,
            log=log,
        )
        ensemble_metrics[tag] = metrics

    avg_ensemble_metrics = {}
    for key in ensemble_metrics[list(ensemble_metrics.keys())[0]].keys():
        if key not in ["mode", "epoch", "predictions"]:
            avg_ensemble_metrics[key] = np.mean(
                [m[key] for m in ensemble_metrics.values()]
            )
        if return_predictions:
            avg_ensemble_metrics["predictions"] = {
                key: np.mean(
                    [m["predictions"][key] for m in ensemble_metrics.values()],
                    axis=0,
                )
                for key in ensemble_metrics[list(ensemble_metrics.keys())[0]][
                    "predictions"
                ].keys()
            }
    if logger is not None:
        logger.log(avg_ensemble_metrics)
        if log_errors == "PerAtomRMSE":
            error_e = avg_ensemble_metrics["rmse_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            logging.info(
                f"RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f}"
                "meV / A"
            )
        elif (
            log_errors == "PerAtomRMSEstressvirials"
            and avg_ensemble_metrics["rmse_stress_per_atom"] is not None
        ):
            error_e = avg_ensemble_metrics["rmse_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            error_stress = avg_ensemble_metrics["rmse_stress_per_atom"] * 1e3
            logging.info(
                f"RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f} "
                f"meV / A, RMSE_stress_per_atom={error_stress:.1f} meV / A^3"
            )
        elif (
            log_errors == "PerAtomRMSEstressvirials"
            and avg_ensemble_metrics["rmse_virials_per_atom"] is not None
        ):
            error_e = avg_ensemble_metrics["rmse_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            error_virials = avg_ensemble_metrics["rmse_virials_per_atom"] * 1e3
            logging.info(
                f"RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f}"
                f"meV / A, RMSE_virials_per_atom={error_virials:.1f} meV"
            )
        elif log_errors == "TotalRMSE":
            error_e = avg_ensemble_metrics["rmse_e"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            logging.info(
                f"RMSE_E={error_e:.1f} meV, RMSE_F={error_f:.1f} meV / A"
            )
        elif log_errors == "PerAtomMAE":
            error_e = avg_ensemble_metrics["mae_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["mae_f"] * 1e3
            logging.info(
                f"MAE_E_per_atom={error_e:.1f} meV, MAE_F={error_f:.1f}"
                "meV / A"
            )
        elif log_errors == "TotalMAE":
            error_e = avg_ensemble_metrics["mae_e"] * 1e3
            error_f = avg_ensemble_metrics["mae_f"] * 1e3
            logging.info(
                f"MAE_E={error_e:.1f} meV, MAE_F={error_f:.1f} meV / A"
            )
        elif log_errors == "DipoleRMSE":
            error_mu = avg_ensemble_metrics["rmse_dipole"] * 1e3
            logging.info(f"RMSE_dipole={error_mu:.2f} mDebye")
        elif log_errors == "EnergyDipoleRMSE":
            error_e = avg_ensemble_metrics["rmse_e_per_atom"] * 1e3
            error_f = avg_ensemble_metrics["rmse_f"] * 1e3
            error_mu = avg_ensemble_metrics["rmse_dipole"] * 1e3
            logging.info(
                f"RMSE_E_per_atom={error_e:.1f} meV, RMSE_F={error_f:.1f}"
                f"meV / A, RMSE_dipole={error_mu:.2f} mDebye"
            )

    return (avg_ensemble_metrics, ensemble_metrics)
