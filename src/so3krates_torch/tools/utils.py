# Taken from MACE package: https://github.com/ACEsuit/mace
# and modified for SO3Krates, SO3LR

from __future__ import annotations

import json
import logging
import os
import sys
from pathlib import Path
from typing import (
    Any,
    Dict,
    Iterable,
    List,
    Optional,
    Sequence,
    Tuple,
    Union,
)

import h5py
import numpy as np
import torch
from ase.io import read, write

from so3krates_torch.tools.torch_geometric import DataLoader

from .torch_tools import to_numpy


def compute_forces(
    energy: torch.Tensor, positions: torch.Tensor, training: bool = True
) -> torch.Tensor:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    gradient = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,  # For complete dissociation turn to true
    )[
        0
    ]  # [n_nodes, 3]
    if gradient is None:
        return torch.zeros_like(positions)
    return -1 * gradient


def get_symmetric_displacement(
    positions: torch.Tensor,
    unit_shifts: torch.Tensor,
    cell: Optional[torch.Tensor],
    edge_index: torch.Tensor,
    num_graphs: int,
    batch: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if cell is None:
        cell = torch.zeros(
            num_graphs * 3,
            3,
            dtype=positions.dtype,
            device=positions.device,
        )
    sender = edge_index[0]
    displacement = torch.zeros(
        (num_graphs, 3, 3),
        dtype=positions.dtype,
        device=positions.device,
    )
    displacement.requires_grad_(True)
    symmetric_displacement = 0.5 * (
        displacement + displacement.transpose(-1, -2)
    )  # From https://github.com/mir-group/nequip
    positions = positions + torch.einsum(
        "be,bec->bc", positions, symmetric_displacement[batch]
    )
    cell = cell.view(-1, 3, 3)
    cell = cell + torch.matmul(cell, symmetric_displacement)
    shifts = torch.einsum(
        "be,bec->bc",
        unit_shifts,
        cell[batch[sender]],
    )
    return positions, shifts, displacement


@torch.jit.unused
def compute_hessians_vmap(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    forces_flatten = forces.view(-1)
    num_elements = forces_flatten.shape[0]

    def get_vjp(v):
        return torch.autograd.grad(
            -1 * forces_flatten,
            positions,
            v,
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )

    I_N = torch.eye(num_elements).to(forces.device)
    try:
        chunk_size = 1 if num_elements < 64 else 16
        gradient = torch.vmap(
            get_vjp, in_dims=0, out_dims=0, chunk_size=chunk_size
        )(I_N)[0]
    except RuntimeError:
        gradient = compute_hessians_loop(forces, positions)
    if gradient is None:
        return torch.zeros((positions.shape[0], forces.shape[0], 3, 3))
    return gradient


@torch.jit.unused
def compute_hessians_loop(
    forces: torch.Tensor,
    positions: torch.Tensor,
) -> torch.Tensor:
    hessian = []
    for grad_elem in forces.view(-1):
        hess_row = torch.autograd.grad(
            outputs=[-1 * grad_elem],
            inputs=[positions],
            grad_outputs=torch.ones_like(grad_elem),
            retain_graph=True,
            create_graph=False,
            allow_unused=False,
        )[0]
        hess_row = (
            hess_row.detach()
        )  # this makes it very slow? but needs less memory
        if hess_row is None:
            hessian.append(torch.zeros_like(positions))
        else:
            hessian.append(hess_row)
    hessian = torch.stack(hessian)
    return hessian


activation_fn_dict = {
    "silu": torch.nn.SiLU,
    "relu": torch.nn.ReLU,
    "gelu": torch.nn.GELU,
    "tanh": torch.nn.Tanh,
    "identity": torch.nn.Identity,
}


def report_count_params(
    model: torch.nn.Module,
    num_elements: int,
    use_eletrostatics: bool,
    use_dispersion: bool,
) -> int:
    # log number of trainable params, absolute and percentage
    trainable_params = 0
    total_params = 0
    for name, param in model.named_parameters():
        if not use_eletrostatics and "partial_charges_output_block" in name:
            continue
        if not use_dispersion and "hirshfeld_output_block" in name:
            continue
        total_params += param.numel()
        if param.requires_grad:
            if "embedding" in name:
                if param.shape[1] == 118:
                    trainable_params += param[:, :num_elements].numel()
            else:
                trainable_params += param.numel()
    logging.info(f"Total model parameters: {total_params}")
    logging.info(f"Trainable model parameters: {trainable_params}")
    logging.info(
        f"Percentage of trainable parameters: {100 * trainable_params / total_params:.2f}%"
    )


def compute_forces_virials(
    energy: torch.Tensor,
    positions: torch.Tensor,
    displacement: torch.Tensor,
    cell: torch.Tensor,
    training: bool = True,
    compute_stress: bool = False,
) -> Tuple[torch.Tensor, Optional[torch.Tensor], Optional[torch.Tensor]]:
    grad_outputs: List[Optional[torch.Tensor]] = [torch.ones_like(energy)]
    forces, virials = torch.autograd.grad(
        outputs=[energy],  # [n_graphs, ]
        inputs=[positions, displacement],  # [n_nodes, 3]
        grad_outputs=grad_outputs,
        retain_graph=training,  # Make sure the graph is not destroyed during training
        create_graph=training,  # Create graph for second derivative
        allow_unused=True,
    )
    stress = torch.zeros_like(displacement)
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(-1, 1, 1)
        stress = torch.where(
            torch.abs(stress) < 1e10, stress, torch.zeros_like(stress)
        )
    if forces is None:
        forces = torch.zeros_like(positions)
    if virials is None:
        virials = torch.zeros((1, 3, 3))

    return -1 * forces, -1 * virials, stress


def compute_multihead_forces_stress(
    energy,
    positions,
    displacement,
    cell: torch.Tensor,
    training=True,
    compute_stress: bool = False,
):
    num_graphs, num_heads = energy.shape
    # change shape to [num_heads,num_graphs]
    energy_for_grad = energy.view(num_graphs, num_heads).permute(1, 0)
    grad_outputs = torch.zeros(
        num_heads,
        num_heads,
        num_graphs,
        device=energy.device,
        dtype=energy.dtype,
    )
    # picks the gradient for a specific head
    eye_for_heads = torch.eye(
        num_heads, device=energy.device, dtype=energy.dtype
    )
    # picks the gradient for a specific graph and head
    grad_outputs[:, :, :] = eye_for_heads.unsqueeze(-1).expand(
        -1, -1, num_graphs
    )

    forces, virials = torch.autograd.grad(
        outputs=[energy_for_grad],
        inputs=[positions, displacement],
        grad_outputs=grad_outputs,
        retain_graph=training,
        create_graph=training,
        allow_unused=True,
        is_grads_batched=True,  # treat the first dim (heads) as batch
    )

    # virials shape: [num_heads, batch, 3, 3]
    # cell shape : [batch, 3, 3]
    if compute_stress and virials is not None:
        cell = cell.view(-1, 3, 3)
        volume = torch.linalg.det(cell).abs().unsqueeze(-1)
        stress = virials / volume.view(1, -1, 1, 1)
        stress = torch.where(
            torch.abs(stress) < 1e10, stress, torch.zeros_like(stress)
        ).squeeze()
    return -1 * forces, -1 * virials, stress


def compute_multihead_forces(energy, positions, batch, training=True):
    num_graphs, num_heads = energy.shape
    # change shape to [num_heads,num_graphs]
    energy_for_grad = energy.view(num_graphs, num_heads).permute(1, 0)
    grad_outputs = torch.zeros(
        num_heads,
        num_heads,
        num_graphs,
        device=energy.device,
        dtype=energy.dtype,
    )
    # picks the gradient for a specific head
    eye_for_heads = torch.eye(
        num_heads, device=energy.device, dtype=energy.dtype
    )
    # picks the gradient for a specific graph and head
    grad_outputs[:, :, :] = eye_for_heads.unsqueeze(-1).expand(
        -1, -1, num_graphs
    )

    forces_pos = torch.autograd.grad(
        outputs=[energy_for_grad],
        inputs=[positions],
        grad_outputs=grad_outputs,
        retain_graph=training,
        create_graph=training,
        allow_unused=True,
        is_grads_batched=True,  # treat the first dim (heads) as batch
    )[0]
    forces = -forces_pos
    return forces


def get_outputs(
    energy: torch.Tensor,
    positions: torch.Tensor,
    cell: torch.Tensor,
    displacement: Optional[torch.Tensor],
    vectors: Optional[torch.Tensor] = None,
    training: bool = False,
    compute_force: bool = True,
    compute_virials: bool = True,
    compute_stress: bool = True,
    compute_hessian: bool = False,
    compute_edge_forces: bool = False,
    is_multihead: bool = False,
    batch: Optional[torch.Tensor] = None,
) -> Tuple[
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
    Optional[torch.Tensor],
]:
    if (compute_virials or compute_stress) and displacement is not None:
        if is_multihead:
            forces, virials, stress = compute_multihead_forces_stress(
                energy=energy,
                positions=positions,
                displacement=displacement,
                cell=cell,
                compute_stress=compute_stress,
                training=(training or compute_hessian or compute_edge_forces),
            )
        else:
            forces, virials, stress = compute_forces_virials(
                energy=energy,
                positions=positions,
                displacement=displacement,
                cell=cell,
                compute_stress=compute_stress,
                training=(training or compute_hessian or compute_edge_forces),
            )

    elif compute_force:
        if is_multihead:
            forces, virials, stress = (
                compute_multihead_forces(
                    energy=energy,
                    positions=positions,
                    training=(
                        training or compute_hessian or compute_edge_forces
                    ),
                    batch=batch,
                ),
                None,
                None,
            )
        else:
            forces, virials, stress = (
                compute_forces(
                    energy=energy,
                    positions=positions,
                    training=(
                        training or compute_hessian or compute_edge_forces
                    ),
                ),
                None,
                None,
            )
    else:
        forces, virials, stress = (None, None, None)
    if compute_hessian:
        assert forces is not None, "Forces must be computed to get the hessian"
        hessian = compute_hessians_vmap(forces, positions)
    else:
        hessian = None
    if compute_edge_forces and vectors is not None:
        edge_forces = compute_forces(
            energy=energy,
            positions=vectors,
            training=(training or compute_hessian),
        )
        if edge_forces is not None:
            edge_forces = -1 * edge_forces  # Match LAMMPS sign convention
    else:
        edge_forces = None
    return forces, virials, stress, hessian, edge_forces


def load_results_hdf5(filename, is_ensemble: bool = False):
    """Load results from HDF5 format."""
    loaded_data = {}

    with h5py.File(filename, "r") as f:
        for key in f.keys():
            if key == "att_scores":
                # Special handling for attention scores
                if is_ensemble:
                    # Ensemble attention scores
                    att_models = {}
                    att_grp = f[key]

                    for model_name in att_grp.keys():
                        model_grp = att_grp[model_name]
                        att_list = []

                        item_names = sorted(
                            [
                                name
                                for name in model_grp.keys()
                                if name.startswith("item_")
                            ]
                        )
                        for item_name in item_names:
                            item_grp = model_grp[item_name]
                            att_dict = {}

                            # Load 'ev' and 'inv' nested dictionaries
                            for key_type in ["ev", "inv"]:
                                if key_type in item_grp:
                                    key_grp = item_grp[key_type]
                                    att_dict[key_type] = {}
                                    for layer_idx in key_grp.keys():
                                        att_dict[key_type][int(layer_idx)] = (
                                            key_grp[layer_idx][()]
                                        )

                            # Load 'senders' and 'receivers' tensors
                            for key_type in ["senders", "receivers"]:
                                if key_type in item_grp:
                                    att_dict[key_type] = item_grp[key_type][()]

                            att_list.append(att_dict)

                        att_models[model_name] = att_list

                    loaded_data[key] = att_models
                else:
                    # Single model attention scores
                    att_list = []
                    att_grp = f[key]

                    item_names = sorted(
                        [
                            name
                            for name in att_grp.keys()
                            if name.startswith("item_")
                        ]
                    )
                    for item_name in item_names:
                        item_grp = att_grp[item_name]
                        att_dict = {}

                        # Load 'ev' and 'inv' nested dictionaries
                        for key_type in ["ev", "inv"]:
                            if key_type in item_grp:
                                key_grp = item_grp[key_type]
                                att_dict[key_type] = {}
                                for layer_idx in key_grp.keys():
                                    att_dict[key_type][int(layer_idx)] = (
                                        key_grp[layer_idx][()]
                                    )

                        # Load 'senders' and 'receivers' tensors
                        for key_type in ["senders", "receivers"]:
                            if key_type in item_grp:
                                att_dict[key_type] = item_grp[key_type][()]

                        att_list.append(att_dict)

                    loaded_data[key] = att_list

            elif is_ensemble:
                # Handle ensemble results for other keys
                loaded_data[key] = {}
                if isinstance(f[key], h5py.Group):
                    ensemble_grp = f[key]
                    for model_name in ensemble_grp.keys():
                        model_grp = ensemble_grp[model_name]
                        if isinstance(model_grp, h5py.Group):
                            # Multiple items per model
                            if "result" in model_grp:
                                # Single result per model
                                loaded_data[key][model_name] = model_grp[
                                    "result"
                                ][()]
                            else:
                                # Multiple items per model
                                items = []
                                item_names = sorted(model_grp.keys())
                                for item_name in item_names:
                                    items.append(model_grp[item_name][()])
                                loaded_data[key][model_name] = items
                        else:
                            # Single item per model
                            loaded_data[key][model_name] = model_grp[()]

            else:
                # Single model results for other keys
                if isinstance(f[key], h5py.Group):
                    # List of items
                    items = []
                    item_names = sorted(f[key].keys())
                    for item_name in item_names:
                        items.append(f[key][item_name][()])
                    loaded_data[key] = items
                else:
                    # Single item or None
                    if "is_none" in f[key].attrs:
                        loaded_data[key] = None
                    else:
                        loaded_data[key] = f[key][()]

    return loaded_data


def save_results_hdf5(results, filename, is_ensemble: bool = False):
    with h5py.File(filename, "w") as f:
        for k, v in results.items():
            if v is not None:
                if k == "att_scores":
                    continue
                    # Special handling for attention scores
                    if is_ensemble:
                        # Ensemble attention scores: list of lists of dicts
                        att_grp = f.create_group(k)
                        for model_idx, model_att_scores in enumerate(v):
                            model_grp = att_grp.create_group(
                                f"model_{model_idx}"
                            )
                            for i, att_dict in enumerate(model_att_scores):
                                item_grp = model_grp.create_group(
                                    f"item_{i:06d}"
                                )

                                # Handle 'ev' and 'inv' nested dictionaries
                                for key_type in ["ev", "inv"]:
                                    if key_type in att_dict:
                                        key_grp = item_grp.create_group(
                                            key_type
                                        )
                                        for layer_idx, tensor in att_dict[
                                            key_type
                                        ].items():
                                            if isinstance(
                                                tensor, torch.Tensor
                                            ):
                                                tensor_data = (
                                                    tensor.detach()
                                                    .cpu()
                                                    .numpy()
                                                )
                                            else:
                                                tensor_data = np.array(tensor)
                                            key_grp.create_dataset(
                                                str(layer_idx),
                                                data=tensor_data,
                                            )

                                # Handle 'senders' and 'receivers' tensors
                                for key_type in ["senders", "receivers"]:
                                    if key_type in att_dict:
                                        if isinstance(
                                            att_dict[key_type], torch.Tensor
                                        ):
                                            tensor_data = (
                                                att_dict[key_type]
                                                .detach()
                                                .cpu()
                                                .numpy()
                                            )
                                        else:
                                            tensor_data = np.array(
                                                att_dict[key_type]
                                            )
                                        item_grp.create_dataset(
                                            key_type, data=tensor_data
                                        )
                    else:
                        # Single model attention scores: list of dicts
                        att_grp = f.create_group(k)
                        for i, att_dict in enumerate(v):
                            item_grp = att_grp.create_group(f"item_{i:06d}")

                            # Handle 'ev' and 'inv' nested dictionaries
                            for key_type in ["ev", "inv"]:
                                if key_type in att_dict:
                                    key_grp = item_grp.create_group(key_type)
                                    for layer_idx, tensor in att_dict[
                                        key_type
                                    ].items():
                                        if isinstance(tensor, torch.Tensor):
                                            tensor_data = (
                                                tensor.detach().cpu().numpy()
                                            )
                                        else:
                                            tensor_data = np.array(tensor)
                                        key_grp.create_dataset(
                                            str(layer_idx), data=tensor_data
                                        )

                            # Handle 'senders' and 'receivers' tensors
                            for key_type in ["senders", "receivers"]:
                                if key_type in att_dict:
                                    if isinstance(
                                        att_dict[key_type], torch.Tensor
                                    ):
                                        tensor_data = (
                                            att_dict[key_type]
                                            .detach()
                                            .cpu()
                                            .numpy()
                                        )
                                    else:
                                        tensor_data = np.array(
                                            att_dict[key_type]
                                        )
                                    item_grp.create_dataset(
                                        key_type, data=tensor_data
                                    )

                elif is_ensemble:
                    # Handle ensemble results for other keys
                    ensemble_grp = f.create_group(k)
                    for model_idx, model_results in enumerate(v):
                        model_grp = ensemble_grp.create_group(
                            f"model_{model_idx}"
                        )
                        if isinstance(model_results, list):
                            for j, result in enumerate(model_results):
                                if isinstance(result, torch.Tensor):
                                    result = result.detach().cpu().numpy()
                                model_grp.create_dataset(
                                    f"item_{j:06d}", data=result
                                )
                        else:
                            # Single result per model
                            if isinstance(model_results, torch.Tensor):
                                model_results = (
                                    model_results.detach().cpu().numpy()
                                )
                            model_grp.create_dataset(
                                "result", data=model_results
                            )

                else:
                    # Single model results for other keys
                    if isinstance(v, list):
                        grp = f.create_group(k)
                        for i, result in enumerate(v):
                            if isinstance(result, torch.Tensor):
                                result = result.detach().cpu().numpy()
                            elif isinstance(result, list):
                                result = np.array(result)
                            grp.create_dataset(f"item_{i:06d}", data=result)
                    else:
                        # Single array/tensor
                        if isinstance(v, torch.Tensor):
                            v = v.detach().cpu().numpy()
                        f.create_dataset(k, data=v)
            else:
                # Store None as an empty dataset with attribute
                dset = f.create_dataset(k, data=np.array([]))
                dset.attrs["is_none"] = True


# TODO: Add support for multi-head outputs
# TODO: Add support from more output types
def save_results_xyz(input_data, results, filename):
    """Save results to an XYZ file."""
    scalar_keys = [
        "energies",
    ]
    tensor_keys = [
        "forces",
    ]
    input_configs = read(input_data, index=":")
    output_configs = []
    for i, config in enumerate(input_configs):
        atoms = config.copy()
        for key, value in results.items():
            if key in scalar_keys:
                if key == "energies":
                    atoms.info[f"SO3_energy"] = value[i].item()
                else:
                    atoms.info[f"SO3_{key}"] = value[i].item()
            elif key in tensor_keys:
                atoms.arrays[f"SO3_{key}"] = value[i]
        output_configs.append(atoms)
    write(filename, output_configs)

# when evaluating, predicted "REF_{key}" will replace the "real {key}". So I modifed this refering to MACE_{key}.

def ensemble_from_folder(path_to_models: str, device: str, dtype: str) -> dict:
    """
    Load an ensemble of models from a folder.

    Args:
        path_to_models (str): Path to the folder containing the models.
        device (str): Device to load the models on.
        dtype (str): Data type to load the models as.

    Returns:
        dict: Dictionary of models.
    """

    assert os.path.exists(path_to_models)
    assert os.listdir(Path(path_to_models))

    ensemble = {}
    for filename in os.listdir(path_to_models):
        if os.path.isfile(os.path.join(path_to_models, filename)):
            complete_path = os.path.join(path_to_models, filename)
            model = torch.load(
                complete_path, map_location=device, weights_only=False
            ).to(dtype)
            filename_without_suffix = os.path.splitext(filename)[0]
            ensemble[filename_without_suffix] = model
    return ensemble


def create_configs_from_list(
    atoms_list: list,
    key_specification=None,
    head_name: str = None,
):
    from so3krates_torch.data.utils import (
        KeySpecification,
        config_from_atoms,
    )

    if key_specification is None:
        key_specification = KeySpecification()
    configs = [
        config_from_atoms(
            atoms,
            key_specification=key_specification,
            head_name=head_name,
        )
        for atoms in atoms_list
    ]
    return configs


def create_data_from_configs(
    config_list: list,
    r_max: float,
    r_max_lr: float,
    all_heads: list = None,
    z_table: AtomicNumberTable = None,
):
    from so3krates_torch.data.atomic_data import AtomicData

    if z_table is None:
        z_table = AtomicNumberTable([int(z) for z in range(1, 119)])
    return [
        AtomicData.from_config(
            config,
            z_table=z_table,
            cutoff=float(r_max),
            cutoff_lr=r_max_lr,
            heads=all_heads,
        )
        for config in config_list
    ]


def create_data_from_list(
    atoms_list: list,
    r_max: float,
    r_max_lr: float,
    head_name: str = None,
    all_heads: list = None,
    key_specification=None,
    z_table: AtomicNumberTable = None,
):
    configs = create_configs_from_list(
        atoms_list,
        key_specification=key_specification,
        head_name=head_name,
    )
    return create_data_from_configs(
        configs,
        r_max,
        r_max_lr,
        all_heads=all_heads,
        z_table=z_table,
    )


def create_dataloader_from_list(
    atoms_list: list,
    batch_size: int,
    r_max: float,
    r_max_lr: float,
    key_specification=None,
    shuffle: bool = False,
    drop_last: bool = False,
    z_table: AtomicNumberTable = None,
    head_name: str = None,
    sampler=None,
):
    data_loader = DataLoader(
        dataset=create_data_from_list(
            atoms_list,
            r_max,
            r_max_lr,
            key_specification=key_specification,
            z_table=z_table,
            head_name=head_name,
        ),
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        drop_last=drop_last,
        sampler=sampler,
    )
    return data_loader


def create_dataloader_from_data(
    config_list: list,
    batch_size: int,
    shuffle: bool = False,
    drop_last: bool = False,
    sampler=None,
):
    data_loader = DataLoader(
        dataset=config_list,
        batch_size=batch_size,
        shuffle=(shuffle if sampler is None else False),
        drop_last=drop_last,
        sampler=sampler,
    )
    return data_loader


# the following was taken from MACE package: https://github.com/ACEsuit/mace
###########################################################################################
# Statistics utilities
# Authors: Ilyes Batatia, Gregor Simm, David Kovacs
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################


def compute_avg_num_neighbors(
    data_loader: torch.utils.data.DataLoader,
) -> float:
    num_neighbors = []
    for batch in data_loader:
        _, receivers = batch.edge_index
        _, counts = torch.unique(receivers, return_counts=True)
        num_neighbors.append(counts)

    avg_num_neighbors = torch.mean(
        torch.cat(num_neighbors, dim=0).type(torch.get_default_dtype())
    )
    return to_numpy(avg_num_neighbors).item()


def compute_mae(delta: np.ndarray) -> float:
    return np.mean(np.abs(delta)).item()


def compute_rel_mae(delta: np.ndarray, target_val: np.ndarray) -> float:
    target_norm = np.mean(np.abs(target_val))
    return np.mean(np.abs(delta)).item() / (target_norm + 1e-9) * 100


def compute_rmse(delta: np.ndarray) -> float:
    return np.sqrt(np.mean(np.square(delta))).item()


def compute_rel_rmse(delta: np.ndarray, target_val: np.ndarray) -> float:
    target_norm = np.sqrt(np.mean(np.square(target_val))).item()
    return (
        np.sqrt(np.mean(np.square(delta))).item() / (target_norm + 1e-9) * 100
    )


def compute_q95(delta: np.ndarray) -> float:
    return np.percentile(np.abs(delta), q=95)


def compute_c(delta: np.ndarray, eta: float) -> float:
    return np.mean(np.abs(delta) < eta).item()


def get_tag(name: str, seed: int) -> str:
    return f"{name}_run-{seed}"


def setup_logger(
    level: Union[int, str] = logging.INFO,
    tag: Optional[str] = None,
    directory: Optional[str] = None,
    rank: Optional[int] = 0,
):
    # Create a logger
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)  # Set to DEBUG to capture all levels

    # Create formatters
    formatter = logging.Formatter(
        "%(asctime)s.%(msecs)03d %(levelname)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # Add filter for rank
    logger.addFilter(lambda _: rank == 0)

    # Create console handler
    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(level)
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    if directory is not None and tag is not None:
        os.makedirs(name=directory, exist_ok=True)

        # Create file handler for non-debug logs
        main_log_path = os.path.join(directory, f"{tag}.log")
        fh_main = logging.FileHandler(main_log_path)
        fh_main.setLevel(level)
        fh_main.setFormatter(formatter)
        logger.addHandler(fh_main)

        # Create file handler for debug logs
        debug_log_path = os.path.join(directory, f"{tag}_debug.log")
        fh_debug = logging.FileHandler(debug_log_path)
        fh_debug.setLevel(logging.DEBUG)
        fh_debug.setFormatter(formatter)
        fh_debug.addFilter(lambda record: record.levelno >= logging.DEBUG)
        logger.addHandler(fh_debug)


class AtomicNumberTable:
    def __init__(self, zs: Sequence[int]):
        self.zs = zs

    def __len__(self) -> int:
        return len(self.zs)

    def __str__(self):
        return f"AtomicNumberTable: {tuple(s for s in self.zs)}"

    def index_to_z(self, index: int) -> int:
        return self.zs[index]

    def z_to_index(self, atomic_number: str) -> int:
        return self.zs.index(atomic_number)


def get_atomic_number_table_from_zs(zs: Iterable[int]) -> AtomicNumberTable:
    z_set = set()
    for z in zs:
        z_set.add(z)
    return AtomicNumberTable(sorted(list(z_set)))


def atomic_numbers_to_indices(
    atomic_numbers: np.ndarray, z_table: AtomicNumberTable
) -> np.ndarray:
    to_index_fn = np.vectorize(z_table.z_to_index)
    return to_index_fn(atomic_numbers)


class UniversalEncoder(json.JSONEncoder):
    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()
        if isinstance(o, torch.Tensor):
            return to_numpy(o)
        return json.JSONEncoder.default(self, o)


class MetricsLogger:
    def __init__(self, directory: str, tag: str) -> None:
        self.directory = directory
        self.filename = tag + ".txt"
        self.path = os.path.join(self.directory, self.filename)

    def log(self, d: Dict[str, Any]) -> None:
        os.makedirs(name=self.directory, exist_ok=True)
        with open(self.path, mode="a", encoding="utf-8") as f:
            f.write(json.dumps(d, cls=UniversalEncoder))
            f.write("\n")


# pylint: disable=abstract-method, arguments-differ
class LAMMPS_MP(torch.autograd.Function):
    @staticmethod
    def forward(ctx, *args):
        feats, data = args  # unpack
        ctx.vec_len = feats.shape[-1]
        ctx.data = data
        out = torch.empty_like(feats)
        data.forward_exchange(feats, out, ctx.vec_len)
        return out

    @staticmethod
    def backward(ctx, *grad_outputs):
        (grad,) = grad_outputs  # unpack
        gout = torch.empty_like(grad)
        ctx.data.reverse_exchange(grad, gout, ctx.vec_len)
        return gout, None


def get_cache_dir() -> Path:
    # get cache dir from XDG_CACHE_HOME if set, otherwise appropriate default
    return (
        Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache"))
        / "torchkrates"
    )


def filter_nonzero_weight(
    batch,
    quantity_l,
    weight,
    quantity_weight,
    spread_atoms=False,
    spread_quantity_vector=True,
) -> float:
    quantity = quantity_l[-1]
    # repeat with interleaving for per-atom quantities
    if spread_atoms:
        weight = torch.repeat_interleave(
            weight, batch.ptr[1:] - batch.ptr[:-1]
        ).unsqueeze(-1)
        quantity_weight = torch.repeat_interleave(
            quantity_weight, batch.ptr[1:] - batch.ptr[:-1]
        ).unsqueeze(-1)

    # repeat for additional dimensions
    if len(quantity.shape) > 1:
        repeats = [1] + list(quantity.shape[1:])
        view = [-1] + [1] * (len(quantity.shape) - 1)
        weight = weight.view(*view).repeat(*repeats)
        if spread_quantity_vector:
            quantity_weight = quantity_weight.view(*view).repeat(*repeats)

    filtered_q = quantity[weight * quantity_weight > 0]
    if len(filtered_q) == 0:
        quantity_l.pop()
        return 0.0

    quantity_l[-1] = filtered_q
    return 1.0
