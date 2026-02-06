###########################################################################################

# Taken from MACE package: https://github.com/ACEsuit/mace and modified for So3krates/SO3LR

# Atomic Data Class for handling molecules as graphs
# Authors: Ilyes Batatia, Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

from copy import deepcopy
from typing import Optional, Sequence

import torch.utils.data

from so3krates_torch.tools.utils import (
    AtomicNumberTable,
    atomic_numbers_to_indices,
)
from so3krates_torch.tools.torch_tools import to_one_hot, voigt_to_matrix
from so3krates_torch.tools import torch_geometric
from .neighborhood import get_neighborhood
from .utils import Configuration


class AtomicData(torch_geometric.data.Data):
    num_graphs: torch.Tensor
    batch: torch.Tensor
    edge_index: torch.Tensor
    node_attrs: torch.Tensor
    atomic_numbers: torch.Tensor
    edge_vectors: torch.Tensor
    edge_lengths: torch.Tensor
    positions: torch.Tensor
    shifts: torch.Tensor
    unit_shifts: torch.Tensor
    cell: torch.Tensor
    forces: torch.Tensor
    energy: torch.Tensor
    stress: torch.Tensor
    virials: torch.Tensor
    dipole: torch.Tensor
    hirshfeld_ratios: torch.Tensor
    charges: torch.Tensor
    total_charge: torch.Tensor
    total_spin: torch.Tensor
    weight: torch.Tensor
    energy_weight: torch.Tensor
    forces_weight: torch.Tensor
    stress_weight: torch.Tensor
    virials_weight: torch.Tensor
    dipole_weight: torch.Tensor
    charges_weight: torch.Tensor

    def __init__(
        self,
        edge_index: torch.Tensor,  # [2, n_edges]
        node_attrs: torch.Tensor,  # [n_nodes, n_node_feats]
        atomic_numbers: torch.Tensor,  # [n_nodes, 1]
        positions: torch.Tensor,  # [n_nodes, 3]
        shifts: torch.Tensor,  # [n_edges, 3],
        unit_shifts: torch.Tensor,  # [n_edges, 3]
        cell: Optional[torch.Tensor],  # [3,3]
        weight: Optional[torch.Tensor],  # [,]
        head: Optional[torch.Tensor],  # [,]
        energy_weight: Optional[torch.Tensor],  # [,]
        forces_weight: Optional[torch.Tensor],  # [,]
        stress_weight: Optional[torch.Tensor],  # [,]
        virials_weight: Optional[torch.Tensor],  # [,]
        dipole_weight: Optional[torch.Tensor],  # [,]
        charges_weight: Optional[torch.Tensor],  # [,]
        hirshfeld_ratios_weight: Optional[torch.Tensor],  # [,]
        forces: Optional[torch.Tensor],  # [n_nodes, 3]
        energy: Optional[torch.Tensor],  # [, ]
        stress: Optional[torch.Tensor],  # [1,3,3]
        virials: Optional[torch.Tensor],  # [1,3,3]
        dipole: Optional[torch.Tensor],  # [, 3]
        hirshfeld_ratios: Optional[torch.Tensor],  # [n_nodes, ]
        charges: Optional[torch.Tensor],  # [n_nodes, ]
        elec_temp: Optional[torch.Tensor],  # [,]
        edge_index_lr: Optional[torch.Tensor] = None,  # [2, n_edges_lr]
        shifts_lr: Optional[torch.Tensor] = None,  # [n_edges_lr, 3]
        unit_shifts_lr: Optional[torch.Tensor] = None,  # [n_edges_lr, 3]
        total_charge: Optional[torch.Tensor] = None,  # [,]
        total_spin: Optional[torch.Tensor] = None,  # [,]
    ):
        # Check shapes
        num_nodes = node_attrs.shape[0]
        assert atomic_numbers.shape == (num_nodes,)
        assert edge_index.shape[0] == 2 and len(edge_index.shape) == 2
        assert positions.shape == (num_nodes, 3)
        assert shifts.shape[1] == 3
        assert unit_shifts.shape[1] == 3
        assert len(node_attrs.shape) == 2
        assert weight is None or len(weight.shape) == 0
        assert head is None or len(head.shape) == 0
        assert energy_weight is None or len(energy_weight.shape) == 0
        assert forces_weight is None or len(forces_weight.shape) == 0
        assert stress_weight is None or len(stress_weight.shape) == 0
        assert virials_weight is None or len(virials_weight.shape) == 0
        assert dipole_weight is None or dipole_weight.shape == (
            1,
            3,
        ), dipole_weight
        assert charges_weight is None or len(charges_weight.shape) == 0
        assert (
            hirshfeld_ratios_weight is None
            or len(hirshfeld_ratios_weight.shape) == 0
        )
        assert cell is None or cell.shape == (3, 3)
        assert forces is None or forces.shape == (num_nodes, 3)
        assert energy is None or len(energy.shape) == 0
        assert stress is None or stress.shape == (1, 3, 3)
        assert virials is None or virials.shape == (1, 3, 3)
        assert dipole is None or dipole.shape[-1] == 3
        assert charges is None or charges.shape == (num_nodes,)
        assert hirshfeld_ratios is None or hirshfeld_ratios.shape == (
            num_nodes,
        )
        assert elec_temp is None or len(elec_temp.shape) == 0
        assert total_charge is None or len(total_charge.shape) == 0
        assert total_spin is None or len(total_spin.shape) == 0
        assert edge_index_lr is None or edge_index_lr.shape[0] == 2
        assert shifts_lr is None or shifts_lr.shape[1] == 3
        assert unit_shifts_lr is None or unit_shifts_lr.shape[1] == 3

        # Aggregate data
        data = {
            "num_nodes": num_nodes,
            "edge_index": edge_index,
            "positions": positions,
            "shifts": shifts,
            "unit_shifts": unit_shifts,
            "edge_index_lr": edge_index_lr,
            "shifts_lr": shifts_lr,
            "unit_shifts_lr": unit_shifts_lr,
            "cell": cell,
            "node_attrs": node_attrs,
            "atomic_numbers": atomic_numbers,
            "weight": weight,
            "head": head,
            "energy_weight": energy_weight,
            "forces_weight": forces_weight,
            "stress_weight": stress_weight,
            "virials_weight": virials_weight,
            "dipole_weight": dipole_weight,
            "charges_weight": charges_weight,
            "hirshfeld_ratios_weight": hirshfeld_ratios_weight,
            "forces": forces,
            "energy": energy,
            "stress": stress,
            "virials": virials,
            "dipole": dipole,
            "charges": charges,
            "hirshfeld_ratios": hirshfeld_ratios,
            "elec_temp": elec_temp,
            "total_charge": total_charge,
            "total_spin": total_spin,
        }
        super().__init__(**data)

    @classmethod
    def from_config(
        cls,
        config: Configuration,
        z_table: AtomicNumberTable,
        cutoff: float,
        cutoff_lr: Optional[float] = None,
        heads: Optional[list] = None,
        **kwargs,  # pylint: disable=unused-argument
    ) -> "AtomicData":
        if heads is None:
            heads = ["Default"]
        (
            edge_index,
            shifts,
            unit_shifts,
            cell,
            edge_index_lr,
            shifts_lr,
            unit_shifts_lr,
        ) = get_neighborhood(
            positions=config.positions,
            cutoff=cutoff,
            cutoff_lr=cutoff_lr,
            pbc=deepcopy(config.pbc),
            cell=deepcopy(config.cell),
        )
        edge_index_lr = (
            torch.tensor(edge_index_lr, dtype=torch.long)
            if edge_index_lr is not None
            else None
        )
        shifts_lr = (
            torch.tensor(shifts_lr, dtype=torch.get_default_dtype())
            if shifts_lr is not None
            else None
        )
        unit_shifts_lr = (
            torch.tensor(unit_shifts_lr, dtype=torch.get_default_dtype())
            if unit_shifts_lr is not None
            else None
        )

        indices = atomic_numbers_to_indices(
            config.atomic_numbers, z_table=z_table
        )
        one_hot = to_one_hot(
            torch.tensor(indices, dtype=torch.long).unsqueeze(-1),
            num_classes=len(z_table),
        )
        atomic_numbers = torch.argmax(one_hot, dim=-1) + 1
        try:
            head = torch.tensor(heads.index(config.head), dtype=torch.long)
        except ValueError:
            head = torch.tensor(len(heads) - 1, dtype=torch.long)
        cell = (
            torch.tensor(cell, dtype=torch.get_default_dtype())
            if cell is not None
            else torch.tensor(
                3 * [0.0, 0.0, 0.0], dtype=torch.get_default_dtype()
            ).view(3, 3)
        )

        num_atoms = len(config.atomic_numbers)

        weight = (
            torch.tensor(config.weight, dtype=torch.get_default_dtype())
            if config.weight is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        energy_weight = (
            torch.tensor(
                config.property_weights.get("energy"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("energy") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        forces_weight = (
            torch.tensor(
                config.property_weights.get("forces"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("forces") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        stress_weight = (
            torch.tensor(
                config.property_weights.get("stress"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("stress") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        virials_weight = (
            torch.tensor(
                config.property_weights.get("virials"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("virials") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        dipole_weight = (
            torch.tensor(
                config.property_weights.get("dipole"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("dipole") is not None
            else torch.tensor(
                [[1.0, 1.0, 1.0]], dtype=torch.get_default_dtype()
            )
        )
        if len(dipole_weight.shape) == 0:
            dipole_weight = dipole_weight * torch.tensor(
                [[1.0, 1.0, 1.0]], dtype=torch.get_default_dtype()
            )
        elif len(dipole_weight.shape) == 1:
            dipole_weight = dipole_weight.unsqueeze(0)

        charges_weight = (
            torch.tensor(
                config.property_weights.get("charges"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("charges") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        hirshfeld_ratios_weight = (
            torch.tensor(
                config.property_weights.get("hirshfeld_ratios"),
                dtype=torch.get_default_dtype(),
            )
            if config.property_weights.get("hirshfeld_ratios") is not None
            else torch.tensor(1.0, dtype=torch.get_default_dtype())
        )

        forces = (
            torch.tensor(
                config.properties.get("forces"),
                dtype=torch.get_default_dtype(),
            )
            if config.properties.get("forces") is not None
            else torch.zeros(num_atoms, 3, dtype=torch.get_default_dtype())
        )
        energy = (
            torch.tensor(
                config.properties.get("energy"),
                dtype=torch.get_default_dtype(),
            )
            if config.properties.get("energy") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )
        stress = (
            voigt_to_matrix(
                torch.tensor(
                    config.properties.get("stress"),
                    dtype=torch.get_default_dtype(),
                )
            ).unsqueeze(0)
            if config.properties.get("stress") is not None
            else torch.zeros(1, 3, 3, dtype=torch.get_default_dtype())
        )
        virials = (
            voigt_to_matrix(
                torch.tensor(
                    config.properties.get("virials"),
                    dtype=torch.get_default_dtype(),
                )
            ).unsqueeze(0)
            if config.properties.get("virials") is not None
            else torch.zeros(1, 3, 3, dtype=torch.get_default_dtype())
        )
        dipole = (
            torch.tensor(
                config.properties.get("dipole"),
                dtype=torch.get_default_dtype(),
            ).unsqueeze(0)
            if config.properties.get("dipole") is not None
            else torch.zeros(1, 3, dtype=torch.get_default_dtype())
        )
        charges = (
            torch.tensor(
                config.properties.get("charges"),
                dtype=torch.get_default_dtype(),
            )
            if config.properties.get("charges") is not None
            else torch.zeros(num_atoms, dtype=torch.get_default_dtype())
        )
        hirshfeld_ratios = (
            torch.tensor(
                config.properties.get("hirshfeld_ratios"),
                dtype=torch.get_default_dtype(),
            )
            if config.properties.get("hirshfeld_ratios") is not None
            else torch.zeros(num_atoms, dtype=torch.get_default_dtype())
        )
        elec_temp = (
            torch.tensor(
                config.properties.get("elec_temp"),
                dtype=torch.get_default_dtype(),
            )
            if config.properties.get("elec_temp") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )

        total_charge = (
            torch.tensor(
                config.properties.get("total_charge"),
                dtype=torch.get_default_dtype(),
            )
            if config.properties.get("total_charge") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )
        total_spin = (
            torch.tensor(
                config.properties.get("total_spin"),
                dtype=torch.get_default_dtype(),
            )
            if config.properties.get("total_spin") is not None
            else torch.tensor(0.0, dtype=torch.get_default_dtype())
        )

        return cls(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            positions=torch.tensor(
                config.positions, dtype=torch.get_default_dtype()
            ),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(
                unit_shifts, dtype=torch.get_default_dtype()
            ),
            edge_index_lr=edge_index_lr,
            shifts_lr=shifts_lr,
            unit_shifts_lr=unit_shifts_lr,
            cell=cell,
            node_attrs=one_hot,
            atomic_numbers=atomic_numbers,
            weight=weight,
            head=head,
            energy_weight=energy_weight,
            forces_weight=forces_weight,
            stress_weight=stress_weight,
            virials_weight=virials_weight,
            dipole_weight=dipole_weight,
            charges_weight=charges_weight,
            hirshfeld_ratios_weight=hirshfeld_ratios_weight,
            forces=forces,
            energy=energy,
            stress=stress,
            virials=virials,
            dipole=dipole,
            hirshfeld_ratios=hirshfeld_ratios,
            charges=charges,
            elec_temp=elec_temp,
            total_charge=total_charge,
            total_spin=total_spin,
        )


def get_data_loader(
    dataset: Sequence[AtomicData],
    batch_size: int,
    shuffle=True,
    drop_last=False,
) -> torch.utils.data.DataLoader:
    return torch_geometric.dataloader.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
