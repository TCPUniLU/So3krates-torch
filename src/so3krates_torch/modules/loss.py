###########################################################################
# Parts taken from MACE package: https://github.com/ACEsuit/mace
###########################################################################

import torch
from typing import Optional, Dict
from torch_geometric.data import Batch
import torch.distributed as dist

TensorDict = Dict[str, torch.Tensor]


def is_ddp_enabled():
    return dist.is_initialized() and dist.get_world_size() > 1


def reduce_loss(
    raw_loss: torch.Tensor, ddp: Optional[bool] = None
) -> torch.Tensor:
    """
    Reduces an element-wise loss tensor.

    If ddp is True and distributed is initialized, the function computes:

        loss = (local_sum * world_size) / global_num_elements

    Otherwise, it returns the regular mean.
    """
    ddp = is_ddp_enabled() if ddp is None else ddp
    if ddp and dist.is_initialized():
        world_size = dist.get_world_size()
        n_local = raw_loss.numel()
        loss_sum = raw_loss.sum()
        total_samples = torch.tensor(
            n_local, device=raw_loss.device, dtype=raw_loss.dtype
        )
        dist.all_reduce(total_samples, op=dist.ReduceOp.SUM)
        return loss_sum * world_size / total_samples
    return raw_loss.mean()


def weighted_mean_squared_error_energy(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Calculate per-graph number of atoms.
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]  # shape: [n_graphs]
    raw_loss = (
        ref.weight
        * ref.energy_weight
        * torch.square((ref["energy"] - pred["energy"]) / num_atoms)
    )
    return reduce_loss(raw_loss, ddp)


def mean_squared_error_forces(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Repeat per-graph weights to per-atom level.
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, ref.ptr[1:] - ref.ptr[:-1]
    ).unsqueeze(-1)
    raw_loss = (
        configs_weight
        * configs_forces_weight
        * torch.square(ref["forces"] - pred["forces"])
    )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_error_dipole(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    num_atoms = (ref.ptr[1:] - ref.ptr[:-1]).unsqueeze(-1)
    raw_loss = torch.square((ref["dipole"] - pred["dipole"]) / num_atoms)
    return reduce_loss(raw_loss, ddp)


def reduce_masked_loss(
    raw_loss: torch.Tensor,
    mask: torch.Tensor,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    """Mean of `raw_loss` over the entries where `mask` is True.

    Masked-out entries must already be finite (zeroed) in `raw_loss`.
    Returns 0 when nothing is selected. Under DDP the selected count is
    all-reduced and the local sum scaled like `reduce_loss`:
    ``local_sum * world_size / global_count``.
    """
    ddp = is_ddp_enabled() if ddp is None else ddp
    loss_sum = torch.where(mask, raw_loss, torch.zeros_like(raw_loss)).sum()
    count = mask.sum().to(raw_loss.dtype)
    scale = 1.0
    if ddp and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
        scale = dist.get_world_size()
    return loss_sum * scale / torch.clamp(count, min=1.0)


def masked_mse_per_atom(
    ref: Batch,
    pred: TensorDict,
    ref_key: str,
    pred_key: str,
    weight_key: str,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    """Weighted per-atom MSE that skips atoms without a label.

    An atom is skipped when its label is non-finite (NaN marks a
    missing label, as in JAX so3lr) or its structure's combined weight
    (`weight * <weight_key>`) is zero. Skipped atoms count in neither
    the numerator nor the denominator. The difference is zeroed with
    `torch.where` before weighting so NaN labels never reach the sum
    or the gradient.
    """
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    atom_weight = torch.repeat_interleave(
        ref.weight * ref[weight_key], num_atoms
    )
    label = ref[ref_key]
    mask = torch.isfinite(label) & (atom_weight != 0)
    diff = torch.where(
        mask, label - pred[pred_key], torch.zeros_like(pred[pred_key])
    )
    raw_loss = atom_weight * torch.square(diff)
    return reduce_masked_loss(raw_loss, mask, ddp)


def weighted_mean_squared_error_hirshfeld(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    """Polarizability-ratio loss (`a0_ratios` in JAX/tauri)."""
    return masked_mse_per_atom(
        ref,
        pred,
        "hirshfeld_ratios",
        "hirshfeld_ratios",
        "hirshfeld_ratios_weight",
        ddp,
    )


def weighted_mean_squared_error_c6_ratios(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    return masked_mse_per_atom(
        ref, pred, "c6_ratios", "c6_ratios", "c6_ratios_weight", ddp
    )


def weighted_mean_squared_error_charges(
    ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
) -> torch.Tensor:
    # Repeat per-graph weights to per-atom level.
    configs_weight = torch.repeat_interleave(
        ref.weight, ref.ptr[1:] - ref.ptr[:-1]
    )
    configs_charges_weight = torch.repeat_interleave(
        ref.charges_weight, ref.ptr[1:] - ref.ptr[:-1]
    )
    raw_loss = (
        configs_weight
        * configs_charges_weight
        * torch.square(ref["charges"] - pred["partial_charges"])
    )
    return reduce_loss(raw_loss, ddp)


class WeightedEnergyForcesLoss(torch.nn.Module):
    def __init__(self, energy_weight=1.0, forces_weight=1.0) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy + self.forces_weight * loss_forces
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f})"
        )


class WeightedEnergyForcesDipoleLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, dipole_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_dipole = weighted_mean_squared_error_dipole(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.dipole_weight * loss_dipole
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"dipole_weight={self.dipole_weight:.3f})"
        )


class WeightedEnergyForcesHirshfeldLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, hirshfeld_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "hirshfeld_weight",
            torch.tensor(hirshfeld_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_hirshfeld = weighted_mean_squared_error_hirshfeld(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.hirshfeld_weight * loss_hirshfeld
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"hirshfeld_weight={self.hirshfeld_weight:.3f})"
        )


class WeightedEnergyForcesDipoleHirshfeldLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        dipole_weight=1.0,
        hirshfeld_weight=1.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "dipole_weight",
            torch.tensor(dipole_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "hirshfeld_weight",
            torch.tensor(hirshfeld_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_dipole = weighted_mean_squared_error_dipole(ref, pred, ddp)
        loss_hirshfeld = weighted_mean_squared_error_hirshfeld(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.dipole_weight * loss_dipole
            + self.hirshfeld_weight * loss_hirshfeld
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"dipole_weight={self.dipole_weight:.3f}, "
            f"hirshfeld_weight={self.hirshfeld_weight:.3f})"
        )


class WeightedEnergyForcesChargesLoss(torch.nn.Module):
    def __init__(
        self, energy_weight=1.0, forces_weight=1.0, charges_weight=1.0
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "charges_weight",
            torch.tensor(charges_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_charges = weighted_mean_squared_error_charges(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.charges_weight * loss_charges
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"charges_weight={self.charges_weight:.3f})"
        )


class WeightedEnergyForcesChargesHirshfeldLoss(torch.nn.Module):
    def __init__(
        self,
        energy_weight=1.0,
        forces_weight=1.0,
        charges_weight=1.0,
        hirshfeld_weight=1.0,
    ) -> None:
        super().__init__()
        self.register_buffer(
            "energy_weight",
            torch.tensor(energy_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "forces_weight",
            torch.tensor(forces_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "charges_weight",
            torch.tensor(charges_weight, dtype=torch.get_default_dtype()),
        )
        self.register_buffer(
            "hirshfeld_weight",
            torch.tensor(hirshfeld_weight, dtype=torch.get_default_dtype()),
        )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        loss_energy = weighted_mean_squared_error_energy(ref, pred, ddp)
        loss_forces = mean_squared_error_forces(ref, pred, ddp)
        loss_charges = weighted_mean_squared_error_charges(ref, pred, ddp)
        loss_hirshfeld = weighted_mean_squared_error_hirshfeld(ref, pred, ddp)
        return (
            self.energy_weight * loss_energy
            + self.forces_weight * loss_forces
            + self.charges_weight * loss_charges
            + self.hirshfeld_weight * loss_hirshfeld
        )

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(energy_weight={self.energy_weight:.3f}, "
            f"forces_weight={self.forces_weight:.3f}, "
            f"charges_weight={self.charges_weight:.3f}, "
            f"hirshfeld_weight={self.hirshfeld_weight:.3f})"
        )


# property name -> (loss term, model output key, config flag enabling it)
_PROPERTY_LOSS_TERMS = {
    "energy": (weighted_mean_squared_error_energy, "energy", None),
    "forces": (mean_squared_error_forces, "forces", None),
    "dipole": (
        weighted_mean_squared_error_dipole,
        "dipole",
        "electrostatic_energy_bool",
    ),
    "hirshfeld_ratios": (
        weighted_mean_squared_error_hirshfeld,
        "hirshfeld_ratios",
        "dispersion_energy_bool",
    ),
    "c6_ratios": (
        weighted_mean_squared_error_c6_ratios,
        "c6_ratios",
        "c6_ratios_bool",
    ),
    "charges": (
        weighted_mean_squared_error_charges,
        "partial_charges",
        "electrostatic_energy_bool",
    ),
}


class WeightedPropertiesLoss(torch.nn.Module):
    """Weighted sum of per-property losses, one term per non-zero
    weight. Keys of `weights` are property names (`energy`, `forces`,
    `dipole`, `hirshfeld_ratios`, `c6_ratios`, `charges`); each weight
    is stored as a buffer `<name>_weight`. Terms with weight 0 are not
    computed, so their model outputs may be absent."""

    def __init__(self, weights: Dict[str, float]) -> None:
        super().__init__()
        unknown = set(weights) - set(_PROPERTY_LOSS_TERMS)
        if unknown:
            raise ValueError(
                f"Unknown loss properties {sorted(unknown)}; supported: "
                f"{sorted(_PROPERTY_LOSS_TERMS)}"
            )
        self.properties = [
            name for name in _PROPERTY_LOSS_TERMS if weights.get(name, 0.0)
        ]
        for name in _PROPERTY_LOSS_TERMS:
            self.register_buffer(
                f"{name}_weight",
                torch.tensor(
                    float(weights.get(name, 0.0)),
                    dtype=torch.get_default_dtype(),
                ),
            )

    def forward(
        self, ref: Batch, pred: TensorDict, ddp: Optional[bool] = None
    ) -> torch.Tensor:
        total = 0.0
        for name in self.properties:
            term, pred_key, flag = _PROPERTY_LOSS_TERMS[name]
            if pred.get(pred_key) is None:
                hint = f" (enable ARCHITECTURE.{flag})" if flag else ""
                raise KeyError(
                    f"Loss weight for '{name}' is > 0 but the model output "
                    f"has no '{pred_key}'{hint}."
                )
            total = total + getattr(self, f"{name}_weight") * term(
                ref, pred, ddp
            )
        return total

    def __repr__(self):
        weights = ", ".join(
            f"{name}={getattr(self, f'{name}_weight'):.3f}"
            for name in self.properties
        )
        return f"{self.__class__.__name__}({weights})"
