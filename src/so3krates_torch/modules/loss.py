###########################################################################
# Parts taken from MACE package: https://github.com/ACEsuit/mace
###########################################################################

import torch
from typing import Callable, Dict, Optional
from torch_geometric.data import Batch
import torch.distributed as dist

import math
from typing import Sequence, Tuple

from so3krates_torch.modules.robust_loss import Distribution, general_loss
from so3krates_torch.modules.loss_reduction import (
    is_ddp_enabled,
    per_graph_node_mean,
    reduce_graphs,
    reduce_valid_elements,
)

TensorDict = Dict[str, torch.Tensor]

# (residual, property name) -> per-element penalty
Penalty = Callable[[torch.Tensor, str], torch.Tensor]
REDUCTIONS = ("flat", "per_graph")


def squared_penalty(x: torch.Tensor, prop: str) -> torch.Tensor:
    return torch.square(x)


ROBUST_MODES = ("none", "fixed", "adaptive")


def fixed_robust_penalty(alpha: float) -> Penalty:
    """JAX `use_robust_loss`: 2 * rho(x, alpha, 1.0)."""

    def _penalty(x: torch.Tensor, prop: str) -> torch.Tensor:
        return 2.0 * general_loss(x, alpha, 1.0)

    return _penalty


class AdaptiveRobustParams(torch.nn.Module):
    """Learned per-property shape alpha and scale c of the adaptive robust
    loss (JAX `use_adaptive_robust_loss`). Stored unconstrained, keyed by
    property name: alpha = 2*sigmoid(alpha_raw), c = softplus(scale_raw)
    + 1e-6. The penalty is 2 * NLL(x | alpha, c)."""

    def __init__(
        self,
        properties: Sequence[str],
        init_alpha: float = 1.0,
        init_scale: float = 1.0,
    ) -> None:
        super().__init__()
        a = min(max(float(init_alpha), 0.01), 1.99)
        alpha_raw = math.log(a / (2.0 - a))
        scale_raw = math.log(math.expm1(max(float(init_scale), 0.1)))
        dtype = torch.get_default_dtype()
        self.alpha_raw = torch.nn.ParameterDict(
            {
                p: torch.nn.Parameter(torch.tensor(alpha_raw, dtype=dtype))
                for p in properties
            }
        )
        self.scale_raw = torch.nn.ParameterDict(
            {
                p: torch.nn.Parameter(torch.tensor(scale_raw, dtype=dtype))
                for p in properties
            }
        )
        self._distribution = Distribution()

    def alpha_scale(self, prop: str) -> Tuple[torch.Tensor, torch.Tensor]:
        alpha = 2.0 * torch.sigmoid(self.alpha_raw[prop])
        scale = torch.nn.functional.softplus(self.scale_raw[prop]) + 1e-6
        return alpha, scale

    def penalty(self, x: torch.Tensor, prop: str) -> torch.Tensor:
        alpha, scale = self.alpha_scale(prop)
        return 2.0 * self._distribution.nllfun(
            x, alpha.to(x.dtype), scale.to(x.dtype)
        )


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


def _num_atoms(ref: Batch) -> torch.Tensor:
    return ref.ptr[1:] - ref.ptr[:-1]


def _check_reduction(reduction: str) -> None:
    if reduction not in REDUCTIONS:
        raise ValueError(
            f"Unknown loss reduction '{reduction}'; use one of {REDUCTIONS}."
        )


def _per_graph_atoms(
    ref: Batch,
    label: torch.Tensor,
    pred_value: torch.Tensor,
    graph_weight: torch.Tensor,
    penalty: Penalty,
    prop: str,
    ddp: Optional[bool],
) -> torch.Tensor:
    """JAX `node_mse_loss`: per-graph mean of per-atom penalties, mean
    over graphs that carry the label (a graph with any NaN is dropped)."""
    finite = torch.isfinite(label)
    diff = torch.where(
        finite, label - pred_value, torch.zeros_like(pred_value)
    )
    per_graph, has_data = per_graph_node_mean(
        penalty(diff, prop), ref, label, graph_weight
    )
    return reduce_graphs(per_graph, has_data, ddp)


def _per_graph_per_atom_graph_property(
    label: torch.Tensor,
    pred_value: torch.Tensor,
    num_atoms: torch.Tensor,
    weight: torch.Tensor,
    penalty: Penalty,
    prop: str,
    ddp: Optional[bool],
) -> torch.Tensor:
    """JAX `graph_mse_loss_per_atom`: residual divided by the number of
    atoms, mean over valid (finite, non-zero weight) elements."""
    valid = torch.isfinite(label) & (weight != 0)
    residual = torch.where(
        valid, (label - pred_value) / num_atoms, torch.zeros_like(pred_value)
    )
    return reduce_valid_elements(penalty(residual, prop), valid, weight, ddp)


def weighted_mean_squared_error_energy(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
    reduction: str = "flat",
    penalty: Penalty = squared_penalty,
) -> torch.Tensor:
    _check_reduction(reduction)
    num_atoms = _num_atoms(ref)  # shape: [n_graphs]
    weight = ref.weight * ref.energy_weight
    if reduction == "per_graph":
        return _per_graph_per_atom_graph_property(
            ref["energy"],
            pred["energy"],
            num_atoms,
            weight,
            penalty,
            "energy",
            ddp,
        )
    raw_loss = weight * penalty(
        (ref["energy"] - pred["energy"]) / num_atoms, "energy"
    )
    return reduce_loss(raw_loss, ddp)


def mean_squared_error_forces(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
    reduction: str = "flat",
    penalty: Penalty = squared_penalty,
) -> torch.Tensor:
    _check_reduction(reduction)
    if reduction == "per_graph":
        return _per_graph_atoms(
            ref,
            ref["forces"],
            pred["forces"],
            ref.weight * ref.forces_weight,
            penalty,
            "forces",
            ddp,
        )
    # Repeat per-graph weights to per-atom level.
    configs_weight = torch.repeat_interleave(
        ref.weight, _num_atoms(ref)
    ).unsqueeze(-1)
    configs_forces_weight = torch.repeat_interleave(
        ref.forces_weight, _num_atoms(ref)
    ).unsqueeze(-1)
    raw_loss = (
        configs_weight
        * configs_forces_weight
        * penalty(ref["forces"] - pred["forces"], "forces")
    )
    return reduce_loss(raw_loss, ddp)


def weighted_mean_squared_error_dipole(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
    reduction: str = "flat",
    penalty: Penalty = squared_penalty,
) -> torch.Tensor:
    _check_reduction(reduction)
    num_atoms = _num_atoms(ref).unsqueeze(-1)
    if reduction == "per_graph":
        weight = ref.weight.unsqueeze(-1) * ref.dipole_weight
        return _per_graph_per_atom_graph_property(
            ref["dipole"],
            pred["dipole"],
            num_atoms,
            weight,
            penalty,
            "dipole",
            ddp,
        )
    raw_loss = penalty((ref["dipole"] - pred["dipole"]) / num_atoms, "dipole")
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


def ratio_label_mask(
    ref: Batch, label_key: str, weight_key: str
) -> torch.Tensor:
    """Per-atom bool mask of usable labels: finite label and non-zero
    structure weight (`weight * <weight_key>`)."""
    num_atoms = ref.ptr[1:] - ref.ptr[:-1]
    atom_weight = torch.repeat_interleave(
        ref.weight * ref[weight_key], num_atoms
    )
    return torch.isfinite(ref[label_key]) & (atom_weight != 0)


def masked_mse_per_atom(
    ref: Batch,
    pred: TensorDict,
    ref_key: str,
    pred_key: str,
    weight_key: str,
    ddp: Optional[bool] = None,
    penalty: Penalty = squared_penalty,
    prop: Optional[str] = None,
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
    mask = ratio_label_mask(ref, ref_key, weight_key)
    diff = torch.where(
        mask, label - pred[pred_key], torch.zeros_like(pred[pred_key])
    )
    raw_loss = atom_weight * penalty(diff, prop or ref_key)
    return reduce_masked_loss(raw_loss, mask, ddp)


def _ratio_term(
    ref: Batch,
    pred: TensorDict,
    key: str,
    ddp: Optional[bool],
    reduction: str,
    penalty: Penalty,
) -> torch.Tensor:
    _check_reduction(reduction)
    weight_key = f"{key}_weight"
    if reduction == "per_graph":
        return _per_graph_atoms(
            ref,
            ref[key],
            pred[key],
            ref.weight * ref[weight_key],
            penalty,
            key,
            ddp,
        )
    return masked_mse_per_atom(
        ref, pred, key, key, weight_key, ddp, penalty=penalty, prop=key
    )


def weighted_mean_squared_error_hirshfeld(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
    reduction: str = "flat",
    penalty: Penalty = squared_penalty,
) -> torch.Tensor:
    """Polarizability-ratio loss (`a0_ratios` in JAX/tauri)."""
    return _ratio_term(ref, pred, "hirshfeld_ratios", ddp, reduction, penalty)


def weighted_mean_squared_error_c6_ratios(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
    reduction: str = "flat",
    penalty: Penalty = squared_penalty,
) -> torch.Tensor:
    return _ratio_term(ref, pred, "c6_ratios", ddp, reduction, penalty)


def weighted_mean_squared_error_charges(
    ref: Batch,
    pred: TensorDict,
    ddp: Optional[bool] = None,
    reduction: str = "flat",
    penalty: Penalty = squared_penalty,
) -> torch.Tensor:
    _check_reduction(reduction)
    if reduction == "per_graph":
        return _per_graph_atoms(
            ref,
            ref["charges"],
            pred["partial_charges"],
            ref.weight * ref.charges_weight,
            penalty,
            "charges",
            ddp,
        )
    # Repeat per-graph weights to per-atom level.
    configs_weight = torch.repeat_interleave(ref.weight, _num_atoms(ref))
    configs_charges_weight = torch.repeat_interleave(
        ref.charges_weight, _num_atoms(ref)
    )
    raw_loss = (
        configs_weight
        * configs_charges_weight
        * penalty(ref["charges"] - pred["partial_charges"], "charges")
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

    def __init__(
        self,
        weights: Dict[str, float],
        reduction: str = "flat",
        penalty: Optional[Penalty] = None,
        robust: str = "none",
        robust_alpha: float = 1.0,
        adaptive_init_alpha: float = 1.0,
        adaptive_init_scale: float = 1.0,
        adaptive: Optional["AdaptiveRobustParams"] = None,
    ) -> None:
        super().__init__()
        _check_reduction(reduction)
        if robust not in ROBUST_MODES:
            raise ValueError(
                f"Unknown robust mode '{robust}'; use one of {ROBUST_MODES}."
            )
        if penalty is not None and robust != "none":
            raise ValueError(
                "Pass either an explicit penalty or robust != 'none', "
                "not both."
            )
        self.reduction = reduction
        self.robust = robust
        unknown = set(weights) - set(_PROPERTY_LOSS_TERMS)
        if unknown:
            raise ValueError(
                f"Unknown loss properties {sorted(unknown)}; supported: "
                f"{sorted(_PROPERTY_LOSS_TERMS)}"
            )
        self.properties = [
            name for name in _PROPERTY_LOSS_TERMS if weights.get(name, 0.0)
        ]
        self.adaptive = None
        if robust == "adaptive":
            if adaptive is None:
                adaptive = AdaptiveRobustParams(
                    self.properties, adaptive_init_alpha, adaptive_init_scale
                )
            missing = set(self.properties) - set(adaptive.alpha_raw.keys())
            if missing:
                raise ValueError(
                    "Shared adaptive robust-loss parameters lack "
                    f"{sorted(missing)}."
                )
            # May be shared with another loss (two-stage training).
            self.adaptive = adaptive
            self.penalty = self.adaptive.penalty
        elif robust == "fixed":
            self.robust_alpha = float(robust_alpha)
            self.penalty = fixed_robust_penalty(self.robust_alpha)
        else:
            self.penalty = penalty or squared_penalty
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
        self.check_outputs(pred)
        total = 0.0
        for name in self.properties:
            term, pred_key, flag = _PROPERTY_LOSS_TERMS[name]
            total = total + getattr(self, f"{name}_weight") * term(
                ref,
                pred,
                ddp,
                reduction=self.reduction,
                penalty=self.penalty,
            )
        return total

    def check_outputs(self, pred: TensorDict) -> None:
        """Raise KeyError if the model output lacks a weighted property."""
        for name in self.properties:
            _term, pred_key, flag = _PROPERTY_LOSS_TERMS[name]
            if pred.get(pred_key) is None:
                hint = f" (enable ARCHITECTURE.{flag})" if flag else ""
                raise KeyError(
                    f"Loss weight for '{name}' is > 0 but the model output "
                    f"has no '{pred_key}'{hint}."
                )

    def __repr__(self):
        weights = ", ".join(
            f"{name}={getattr(self, f'{name}_weight'):.3f}"
            for name in self.properties
        )
        return (
            f"{self.__class__.__name__}({weights}, "
            f"reduction={self.reduction}, robust={self.robust})"
        )


def robust_param_metrics(loss_fn: torch.nn.Module) -> Dict[str, float]:
    """Current adaptive robust-loss alpha/scale per property, as
    `robust_alpha_<prop>` / `robust_scale_<prop>` (empty otherwise)."""
    adaptive = getattr(loss_fn, "adaptive", None)
    if adaptive is None:
        return {}
    metrics = {}
    with torch.no_grad():
        for prop in adaptive.alpha_raw.keys():
            alpha, scale = adaptive.alpha_scale(prop)
            metrics[f"robust_alpha_{prop}"] = float(alpha)
            metrics[f"robust_scale_{prop}"] = float(scale)
    return metrics
