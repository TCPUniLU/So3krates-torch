import torch
import torch.nn as nn
import math
from so3krates_torch.tools import scatter
from typing import Callable, Dict, Optional, Union


class AtomicEnergyOutputHead(nn.Module):
    def __init__(
        self,
        num_features: int,
        energy_regression_dim: Optional[int] = None,
        final_output_features: int = 1,
        layers: int = 2,
        bias: bool = True,
        use_non_linearity: bool = True,
        non_linearity: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        final_non_linearity: bool = False,
        atomic_type_shifts: Optional[Dict[str, float]] = None,
        energy_learn_atomic_type_shifts: bool = False,
        energy_learn_atomic_type_scales: bool = False,
        num_elements: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.device = device
        self.layers = nn.ModuleList()
        if energy_regression_dim is None:
            energy_regression_dim = num_features

        for _ in range(layers - 1):
            self.layers.append(
                nn.Linear(num_features, energy_regression_dim, bias=bias)
            )
        self.final_layer = nn.Linear(
            energy_regression_dim, final_output_features, bias=bias
        )
        # make sure non_linearity is not None if use_non_linearity is True
        if use_non_linearity and non_linearity is None:
            raise ValueError(
                "If use_non_linearity is True, non_linearity must be provided."
            )
        self.use_non_linearity = use_non_linearity
        self.non_linearity = non_linearity()
        self.final_non_linearity = final_non_linearity

        self.energy_learn_atomic_type_shifts = energy_learn_atomic_type_shifts
        self.energy_learn_atomic_type_scales = energy_learn_atomic_type_scales

        self.energy_shifts = nn.Parameter(
            torch.zeros(
                num_elements,
                dtype=torch.get_default_dtype(),
                device=self.device,
            ),
            requires_grad=False,
        )
        self.energy_scales = nn.Linear(
            num_elements,
            1,
            bias=False,
            dtype=torch.get_default_dtype(),
            device=self.device,
        )
        nn.init.ones_(self.energy_scales.weight)
        self.energy_scales.weight.requires_grad = False

        self.use_defined_shifts = False
        if atomic_type_shifts is not None:
            self.use_defined_shifts = True
            # so3lr uses 119 elements
            # starts from 0 index (element zero from mass effect maybe)
            if atomic_type_shifts.get("0") is not None:
                atomic_type_shifts.pop("0")
            if num_elements is not None:
                assert len(atomic_type_shifts) == num_elements
            # sort atomic_type_shifts based on element number (keys)
            atomic_type_shifts = {
                k: atomic_type_shifts[k]
                for k in sorted(atomic_type_shifts.keys())
            }
            self.energy_shifts = nn.Parameter(
                torch.tensor(
                    list(atomic_type_shifts.values()),
                    dtype=torch.get_default_dtype(),
                    requires_grad=False,
                    device=self.device,
                )
            )

            self.set_defined_energy_shifts(
                atomic_type_shifts=atomic_type_shifts
            )

        self.enable_learned_energy_shifts_and_scales()

    def _apply(self, fn):
        """Keep self.device in sync when model.to() / .cuda() / .cpu() is called."""
        super()._apply(fn)
        # fn is a tensor transform; recover the new device from energy_shifts
        # which _apply has already moved.
        self.device = self.energy_shifts.device
        return self

    def set_defined_energy_shifts(
        self,
        atomic_type_shifts: Union[
            Dict[str, float], torch.Tensor, nn.Parameter
        ],
    ):
        if isinstance(atomic_type_shifts, dict):
            self.energy_shifts = nn.Parameter(
                torch.tensor(
                    list(atomic_type_shifts.values()),
                    dtype=torch.get_default_dtype(),
                    requires_grad=False,
                    device=self.device,
                )
            )
        elif isinstance(atomic_type_shifts, torch.Tensor) or isinstance(
            atomic_type_shifts, nn.Parameter
        ):
            self.energy_shifts = nn.Parameter(
                atomic_type_shifts.to(
                    device=self.device,
                    dtype=torch.get_default_dtype(),
                ),
                requires_grad=False,
            )
        else:
            raise ValueError(
                "atomic_type_shifts must be either a dict, torch.Tensor, or nn.Parameter."
            )
        self.enable_learned_energy_shifts_and_scales()

    def enable_learned_energy_shifts_and_scales(self):
        if self.energy_learn_atomic_type_shifts:
            self.energy_shifts.requires_grad = True
        if self.energy_learn_atomic_type_scales:
            self.energy_scales.weight.requires_grad = True

    def reset_parameters(self):
        # JAX init (lecun normal)
        for m in self.layers:
            if isinstance(m, torch.nn.Linear):
                std = 1.0 / (m.in_features**0.5)
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        # Final layer
        m = self.final_layer
        if isinstance(m, torch.nn.Linear):
            std = 1.0 / (m.in_features**0.5)
            torch.nn.init.normal_(m.weight, mean=0.0, std=std)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        inv_features: torch.Tensor,
        data: Dict[str, torch.Tensor],
        atomic_numbers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        for layer in self.layers:
            inv_features = layer(inv_features)
            if self.use_non_linearity:
                inv_features = self.non_linearity(inv_features)

        atomic_energies = self.final_layer(inv_features)
        if self.final_non_linearity:
            atomic_energies = self.non_linearity(inv_features)

        one_hot = data["node_attrs"]
        atomic_number_idx = torch.argmax(one_hot, dim=1)
        atomic_energies *= self.energy_scales(data["node_attrs"])
        atomic_energies += self.energy_shifts[atomic_number_idx].unsqueeze(1)

        return atomic_energies


class MultiAtomicEnergyOutputHead(AtomicEnergyOutputHead):
    def __init__(
        self,
        num_output_heads: int,
        num_features: int,
        energy_regression_dim: Optional[int] = None,
        final_output_features: int = 1,
        layers: int = 2,
        bias: bool = True,
        use_non_linearity: bool = True,
        non_linearity: Optional[Callable[[torch.Tensor], torch.Tensor]] = None,
        final_non_linearity: bool = False,
        atomic_type_shifts: Optional[Dict[str, float]] = None,
        energy_learn_atomic_type_shifts: bool = False,
        energy_learn_atomic_type_scales: bool = False,
        num_elements: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__(
            num_features=num_features,
            energy_regression_dim=energy_regression_dim,
            final_output_features=final_output_features,
            layers=layers,
            bias=bias,
            use_non_linearity=use_non_linearity,
            non_linearity=non_linearity,
            final_non_linearity=final_non_linearity,
            atomic_type_shifts=atomic_type_shifts,
            energy_learn_atomic_type_shifts=energy_learn_atomic_type_shifts,
            energy_learn_atomic_type_scales=energy_learn_atomic_type_scales,
            num_elements=num_elements,
            device=device,
        )
        self.num_output_heads = num_output_heads

        self.layers_weights = nn.ParameterList()
        self.layers_bias = nn.ParameterList()
        for i in range(layers - 1):
            in_dim = num_features if i == 0 else energy_regression_dim
            multi_head_weights = nn.Parameter(
                torch.empty(
                    self.num_output_heads,
                    in_dim,
                    energy_regression_dim,
                    dtype=torch.get_default_dtype(),
                    device=device,
                )
            )
            multi_head_bias = nn.Parameter(
                torch.empty(
                    self.num_output_heads,
                    energy_regression_dim,
                    dtype=torch.get_default_dtype(),
                    device=device,
                )
            )

            # JAX init (lecun normal)
            std = 1.0 / (multi_head_weights.size(1) ** 0.5)
            torch.nn.init.normal_(multi_head_weights, mean=0.0, std=std)
            torch.nn.init.zeros_(multi_head_bias)

            self.layers_weights.append(multi_head_weights)
            self.layers_bias.append(multi_head_bias)

        final_input_dim = (
            num_features if layers == 1 else energy_regression_dim
        )
        multi_head_final_weights = nn.Parameter(
            torch.empty(
                self.num_output_heads,
                final_input_dim,
                final_output_features,
                dtype=torch.get_default_dtype(),
                device=device,
            )
        )
        multi_head_final_bias = nn.Parameter(
            torch.empty(
                self.num_output_heads,
                final_output_features,
                dtype=torch.get_default_dtype(),
                device=device,
            )
        )

        # JAX init (lecun normal)
        std = 1.0 / (multi_head_final_weights.size(1) ** 0.5)
        torch.nn.init.normal_(multi_head_final_weights, mean=0.0, std=std)
        torch.nn.init.zeros_(multi_head_final_bias)

        self.final_layer_weights = multi_head_final_weights
        self.final_layer_bias = multi_head_final_bias

        # Override parent class initialization for multi-head support
        # Initialize energy_shifts for all heads with zeros (or defined shifts)
        self.energy_shifts = nn.Parameter(
            torch.zeros(
                num_elements, dtype=torch.get_default_dtype(), device=device
            ),
            requires_grad=False,
        )

        # Initialize energy_scales for all heads
        self.energy_scales = nn.Parameter(
            torch.ones(
                num_elements, 1, dtype=torch.get_default_dtype(), device=device
            ),
            requires_grad=False,
        )

        # Handle defined atomic type shifts (same for all heads)
        if atomic_type_shifts is not None:
            # Remove element zero if present
            if atomic_type_shifts.get("0") is not None:
                atomic_type_shifts.pop("0")
            if num_elements is not None:
                assert len(atomic_type_shifts) == num_elements
            # Sort atomic_type_shifts based on element number (keys)
            atomic_type_shifts = {
                k: atomic_type_shifts[k]
                for k in sorted(atomic_type_shifts.keys())
            }
            self.energy_shifts = nn.Parameter(
                torch.tensor(
                    list(atomic_type_shifts.values()),
                    dtype=torch.get_default_dtype(),
                    device=device,
                ),
                requires_grad=False,
            )

        # Enable gradient if requested
        self.enable_learned_energy_shifts_and_scales()

    def enable_learned_energy_shifts_and_scales(self):
        if self.energy_learn_atomic_type_shifts:
            self.energy_shifts.requires_grad = True
        if self.energy_learn_atomic_type_scales:
            self.energy_scales.requires_grad = True

    def forward(
        self,
        inv_features: torch.Tensor,
        data: Dict[str, torch.Tensor],
        atomic_numbers: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        for i, (layer_weights, layer_bias) in enumerate(
            zip(self.layers_weights, self.layers_bias)
        ):
            if i == 0:
                # First layer: 2D→3D expansion
                # (n,f) @ (h,f,d) → (n,h,d)
                inv_features = torch.einsum(
                    "nf, hfd -> nhd",
                    inv_features,
                    layer_weights,
                ) + layer_bias.unsqueeze(0)
            else:
                # Subsequent layers: 3D→3D
                # (n,h,d) @ (h,d,f) → (n,h,f)
                inv_features = torch.einsum(
                    "nhd, hdf -> nhf",
                    inv_features,
                    layer_weights,
                ) + layer_bias.unsqueeze(0)
            if self.use_non_linearity:
                inv_features = self.non_linearity(inv_features)

        if len(self.layers_weights) == 0:
            # layers=1: no intermediate layers, 2D→3D
            # (n,f) @ (h,f,d) → (n,h,d)
            atomic_energies = torch.einsum(
                "nf, hfd -> nhd",
                inv_features,
                self.final_layer_weights,
            ) + self.final_layer_bias.unsqueeze(0)
        else:
            # (n,h,d) @ (h,d,f) → (n,h,f)
            atomic_energies = torch.einsum(
                "nhd, hdf -> nhf",
                inv_features,
                self.final_layer_weights,
            ) + self.final_layer_bias.unsqueeze(0)
        if self.final_non_linearity:
            atomic_energies = self.non_linearity(atomic_energies)

        one_hot = data["node_attrs"]
        atomic_number_idx = torch.argmax(one_hot, dim=1)

        one_hot = data["node_attrs"]
        scales = torch.matmul(one_hot, self.energy_scales)
        atomic_energies *= scales.view(-1, 1, 1)
        atomic_energies += self.energy_shifts[atomic_number_idx].view(-1, 1, 1)

        return atomic_energies.squeeze(
            0
        )  # shape: (num_nodes, num_heads, final_output_features)


class PartialChargesOutputHead(nn.Module):
    def __init__(
        self,
        num_features: int,
        regression_dim: Optional[int] = None,
        activation_fn: torch.nn.Module = torch.nn.Identity,
    ):
        super().__init__()

        self.regression_dim = regression_dim
        self.activation_fn = activation_fn

        # Element-dependent bias embedding (for atomic numbers up to 100)
        self.atomic_embedding = nn.Embedding(
            num_embeddings=100, embedding_dim=1
        )

        # Build the network layers
        if self.regression_dim is not None:
            self.transform_inv_features = nn.Sequential(
                nn.Linear(num_features, regression_dim),
                activation_fn(),
                nn.Linear(regression_dim, 1),
            )
        else:
            self.transform_inv_features = nn.Linear(num_features, 1)

    def reset_parameters(self):
        # JAX init (lecun normal) - embedding
        std = 1.0 / (self.atomic_embedding.embedding_dim**0.5)
        torch.nn.init.normal_(self.atomic_embedding.weight, mean=0.0, std=std)
        # JAX init (lecun normal) - linear layers
        if isinstance(self.transform_inv_features, nn.Sequential):
            for m in self.transform_inv_features:
                if isinstance(m, torch.nn.Linear):
                    std = 1.0 / (m.in_features**0.5)
                    torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        else:
            m = self.transform_inv_features
            if isinstance(m, torch.nn.Linear):
                std = 1.0 / (m.in_features**0.5)
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        inv_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
        total_charge: torch.Tensor,
        batch_segments: torch.Tensor,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:

        # q_ - element-dependent bias
        q_ = self.atomic_embedding(atomic_numbers).squeeze(-1)
        x_ = self.transform_inv_features(inv_features).squeeze(-1)
        x_q = x_ + q_

        total_charge_predicted = scatter.scatter_sum(
            src=x_q, index=batch_segments, dim=0, dim_size=num_graphs
        )  # (num_graphs)

        number_of_atoms_in_molecule = scatter.scatter_sum(
            src=torch.ones_like(batch_segments),
            index=batch_segments,
            dim=0,
            dim_size=num_graphs,
        )

        charge_conservation = (1 / number_of_atoms_in_molecule) * (
            total_charge - total_charge_predicted
        )

        # Repeat charge conservation for each atom in molecule
        partial_charges = x_q + charge_conservation[batch_segments]
        return partial_charges


class DipoleVecOutputHead(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(
        self,
        partial_charges: torch.Tensor,
        positions: torch.Tensor,
        batch_segments: torch.Tensor,
        num_graphs: int,
    ) -> Dict[str, torch.Tensor]:

        mu_i = positions * partial_charges[:, None]

        # Compute the total dipole moment for each molecule
        total_dipole = scatter.scatter_sum(
            src=mu_i, index=batch_segments, dim=0, dim_size=num_graphs
        )

        return total_dipole


class HirshfeldOutputHead(nn.Module):
    def __init__(
        self,
        num_features: int,
        regression_dim: Optional[int] = None,
        activation_fn: torch.nn.Module = torch.nn.Identity,
    ):
        super().__init__()

        self.regression_dim = regression_dim
        self.activation_fn = activation_fn
        self.num_features = num_features

        self.v_shift_embedding = nn.Embedding(
            num_embeddings=100, embedding_dim=1
        )

        self.q_embedding = nn.Embedding(
            num_embeddings=100, embedding_dim=num_features // 2
        )

        # Build the network layers for k
        if self.regression_dim is not None:
            self.transform_features = nn.Sequential(
                nn.Linear(num_features, regression_dim // 2),
                activation_fn(),
                nn.Linear(regression_dim // 2, num_features // 2),
            )
        else:
            self.transform_features = nn.Linear(
                num_features, num_features // 2
            )

    def reset_parameters(self):
        # JAX init (lecun normal) - embeddings
        std = 1.0 / (self.v_shift_embedding.embedding_dim**0.5)
        torch.nn.init.normal_(self.v_shift_embedding.weight, mean=0.0, std=std)
        std = 1.0 / (self.q_embedding.embedding_dim**0.5)
        torch.nn.init.normal_(self.q_embedding.weight, mean=0.0, std=std)
        # JAX init (lecun normal) - linear layers
        if isinstance(self.transform_features, nn.Sequential):
            for m in self.transform_features:
                if isinstance(m, torch.nn.Linear):
                    std = 1.0 / (m.in_features**0.5)
                    torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        else:
            m = self.transform_features
            if isinstance(m, torch.nn.Linear):
                std = 1.0 / (m.in_features**0.5)
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        inv_features: torch.Tensor,
        atomic_numbers: torch.Tensor,
    ) -> torch.Tensor:
        """
        Predict Hirshfeld ratios from atom-wise features and atomic types.

        Args:
            inv_features: Atomic features, shape: (num_nodes, num_features)
            atomic_numbers: Atomic types, shape: (num_nodes)
            node_mask: Node mask, shape: (num_nodes), optional

        Returns:
            hirshfeld_ratios: Predicted Hirshfeld ratios, shape: (num_nodes)
        """

        v_shift = self.v_shift_embedding(atomic_numbers).squeeze(
            -1
        )  # (num_nodes)

        q = self.q_embedding(atomic_numbers)  # (num_nodes, num_features//2)

        k = self.transform_features(
            inv_features
        )  # (num_nodes, num_features//2)

        qk = (q * k * (1.0 / math.sqrt(k.shape[-1]))).sum(dim=-1)

        v_eff = v_shift + qk  # (num_nodes)

        hirshfeld_ratios = torch.abs(v_eff)

        return hirshfeld_ratios


class DirectForceOutputHead(nn.Module):
    def __init__(self, num_features: int):
        raise NotImplementedError(
            "DirectForceOutputHead is not implemented yet."
        )
