import torch
from typing import Callable, List, Dict, Optional
from so3krates_torch.tools import scatter
import math


class InvariantEmbedding(torch.nn.Module):
    """
    Eq. 10 in https://doi.org/10.1038/s41467-024-50620-6
    """

    def __init__(
        self, num_elements: int, out_features: int, bias: bool = False
    ):
        super().__init__()
        self.embedding = torch.nn.Linear(
            num_elements,
            out_features,
            bias=bias,
            dtype=torch.get_default_dtype(),
        )

    def reset_parameters(self):
        # JAX init for embedding
        std = 1.0 / (self.embedding.out_features**0.5)
        torch.nn.init.normal_(self.embedding.weight, mean=0.0, std=std)
        if self.embedding.bias is not None:
            torch.nn.init.zeros_(self.embedding.bias)

    def forward(self, one_hot: torch.Tensor) -> torch.Tensor:
        return self.embedding(one_hot)


class EuclideanEmbedding(torch.nn.Module):
    """
    Eq. 11 in https://doi.org/10.1038/s41467-024-50620-6

    In the paper it is said that the initialization is done trough
    Eq. 11, as stated above. However, in the code it is hardcoded to
    be zeros.
    """

    def __init__(
        self,
        initialization_to_zeros: bool = True,
    ):
        super().__init__()
        self.initialization_to_zeros = initialization_to_zeros

    def forward(
        self,
        sh_vectors: torch.Tensor,
        cutoffs: torch.Tensor,
        receivers: torch.Tensor,
        avg_num_neighbors: float,
        num_nodes: int,
    ) -> torch.Tensor:
        inv_avg_num_neighbors = 1.0 / avg_num_neighbors
        if self.initialization_to_zeros:
            ev_embedding = torch.zeros(
                (num_nodes, sh_vectors.shape[1]),
                dtype=sh_vectors.dtype,
                device=sh_vectors.device,
            )
        else:
            # This is actually Eq. 11 in the paper:
            scaled_neighbors = sh_vectors * cutoffs
            ev_embedding = (
                scatter.scatter_sum(
                    src=scaled_neighbors,
                    index=receivers,
                    dim=0,
                    dim_size=num_nodes,
                )
                * inv_avg_num_neighbors
            )

        return ev_embedding


class ChargeSpinEmbedding(torch.nn.Module):
    """
    As introduced in doi.org/10.1038/s41467-021-27504-0.
    """

    def __init__(
        self,
        num_features: int,
        activation_fn: Callable = torch.nn.SiLU,
        num_elements: int = 118,
    ):
        super().__init__()

        self.Wq = torch.nn.Linear(
            in_features=num_elements,
            out_features=num_features,
            bias=False,
        )

        # JAX init for Wq
        std = 1.0 / (self.Wq.out_features**0.5)
        torch.nn.init.normal_(self.Wq.weight, mean=0.0, std=std)

        self.Wk = torch.nn.Parameter(torch.empty(size=(2, num_features)))
        self.Wv = torch.nn.Parameter(torch.empty(size=(2, num_features)))

        # JAX init for Wk and Wv
        std = 1.0 / (self.Wk.size(1) ** 0.5)
        torch.nn.init.normal_(self.Wk, mean=0.0, std=std)
        torch.nn.init.normal_(self.Wv, mean=0.0, std=std)

        self.sqrt_dim = num_features**0.5
        self.activation_fn = activation_fn
        self.mlp = torch.nn.Sequential(
            self.activation_fn(),
            torch.nn.Linear(num_features, num_features, bias=False),
            self.activation_fn(),
            torch.nn.Linear(num_features, num_features, bias=False),
        )

        # JAX init for MLP
        for m in self.mlp:
            if isinstance(m, torch.nn.Linear):
                std = 1.0 / (m.in_features**0.5)
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    @torch.compiler.disable()
    def forward(
        self,
        elements_one_hot: torch.Tensor,
        psi: torch.Tensor,
        batch_segments: torch.Tensor,
        num_graphs: Optional[int] = None,
        eps: float = 1e-6,
    ) -> torch.Tensor:
        """
        Computes the charge-spin embedding as described in
        doi.org/10.1038/s41467-021-27504-0.

        Args:
            elements_one_hot (torch.Tensor): One-hot encoded atomic numbers.
            psi (torch.Tensor): Charge or number of unpaired electrons.
            batch_segments (torch.Tensor): Indices of the batches inside the tensor.

        Returns:
            torch.Tensor: The charge-spin embedding for the input.
        """
        q = self.Wq(elements_one_hot)
        idx = (psi // torch.inf).type(torch.int)
        # Expand graph-level idx to atom-level using batch_segments
        idx_per_atom = idx[batch_segments]
        k = self.Wk[idx_per_atom]
        v = self.Wv[idx_per_atom]
        q_x_k = (q * k).sum(dim=-1) / self.sqrt_dim
        y = torch.nn.functional.softplus(q_x_k)
        # Use batch size information to avoid FX specialization
        if num_graphs is None:
            # Fallback to data-dependent computation
            if batch_segments.numel() == 0:
                computed_num_graphs = 0
            else:
                # This fallback should not be reached in normal operation
                computed_num_graphs = int(batch_segments.max()) + 1
        else:
            computed_num_graphs = num_graphs
        denominator = (
            scatter.scatter_sum(
                src=y,
                index=batch_segments,
                dim=0,
                dim_size=computed_num_graphs,
            )
            + eps
        )
        att = psi[batch_segments] * y / denominator[batch_segments]
        v_att = att[:, None] * v
        v_att_temp = v_att.clone()
        e_psi = v_att_temp + self.mlp(v_att)
        return e_psi
