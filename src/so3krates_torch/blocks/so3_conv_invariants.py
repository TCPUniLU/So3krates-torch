import torch
from typing import Callable, List, Dict, Optional
import torch.nn as nn
import itertools as it
import numpy as np
import importlib.resources
from so3krates_torch.tools.scatter import scatter_sum


class SO3ConvolutionInvariants(torch.nn.Module):
    def __init__(
        self,
        degrees: List[int],
    ):
        super().__init__()
        import e3nn.o3 as o3

        irreps_list = []
        for l in degrees:
            standard_parity = "e" if l % 2 == 0 else "o"
            irreps_list.append(o3.Irrep(f"{l}{standard_parity}"))
        self.irreps_in = o3.Irreps(irreps_list)
        self.tensor_product = o3.FullTensorProduct(
            self.irreps_in,
            self.irreps_in,
            filter_ir_out=["0e"],
            internal_weights=False,
        )

    def forward(
        self, ev_features_1: torch.Tensor, ev_features_2: torch.Tensor
    ) -> torch.Tensor:

        return self.tensor_product(ev_features_1, ev_features_2)


indx_fn = lambda x: int((x + 1) ** 2) if x >= 0 else 0


def load_cgmatrix():
    """Load the precomputed Clebsch-Gordan matrix from the bundled resource file."""
    ref = importlib.resources.files(__package__).joinpath("cgmatrix.npz")
    with importlib.resources.as_file(ref) as path:
        return np.load(path)["cg"]


def init_clebsch_gordan_matrix(degrees, l_out_max=0):
    l_in_max = max(degrees)
    l_in_min = min(degrees)
    offset_corr = indx_fn(l_in_min - 1)
    cg_full = load_cgmatrix()
    return cg_full[
        offset_corr : indx_fn(l_out_max),
        offset_corr : indx_fn(l_in_max),
        offset_corr : indx_fn(l_in_max),
    ]


class L0Contraction(nn.Module):
    def __init__(self, degrees, dtype=torch.float32, device="cpu"):
        super().__init__()
        self.degrees = degrees
        self.num_segments = len(degrees)

        # Always include l=0 in CG matrix construction (mimicking {0, *degrees})
        cg_matrix = init_clebsch_gordan_matrix(
            degrees=list({0, *degrees}), l_out_max=0
        )
        cg_diag = np.diagonal(cg_matrix, axis1=1, axis2=2)[
            0
        ]  # shape: (m_tot,)

        # Tile CG blocks exactly as in JAX logic
        cg_rep = []
        degrees_np = np.array(degrees)
        unique_degrees, counts = np.unique(degrees_np, return_counts=True)
        for d, r in zip(unique_degrees, counts):
            block = cg_diag[
                indx_fn(d - 1) : indx_fn(d)
            ]  # only select CG for degree d
            tiled = np.tile(block, r)
            cg_rep.append(tiled)

        cg_rep = np.concatenate(cg_rep)
        self.register_buffer(
            "cg_rep", torch.tensor(cg_rep, dtype=dtype, device=device)
        )

        # Segment IDs
        segment_ids = list(
            it.chain(
                *[[n] * (2 * degrees[n] + 1) for n in range(len(degrees))]
            )
        )
        self.register_buffer(
            "segment_ids",
            torch.tensor(segment_ids, dtype=torch.long, device=device),
        )

    def forward(self, sphc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sphc: shape (B, m_tot)
        Returns:
            shape (B, len(degrees))
        """
        B, m_tot = sphc.shape
        weighted = sphc * sphc * self.cg_rep[None, :]  # (B, m_tot)

        flat = weighted.reshape(-1)
        batch_ids = torch.arange(B, device=sphc.device).repeat_interleave(
            m_tot
        )
        seg_ids = self.segment_ids.repeat(B)
        scatter_ids = batch_ids * self.num_segments + seg_ids

        out = scatter_sum(
            flat, index=scatter_ids, dim=0, dim_size=B * self.num_segments
        )
        return out.view(B, self.num_segments)
