import torch
from typing import Callable, List, Dict, Optional
import torch.nn as nn
import itertools as it
import numpy as np
import importlib.resources

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
                *[
                    [n] * (2 * degrees[n] + 1)
                    for n in range(len(degrees))
                ]
            )
        )
        self.register_buffer(
            "segment_ids",
            torch.tensor(
                segment_ids, dtype=torch.long, device=device
            ),
        )

        # Segment sum matrix for efficient matmul
        m_tot = len(segment_ids)
        S = torch.zeros(
            m_tot, self.num_segments, dtype=dtype, device=device
        )
        S[
            torch.arange(m_tot, device=device),
            self.segment_ids,
        ] = 1.0
        self.register_buffer("segment_sum_matrix", S)

    def forward(self, sphc: torch.Tensor) -> torch.Tensor:
        """
        Args:
            sphc: shape (B, m_tot)
        Returns:
            shape (B, len(degrees))
        """
        if not hasattr(self, "segment_sum_matrix"):
            m = self.segment_ids.shape[0]
            S = torch.zeros(
                m,
                self.num_segments,
                dtype=self.cg_rep.dtype,
                device=self.cg_rep.device,
            )
            S[
                torch.arange(m, device=S.device),
                self.segment_ids,
            ] = 1.0
            self.register_buffer("segment_sum_matrix", S)
        weighted = sphc * sphc * self.cg_rep
        return weighted @ self.segment_sum_matrix.to(
            weighted.dtype
        )
