"""NVAlchemi integration for So3krates-torch models.

Wraps SO3LR (and So3krates) as a NVAlchemi BaseModelMixin so that
nvalchemi's MD integrators (NVTLangevin, NPT, …) can drive the model
directly on GPU without LAMMPS or its ghost-atom machinery.

Usage::

    from so3krates_torch.calculator.so3 import SO3LRCalculator
    from so3krates_torch.calculator.nvalchemi_so3 import NVAlchemiSO3LR
    from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
    from nvalchemi.data.atomic_data import AtomicData
    import torch

    calc = SO3LRCalculator(device="cuda", dtype=torch.float64)
    model = NVAlchemiSO3LR(calc.models[0])

    # Build AtomicData from an ASE Atoms object
    data = AtomicData.from_ase(atoms, device="cuda")

    integrator = NVTLangevin(
        model=model,
        dt=0.5e-3,          # ps
        temperature=300.0,  # K
        friction=1e-2,      # 1/ps
    )
    integrator.run(data, n_steps=10_000)
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.models.base import BaseModelMixin, ModelConfig, NeighborConfig
from nvalchemi.models.base import NeighborListFormat


class NVAlchemiSO3LR(nn.Module, BaseModelMixin):
    """Wraps a So3krates / SO3LR model for NVAlchemi MD.

    nvalchemi builds and updates the neighbor list every step using its
    GPU-native cell-list kernel at ``r_max_lr`` (or ``r_max`` for
    short-range-only models).  ``adapt_input`` then splits the full
    neighbor list into SR and LR sublists and assembles the dict that
    ``SO3LR.forward`` expects.

    Parameters
    ----------
    model:
        An instantiated So3krates or SO3LR ``nn.Module``.  Must expose
        ``r_max`` (and optionally ``r_max_lr`` for long-range models).
    """

    def __init__(self, model: nn.Module) -> None:
        super().__init__()
        self.model = model
        self._cached_dtype: torch.dtype = next(model.parameters()).dtype

        self.r_sr: float = float(model.r_max)
        self.has_lr: bool = hasattr(model, "r_max_lr")
        self.r_lr: float = float(model.r_max_lr) if self.has_lr else self.r_sr
        self.num_elements: int = int(model.num_elements)

        # Pre-build one-hot lookup table: _node_emb[z] → one-hot row.
        # z runs 1..118 (atomic number); index 0 is a zero-pad.
        node_emb = torch.zeros(
            self.num_elements + 1,
            self.num_elements,
            dtype=self._cached_dtype,
        )
        for z in range(1, self.num_elements + 1):
            node_emb[z, z - 1] = 1.0
        model_device = next(model.parameters()).device
        node_emb = node_emb.to(device=model_device)
        self.register_buffer("_node_emb", node_emb, persistent=False)

        self.model_config = ModelConfig(
            outputs=frozenset({"energy", "forces"}),
            active_outputs={"energy", "forces"},
            autograd_outputs=frozenset({"forces"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset(),
            optional_inputs=frozenset({"cell"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=self.r_lr,
                format=NeighborListFormat.COO,
                half_list=False,
            ),
        )

    # ------------------------------------------------------------------
    # BaseModelMixin required interface
    # ------------------------------------------------------------------

    @property
    def embedding_shapes(self) -> dict[str, tuple[int, ...]]:
        return {"node_embeddings": (int(self.model.num_features),)}

    def compute_embeddings(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> AtomicData | Batch:
        raise NotImplementedError(
            "NVAlchemiSO3LR does not expose intermediate embeddings."
        )

    # ------------------------------------------------------------------
    # Input / output adaptation
    # ------------------------------------------------------------------

    def adapt_input(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> dict[str, Any]:
        if isinstance(data, AtomicData):
            data = Batch.from_data_list([data])

        dtype = self._cached_dtype
        device = data.positions.device

        # ── Positions (requires_grad for force autograd) ──────────────
        positions = data.positions.to(dtype=dtype).clone()
        positions.requires_grad_(True)

        # ── Neighbor list: nvalchemi COO [E, 2] → [2, E] ─────────────
        # neighbor_list_shifts: integer PBC image indices [E, 3]
        edge_index_lr = data.neighbor_list.long().T  # [2, E]
        ni, nj = edge_index_lr[0], edge_index_lr[1]

        # ── Shift vectors (Cartesian) ─────────────────────────────────
        # neighbor_list_shifts @ cell[graph_of_sender] gives the
        # Cartesian offset for the periodic image.
        nl_shifts = getattr(data, "neighbor_list_shifts", None)
        E = ni.shape[0]
        if nl_shifts is not None:
            nl_shifts = nl_shifts.to(dtype=dtype, device=device)
        else:
            nl_shifts = torch.zeros(E, 3, dtype=dtype, device=device)

        cell_raw = getattr(data, "cell", None)
        B = data.num_graphs
        if cell_raw is not None:
            cell = cell_raw.to(dtype=dtype, device=device)
            if cell.dim() == 2:
                cell = cell.unsqueeze(0)  # [1, 3, 3]
        else:
            cell = (
                torch.eye(3, dtype=dtype, device=device)
                .unsqueeze(0)
                .expand(B, -1, -1)
                .contiguous()
            )

        batch_per_edge = data.batch_idx[ni]
        # shifts[e] = integer_shift[e] @ cell[graph_of_ni[e]]
        shifts = torch.einsum(
            "eb,ebc->ec", nl_shifts, cell[batch_per_edge]
        )  # [E, 3]

        # Full displacement vectors (i→j, PBC-wrapped) [E, 3]
        vectors_lr = positions[nj] - positions[ni] + shifts

        # ── SR / LR split by distance ─────────────────────────────────
        lengths_lr = torch.linalg.norm(vectors_lr, dim=1)
        sr_mask = lengths_lr < self.r_sr

        # prepare_graph (input_convention="positions") computes edge vectors
        # internally as:  vectors[e] = positions[nj] - positions[ni] + shifts[e]
        # So we pass shifts (Cartesian) and unit_shifts (integer) rather than
        # pre-computed vectors. unit_shifts are needed for the stress path.
        shifts_sr = shifts[sr_mask]  # [E_sr, 3] Cartesian
        unit_shifts_sr = nl_shifts[sr_mask]  # [E_sr, 3] integer

        d: dict[str, Any] = {
            # One-hot via lookup table — single GPU op, no CPU round-trip
            "node_attrs": self._node_emb.index_select(
                0, data.atomic_numbers.long()
            ),
            "atomic_numbers": data.atomic_numbers.long(),
            "batch": data.batch_idx.long(),
            # CSR-style batch pointer [B+1]: needed by So3krates.forward
            "ptr": data.batch_ptr.long(),
            # Head index per graph (0 for single-head models)
            "head": torch.zeros(B, dtype=torch.long, device=device),
            "edge_index": edge_index_lr[:, sr_mask],  # [2, E_sr]
            "shifts": shifts_sr,  # [E_sr, 3]
            "unit_shifts": unit_shifts_sr,  # [E_sr, 3]
            # PME and cell-dependent terms need real positions and cell
            "positions": positions,  # [N, 3]
            "cell": cell,  # [B, 3, 3]
        }
        if self.has_lr:
            d["edge_index_lr"] = edge_index_lr  # [2, E_lr]
            d["shifts_lr"] = shifts  # [E_lr, 3]
            d["unit_shifts_lr"] = nl_shifts  # [E_lr, 3]

        return d

    def adapt_output(
        self,
        raw_output: dict[str, Any],
        data: AtomicData | Batch,
    ) -> dict[str, Any]:
        return {
            "energy": raw_output["energy"],
            "forces": raw_output["forces"],
        }

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> dict[str, Any]:
        model_inputs = self.adapt_input(data, **kwargs)
        raw = self.model(model_inputs, compute_force=True)
        return self.adapt_output(raw, data)
