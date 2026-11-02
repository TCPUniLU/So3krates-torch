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

    # Build a (multi-graph) Batch from one or more ASE Atoms objects
    data = AtomicData.from_atoms(atoms, device="cuda", dtype=torch.float64)
    batch = Batch.from_data_list([data])

    integrator = NVTLangevin(
        model=model,
        dt=0.5,             # fs
        temperature=300.0,  # K
        friction=1e-2,      # 1/fs
        n_steps=10_000,
    )
    batch = integrator.run(batch)
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
    electrostatics:
        Long-range electrostatics backend.

        - ``"model"`` (default): the model computes its own electrostatics
          (real-space erf-Coulomb when ``use_pme=False`` or the model's
          torch-pme when ``use_pme=True``).
        - ``"nvalchemi"``: the model emits partial charges only (no internal
          electrostatic energy) so that NVAlchemi's native PME can supply the
          long-range Coulomb energy.  Requires the model to be configured with
          ``electrostatic_energy_bool=False`` and ``output_partial_charges=True``
          (use :func:`build_nvalchemi_pme_model`, which sets this up and
          composes the PME step).
    skin:
        Verlet skin (Å) forwarded to ``model_config.neighbor_config.skin``,
        so :meth:`make_neighbor_hooks` builds the neighbor list out to
        ``r_lr + skin`` and only rebuilds it once an atom has moved more
        than ``skin / 2``.  ``adapt_input`` drops every pair with
        ``r >= r_lr`` (the skin shell) before calling the model, so the
        result does not depend on ``skin``.  Default ``0.0`` (rebuild every
        step).

    Notes
    -----
    *Active outputs.*  ``forward`` requests forces / stress from the model
    only when they are in ``model_config.active_outputs``.  The default is
    all outputs (including stress); nvalchemi-toolkit 0.2 integrators do
    **not** narrow it, so callers that do not need stress (e.g. NVT) should
    call :func:`set_active_outputs` (the ``torchkrates-md-nvalchemi`` CLI
    does this per ensemble).  With neither active, the wrapper runs
    an energy-only forward and returns an ``energy`` that stays attached to
    ``data.positions`` (used as given, not cloned, when it already requires
    grad), so autograd pipeline groups can differentiate through it.

    *Charge and spin.*  The total charge is read from ``data.charge``
    (``[B, 1]``; :meth:`AtomicData.from_atoms` fills it from the integer
    ``atoms.info["charge"]``, the same key :class:`TorchkratesCalculator`
    uses).  The total spin is read from ``data.spin`` (``[B, 1]``) and is
    interpreted like the calculator's ``atoms.info["total_spin"]``: the
    total spin *S* (the model uses ``2 * S`` unpaired electrons), **not**
    the multiplicity.  ``AtomicData.from_atoms`` does not populate
    ``spin``; :func:`so3krates_torch.cli.run_md_nvalchemi.atoms_to_data`
    fills it from ``atoms.info["total_spin"]``, otherwise set it explicitly
    (e.g. ``data.spin = torch.tensor([[S]])``) for spin-embedding models.
    Both default to 0 when absent.

    *Dtype.*  Inputs are cast to the dtype of the wrapped model's
    parameters at call time, so ``wrapper.to(dtype)`` (or converting the
    inner model) is honoured; there is no dtype frozen at construction.
    """

    def __init__(
        self,
        model: nn.Module,
        electrostatics: str = "model",
        skin: float = 0.0,
    ) -> None:
        super().__init__()
        if electrostatics not in ("model", "nvalchemi"):
            raise ValueError(
                "electrostatics must be 'model' or 'nvalchemi', got "
                f"{electrostatics!r}"
            )
        self.electrostatics = electrostatics
        if electrostatics == "nvalchemi":
            if getattr(model, "electrostatic_energy_bool", False):
                raise ValueError(
                    "electrostatics='nvalchemi' requires the model's own "
                    "electrostatics to be disabled "
                    "(electrostatic_energy_bool=False) to avoid "
                    "double-counting; use build_nvalchemi_pme_model()."
                )
            if not getattr(model, "output_partial_charges", False):
                raise ValueError(
                    "electrostatics='nvalchemi' requires the model to emit "
                    "charges (output_partial_charges=True); use "
                    "build_nvalchemi_pme_model()."
                )
        self.model = model

        self.r_sr: float = float(model.r_max)
        self.has_lr: bool = (
            hasattr(model, "r_max_lr") and model.r_max_lr is not None
        )
        self.r_lr: float = float(model.r_max_lr) if self.has_lr else self.r_sr
        self.num_elements: int = int(model.num_elements)

        # Pre-build one-hot lookup table: _node_emb[z] → one-hot row.
        # z runs 1..118 (atomic number); index 0 is a zero-pad.
        node_emb = torch.zeros(
            self.num_elements + 1,
            self.num_elements,
            dtype=self._model_dtype,
        )
        for z in range(1, self.num_elements + 1):
            node_emb[z, z - 1] = 1.0
        model_device = next(model.parameters()).device
        node_emb = node_emb.to(device=model_device)
        self.register_buffer("_node_emb", node_emb, persistent=False)

        outputs = {"energy", "forces", "stress"}
        if electrostatics == "nvalchemi":
            # Partial charges are wired into the downstream PME step.
            outputs.add("charges")
        self.model_config = ModelConfig(
            outputs=frozenset(outputs),
            active_outputs=set(outputs),
            autograd_outputs=frozenset({"forces", "stress"}),
            autograd_inputs=frozenset({"positions"}),
            required_inputs=frozenset(),
            optional_inputs=frozenset({"cell"}),
            supports_pbc=True,
            needs_pbc=False,
            neighbor_config=NeighborConfig(
                cutoff=self.r_lr,
                format=NeighborListFormat.COO,
                half_list=False,
                skin=skin,
            ),
        )

    @property
    def _model_dtype(self) -> torch.dtype:
        """Floating dtype of the wrapped model's parameters (read live, so
        it follows ``.to(dtype)`` / ``.double()`` / ``.float()``)."""
        return next(self.model.parameters()).dtype

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

        dtype = self._model_dtype
        device = data.positions.device

        # ── Positions ─────────────────────────────────────────────────
        # A tensor that already requires grad (pipeline autograd group) is
        # used as-is so the caller can differentiate through it.  Otherwise
        # clone, so neither our requires_grad_ nor prepare_graph's (which
        # always enables grad on data["positions"]) mutates a caller leaf.
        active = self.model_config.active_outputs
        needs_grad = bool({"forces", "stress"} & active)
        positions = data.positions.to(dtype=dtype)
        if not positions.requires_grad:
            positions = positions.clone()
            if needs_grad:
                positions.requires_grad_(True)

        # ── Neighbor list: nvalchemi COO [E, 2] → [2, E] ─────────────
        # neighbor_list_shifts: integer PBC image indices [E, 3]
        edge_index_all = data.neighbor_list.long().T  # [2, E]
        E = edge_index_all.shape[1]
        nl_shifts = getattr(data, "neighbor_list_shifts", None)
        if nl_shifts is not None:
            nl_shifts = nl_shifts.to(dtype=dtype, device=device)
        else:
            nl_shifts = torch.zeros(E, 3, dtype=dtype, device=device)

        # Drop sentinel / padding edges (endpoint == n_atoms) — they would
        # index out of bounds in the gathers below.
        n_atoms = positions.shape[0]
        valid = (edge_index_all[0] < n_atoms) & (edge_index_all[1] < n_atoms)
        edge_index_all = edge_index_all[:, valid]
        nl_shifts = nl_shifts[valid]
        ni, nj = edge_index_all[0], edge_index_all[1]

        # ── Shift vectors (Cartesian) ─────────────────────────────────
        # neighbor_list_shifts @ cell[graph_of_sender] gives the
        # Cartesian offset for the periodic image.
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

        # ── SR / LR split by distance ─────────────────────────────────
        # NeighborListHook builds out to cutoff + skin; drop the skin
        # shell so the model sees exactly the pairs with r < r_lr (as
        # the ASE calculator path does).  The masks only select edges —
        # prepare_graph recomputes the vectors with autograd.
        with torch.no_grad():
            # Full displacement vectors (i→j, PBC-wrapped) [E, 3]
            vectors_all = positions[nj] - positions[ni] + shifts
            lengths_lr = torch.linalg.norm(vectors_all, dim=1)
        lr_mask = lengths_lr < self.r_lr
        sr_mask = lengths_lr < self.r_sr

        # prepare_graph (input_convention="positions") computes edge vectors
        # internally as:  vectors[e] = positions[nj] - positions[ni] + shifts[e]
        # So we pass shifts (Cartesian) and unit_shifts (integer) rather than
        # pre-computed vectors. unit_shifts are needed for the stress path.
        shifts_sr = shifts[sr_mask]  # [E_sr, 3] Cartesian
        unit_shifts_sr = nl_shifts[sr_mask]  # [E_sr, 3] integer

        charge = getattr(data, "charge", None)
        spin = getattr(data, "spin", None)

        d: dict[str, Any] = {
            # One-hot via lookup table — single GPU op, no CPU round-trip
            # (.to is a no-op unless the inner model was converted
            # directly, bypassing this wrapper's .to()).
            "node_attrs": self._node_emb.index_select(
                0, data.atomic_numbers.long()
            ).to(dtype=dtype, device=device),
            "atomic_numbers": data.atomic_numbers.long(),
            "batch": data.batch_idx.long(),
            # CSR-style batch pointer [B+1]: needed by So3krates.forward
            "ptr": data.batch_ptr.long(),
            # Head index per graph (0 for single-head models)
            "head": torch.zeros(B, dtype=torch.long, device=device),
            "edge_index": edge_index_all[:, sr_mask],  # [2, E_sr]
            "shifts": shifts_sr,  # [E_sr, 3]
            "unit_shifts": unit_shifts_sr,  # [E_sr, 3]
            # PME and cell-dependent terms need real positions and cell
            "positions": positions,  # [N, 3]
            "cell": cell,  # [B, 3, 3]
            # Total charge / spin per graph (see class docstring).
            "total_charge": (
                charge.reshape(B).to(dtype=dtype, device=device)
                if charge is not None
                else torch.zeros(B, dtype=dtype, device=device)
            ),
            "total_spin": (
                spin.reshape(B).to(dtype=dtype, device=device)
                if spin is not None
                else torch.zeros(B, dtype=dtype, device=device)
            ),
        }
        if self.has_lr:
            d["edge_index_lr"] = edge_index_all[:, lr_mask]  # [2, E_lr]
            d["shifts_lr"] = shifts[lr_mask]  # [E_lr, 3]
            d["unit_shifts_lr"] = nl_shifts[lr_mask]  # [E_lr, 3]

        return d

    def adapt_output(
        self,
        raw_output: dict[str, Any],
        data: AtomicData | Batch,
    ) -> dict[str, Any]:
        out = {
            # nvalchemi expects [B, 1]; LoggingHook squeezes to [B]
            "energy": raw_output["energy"].unsqueeze(-1),
            # None when not requested (inactive in model_config).
            "forces": raw_output.get("forces"),
            # The model emits ASE-convention stress (+dE/de / V).
            # nvalchemi-toolkit>=0.2 uses the same convention
            # (tensile-positive Cauchy stress, virial W = -stress * V),
            # so it is passed through unchanged.
            "stress": raw_output.get("stress"),  # [B, 3, 3]
        }
        if self.electrostatics == "nvalchemi":
            # Per-atom charges [N], wired into the PME step downstream.
            out["charges"] = raw_output["partial_charges"].reshape(-1)
        # Base mapper keeps only active & supported keys (inactive /
        # uncomputed ones come back as None).
        return super().adapt_output(out, data)

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------

    def forward(
        self, data: AtomicData | Batch, **kwargs: Any
    ) -> dict[str, Any]:
        active = self.model_config.active_outputs & self.model_config.outputs
        model_inputs = self.adapt_input(data, **kwargs)
        raw = self.model(
            model_inputs,
            compute_force="forces" in active,
            compute_stress="stress" in active,
        )
        return self.adapt_output(raw, data)


def set_active_outputs(model: nn.Module, outputs) -> None:
    """Narrow ``model_config.active_outputs`` for MD.

    nvalchemi-toolkit 0.2 integrators do not set ``active_outputs``
    themselves, so without this every step computes stress (needed only
    for NPT).  Use ``{"energy", "forces"}`` for NVT and
    ``{"energy", "forces", "stress"}`` for NPT.

    Works on a bare :class:`NVAlchemiSO3LR` and on the
    ``PipelineModelWrapper`` from :func:`build_nvalchemi_pme_model`.  For a
    pipeline, every step's model is narrowed too (direct groups use each
    sub-model's own ``active_outputs``), but a step's ``charges`` output is
    kept active so the SR -> PME charge wiring stays intact.
    """
    outputs = set(outputs)
    for group in getattr(model, "groups", ()):
        for step in group.steps:
            sub = step.model
            want = (outputs | {"charges"}) & set(sub.model_config.outputs)
            sub.set_config("active_outputs", want)
    model.set_config(
        "active_outputs", outputs & set(model.model_config.outputs)
    )


# Standard Coulomb constant in eV·Å/e² — matches the model's `ke`
# (so3krates_torch.blocks.physical_potentials).
_KE_EV_ANG = 14.399645351950548


def build_nvalchemi_pme_model(
    model: nn.Module,
    *,
    mesh_spacing: float = 1.0,
    alpha: float = None,
    spline_order: int = 4,
    accuracy: float = 1e-6,
    coulomb_constant: float = _KE_EV_ANG,
    hybrid_forces: bool = True,
    skin: float = 0.0,
):
    """Compose an SO3LR model with NVAlchemi-native PME electrostatics.

    The model's *own* electrostatic energy is disabled and it is switched to
    emit partial charges; NVAlchemi's GPU-native PME (full Ewald) then supplies
    the long-range Coulomb energy from those charges.  ZBL repulsion and the
    (real-space) dispersion term are unchanged.

    The returned object is a NVAlchemi ``PipelineModelWrapper`` that runs the
    short-range model first, wires its ``charges`` output into the PME step,
    and sums the energy/forces/stress of both.

    .. warning::
        This MUTATES ``model`` in place (sets ``electrostatic_energy_bool=False``
        and ``output_partial_charges=True``).  It is a *different* electrostatics
        backend than the model's own erf-Coulomb / torch-pme term and does not
        reproduce them numerically — only use it deliberately.

    Parameters
    ----------
    model:
        An SO3LR model exposing ``r_max_lr`` (the PME real-space cutoff).
    alpha:
        Ewald splitting parameter (inverse Å) — NVAlchemi's analogue of the
        Gaussian smearing.  ``None`` (default) auto-estimates it from
        ``accuracy`` via the Kolafa-Perram formula each time the cell changes.
    mesh_spacing, spline_order, accuracy, coulomb_constant, hybrid_forces:
        Forwarded to :class:`nvalchemi.models.pme.PMEModelWrapper`.  The
        default ``coulomb_constant`` matches the model's ``ke``.
    skin:
        Verlet skin (Å) forwarded to the short-range :class:`NVAlchemiSO3LR`
        step's ``model_config.neighbor_config.skin`` (see its docstring).
        Default ``0.0``.
    """
    from nvalchemi.models.pme import PMEModelWrapper
    from nvalchemi.models.pipeline import (
        PipelineGroup,
        PipelineModelWrapper,
        PipelineStep,
    )

    if getattr(model, "r_max_lr", None) is None:
        raise ValueError(
            "build_nvalchemi_pme_model requires a long-range cutoff "
            "(model.r_max_lr); short-range-only models have no electrostatics."
        )

    model.electrostatic_energy_bool = False
    model.output_partial_charges = True

    sr = NVAlchemiSO3LR(model, electrostatics="nvalchemi", skin=skin)
    pme = PMEModelWrapper(
        cutoff=float(model.r_max_lr),
        mesh_spacing=mesh_spacing,
        alpha=alpha,
        spline_order=spline_order,
        accuracy=accuracy,
        coulomb_constant=coulomb_constant,
        hybrid_forces=hybrid_forces,
    )
    group = PipelineGroup(
        steps=[PipelineStep(sr, wire={"charges": "charges"}), pme],
        use_autograd=False,
    )
    pipeline = PipelineModelWrapper(
        [group], additive_keys={"energy", "forces", "stress"}
    )
    # The pipeline unions sub-model active_outputs (which includes the SR
    # step's "charges"); narrow it to the MD-relevant outputs so integrators
    # only request energy/forces/stress.
    pipeline.model_config.active_outputs = {"energy", "forces", "stress"}
    return pipeline
