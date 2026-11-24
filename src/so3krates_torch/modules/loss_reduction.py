"""Loss reductions matching the JAX so3lr trainer (`training_utils.py`).

`per_graph` reduction: per-atom penalties are averaged per graph, then
averaged over the graphs that carry the label (`node_mse_loss`);
graph-level penalties are averaged over valid elements
(`graph_mse_loss`). Under DDP the number of labelled graphs/elements is
all-reduced so every structure gets the same weight regardless of which
rank it lands on: each rank returns ``local_sum * world_size /
global_count`` and DDP's gradient averaging turns that into
``global_sum / global_count``.
"""

from typing import Optional, Tuple

import torch
import torch.distributed as dist
from torch_geometric.data import Batch


def is_ddp_enabled() -> bool:
    return dist.is_initialized() and dist.get_world_size() > 1


def _use_ddp(ddp: Optional[bool]) -> bool:
    return is_ddp_enabled() if ddp is None else bool(ddp)


def global_count(
    count: torch.Tensor, ddp: Optional[bool] = None
) -> torch.Tensor:
    """Sum `count` over ranks (no autograd). Collective: every rank must
    call it for the same term on every step, also when its count is 0."""
    count = count.detach().clone()
    if _use_ddp(ddp) and dist.is_initialized():
        dist.all_reduce(count, op=dist.ReduceOp.SUM)
    return count


def _world_size(ddp: Optional[bool]) -> int:
    if _use_ddp(ddp) and dist.is_initialized():
        return dist.get_world_size()
    return 1


def per_graph_node_mean(
    penalty: torch.Tensor,
    ref: Batch,
    label: torch.Tensor,
    graph_weight: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Per-graph mean of per-atom penalties.

    Each atom's penalty is first averaged over its component dims (e.g.
    x/y/z), then averaged over all atoms of its graph. A graph has data
    iff all its labels are finite and its combined weight is non-zero;
    otherwise its entry is 0 (JAX drops a graph with any NaN label).
    `penalty` must already be finite (zeroed where the label is NaN).

    Returns (graph_weight * per-graph mean [n_graphs], has_data [n_graphs]).
    """
    n_graphs = graph_weight.shape[0]
    n_atoms = penalty.shape[0]
    per_atom = penalty.reshape(n_atoms, -1).mean(dim=-1)
    finite_atom = torch.isfinite(label.reshape(n_atoms, -1)).all(dim=-1)

    sums = torch.zeros(
        n_graphs, dtype=per_atom.dtype, device=per_atom.device
    ).index_add(0, ref.batch, per_atom)
    counts = torch.bincount(ref.batch, minlength=n_graphs).to(per_atom.dtype)
    n_finite = torch.bincount(
        ref.batch, weights=finite_atom.to(per_atom.dtype), minlength=n_graphs
    )
    has_data = (n_finite == counts) & (graph_weight != 0) & (counts > 0)

    mean = sums / torch.clamp(counts, min=1.0)
    per_graph = torch.where(
        has_data, graph_weight * mean, torch.zeros_like(mean)
    )
    return per_graph, has_data


def reduce_graphs(
    per_graph: torch.Tensor,
    has_data: torch.Tensor,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    """Mean of `per_graph` over graphs with data, globally across ranks."""
    local_sum = torch.where(
        has_data, per_graph, torch.zeros_like(per_graph)
    ).sum()
    count = global_count(has_data.sum().to(per_graph.dtype), ddp)
    return local_sum * _world_size(ddp) / torch.clamp(count, min=1.0)


def reduce_valid_elements(
    penalty: torch.Tensor,
    valid: torch.Tensor,
    weight: torch.Tensor,
    ddp: Optional[bool] = None,
) -> torch.Tensor:
    """Weighted mean of `penalty` over `valid` elements, globally across
    ranks (JAX `graph_mse_loss`). `penalty` must be finite everywhere:
    zero the residual of invalid entries *before* computing the penalty,
    since NaN residuals give NaN gradients even when masked out here."""
    contrib = torch.where(valid, weight * penalty, torch.zeros_like(penalty))
    count = global_count(valid.sum().to(penalty.dtype), ddp)
    return contrib.sum() * _world_size(ddp) / torch.clamp(count, min=1.0)
