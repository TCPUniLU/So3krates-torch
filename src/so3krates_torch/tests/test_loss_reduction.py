"""Per-graph loss reduction (JAX so3lr parity) and DDP global counts."""

import pytest
import torch
import torch.distributed as dist

from so3krates_torch.modules.loss_reduction import (
    global_count,
    per_graph_node_mean,
    reduce_graphs,
    reduce_valid_elements,
)
from so3krates_torch.tools.torch_geometric import Batch


def _batch():
    # 2 graphs: 3 atoms + 2 atoms
    return Batch(
        ptr=torch.tensor([0, 3, 5]),
        batch=torch.tensor([0, 0, 0, 1, 1]),
    )


@pytest.fixture
def fake_ddp(monkeypatch):
    """Two ranks; the other rank's contribution to every all_reduce is
    taken from `other` (a list consumed in call order)."""
    calls = {"n": 0, "other": []}

    def all_reduce(tensor, op=None):
        calls["n"] += 1
        tensor += calls["other"].pop(0)

    monkeypatch.setattr(dist, "is_initialized", lambda: True)
    monkeypatch.setattr(dist, "get_world_size", lambda: 2)
    monkeypatch.setattr(dist, "all_reduce", all_reduce)
    return calls


def test_per_graph_node_mean_values():
    ref = _batch()
    penalty = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [2.0, 2.0, 2.0],
            [3.0, 3.0, 3.0],
            [4.0, 4.0, 4.0],
            [0.0, 0.0, 6.0],
        ]
    )
    label = torch.zeros(5, 3)
    per_graph, has_data = per_graph_node_mean(
        penalty, ref, label, torch.tensor([1.0, 2.0])
    )
    # graph 0: mean(1, 2, 3) = 2 ; graph 1: mean(4, 2) = 3, weight 2
    assert torch.allclose(per_graph, torch.tensor([2.0, 6.0]))
    assert has_data.tolist() == [True, True]


def test_per_graph_zero_weight_graph_has_no_data():
    ref = _batch()
    per_graph, has_data = per_graph_node_mean(
        torch.ones(5), ref, torch.zeros(5), torch.tensor([1.0, 0.0])
    )
    assert has_data.tolist() == [True, False]
    assert per_graph[1].item() == 0.0


def test_per_graph_nan_label_excludes_whole_graph():
    ref = _batch()
    label = torch.tensor([0.0, 0.0, 0.0, float("nan"), 0.0])
    per_graph, has_data = per_graph_node_mean(
        torch.ones(5), ref, label, torch.ones(2)
    )
    assert has_data.tolist() == [True, False]
    assert torch.isfinite(per_graph).all()


def test_reduce_graphs_single_process():
    loss = reduce_graphs(
        torch.tensor([1.0, 3.0, 0.0]),
        torch.tensor([True, True, False]),
        ddp=False,
    )
    assert loss.item() == pytest.approx(2.0)


def test_reduce_graphs_ddp_matches_single_process(fake_ddp):
    # local: 2 data graphs summing to 4; other rank: 3 data graphs
    fake_ddp["other"] = [torch.tensor(3.0)]
    loss = reduce_graphs(
        torch.tensor([1.0, 3.0]), torch.tensor([True, True]), ddp=True
    )
    # DDP averages rank losses: this rank must hold 4 * 2 / 5
    assert loss.item() == pytest.approx(1.6)
    assert fake_ddp["n"] == 1


def test_reduce_graphs_ddp_zero_local_count(fake_ddp):
    fake_ddp["other"] = [torch.tensor(3.0)]
    loss = reduce_graphs(torch.tensor([5.0]), torch.tensor([False]), ddp=True)
    assert loss.item() == 0.0
    assert fake_ddp["n"] == 1  # still joined the collective


def test_global_count_no_ddp_is_identity():
    assert global_count(torch.tensor(4.0), ddp=False).item() == 4.0


def test_reduce_valid_elements_dipole_like():
    penalty = torch.tensor([[1.0, 2.0, 3.0], [10.0, 10.0, 10.0]])
    valid = torch.tensor([[True] * 3, [False] * 3])
    weight = torch.tensor([[2.0] * 3, [1.0] * 3])
    loss = reduce_valid_elements(penalty, valid, weight, ddp=False)
    assert loss.item() == pytest.approx(2.0 * 6.0 / 3)


def test_reduce_valid_elements_nothing_valid_is_zero():
    loss = reduce_valid_elements(
        torch.ones(2),
        torch.zeros(2, dtype=torch.bool),
        torch.ones(2),
        ddp=False,
    )
    assert loss.item() == 0.0
