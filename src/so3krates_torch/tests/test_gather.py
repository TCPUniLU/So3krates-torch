"""Tests for ``gather_rows`` and the gather sites that use it.

``x[idx]`` backpropagates through ``index_put_(accumulate=True)``, which is
extremely slow on CUDA when millions of edges accumulate onto a few thousand
rows.  ``x.index_select(0, idx)`` computes the same values, but its backward
is ``index_add_``.  The tests below pin (a) equal values and gradients and
(b) the backward op that makes the difference.
"""

import pytest
import torch

from so3krates_torch.tools.gather import gather_rows

INDEX_PUT_OPS = {"aten::_index_put_impl_", "aten::index_put_"}


def _backward_ops(fn):
    """Names of all ops executed while running ``fn`` (a backward pass)."""
    with torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.CPU]
    ) as prof:
        fn()
    return {e.key for e in prof.key_averages()}


def _grad_pair(x, idx, w):
    new = torch.autograd.grad((gather_rows(x, idx) * w).sum(), x)[0]
    ref = torch.autograd.grad((x[idx] * w).sum(), x)[0]
    return new, ref


@pytest.mark.parametrize("shape", [(7,), (7, 3), (7, 4, 5)])
@pytest.mark.parametrize("idx_dtype", [torch.int64, torch.int32])
def test_gather_rows_values_match_advanced_indexing(shape, idx_dtype):
    g = torch.Generator().manual_seed(0)
    x = torch.randn(shape, generator=g, dtype=torch.float64)
    idx = torch.randint(0, 7, (40,), generator=g).to(idx_dtype)
    assert torch.equal(gather_rows(x, idx), x[idx.long()])


def test_gather_rows_gradient_matches_advanced_indexing():
    g = torch.Generator().manual_seed(1)
    x = torch.randn(50, 3, generator=g, dtype=torch.float64).requires_grad_()
    idx = torch.randint(0, 50, (5000,), generator=g)
    w = torch.randn(5000, 3, generator=g, dtype=torch.float64)
    new, ref = _grad_pair(x, idx, w)
    torch.testing.assert_close(new, ref, atol=1e-12, rtol=0)


def test_gather_rows_all_indices_equal():
    g = torch.Generator().manual_seed(2)
    x = torch.randn(1, 3, 3, generator=g, dtype=torch.float64).requires_grad_()
    idx = torch.zeros(1000, dtype=torch.long)
    w = torch.randn(1000, 3, 3, generator=g, dtype=torch.float64)
    new, ref = _grad_pair(x, idx, w)
    torch.testing.assert_close(new, ref, atol=1e-12, rtol=0)


def test_gather_rows_empty_index():
    x = torch.randn(5, 3, dtype=torch.float64, requires_grad=True)
    idx = torch.empty(0, dtype=torch.long)
    out = gather_rows(x, idx)
    assert out.shape == (0, 3)
    (grad,) = torch.autograd.grad(out.sum(), x)
    assert torch.equal(grad, torch.zeros_like(x))


def test_gather_rows_indexes_beyond_real_atoms():
    """LAMMPS ML-IAP: indices run over real + ghost atoms (``n_total``)."""
    g = torch.Generator().manual_seed(3)
    x = torch.randn(10, 3, generator=g, dtype=torch.float64).requires_grad_()
    idx = torch.tensor([9, 9, 0, 5, 9, 7], dtype=torch.int32)
    assert torch.equal(gather_rows(x, idx), x[idx.long()])
    w = torch.randn(6, 3, generator=g, dtype=torch.float64)
    new, ref = _grad_pair(x, idx.long(), w)
    torch.testing.assert_close(new, ref, atol=1e-12, rtol=0)


def test_gather_rows_traces_without_graph_break():
    x = torch.randn(9, 3, dtype=torch.float64)
    idx = torch.randint(0, 9, (30,))
    compiled = torch.compile(gather_rows, backend="eager", fullgraph=True)
    assert torch.equal(compiled(x, idx), gather_rows(x, idx))


def test_gather_rows_float32():
    g = torch.Generator().manual_seed(4)
    x = torch.randn(50, 3, generator=g).requires_grad_()
    idx = torch.randint(0, 50, (3000,), generator=g)
    w = torch.randn(3000, 3, generator=g)
    new, ref = _grad_pair(x, idx, w)
    torch.testing.assert_close(new, ref, atol=1e-4, rtol=1e-5)


def test_gather_rows_backward_has_no_index_put():
    x = torch.randn(20, 3, dtype=torch.float64, requires_grad=True)
    idx = torch.randint(0, 20, (500,))

    new_loss = gather_rows(x, idx).sum()
    ops = _backward_ops(lambda: new_loss.backward())
    assert "aten::index_add_" in ops
    assert not (ops & INDEX_PUT_OPS)

    # Premise of the whole change, pinned on the installed torch:
    ref_loss = x[idx].sum()
    ref_ops = _backward_ops(lambda: ref_loss.backward())
    assert ref_ops & INDEX_PUT_OPS


# ---------------------------------------------------------------------------
# Gather sites.  The equivalence tests compare each function with itself under
# plain advanced indexing (``gather_rows`` swapped for ``x[idx]``), so the old
# formulas are not duplicated here.  ``monkeypatch(..., raising=False)`` keeps
# them meaningful (and passing) before a module imports ``gather_rows``.
# ---------------------------------------------------------------------------

import so3krates_torch.blocks.physical_potentials as pp
import so3krates_torch.data.utils as du


def _plain(x, idx):
    return x[idx]


def _new_and_reference(module, monkeypatch, fn):
    """``fn()`` as implemented now, and with plain-indexing gathers."""
    new = fn()
    with monkeypatch.context() as m:
        m.setattr(module, "gather_rows", _plain, raising=False)
        ref = fn()
    return new, ref


def _assert_same(new, ref, atol=1e-12):
    assert len(new) == len(ref)
    for a, b in zip(new, ref):
        torch.testing.assert_close(a, b, atol=atol, rtol=0)


def _rng(seed):
    return torch.Generator().manual_seed(seed)


def _edge_inputs(n_nodes=30, n_edges=400, seed=10):
    g = _rng(seed)
    positions = torch.randn(n_nodes, 3, generator=g, dtype=torch.float64)
    edge_index = torch.randint(0, n_nodes, (2, n_edges), generator=g)
    shifts = torch.randn(n_edges, 3, generator=g, dtype=torch.float64)
    return positions.requires_grad_(), edge_index, shifts


@pytest.mark.parametrize("n_edges", [400, 0])
def test_get_edge_vectors_and_lengths_equivalent(monkeypatch, n_edges):
    positions, edge_index, shifts = _edge_inputs(n_edges=n_edges)
    w = torch.randn(n_edges, 4, generator=_rng(11), dtype=torch.float64)

    def run():
        vec, length = du.get_edge_vectors_and_lengths(
            positions, edge_index, shifts
        )
        loss = (torch.cat([vec, length], dim=-1) * w).sum()
        (grad,) = torch.autograd.grad(loss, positions)
        return vec.detach(), length.detach(), grad

    _assert_same(*_new_and_reference(du, monkeypatch, run))


def test_get_edge_vectors_and_lengths_backward_has_no_index_put():
    positions, edge_index, shifts = _edge_inputs()
    vec, length = du.get_edge_vectors_and_lengths(
        positions, edge_index, shifts
    )
    ops = _backward_ops(lambda: (vec.sum() + length.sum()).backward())
    assert not (ops & INDEX_PUT_OPS)


def _mixing_inputs(n_nodes=30, n_edges=400, seed=20):
    g = _rng(seed)
    z = torch.randint(1, 11, (n_nodes,), generator=g)
    idx_i = torch.randint(0, n_nodes, (n_edges,), generator=g)
    idx_j = torch.randint(0, n_nodes, (n_edges,), generator=g)
    h = 0.5 + torch.rand(n_nodes, generator=g, dtype=torch.float64)
    return z, idx_i, idx_j, h.requires_grad_()


@pytest.mark.parametrize("n_edges", [400, 0])
def test_mixing_rules_equivalent(monkeypatch, n_edges):
    z, idx_i, idx_j, h = _mixing_inputs(n_edges=n_edges)
    w = torch.randn(n_edges, 2, generator=_rng(21), dtype=torch.float64)

    def run():
        alpha_ij, c6_ij = pp.mixing_rules(z, idx_i, idx_j, h)
        loss = (torch.stack([alpha_ij, c6_ij], dim=-1) * w).sum()
        (grad,) = torch.autograd.grad(loss, h)
        return alpha_ij.detach(), c6_ij.detach(), grad

    _assert_same(*_new_and_reference(pp, monkeypatch, run))


def test_mixing_rules_backward_has_no_index_put():
    z, idx_i, idx_j, h = _mixing_inputs()
    alpha_ij, c6_ij = pp.mixing_rules(z, idx_i, idx_j, h)
    ops = _backward_ops(lambda: (alpha_ij.sum() + c6_ij.sum()).backward())
    assert not (ops & INDEX_PUT_OPS)


def _coulomb_inputs(n_nodes=30, n_edges=400, seed=30, trailing_dim=False):
    g = _rng(seed)
    q = torch.randn(n_nodes, generator=g, dtype=torch.float64)
    if trailing_dim:
        q = q.unsqueeze(-1)
    senders = torch.randint(0, n_nodes, (n_edges,), generator=g)
    receivers = torch.randint(0, n_nodes, (n_edges,), generator=g)
    rij = 1.0 + 5.5 * torch.rand(n_edges, generator=g, dtype=torch.float64)
    return q.requires_grad_(), rij, senders, receivers


@pytest.mark.parametrize("trailing_dim", [False, True])
@pytest.mark.parametrize("n_edges", [400, 0])
def test_coulomb_erf_energy_equivalent(monkeypatch, trailing_dim, n_edges):
    q, rij, s, r = _coulomb_inputs(n_edges=n_edges, trailing_dim=trailing_dim)
    w = torch.randn(n_edges, generator=_rng(31), dtype=torch.float64)

    def run():
        e = pp._coulomb_erf_energy(q, rij, s, r, 14.4, 1.5, 0.5)
        (grad,) = torch.autograd.grad((e * w).sum(), q)
        return e.detach(), grad

    _assert_same(*_new_and_reference(pp, monkeypatch, run))


def test_coulomb_erf_energy_backward_has_no_index_put():
    q, rij, s, r = _coulomb_inputs()
    e = pp._coulomb_erf_energy(q, rij, s, r, 14.4, 1.5, 0.5)
    ops = _backward_ops(lambda: e.sum().backward())
    assert not (ops & INDEX_PUT_OPS)


@pytest.mark.parametrize("n_edges", [400, 0])
def test_coulomb_erf_shifted_force_smooth_energy_equivalent(
    monkeypatch, n_edges
):
    q, rij, s, r = _coulomb_inputs(n_edges=n_edges)
    w = torch.randn(n_edges, generator=_rng(32), dtype=torch.float64)

    def run():
        e = pp._coulomb_erf_shifted_force_smooth_energy(
            q, rij, s, r, 14.4, 1.5, 6.0, 4.0, 0.5
        )
        (grad,) = torch.autograd.grad((e * w).sum(), q)
        return e.detach(), grad

    _assert_same(*_new_and_reference(pp, monkeypatch, run))


def test_coulomb_erf_shifted_force_smooth_energy_backward_has_no_index_put():
    q, rij, s, r = _coulomb_inputs()
    e = pp._coulomb_erf_shifted_force_smooth_energy(
        q, rij, s, r, 14.4, 1.5, 6.0, 4.0, 0.5
    )
    ops = _backward_ops(lambda: e.sum().backward())
    assert not (ops & INDEX_PUT_OPS)


def _displacement_inputs(seed=40):
    g = _rng(seed)
    positions = torch.randn(30, 3, generator=g, dtype=torch.float64)
    cell = torch.randn(2, 3, 3, generator=g, dtype=torch.float64)
    batch = torch.randint(0, 2, (30,), generator=g)
    return positions, cell, batch


def test_get_symmetric_displacement_equivalent(monkeypatch):
    positions, cell, batch = _displacement_inputs()
    w = torch.randn(30, 3, generator=_rng(41), dtype=torch.float64)

    def run():
        pos, new_cell, disp = du.get_symmetric_displacement(
            positions, cell, 2, batch
        )
        (grad,) = torch.autograd.grad((pos * w).sum(), disp)
        return pos.detach(), new_cell.detach(), grad

    _assert_same(*_new_and_reference(du, monkeypatch, run))


def test_get_symmetric_displacement_backward_has_no_index_put():
    positions, cell, batch = _displacement_inputs()
    pos, _, _ = du.get_symmetric_displacement(positions, cell, 2, batch)
    ops = _backward_ops(lambda: pos.sum().backward())
    assert not (ops & INDEX_PUT_OPS)


# ---------------------------------------------------------------------------
# compute_shifts: the cell gather sends every edge into one 3x3 row (the worst
# case for index_put_ accumulation).  Single graph, few graphs (dense matmul),
# and many graphs (gather_rows fallback) all have to agree with the einsum.
# ---------------------------------------------------------------------------

SHIFT_GRAPHS = [1, 3, 16, 17, 20]


def _shift_inputs(n_graphs, n_edges=600, n_atoms=50, seed=50):
    g = _rng(seed)
    cell = torch.randn(n_graphs, 3, 3, generator=g, dtype=torch.float64)
    batch = torch.randint(0, n_graphs, (n_atoms,), generator=g)  # unsorted
    sender = torch.randint(0, n_atoms, (n_edges,), generator=g)
    unit_shifts = torch.randint(-2, 3, (n_edges, 3), generator=g).to(
        torch.float64
    )
    return unit_shifts, cell.requires_grad_(), batch, sender


def _reference_shifts(unit_shifts, cell, batch, sender):
    return torch.einsum("be,bec->bc", unit_shifts, cell[batch[sender]])


@pytest.mark.parametrize("n_graphs", SHIFT_GRAPHS)
def test_compute_shifts_equivalent(n_graphs):
    unit_shifts, cell, batch, sender = _shift_inputs(n_graphs)
    w = torch.randn(600, 3, generator=_rng(51), dtype=torch.float64)

    new = du.compute_shifts(unit_shifts, cell, batch, sender)
    ref = _reference_shifts(unit_shifts, cell, batch, sender)
    torch.testing.assert_close(new, ref, atol=1e-12, rtol=0)

    (g_new,) = torch.autograd.grad((new * w).sum(), cell)
    (g_ref,) = torch.autograd.grad((ref * w).sum(), cell)
    torch.testing.assert_close(g_new, g_ref, atol=1e-12, rtol=0)


@pytest.mark.parametrize("n_graphs", SHIFT_GRAPHS)
def test_compute_shifts_zero_edges(n_graphs):
    _, cell, batch, _ = _shift_inputs(n_graphs)
    unit_shifts = torch.empty(0, 3, dtype=torch.float64)
    sender = torch.empty(0, dtype=torch.long)
    out = du.compute_shifts(unit_shifts, cell, batch, sender)
    assert out.shape == (0, 3)
    (grad,) = torch.autograd.grad(out.sum(), cell)
    assert torch.equal(grad, torch.zeros_like(cell))


@pytest.mark.parametrize("n_graphs", SHIFT_GRAPHS)
def test_compute_shifts_backward_has_no_index_put(n_graphs):
    unit_shifts, cell, batch, sender = _shift_inputs(n_graphs)
    shifts = du.compute_shifts(unit_shifts, cell, batch, sender)
    ops = _backward_ops(lambda: shifts.sum().backward())
    assert not (ops & INDEX_PUT_OPS)
