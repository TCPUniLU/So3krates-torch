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
