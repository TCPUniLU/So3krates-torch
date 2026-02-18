"""Tests for scatter_sum in tools/scatter.py.

scatter_sum is the only scatter operation used in production code
(energy aggregation, attention, charge summation). Tests cover
correctness, output sizing, multi-dimensional inputs, and gradient flow.
"""

import torch

from so3krates_torch.tools.scatter import scatter_sum


def test_scatter_sum_basic():
    """Scatter sum groups values correctly."""
    src = torch.tensor([1.0, 2.0, 3.0, 4.0])
    index = torch.tensor([0, 0, 1, 1])

    result = scatter_sum(src, index, dim=0)

    expected = torch.tensor([3.0, 7.0])
    assert torch.allclose(result, expected)


def test_scatter_sum_dim_size():
    """dim_size controls output length; empty bins are zero."""
    src = torch.tensor([1.0, 2.0])
    index = torch.tensor([0, 3])

    result = scatter_sum(src, index, dim=0, dim_size=5)

    expected = torch.tensor([1.0, 0.0, 0.0, 2.0, 0.0])
    assert torch.allclose(result, expected)


def test_scatter_sum_2d():
    """Scatter sum aggregates rows of a 2D tensor."""
    src = torch.tensor(
        [
            [1.0, 2.0, 3.0],
            [4.0, 5.0, 6.0],
            [7.0, 8.0, 9.0],
            [10.0, 11.0, 12.0],
        ]
    )
    index = torch.tensor([0, 0, 1, 1])

    result = scatter_sum(src, index, dim=0)

    expected = torch.tensor([[5.0, 7.0, 9.0], [17.0, 19.0, 21.0]])
    assert result.shape == (2, 3)
    assert torch.allclose(result, expected)


def test_scatter_sum_gradient_flows():
    """Gradients propagate through scatter_sum (required for force training)."""
    src = torch.tensor([1.0, 2.0, 3.0, 4.0], requires_grad=True)
    index = torch.tensor([0, 0, 1, 1])

    result = scatter_sum(src, index, dim=0)
    result.sum().backward()

    assert src.grad is not None
    # Each element contributes exactly once to its group sum
    assert torch.allclose(src.grad, torch.ones_like(src))
