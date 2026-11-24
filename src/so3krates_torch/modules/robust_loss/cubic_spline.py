# coding=utf-8
# Copyright 2024 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Modified for so3krates-torch: PyTorch port of the SO3LR JAX adaptation
# (so3lr/mlff/utils/robust_loss_jax/cubic_spline.py).

"""1D cubic Hermite spline interpolation."""

import torch


def interpolate1d(
    x: torch.Tensor, values: torch.Tensor, tangents: torch.Tensor
) -> torch.Tensor:
    """Cubic Hermite interpolation with knots at [0, len(values) - 1];
    linear extrapolation outside that range."""
    assert values.dim() == 1 and tangents.dim() == 1
    assert values.shape[0] == tangents.shape[0]

    x_lo = torch.floor(torch.clamp(x, 0.0, values.shape[0] - 2)).long()
    x_hi = x_lo + 1

    t = x - x_lo.to(x.dtype)

    t_sq = t**2
    t_cu = t * t_sq
    h01 = -2 * t_cu + 3 * t_sq
    h00 = 1 - h01
    h11 = t_cu - t_sq
    h10 = h11 - t_sq + t

    value_before = tangents[0] * t + values[0]
    value_after = tangents[-1] * (t - 1) + values[-1]

    value_mid = (
        values[x_lo] * h00
        + values[x_hi] * h01
        + tangents[x_lo] * h10
        + tangents[x_hi] * h11
    )

    return torch.where(
        t < 0.0, value_before, torch.where(t > 1.0, value_after, value_mid)
    )
