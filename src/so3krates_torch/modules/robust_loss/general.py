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
# (so3lr/mlff/utils/robust_loss_jax/general.py), same constants and cases.

r"""General form of the robust loss rho(x, alpha, c) (Barron, 2019)."""

from typing import Union

import torch

Number = Union[float, torch.Tensor]

_EPS = torch.finfo(torch.float32).eps  # JAX uses float32 eps as well
_MAXVAL = 1e15


def _fake_clip(x: torch.Tensor, lo: float, hi: float) -> torch.Tensor:
    """Clip in the forward pass, identity gradient (JAX fake_clip)."""
    return x + (torch.clamp(x, lo, hi) - x).detach()


def _expm1_safe(x: torch.Tensor) -> torch.Tensor:
    return torch.expm1(torch.clamp(x, max=43.0))


def general_loss(
    x: torch.Tensor, alpha: Number, scale: Number
) -> torch.Tensor:
    r"""rho(x, alpha, c) of "A General and Adaptive Robust Loss Function",
    https://arxiv.org/abs/1701.03077. alpha = 2: L2 (0.5 (x/c)^2),
    1: Charbonnier, 0: Cauchy, -2: Geman-McClure, -inf: Welsch."""
    alpha = torch.as_tensor(alpha, dtype=x.dtype, device=x.device)
    scale = torch.as_tensor(scale, dtype=x.dtype, device=x.device)
    scale = torch.clamp(scale, min=_EPS)

    # Large values of |x| can cause non-finite gradients.
    x = _fake_clip(x, -_MAXVAL, _MAXVAL)

    loss_two = 0.5 * (x / scale) ** 2

    # Clamp |alpha| and |2 - alpha| away from 0 so it's safe to divide.
    a = torch.where(
        alpha >= 0, torch.ones_like(alpha), -torch.ones_like(alpha)
    ) * torch.clamp(torch.abs(alpha), min=_EPS)
    b = torch.clamp(torch.abs(a - 2), min=_EPS)

    loss_ow = (b / a) * ((loss_two / (0.5 * b) + 1) ** (0.5 * a) - 1)

    return torch.where(
        alpha == -float("inf"),
        -_expm1_safe(-loss_two),
        torch.where(
            alpha == 0,
            torch.log1p(loss_two),
            torch.where(
                alpha == 2,
                loss_two,
                torch.where(
                    alpha == float("inf"), _expm1_safe(loss_two), loss_ow
                ),
            ),
        ),
    )
