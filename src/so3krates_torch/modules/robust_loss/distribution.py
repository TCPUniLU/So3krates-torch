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
# (so3lr/mlff/utils/robust_loss_jax/distribution.py); sampling omitted.

r"""Negative log-likelihood of the robust loss distribution (Barron 2019,
Sec. 2): NLL(x | alpha, c) = rho(x, alpha, c) + log(c) + log Z(alpha),
with log Z(alpha) from a cubic spline fitted by the original authors."""

from importlib import resources

import numpy as np
import torch

from . import cubic_spline
from .general import _EPS, general_loss


def partition_spline_curve(alpha: torch.Tensor) -> torch.Tensor:
    """Compress alpha >= 0 into the spline's x-range (continuously
    differentiable; ~linear in [0, 4], logarithmic above)."""
    return torch.where(
        alpha < 4,
        (2.25 * alpha - 4.5) / (torch.abs(alpha - 2) + 0.25) + alpha + 2,
        (5 / 18) * torch.log(torch.clamp(4 * alpha - 15, max=3e37)) + 8,
    )


def _load_spline():
    path = resources.files(__package__).joinpath("data/partition_spline.npz")
    with resources.as_file(path) as p, np.load(p, allow_pickle=False) as f:
        return (
            np.asarray(f["x_scale"]),
            np.asarray(f["values"]),
            np.asarray(f["tangents"]),
        )


class Distribution:
    """Wrapper around the spline-approximated partition function."""

    def __init__(self) -> None:
        x_scale, values, tangents = _load_spline()
        self._x_scale = torch.as_tensor(x_scale, dtype=torch.float64)
        self._values = torch.as_tensor(values, dtype=torch.float64)
        self._tangents = torch.as_tensor(tangents, dtype=torch.float64)

    def log_base_partition_function(self, alpha: torch.Tensor) -> torch.Tensor:
        """log Z(alpha) for alpha >= 0 (accurate to ~1e-6)."""
        alpha = torch.clamp(alpha, min=0)
        kw = dict(dtype=alpha.dtype, device=alpha.device)
        x = partition_spline_curve(alpha)
        return cubic_spline.interpolate1d(
            x * self._x_scale.to(**kw),
            self._values.to(**kw),
            self._tangents.to(**kw),
        )

    def nllfun(
        self, x: torch.Tensor, alpha: torch.Tensor, scale: torch.Tensor
    ) -> torch.Tensor:
        """-log p(x | 0, alpha, c), Eq. 16 of the paper."""
        alpha = torch.clamp(
            torch.as_tensor(alpha, dtype=x.dtype, device=x.device), min=0
        )
        scale = torch.clamp(
            torch.as_tensor(scale, dtype=x.dtype, device=x.device), min=_EPS
        )
        loss = general_loss(x, alpha, scale)
        return (
            loss + torch.log(scale) + self.log_base_partition_function(alpha)
        )
