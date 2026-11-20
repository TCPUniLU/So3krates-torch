# Robust loss (Google Research, Apache-2.0)

PyTorch port of SO3LR's JAX adaptation (`so3lr/mlff/utils/robust_loss_jax/`)
of the Google Research implementation of *A General and Adaptive Robust Loss
Function* (Barron, 2019):

- `general.py`, `cubic_spline.py`, `distribution.py` (sampling omitted)
- `data/partition_spline.npz` (byte-identical copy)

Unlike the rest of so3krates-torch (MIT), these files remain under the Apache
License 2.0, reproduced in
[`licenses/LICENSE.ROBUST-LOSS-APACHE-2.0`](../../../../licenses/LICENSE.ROBUST-LOSS-APACHE-2.0)
and listed in [`NOTICE.md`](../../../../NOTICE.md), which gives the full provenance. The sources keep the
upstream notice, *Copyright 2024 The Google Research Authors*; they are
modified copies (numerical backend replaced with PyTorch).

Upstream: <https://github.com/google-research/google-research/tree/master/robust_loss>
