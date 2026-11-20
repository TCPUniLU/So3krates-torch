# NOTICE

So3krates-torch combines source code and pretrained model weights under
different licenses. Except where noted below, the code is licensed under the
MIT License in [`LICENSE.md`](LICENSE.md). The full texts of the other licenses
are in [`licenses/`](licenses/) and ship with the package.

## Pretrained model weights

### SO3LR Tauri (CC BY-NC 4.0)

The SO3LR Tauri weights in `src/so3krates_torch/pretrained/so3lr/tauri/`
(every size and release: `so3lr-tauri-{s,m,l}-2602` and
`so3lr-tauri-{s,m,l}-2606`), © 2026 Adil Kabylda, are licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/)
(CC BY-NC 4.0), reproduced in
[`licenses/LICENSE.CC-BY-NC-4.0`](licenses/LICENSE.CC-BY-NC-4.0).

- They are PyTorch conversions of the JAX parameters released with
  [SO3LR](https://github.com/general-molecular-simulations/so3lr).
- The license covers the converted parameters (`*.pt`) and the files that
  describe them in the same directory (`*_settings.yaml`).
- They may be used, shared and adapted for non-commercial purposes with
  attribution; commercial use is not permitted.
- Models fine-tuned from these weights are adaptations and fall under the same
  license.

### Original SO3LR (MIT)

The original SO3LR weights in `src/so3krates_torch/pretrained/so3lr/v1/`,
published with [Kabylda et al., JACS 2025](https://doi.org/10.1021/jacs.5c09558),
remain under the MIT License, as in the SO3LR release.

## Third-party code

### Robust loss (Apache-2.0)

`src/so3krates_torch/modules/robust_loss/` implements *A General and Adaptive
Robust Loss Function* (Barron, 2019). It is licensed under the Apache License
2.0, reproduced in
[`licenses/LICENSE.ROBUST-LOSS-APACHE-2.0`](licenses/LICENSE.ROBUST-LOSS-APACHE-2.0).
Its provenance, each step a modified copy of the previous one:

1. Google Research implementation
   (<https://github.com/google-research/google-research/tree/master/robust_loss>),
   Copyright 2024 The Google Research Authors.
2. JAX adaptation in SO3LR (`so3lr/mlff/utils/robust_loss_jax/`), by the SO3LR
   authors (SO3LR: Copyright (c) 2024 Thorben Frank, Adil Kabylda), distributed
   under Apache-2.0.
3. PyTorch port in So3krates-torch: the numerical backend is replaced with
   PyTorch and distribution sampling is omitted. The spline table
   `data/partition_spline.npz` is an unmodified copy.

### MACE (MIT)

Parts of the code, among them the data pipeline, the training and evaluation
utilities and the loss functions, are adapted from
[MACE](https://github.com/ACEsuit/mace), Copyright (c) 2022 ACEsuit/mace,
under the MIT License reproduced in
[`licenses/LICENSE.MACE-MIT`](licenses/LICENSE.MACE-MIT).

### PyTorch Geometric (MIT)

`src/so3krates_torch/tools/torch_geometric/` is a modified copy of PyTorch
Geometric internals (`Data`, `Batch`, `DataLoader`, `Dataset`),
Copyright (c) 2023 PyG Team, under the MIT License reproduced in
[`licenses/LICENSE.PYG-MIT`](licenses/LICENSE.PYG-MIT).

## Data

These licenses do not relicense the datasets the models were trained on or the
scientific reference data included with the code; those remain under the terms
set by their original authors.
