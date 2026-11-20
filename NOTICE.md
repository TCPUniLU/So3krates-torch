# NOTICE

so3krates-torch is licensed under the MIT License in [`LICENSE.md`](LICENSE.md),
except where noted below. Full texts of other licenses are in
[`licenses/`](licenses/).

## SO3LR Tauri model weights

The pretrained SO3LR Tauri weights in `src/so3krates_torch/pretrained/so3lr/tauri/`
(every size and release date: `so3lr-tauri-{s,m,l}-2602` and
`so3lr-tauri-{s,m,l}-2606`), © 2026 Adil Kabylda, are licensed under the
[Creative Commons Attribution-NonCommercial 4.0 International License](https://creativecommons.org/licenses/by-nc/4.0/)
(CC BY-NC 4.0), reproduced in
[`licenses/LICENSE.CC-BY-NC-4.0`](licenses/LICENSE.CC-BY-NC-4.0). They are
PyTorch conversions of the JAX parameters released with
[SO3LR](https://github.com/general-molecular-simulations/so3lr). They may be
used, shared and adapted for non-commercial purposes with attribution;
commercial use is not permitted. This covers the converted parameters
(`*.pt`) and the files that describe them in the same directory
(`*_settings.yaml`). Models fine-tuned from these weights are adaptations and
fall under the same license.

The original SO3LR weights in `src/so3krates_torch/pretrained/so3lr/v1/`,
published with [Kabylda et al., JACS 2025](https://doi.org/10.1021/jacs.5c09558),
remain under the MIT License, as in the SO3LR release.

## Robust loss

`src/so3krates_torch/modules/robust_loss/` is a PyTorch port of SO3LR's JAX
adaptation of the Google Research implementation of *A General and Adaptive
Robust Loss Function*
(<https://github.com/google-research/google-research/tree/master/robust_loss>),
Copyright 2024 The Google Research Authors. It is licensed under the Apache
License 2.0, reproduced in
[`licenses/LICENSE.ROBUST-LOSS-APACHE-2.0`](licenses/LICENSE.ROBUST-LOSS-APACHE-2.0).
The files are modified copies (numerical backend replaced with PyTorch).
