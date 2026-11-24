# Changelog

## 0.3.0 (unreleased)

### Breaking changes

- **Bundled SO3LR model moved; `SO3LRCalculator` needs `model=`.** The pickled
  `pretrained/so3lr/so3lr.model` is removed. Bundled checkpoints are now a
  `state_dict` (`<stem>.pt`) plus `<stem>_settings.yaml`, selected by keyword:
  `SO3LRCalculator(model="v1")` gives the original SO3LR; `tauri-{s,m,l}-{2602,2606}`
  are the new tauri models. Code that loaded `so3lr.model` by path must switch
  to `model="v1"` or `load_pretrained_so3lr("v1")`.
- **New weights are non-commercial.** The bundled SO3LR tauri weights are
  CC BY-NC 4.0 (see [NOTICE.md](NOTICE.md)). Models fine-tuned from them fall
  under the same license. The code and the `v1` weights stay MIT.
- **Multi-theory-level models need a theory level.** The tauri models have 16
  theory-level heads: pass `theory_level=` to `SO3LRCalculator` (so3lr_dev
  uses `5`), and set `atoms.info["theory_level"]` for `torchkrates-eval` /
  `torchkrates-metric` and for training data.
- **PME electrostatics energies change.** For models with `use_pme: true`:
  - the PME term now applies `electrostatic_energy_scale` damping and drops a
    spurious factor 0.5;
  - the PME mesh is sized with so3lr_dev's rule (`pme_mesh_rule: "jax"`, the
    new default), which can pick a finer mesh near power-of-two boundaries
    (e.g. 128 instead of 64 points, about 0.2 eV in one test cell). Set
    `pme_mesh_rule: "torchpme"` for the old mesh size.

  Energies of PME models therefore differ from 0.2.x and now match so3lr_dev.
  `PMEElectrostaticInteraction` now returns the k-space term only; the
  real-space residual is part of `ElectrostaticInteraction`.
- **Legacy dispersion tables now match official SO3LR.** With
  `legacy_dispersion_bool: true` (the default for v1 and other non-tauri
  models), the free-atom alpha and C6 tables are now the official SO3LR values,
  stored as float64:
  - the actinides Pa–No (Z = 91–102) get different C6 values; the old values,
    from the `mlff` package, were 20–50 % larger;
  - all other elements change only by the former float32 rounding (about 1e-7
    relative).

  `ALPHAS` and `C6_COEF` are gone from `blocks.physical_potentials`; the tables
  are `alphas_legacy` and `C6_coef_legacy` in `blocks.dispersion_ref_data`.
  The NLH repulsion tables are now loaded from `blocks/nhl_tables.npz`.
- **`pip install .[pme]` requires `torch-pme>=0.5`** (was `>=0.4`).
- **Training stops when a loss needs a property the data lacks.** If a loss
  weight is set (e.g. `hirshfeld_weight`) but the property's key is missing
  from the data, training now raises a `ValueError` naming the property.
  Before, the term was silently filled with zeros and contributed nothing.
- **JAX↔PyTorch conversion uses `so3lr_dev`.** `torchkrates-jax2torch` /
  `torchkrates-torch2jax` and their parity check need `jax`, `flax` and
  `so3lr_dev` instead of `mlff`.

### Added

- **GPU-native MD with NVAlchemi:** `torchkrates-md-nvalchemi` (extra
  `[nvalchemi]`, Python ≥ 3.11). NVT (Langevin) and NPT (MTK), batched systems,
  GPU cell-list neighbor search. Supports:
  - bundled keywords and `.pt` checkpoints;
  - multi-theory-level models (`--theory-level`);
  - multi-output-head models (`MultiHeadSO3LR`, from `convert_to_multihead`):
    the atoms move on the mean over heads, as with `MultiHeadSO3LRCalculator`,
    and the CSV log adds the spread over heads (`energy_std_eV`,
    `force_std_max_eV_A`);
  - the model's own PME (`--model-pme`, `--model-pme-dispersion`);
  - NVAlchemi-native PME (`--electrostatics nvalchemi`).
  Programmatic API in `so3krates_torch.calculator.nvalchemi_so3`.
- **Pretrained SO3LR tauri models** (s/m/l, releases 2602 and 2606), usable in
  the ASE calculator, `torchkrates-eval`, `torchkrates-metric`, the MD CLI and
  as `pretrained_weights` for training.
- **SO3LR-tauri architecture and physics:** RMSNorm, QK norm, residual scalars
  (`use_rms_norm`, `qk_norm`, `use_residual_scalars`), NHL repulsion, a C6-ratio
  output head (`c6_ratios_bool`), multiple theory levels (`num_theory_levels`)
  and PME dispersion (`use_pme_dispersion`).
- **Training:**
  - multi-theory-level (multi-fidelity) training;
  - fixed and adaptive robust loss (Barron), with DDP and checkpoint support;
  - per-structure loss reduction (`loss_reduction: per_graph`, as in the JAX
    trainer);
  - two-stage loss weights (`stage_two_start_epoch`, `stage_two_loss_weights`);
  - robust validation loss settings;
  - C6 and a0 ratio losses that skip atoms with missing labels
    (`c6_ratios_weight`, `a0_ratios_weight`);
  - `freeze_c6_ratios` for fine-tuning;
  - custom atomic energy shifts when fine-tuning from a checkpoint.
- **Evaluation:** C6 ratio predictions and metrics (`--compute_c6_ratios`,
  `c6_ratios` in `--output_args`). Ratio metrics skip missing labels.
- **Conversion:** `--check_parity` (on by default) verifies the converted model
  against the JAX model. `--so3lr_dev_checkpoint` converts the released tauri
  checkpoints directly.
- **`torchkrates-tune-pme --tune_dispersion`** for PME dispersion parameters.
- **Experimental `torch.compile`** for `torchkrates-eval` and
  `torchkrates-md-nvalchemi` (`--compile`). It has known problems on current
  PyTorch and no measured speedup yet; see the README section
  "Experimental: torch.compile".
- JAX-vs-PyTorch inference benchmark (`examples/benchmarks/`), LoRA fine-tuning
  example for the bundled models (`examples/finetuning/`).

### Changed

- About 26× faster force and stress evaluation on an A100, from gathers based
  on `index_select`.
- License texts for third-party code and weights (`licenses/`, `NOTICE.md`)
  ship with the package.

### Fixed

- `MultiHeadSO3LRCalculator` returned partial charges and Hirshfeld ratios
  averaged over atoms (a single number instead of one value per atom), and a
  scalar dipole for models without one.

### Known limitations

- `torchkrates-eval` / `torchkrates-metric` have no flag to set the theory level
  for a whole dataset; it must be in each structure's `atoms.info`.
- NVAlchemi-native PME is validated on CPU only; GPU runs and long NPT/NVE runs
  are not yet validated.
