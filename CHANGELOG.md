# Changelog

## Unreleased

### Added

- **Long-range anchoring** (`TRAINING.long_range_anchor_weights`, as in
  so3lr 0.2.0). When fine-tuning from a pretrained model, a frozen copy of
  it predicts the partial charges, a0 ratios, C6 ratios and/or dipole of
  every batch, and the fine-tuned model is pulled towards them. This keeps
  the long-range behaviour when the data has only energies and forces.
  Keys: `partial_charges`, `a0_ratios`/`hirshfeld_ratios`, `c6_ratios`,
  `dipole_vec` (torch aliases `charges`, `dipole`). A warning is logged
  when the data also has labels for an anchored quantity. Validation logs
  `RMSE_anchor_<quantity>`, the deviation from the pretrained model.

### Fixed

- **Training a `convert_to_multihead` model crashed at the first
  validation** when batches differed in atom count. Validation metrics of
  multi-head models now use the mean over heads, as
  `MultiHeadSO3LRCalculator` and `torchkrates-metric` do.

## 0.4.0 (2026-11-25)

First release on PyPI since 0.2.1: it also contains all changes of 0.3.0
below, which was tagged but not published to PyPI.

### Packaging

- License metadata is now an SPDX expression
  (`MIT AND CC-BY-NC-4.0 AND Apache-2.0`); building needs
  `setuptools>=77`.
- New GitHub Actions workflow `publish.yml` builds and checks the sdist
  and wheel for every GitHub release and publishes them to TestPyPI or
  PyPI on manual dispatch (trusted publishing).
- The pytest pre-commit hook runs on demand only:
  `pre-commit run pytest --hook-stage manual`.

### Breaking changes

- **Removed `pretrained_to_mh_model`, `model_to_multihead` and
  `reduce_mh_model_to_sh`** from `so3krates_torch.tools.multihead_utils`.
  Use `to_multihead_model(model, num_output_heads)` and
  `to_single_head_model(model, head_idx)` instead; both take and return
  model objects.

### Fixed

- **`convert_to_multihead` fine-tuning.** Converting a pretrained model to
  multiple energy heads crashed on a validated config
  (`unexpected keyword argument 'compute_avg_num_neighbors'`). The
  conversion now copies the pretrained model and replaces only its energy
  head, so it also no longer:
  - builds the model in float32 (and switches torch's default dtype to
    float32) when training in float64;
  - resets `r_max_lr` to 12.0;
  - drops learned per-element energy scales;
  - takes settings that are not weights (e.g. `cutoff_fn`,
    `electrostatic_energy_scale`) from the config instead of the
    pretrained model.
- **`convert_to_multihead` with `pretrained_weights`** now loads single-head
  weights into a single-head model before converting (the energy scales
  were dropped before), and multi-head weights into a multi-head model.
- **`convert_to_multihead` without `finetune_choice`** converts a pretrained
  model too; before, the setting was silently ignored.

### Added

- **`torchkrates-convert-heads` CLI.** `--to-multihead N` turns a trained
  or bundled model into a multi-head model with the same backbone and N
  fresh energy heads; `--select-head K` keeps one head of a multi-head
  model as a single-head model. See the README.
- `convert_theory_level` (ARCHITECTURE): with `convert_to_multihead`, keep
  this theory level of a multi-theory-level model such as the bundled
  `tauri-*` models, which could not be converted before. Defaults to 5
  (so3lr_dev's level), also for `torchkrates-convert-heads --theory-level`
  and `to_multihead_model(theory_level=...)`.
- `select_theory_level(model, theory_level)` in
  `so3krates_torch.tools.multihead_utils`.

## 0.3.0 (2026-11-24, tagged; not published to PyPI)

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
