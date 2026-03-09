# So3krates-torch

> [!IMPORTANT]
> The code is work in progress! There may be breaking changes!

Lightweight implementation of the So3krates model in pytorch. This package is mostly intended for [aims-PAX](https://github.com/tohenkes/aims-PAX) but is a functional implementation of [So3krates](https://github.com/thorben-frank/mlff) and [SO3LR](https://github.com/general-molecular-simulations/so3lr) in pytorch. For now it uses (modified) source code of the [MACE](https://github.com/ACEsuit/mace) package and follows its style, so many functions are actually compatible.

#### Installation

1. activate your environment
2. clone this repository
3. move to the clone repository
4. `pip install -r requirements.txt`
5. `pip install .`

#### Implemented features:
1. ASE calculator for MD (including pre-trained SO3LR)
2. Inference over ase readable datasets: `torchkrates-eval`
3. Error metrics over ase readable datasets: `torchkrates-metric`
4. Transforming pyTorch and JAX parameter formates: `torchkrates-jax2torch` or `torchkrates-torch2jax` (for these you need to install jax, flax, and mlff (https://github.com/thorben-frank/mlff/tree/v1.0-lrs-gems))
5. Training: `torchkrates-train --config config.yaml` (see example)
6. Data preprocessing: `torchkrates-preprocess`
7. HDF5 file merging: `torchkrates-merge`
8. LAMMPS model export: `torchkrates-create-lammps-model`


> [!IMPORTANT]
> Number 4 means that you can transform the weights from this pytorch version into the JAX version and vice versa. Inference and training is much faster (*at least 1 order of magnitude at the moment*) in the JAX version. This implementation is mostly for prototyping and compatability with other packages.

---

## CLI Reference

### `torchkrates-train` — Training

Train an SO3LR (or multi-head SO3LR) model from a YAML configuration file.

```bash
torchkrates-train --config config.yaml
```

See the **[Training Configuration](#training-configuration)** section below for detailed documentation of all configuration options.

---

### `torchkrates-preprocess` — Data Preprocessing

Convert atomic structure data (XYZ or raw HDF5) into preprocessed HDF5 files with precomputed neighbour lists for faster training.

```bash
# XYZ → preprocessed HDF5
torchkrates-preprocess --input data.xyz --output data.h5 --mode preprocessed --r-max 4.5

# XYZ → raw HDF5 (structures only, no neighbour lists)
torchkrates-preprocess --input data.xyz --output data.h5 --mode raw

# Raw HDF5 → preprocessed HDF5
torchkrates-preprocess --input raw.h5 --output preprocessed.h5 --mode preprocessed --r-max 4.5
```

| Flag | Description |
|------|-------------|
| `--input` | Input file path (`.xyz` or `.h5`/`.hdf5`) |
| `--output` | Output HDF5 file path |
| `--mode` | `raw` (structures only) or `preprocessed` (with neighbour lists) |
| `--r-max` | Short-range cutoff (required for `preprocessed` mode) |
| `--r-max-lr` | Long-range cutoff (optional) |
| `--energy-key` | Key for energy in the input file (default: `REF_energy`) |
| `--forces-key` | Key for forces (default: `REF_forces`) |
| `--stress-key` | Key for stress (default: `REF_stress`) |
| `--dipole-key` | Key for dipole (default: `REF_dipole`) |
| `--charges-key` | Key for charges (default: `REF_charges`) |
| `--description` | Optional dataset description stored in the HDF5 metadata |
| `--dtype` | Data type: `float32` or `float64` (default: `float64`) |
| `--validate` | Validate the output file after creation |

---

### `torchkrates-merge` — HDF5 File Merging

Merge two or more HDF5 files (raw or preprocessed) into a single file. Both formats are supported; all inputs must be the same type. Raw files are merged with streaming writes to avoid loading everything into memory.

```bash
# Merge raw HDF5 files
torchkrates-merge --inputs train_a.h5 train_b.h5 --output train_merged.h5

# Merge preprocessed HDF5 files
torchkrates-merge --inputs part1.h5 part2.h5 part3.h5 --output all.h5

# With optional metadata and custom batch size
torchkrates-merge --inputs a.h5 b.h5 --output merged.h5 \
    --description "combined dataset" --batch-size 50000
```

| Flag | Description |
|------|-------------|
| `--inputs FILE [FILE ...]` | Two or more input HDF5 files to merge (must be the same format) |
| `--output FILE` | Output HDF5 file path |
| `--description TEXT` | Optional description stored in the merged file metadata (raw format only) |
| `--batch-size N` | Structures processed per write batch (raw format only, default: `100000`) |

---

### `torchkrates-create-lammps-model` — LAMMPS Model Export

> [!NOTE]
> More details and how to use the model in LAMMPS are coming ...

Convert a trained SO3LR model to a TorchScript model compatible with the LAMMPS ML-IAP interface.

```bash
torchkrates-create-lammps-model model.pt --elements Si O
```

| Flag | Description |
|------|-------------|
| `model_path` | Path to the trained `.pt` model file |
| `--elements` | Element symbols present in the simulation (must match LAMMPS `pair_coeff` type order) |
| `--head` | Head name for multi-head models (interactive selection if omitted) |
| `--dtype` | `float32` or `float64` (default: `float64`) |

> [!NOTE]
> LAMMPS export only supports **short-range** models. Models with `electrostatic_energy_bool=True` or `dispersion_energy_bool=True` are rejected — retrain without long-range potentials for LAMMPS use.

---

### `torchkrates-eval` — Inference

Run inference over an ASE-readable dataset.

### `torchkrates-metric` — Error Metrics

Compute error metrics over an ASE-readable dataset.

### `torchkrates-jax2torch` / `torchkrates-torch2jax` — Weight Conversion

Convert model weights between the PyTorch and JAX (mlff) implementations. Requires `jax`, `flax`, and [`mlff`](https://github.com/thorben-frank/mlff/tree/v1.0-lrs-gems) to be installed.

---

## Training Configuration

Training is configured via a YAML file with four sections: `GENERAL`, `ARCHITECTURE`, `TRAINING`, and `MISC`. Launch training with:

```bash
torchkrates-train --config config.yaml
```

A full example is provided in [examples/training/train_settings_example.yaml](examples/training/train_settings_example.yaml).


### Model Architecture (`ARCHITECTURE`)

These settings define the SO3LR neural network architecture.

#### Core Transformer

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `degrees` | `list[int]` | *required* | Spherical harmonic degrees included in the equivariant features, e.g. `[1,2,3,4]`. Higher degrees capture more angular information but increase cost. |
| `num_features` | `int` | `128` | Hidden feature dimension of invariant and equivariant representations. |
| `num_heads` | `int` | `4` | Number of attention heads in each Euclidean transformer layer. |
| `num_layers` | `int` | `3` | Number of stacked Euclidean transformer layers. |
| `num_radial_basis_fn` | `int` | `32` | Number of radial basis functions used to expand interatomic distances. |
| `energy_regression_dim` | `int` | `128` | Hidden dimension of the MLP in the atomic energy output head. |

#### Cutoffs and Basis Functions

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `r_max` | `float` | `4.5` | Short-range cutoff radius in Angstrom. Atoms beyond this distance do not interact through the neural network. |
| `r_max_lr` | `float` | `None` | Long-range cutoff for electrostatics and dispersion. Required when `electrostatic_energy_bool` or `dispersion_energy_bool` is enabled. |
| `radial_basis_fn` | `str` | `"bernstein"` | Radial basis function type. Options: `bernstein`, `gaussian`, `bessel`. |
| `cutoff_fn` | `str` | `"cosine"` | Envelope function that smoothly decays interactions to zero at the cutoff. Options: `cosine`, `phys`, `polynomial`, `exponential`. |
| `trainable_rbf` | `bool` | `False` | Whether radial basis function parameters are trainable. |

#### Activation Functions

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `activation_fn` | `str` | `"silu"` | Activation function in transformer layers. Options: `silu`, `relu`, `gelu`, `tanh`, `identity`. |
| `energy_activation_fn` | `str` | `"silu"` | Activation function in the energy output head MLP. Same options as above. |
| `qk_non_linearity` | `str` | `"identity"` | Non-linearity applied to query and key projections in attention. `identity` means linear attention. |

#### Normalization and Residual Connections

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `message_normalization` | `str` | `"avg_num_neighbors"` | How messages are normalized after aggregation. Options: `avg_num_neighbors` (divide by mean neighbor count), `sqrt_num_features`, `identity`. |
| `layer_normalization_1` | `bool` | `False` | Apply layer normalization after the first MLP in each transformer layer. |
| `layer_normalization_2` | `bool` | `False` | Apply layer normalization after the second MLP in each transformer layer. |
| `residual_mlp_1` | `bool` | `False` | Add a residual connection around the first MLP. |
| `residual_mlp_2` | `bool` | `False` | Add a residual connection around the second MLP. |
| `compute_avg_num_neighbors` | `bool` | `True` | Compute the average number of neighbors from the training data (used for `avg_num_neighbors` normalization). |

#### Embeddings and Energy Output

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `use_charge_embed` | `bool` | `False` | Include total charge as an input embedding. Required when training on charged systems. |
| `use_spin_embed` | `bool` | `False` | Include total spin as an input embedding. Required for spin-polarized systems. |
| `energy_learn_atomic_type_shifts` | `bool` | `False` | Learn per-element energy shifts as trainable parameters. When `False`, shifts are fixed from the training data E0s. |
| `energy_learn_atomic_type_scales` | `bool` | `False` | Learn per-element energy scales as trainable parameters. |
| `atomic_energy_shifts` | `dict` | `None` | Manually specify per-element energy shifts, e.g. `{1: -13.6, 6: -1029.5}`. Overrides the values computed from training data. |

#### SO3LR Physical Potentials

These enable the physics-based long-range interactions that distinguish SO3LR from the base So3krates model.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `zbl_repulsion_bool` | `bool` | `True` | Enable the ZBL repulsion potential for short-range nuclear repulsion. |
| `electrostatic_energy_bool` | `bool` | `True` | Enable electrostatic interactions via learned partial charges. Requires `r_max_lr` to be set. |
| `electrostatic_energy_scale` | `float` | `4.0` | Scaling factor for the electrostatic energy contribution. |
| `dispersion_energy_bool` | `bool` | `True` | Enable van der Waals dispersion interactions via learned Hirshfeld ratios. Requires `r_max_lr` to be set. |
| `dispersion_energy_scale` | `float` | `1.2` | Scaling factor for the dispersion energy contribution. |
| `dispersion_energy_cutoff_lr_damping` | `float` | `None` | Damping cutoff for dispersion interactions. |
| `neighborlist_format_lr` | `str` | `"sparse"` | Storage format for the long-range neighbor list. |

#### Multi-Head Ensemble

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `convert_to_multihead` | `bool` | `False` | Convert the single energy output head to multiple independent heads for ensemble predictions. |
| `num_output_heads` | `int` | `None` | Number of output heads. Required when `convert_to_multihead: true`. |
| `use_multihead` | `bool` | `False` | Enable head selection during multi-head training (each sample is assigned to a specific head). |


### Training Procedure (`TRAINING`)

#### Data

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `path_to_train_data` | `str` | *required* | Path to training data. Accepts `.xyz` files (ASE-readable) or `.h5`/`.hdf5` files. Preprocessed HDF5 files with pre-computed neighbor lists are auto-detected and loaded directly, which is significantly faster. |
| `path_to_val_data` | `str` | `None` | Path to a separate validation dataset. If not provided, validation data is split from the training set using `valid_ratio`. |
| `valid_ratio` | `float` | `0.1` | Fraction of training data to use for validation when `path_to_val_data` is not specified. |
| `num_train` | `int` | `None` | Limit the number of training samples. Useful for debugging or ablation studies. |
| `num_valid` | `int` | `None` | Limit the number of validation samples. |
| `batch_size` | `int` | *required* | Number of structures per training batch. |
| `valid_batch_size` | `int` | *required* | Number of structures per validation batch. Can be larger than `batch_size` since no gradients are computed. |
| `lazy_loading` | `bool` | `False` | Enable on-the-fly data loading and preprocessing from raw HDF5 files. Instead of loading all structures into memory upfront, each structure is read and its neighbor list computed on the fly by DataLoader worker processes. Only supported for raw HDF5 files (not XYZ). |
| `num_workers` | `int` | `4` | Number of DataLoader worker processes for parallel preprocessing. Only used when `lazy_loading: true`. Each worker reads structures from HDF5 and computes neighbor lists concurrently. |
| `prefetch_factor` | `int` | `2` | Number of batches each worker prefetches ahead of time. With `num_workers=4` and `prefetch_factor=2`, up to 8 batches are prepared in the background while the GPU trains. Only used when `lazy_loading: true`. |
| `num_neighbor_samples` | `int` | `1000` | Number of structures randomly sampled to estimate the average number of neighbors (used for message normalization). Only used when `lazy_loading: true`. |

For multi-head models, data can be specified per head instead of using `path_to_train_data`:

```yaml
TRAINING:
  heads:
    head_0:
      path_to_train_data: /path/to/head0_train.xyz
      path_to_val_data: /path/to/head0_val.xyz  # optional
      valid_ratio: 0.1  # used if path_to_val_data not given
    head_1:
      path_to_train_data: /path/to/head1_train.xyz
```

#### Data Key Mapping

By default, the trainer reads reference properties using these keys from the ASE `atoms.info` / `atoms.arrays` dictionaries:

| Property | Default Key |
|----------|-------------|
| Energy | `REF_energy` |
| Forces | `REF_forces` |
| Stress | `REF_stress` |
| Virials | `REF_virials` |
| Dipole | `dipole` |
| Charges | `REF_charges` |
| Hirshfeld ratios | `REF_hirsh_ratios` |
| Total charge | `total_charge` |
| Total spin | `total_spin` |

Override any of these via the `keys` dict:

```yaml
TRAINING:
  keys:
    energy_key: "energy"
    forces_key: "forces"
```

#### Optimizer

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `optimizer` | `str` | `"adam"` | Optimizer. Options: `adam`, `adamw`. |
| `lr` | `float` | *required* | Initial learning rate. |
| `weight_decay` | `float` | `0.0` | L2 regularization weight. Applied to all parameters. |
| `amsgrad` | `bool` | `False` | Use the AMSGrad variant of Adam, which keeps a running maximum of the second moment to prevent learning rate from increasing. |

#### Learning Rate Scheduler

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `scheduler` | `str` | `"exponential_decay"` | Learning rate scheduler. Options: `exponential_decay`, `reduce_on_plateau`. |
| `lr_scheduler_gamma` | `float` | `0.9993` | Multiplicative decay factor applied every epoch (for `exponential_decay`). An effective learning rate after *N* epochs is `lr * gamma^N`. |
| `scheduler_patience` | `int` | `5` | Number of epochs with no improvement before reducing the learning rate (for `reduce_on_plateau`). |
| `lr_factor` | `float` | `0.85` | Factor by which the learning rate is reduced when the plateau is reached (for `reduce_on_plateau`). |

#### Loss Function

The loss function is automatically determined based on which weights are non-zero, or can be set explicitly via `loss_type`.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `energy_weight` | `float` | `1.0` | Weight of the energy MSE loss term. Energy loss is normalized per atom. |
| `forces_weight` | `float` | `1000.0` | Weight of the forces MSE loss term. Typically much larger than `energy_weight` since force errors are smaller in magnitude. |
| `dipole_weight` | `float` | `0.0` | Weight of the dipole loss term. Set > 0 to train dipole predictions (requires dipole labels in training data). |
| `hirshfeld_weight` | `float` | `0.0` | Weight of the Hirshfeld ratios loss term. Set > 0 to train Hirshfeld volume ratio predictions. |
| `loss_type` | `str` | `"auto"` | Explicit loss type selection. Options: `auto`, `energy_forces`, `energy_forces_dipole`, `energy_forces_hirshfeld`, `energy_forces_dipole_hirshfeld`. When `auto`, the loss is inferred from which weights are non-zero. |

#### Training Loop

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `num_epochs` | `int` | *required* | Maximum number of training epochs. |
| `eval_interval` | `int` | `1` | Run validation every N epochs. |
| `patience` | `int` | `50` | Early stopping patience: training stops after this many consecutive epochs without improvement on the validation loss. |
| `clip_grad` | `float` | `10.0` | Maximum gradient norm for gradient clipping. Set to `null` to disable. |

#### Exponential Moving Average (EMA)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `ema` | `bool` | `False` | Maintain an exponential moving average of model weights. The EMA weights are used for validation and the final saved model. |
| `ema_decay` | `float` | `0.99` | EMA decay factor. Values closer to 1.0 average over more history. |

#### Pre-trained Models

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `pretrained_model` | `str` | `None` | Path to a complete pre-trained model (`.model` file). The full model object is loaded, including architecture and weights. Cannot be combined with `pretrained_weights`. |
| `pretrained_weights` | `str` | `None` | Path to pre-trained weights (state dict). Weights are loaded into the model defined by the `ARCHITECTURE` section. Cannot be combined with `pretrained_model`. |
| `ft_update_avg_num_neighbors` | `bool` | `False` | Recompute the average number of neighbors from the new training data instead of keeping the value from the pre-trained model. |
| `force_use_average_shifts` | `bool` | `False` | Use E0 shifts computed from the new training data instead of the pre-trained model's shifts. |

#### Fine-Tuning Strategy

When loading a pre-trained model, `finetune_choice` controls which parameters remain trainable.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `finetune_choice` | `str` | `None` | Fine-tuning strategy. Options: `naive` (all parameters trainable), `last_layer` (only last transformer layer), `mlp` (only MLP weights), `qkv` (only query/key/value projections), `lora` (low-rank adaptation), `dora` (weight-decomposed LoRA), `vera` (vector-based random matrix adaptation). Combinations with `+mlp` are also supported: `last_layer+mlp`, `qkv+mlp`, `lora+mlp`. |
| `freeze_embedding` | `bool` | `True` | Freeze the atomic embedding layers during fine-tuning. |
| `freeze_zbl` | `bool` | `True` | Freeze ZBL repulsion parameters. |
| `freeze_partial_charges` | `bool` | `True` | Freeze the partial charges output head. |
| `freeze_hirshfeld` | `bool` | `True` | Freeze the Hirshfeld ratios output head. |
| `freeze_shifts` | `bool` | `False` | Freeze learned atomic energy shifts. |
| `freeze_scales` | `bool` | `False` | Freeze learned atomic energy scales. |

#### LoRA / DoRA / VeRA Parameters

These apply when `finetune_choice` is one of `lora`, `dora`, `vera`, or their `+mlp` variants.

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `lora_rank` | `int` | `4` | Rank of the low-rank adaptation matrices. Lower rank means fewer trainable parameters. |
| `lora_alpha` | `float` | `8.0` | Scaling factor. The effective adaptation is scaled by `alpha / rank`. |
| `lora_freeze_A` | `bool` | `False` | Freeze the A (down-projection) matrices and only train B. Reduces trainable parameters by half. |
| `dora_scaling_to_one` | `bool` | `True` | Initialize DoRA magnitude vectors to normalize columns to unit norm. |


### General Settings (`GENERAL`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `name_exp` | `str` | *required* | Experiment name. Used for checkpoint filenames, log files, and the final saved model. |
| `checkpoints_dir` | `str` | *required* | Directory where training checkpoints are saved. |
| `model_dir` | `str` | *required* | Directory for the final trained model. |
| `log_dir` | `str` | *required* | Directory for training and validation log files. |
| `default_dtype` | `str` | `"float64"` | Default floating-point precision. Options: `float32`, `float64`, `float16`, `bfloat16`. Training typically uses `float64` for numerical stability. |
| `seed` | `int` | `42` | Random seed for reproducibility (weight initialization, data shuffling). |
| `compute_stress` | `bool` | `False` | Compute stress tensors during training. Required when training with stress/virial labels. |


### Runtime and Logging (`MISC`)

| Key | Type | Default | Description |
|-----|------|---------|-------------|
| `device` | `str` | `"cpu"` | Device for training: `cpu`, `cuda`, `cuda:0`, etc. Ignored when `distributed: true`. |
| `distributed` | `bool` | `False` | Enable multi-GPU training with DistributedDataParallel (DDP). |
| `launcher` | `str` | `None` | Distributed launcher. Required when `distributed: true`. Options: `torchrun`, `slurm`, `mpi`. |
| `log_level` | `str` | `"INFO"` | Python logging level: `DEBUG`, `INFO`, `WARNING`, `ERROR`. |
| `error_table` | `str` | `"PerAtomMAE"` | Format for validation error reporting. Options: `PerAtomMAE`, `TotalMAE`, `PerAtomRMSE`, `TotalRMSE`, `PerAtomMAEstressvirials`, `PerAtomRMSEstressvirials`, `EnergyForceDipoleMAE`, `EnergyForceHirshfeldMAE`, `EnergyForceDipoleHirshfeldMAE`. |
| `log_wandb` | `bool` | `False` | Enable [Weights & Biases](https://wandb.ai) logging. Requires `wandb` to be installed and configured. |
| `restart_latest` | `bool` | `True` | Automatically resume from the latest checkpoint in `checkpoints_dir` if one exists. |
| `keep_checkpoints` | `bool` | `False` | Keep all checkpoints. When `False`, only the best checkpoint (lowest validation loss) is kept. |
| `no_checkpoint` | `bool` | `False` | Disable checkpoint loading entirely (overrides `restart_latest`). Useful for forcing a fresh start. |


## Cite
If you are using the models implemented here please cite:

```bibtex
@article{doi:10.1021/jacs.5c09558,
author = {Kabylda, Adil and Frank, J. Thorben and Suárez-Dou, Sergio and Khabibrakhmanov, Almaz and Medrano Sandonas, Leonardo and Unke, Oliver T. and Chmiela, Stefan and M{\"u}ller, Klaus-Robert and Tkatchenko, Alexandre},
title = {Molecular Simulations with a Pretrained Neural Network and Universal Pairwise Force Fields},
journal = {Journal of the American Chemical Society},
volume = {0},
number = {0},
pages = {null},
year = {0},
doi = {10.1021/jacs.5c09558},
    note ={PMID: 40886167},
URL = { 
    
        https://doi.org/10.1021/jacs.5c09558
},
eprint = { 
    
        https://doi.org/10.1021/jacs.5c09558
}
}

@article{frank2024euclidean,
  title={A Euclidean transformer for fast and stable machine learned force fields},
  author={Frank, Thorben and Unke, Oliver and M{\"u}ller, Klaus-Robert and Chmiela, Stefan},
  journal={Nature Communications},
  volume={15},
  number={1},
  pages={6539},
  year={2024}
}
```

Also consider citing MACE, as this software heavily leans on or uses its code:


```bibtex
@inproceedings{Batatia2022mace,
  title={{MACE}: Higher Order Equivariant Message Passing Neural Networks for Fast and Accurate Force Fields},
  author={Ilyes Batatia and David Peter Kovacs and Gregor N. C. Simm and Christoph Ortner and Gabor Csanyi},
  booktitle={Advances in Neural Information Processing Systems},
  editor={Alice H. Oh and Alekh Agarwal and Danielle Belgrave and Kyunghyun Cho},
  year={2022},
  url={https://openreview.net/forum?id=YPpSngE-ZU}
}

@misc{Batatia2022Design,
  title = {The Design Space of E(3)-Equivariant Atom-Centered Interatomic Potentials},
  author = {Batatia, Ilyes and Batzner, Simon and Kov{\'a}cs, D{\'a}vid P{\'e}ter and Musaelian, Albert and Simm, Gregor N. C. and Drautz, Ralf and Ortner, Christoph and Kozinsky, Boris and Cs{\'a}nyi, G{\'a}bor},
  year = {2022},
  number = {arXiv:2205.06643},
  eprint = {2205.06643},
  eprinttype = {arxiv},
  doi = {10.48550/arXiv.2205.06643},
  archiveprefix = {arXiv}
 }
```

## Contact

If you have questions you can reach me at: tobias.henkes@uni.lu

For bugs or feature requests, please use [GitHub Issues](https://github.com/tohenkes/So3krates-torch/issues).

## License

The code is published and distributed under the [MIT License](MIT.md).
