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
3. Error metrics over ase readable datasets: `torchkrates-test`
4. Transforming pyTorch and JAX parameter formates: `torchkrates-jax2torch` or `torchkrates-torch2jax` (for these you need to install jax, flax, and mlff (https://github.com/thorben-frank/mlff/tree/v1.0-lrs-gems))
5. Training: `torchkrates-train --config config.yaml` (see example)


> [!IMPORTANT]
> Number 4 means that you can transform the weights from this pytorch version into the JAX version and vice versa. Inference and training is much faster (*at least 1 order of magnitude at the moment*) in the JAX version. This implementation is mostly for prototyping and compatability with other packages.

---

## CLI Reference

### `torchkrates-train` — Training

Train an SO3LR (or multi-head SO3LR) model from a YAML configuration file.

```bash
torchkrates-train --config config.yaml
```

The YAML file is organised into sections: `GENERAL`, `ARCHITECTURE`, `DATA`, `TRAINING`, `LOSS`, `OPTIMIZER`, `SCHEDULER`, `FINETUNING`, and `MISC`. An example is provided in `examples/training/train_settings_example.yaml`.

#### `ARCHITECTURE` settings

These keywords map directly to the constructor arguments of the `So3krates` / `SO3LR` model classes.

##### Core Euclidean Transformer parameters

| Keyword | Default | Description |
|---------|---------|-------------|
| `degrees` | *(required)* | List of spherical-harmonic degrees used for the equivariant (geometric) features. Determines the angular resolution of the representation. E.g. `[1, 2]` uses $\ell=1$ (vectors) and $\ell=2$ (rank-2 tensors). Higher degrees capture more angular detail at higher cost. The number of equivariant feature channels equals $\sum_\ell (2\ell+1)$ and this also sets the number of equivariant attention heads. |
| `num_features` | `128` | Dimension of the invariant (scalar) feature vector per atom. Must be divisible by 4 (used for internal head splitting in the FilterNet). Also sets the width of the MLP layers inside the Euclidean Transformer. |
| `num_heads` | `4` | Number of attention heads for the **invariant** self-attention. The invariant feature dimension is split into `num_features // num_heads` per head. Each head learns independent query/key/value projections. |
| `num_layers` | `3` | Number of stacked Euclidean Transformer layers. Each layer consists of a Euclidean attention block (with separate invariant and equivariant attention) followed by an interaction block that mixes invariant and equivariant features via SO(3)-equivariant contractions. |
| `r_max` | `4.5` | Short-range cutoff radius in Ångström. Atoms within this distance are connected by edges. Defines the neighbourhood for the graph neural network message passing. |
| `num_radial_basis_fn` | `32` | Number of radial basis functions used to expand interatomic distances. These are fed into the FilterNet (Eq. 20 of the So3krates paper) to produce distance-dependent filter weights for the attention mechanism. |
| `radial_basis_fn` | `"bernstein"` | Type of radial basis expansion. Options: `"gaussian"` (evenly spaced Gaussians), `"bernstein"` (Bernstein polynomials in exponential coordinates, as used in SO3LR), `"bessel"` (sinc-type basis). |
| `cutoff_fn` | `"cosine"` | Smooth cutoff function applied to interatomic distances at `r_max`. Options: `"cosine"` (Eq. 17 of the paper), `"phys"` (PhysNet polynomial), `"polynomial"` (tuneable polynomial order via `cutoff_p`), `"exponential"` (SpookyNet-style). |
| `energy_regression_dim` | `128` | Hidden dimension of the energy readout MLP. If `None`, defaults to `num_features`. The readout consists of `final_mlp_layers` linear layers mapping from `num_features` → `energy_regression_dim` → 1. |
| `message_normalization` | `"avg_num_neighbors"` | Normalisation factor for attention scores (Eq. 21 of the paper). `"sqrt_num_features"` divides by $\sqrt{d_\text{head}}$ (standard transformer scaling), `"avg_num_neighbors"` divides by the average coordination number (dataset-dependent, set automatically from the training data), `"identity"` applies no normalisation. |

##### Activation & non-linearity

| Keyword | Default | Description |
|---------|---------|-------------|
| `activation_fn` | `"silu"` | Activation function used inside the Euclidean Transformer layers (FilterNet MLPs, interaction block, residual MLPs). Options: `"silu"`, `"relu"`, `"gelu"`, `"tanh"`, `"identity"`. |
| `energy_activation_fn` | `"silu"` | Activation function used in the energy readout MLP (output block). Same options as `activation_fn`. |
| `qk_non_linearity` | `"identity"` | Non-linearity applied to query and key projections before computing attention scores. `"identity"` means standard linear attention; setting e.g. `"silu"` adds a non-linearity after the Q/K projections. |

##### Layer enhancements (from the SO3LR paper)

| Keyword | Default | Description |
|---------|---------|-------------|
| `layer_normalization_1` | `false` | Apply LayerNorm (eps=1e-6) to the invariant features **after** the attention residual connection and **before** the interaction block within each Euclidean Transformer layer. |
| `layer_normalization_2` | `false` | Apply LayerNorm (eps=1e-6) to the invariant features **after** the interaction block residual connection (i.e. at the very end of each layer). |
| `residual_mlp_1` | `false` | Add a two-layer residual MLP on the invariant features between the attention block and the interaction block. Provides additional expressivity within each layer. |
| `residual_mlp_2` | `false` | Add a two-layer residual MLP on the invariant features after the interaction block. Applied before `layer_normalization_2` if both are enabled. |

##### Embedding & element handling

| Keyword | Default | Description |
|---------|---------|-------------|
| `use_charge_embed` | `false` | Enable a charge-dependent atomic embedding (as in Unke et al., Nat. Commun. 2021). Requires `total_charge` in the input data. The embedding is added to the invariant features before the transformer layers. |
| `use_spin_embed` | `false` | Enable a spin-dependent atomic embedding (same architecture as charge embedding). Requires `total_spin` in the input data (internally multiplied by 2 to get unpaired electrons). |
| `energy_learn_atomic_type_shifts` | `false` | Make per-element energy shifts (E0s) learnable parameters. When `false`, the shifts are fixed to values computed from the training data (linear regression of total energies on composition). |
| `energy_learn_atomic_type_scales` | `false` | Make per-element energy scales learnable parameters. Adds a learned linear scaling per element on top of the predicted atomic energies. |

##### SO3LR-specific parameters (long-range physics)

These are specific to the `SO3LR` model, which extends `So3krates` with universal pairwise physical potentials.

| Keyword | Default | Description |
|---------|---------|-------------|
| `zbl_repulsion_bool` | `true` | Enable the Ziegler-Biersack-Littmark (ZBL) short-range repulsion potential. Provides a physically correct repulsive wall at very short interatomic distances; uses the short-range cutoff and neighbour list. |
| `electrostatic_energy_bool` | `true` | Enable the long-range electrostatic interaction. Predicts partial atomic charges from the learned invariant features via a dedicated output head, then computes Coulomb interactions over the long-range neighbour list. |
| `electrostatic_energy_scale` | `4.0` | Global scaling factor for the electrostatic energy contribution (multiplied with the Coulomb potential). |
| `dispersion_energy_bool` | `true` | Enable the long-range dispersion (van der Waals) interaction. Predicts Hirshfeld volume ratios from the learned features, then computes many-body dispersion C6-based pairwise interactions over the long-range neighbour list. |
| `dispersion_energy_scale` | `1.2` | Global scaling factor for the dispersion energy contribution. |
| `dispersion_energy_cutoff_lr_damping` | `None` | Optional additional cutoff for damping the dispersion interaction at long range. If `None`, no extra damping beyond the standard MBD damping function is applied. |
| `r_max_lr` | `None` | Long-range cutoff radius in Ångström (e.g. `12.0`). Defines the neighbourhood for electrostatic and dispersion interactions. Must be set when either `electrostatic_energy_bool` or `dispersion_energy_bool` is `true`. |
| `neighborlist_format_lr` | `"sparse"` | Storage format for the long-range neighbour list. `"sparse"` uses an edge-index representation. |

##### Multi-head model

| Keyword | Default | Description |
|---------|---------|-------------|
| `convert_to_multihead` | `false` | If `true`, creates a `MultiHeadSO3LR` model with multiple independent energy output heads sharing the same representation backbone. |
| `num_output_heads` | *(required if multi-head)* | Number of independent energy readout heads. Each head has its own MLP and per-element shifts/scales but shares all transformer layers. |

##### Initialisation & misc

| Keyword | Default | Description |
|---------|---------|-------------|
| `input_convention` | `"positions"` | Input convention. Currently only `"positions"` is supported (Cartesian atomic positions). |
| `layers_behave_like_identity_fn_at_init` | `false` | *(Not yet implemented.)* Initialise transformer layers such that they act as identity functions at the start of training. |
| `output_is_zero_at_init` | `false` | *(Not yet implemented.)* Initialise the output MLP such that predictions are zero before training. |

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

### `torchkrates-lammps` — LAMMPS Model Export

> [!NOTE]
> More details and how to use the model in LAMMPS are coming.

> [!IMPORTANT]  
> Only works with torch==2.6.0 for CUDA 12.6.0 on Meluxina!

Convert a trained SO3LR model to a TorchScript model compatible with the LAMMPS ML-IAP interface.

```bash
torchkrates-lammps model.pt --elements Si O
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

### `torchkrates-test` — Error Metrics

Compute error metrics over an ASE-readable dataset.

### `torchkrates-jax2torch` / `torchkrates-torch2jax` — Weight Conversion

Convert model weights between the PyTorch and JAX (mlff) implementations. Requires `jax`, `flax`, and [`mlff`](https://github.com/thorben-frank/mlff/tree/v1.0-lrs-gems) to be installed.

---

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

Also consider citing MACE, as this software heavlily leans on or uses its code:


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
