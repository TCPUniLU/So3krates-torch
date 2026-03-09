import torch
from typing import Dict, Any, Optional, Callable
from ml_collections import config_dict
import jax
import flax
import json
import numpy as np
import pickle
from so3krates_torch.modules.models import So3krates, SO3LR
import yaml
import torch.nn as nn


def flatten_params(params, prefix=""):
    flat_params = {}

    def _recurse(p, prefix=""):
        if isinstance(p, dict) or isinstance(
            p, flax.core.frozen_dict.FrozenDict
        ):
            for k, v in p.items():
                _recurse(v, prefix=f"{prefix}/{k}" if prefix else k)
        else:
            flat_params[prefix] = p

    _recurse(params, prefix)
    return flat_params


def unflatten_params(
    flat_params: dict,
) -> dict:
    nested = {}
    for flat_key, value in flat_params.items():
        keys = flat_key.split("/")
        d = nested
        for k in keys[:-1]:
            if k not in d:
                d[k] = {}
            d = d[k]
        d[keys[-1]] = value
    return nested


def get_flax_to_torch_mapping(cfg, trainable_rbf: bool):
    num_layers = cfg.model.num_layers
    layer_norm_1 = cfg.model.layer_normalization_1
    layer_norm_2 = cfg.model.layer_normalization_2
    residual_mlp_1 = cfg.model.residual_mlp_1
    residual_mlp_2 = cfg.model.residual_mlp_2
    energy_learn_atomic_type_shifts = cfg.model.energy_learn_atomic_type_shifts
    energy_learn_atomic_type_scales = cfg.model.energy_learn_atomic_type_scales
    use_charge_embed = cfg.model.use_charge_embed
    use_spin_embed = cfg.model.use_spin_embed
    use_zbl = cfg.model.zbl_repulsion_bool
    use_electrostatic_energy = cfg.model.electrostatic_energy_bool
    use_dispersion_energy = cfg.model.dispersion_energy_bool

    mapping = {}

    # Embedding layers
    mapping[
        "params/feature_embeddings_0/Embed_0/embedding"
    ] = "inv_feature_embedding.embedding.weight"
    if trainable_rbf:
        mapping[
            "params/geometry_embeddings_0/rbf_fn/centers"
        ] = "radial_embedding.radial_basis_fn.centers"
        mapping[
            "params/geometry_embeddings_0/rbf_fn/widths"
        ] = "radial_embedding.radial_basis_fn.widths"
    if use_charge_embed and not use_spin_embed:
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_0/embedding"
        ] = "charge_embedding.Wq.weight"
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_1/embedding"
        ] = "charge_embedding.Wk"
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_2/embedding"
        ] = "charge_embedding.Wv"
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_0/kernel"
        ] = "charge_embedding.mlp.1.weight"
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_1/kernel"
        ] = "charge_embedding.mlp.3.weight"
    elif use_spin_embed and not use_charge_embed:
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_0/embedding"
        ] = "spin_embedding.Wq.weight"
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_1/embedding"
        ] = "spin_embedding.Wk"
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_2/embedding"
        ] = "spin_embedding.Wv"
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_0/kernel"
        ] = "spin_embedding.mlp.1.weight"
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_1/kernel"
        ] = "spin_embedding.mlp.3.weight"
    elif use_charge_embed and use_spin_embed:
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_0/embedding"
        ] = "charge_embedding.Wq.weight"
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_1/embedding"
        ] = "charge_embedding.Wk"
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_2/embedding"
        ] = "charge_embedding.Wv"
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_0/kernel"
        ] = "charge_embedding.mlp.1.weight"
        mapping[
            "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_1/kernel"
        ] = "charge_embedding.mlp.3.weight"
        mapping[
            "params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Embed_0/embedding"
        ] = "spin_embedding.Wq.weight"
        mapping[
            "params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Embed_1/embedding"
        ] = "spin_embedding.Wk"
        mapping[
            "params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Embed_2/embedding"
        ] = "spin_embedding.Wv"
        mapping[
            "params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Residual_0/layers_0/kernel"
        ] = "spin_embedding.mlp.1.weight"
        mapping[
            "params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Residual_0/layers_1/kernel"
        ] = "spin_embedding.mlp.3.weight"

    # Per-layer transformer mappings
    for i in range(num_layers):
        flax_prefix = f"params/layers_{i}/attention_block"
        torch_prefix = f"euclidean_transformers.{i}"

        # Radial filters (inv)
        mapping[
            f"{flax_prefix}/radial_filter1_layer_1/kernel"
        ] = f"{torch_prefix}.filter_net_inv.mlp_rbf.0.weight"
        mapping[
            f"{flax_prefix}/radial_filter1_layer_1/bias"
        ] = f"{torch_prefix}.filter_net_inv.mlp_rbf.0.bias"
        mapping[
            f"{flax_prefix}/radial_filter1_layer_2/kernel"
        ] = f"{torch_prefix}.filter_net_inv.mlp_rbf.mlp_rbf_layer_1.0.weight"
        mapping[
            f"{flax_prefix}/radial_filter1_layer_2/bias"
        ] = f"{torch_prefix}.filter_net_inv.mlp_rbf.mlp_rbf_layer_1.0.bias"

        # Radial filters (ev)
        mapping[
            f"{flax_prefix}/radial_filter2_layer_1/kernel"
        ] = f"{torch_prefix}.filter_net_ev.mlp_rbf.0.weight"
        mapping[
            f"{flax_prefix}/radial_filter2_layer_1/bias"
        ] = f"{torch_prefix}.filter_net_ev.mlp_rbf.0.bias"
        mapping[
            f"{flax_prefix}/radial_filter2_layer_2/kernel"
        ] = f"{torch_prefix}.filter_net_ev.mlp_rbf.mlp_rbf_layer_1.0.weight"
        mapping[
            f"{flax_prefix}/radial_filter2_layer_2/bias"
        ] = f"{torch_prefix}.filter_net_ev.mlp_rbf.mlp_rbf_layer_1.0.bias"

        # Spherical filters (inv)
        mapping[
            f"{flax_prefix}/spherical_filter1_layer_1/kernel"
        ] = f"{torch_prefix}.filter_net_inv.mlp_ev.0.weight"
        mapping[
            f"{flax_prefix}/spherical_filter1_layer_1/bias"
        ] = f"{torch_prefix}.filter_net_inv.mlp_ev.0.bias"
        mapping[
            f"{flax_prefix}/spherical_filter1_layer_2/kernel"
        ] = f"{torch_prefix}.filter_net_inv.mlp_ev.mlp_ev_layer_1.0.weight"
        mapping[
            f"{flax_prefix}/spherical_filter1_layer_2/bias"
        ] = f"{torch_prefix}.filter_net_inv.mlp_ev.mlp_ev_layer_1.0.bias"

        # Spherical filters (ev)
        mapping[
            f"{flax_prefix}/spherical_filter2_layer_1/kernel"
        ] = f"{torch_prefix}.filter_net_ev.mlp_ev.0.weight"
        mapping[
            f"{flax_prefix}/spherical_filter2_layer_1/bias"
        ] = f"{torch_prefix}.filter_net_ev.mlp_ev.0.bias"
        mapping[
            f"{flax_prefix}/spherical_filter2_layer_2/kernel"
        ] = f"{torch_prefix}.filter_net_ev.mlp_ev.mlp_ev_layer_1.0.weight"
        mapping[
            f"{flax_prefix}/spherical_filter2_layer_2/bias"
        ] = f"{torch_prefix}.filter_net_ev.mlp_ev.mlp_ev_layer_1.0.bias"

        # Attention weights
        mapping[
            f"{flax_prefix}/Wq1"
        ] = f"{torch_prefix}.euclidean_attention_block.W_q_inv"
        mapping[
            f"{flax_prefix}/Wk1"
        ] = f"{torch_prefix}.euclidean_attention_block.W_k_inv"
        mapping[
            f"{flax_prefix}/Wv1"
        ] = f"{torch_prefix}.euclidean_attention_block.W_v_inv"
        mapping[
            f"{flax_prefix}/Wq2"
        ] = f"{torch_prefix}.euclidean_attention_block.W_q_ev"
        mapping[
            f"{flax_prefix}/Wk2"
        ] = f"{torch_prefix}.euclidean_attention_block.W_k_ev"

        # Exchange block
        mapping[
            f"params/layers_{i}/exchange_block/mlp_layer_2/kernel"
        ] = f"{torch_prefix}.interaction_block.linear_layer.weight"
        mapping[
            f"params/layers_{i}/exchange_block/mlp_layer_2/bias"
        ] = f"{torch_prefix}.interaction_block.linear_layer.bias"

        # Layer normalization
        if layer_norm_1:
            mapping[
                f"params/layers_{i}/layer_normalization_1/scale"
            ] = f"{torch_prefix}.layer_norm_inv_1.weight"
            mapping[
                f"params/layers_{i}/layer_normalization_1/bias"
            ] = f"{torch_prefix}.layer_norm_inv_1.bias"
        if layer_norm_2:
            mapping[
                f"params/layers_{i}/layer_normalization_2/scale"
            ] = f"{torch_prefix}.layer_norm_inv_2.weight"
            mapping[
                f"params/layers_{i}/layer_normalization_2/bias"
            ] = f"{torch_prefix}.layer_norm_inv_2.bias"

        # Residual MLPs
        if residual_mlp_1:
            mapping[
                f"params/layers_{i}/res_mlp_1_layer_1/kernel"
            ] = f"{torch_prefix}.mlp_1.1.weight"
            mapping[
                f"params/layers_{i}/res_mlp_1_layer_1/bias"
            ] = f"{torch_prefix}.mlp_1.1.bias"
            mapping[
                f"params/layers_{i}/res_mlp_1_layer_2/kernel"
            ] = f"{torch_prefix}.mlp_1.3.weight"
            mapping[
                f"params/layers_{i}/res_mlp_1_layer_2/bias"
            ] = f"{torch_prefix}.mlp_1.3.bias"
        if residual_mlp_2:
            mapping[
                f"params/layers_{i}/res_mlp_2_layer_1/kernel"
            ] = f"{torch_prefix}.mlp_2.1.weight"
            mapping[
                f"params/layers_{i}/res_mlp_2_layer_1/bias"
            ] = f"{torch_prefix}.mlp_2.1.bias"
            mapping[
                f"params/layers_{i}/res_mlp_2_layer_2/kernel"
            ] = f"{torch_prefix}.mlp_2.3.weight"
            mapping[
                f"params/layers_{i}/res_mlp_2_layer_2/bias"
            ] = f"{torch_prefix}.mlp_2.3.bias"

    # Output layers
    mapping[
        "params/observables_0/energy_dense_regression/kernel"
    ] = "atomic_energy_output_block.layers.0.weight"
    mapping[
        "params/observables_0/energy_dense_regression/bias"
    ] = "atomic_energy_output_block.layers.0.bias"
    mapping[
        "params/observables_0/energy_dense_final/kernel"
    ] = "atomic_energy_output_block.final_layer.weight"
    mapping[
        "params/observables_0/energy_dense_final/bias"
    ] = "atomic_energy_output_block.final_layer.bias"
    if energy_learn_atomic_type_shifts:
        mapping[
            "params/observables_0/energy_offset"
        ] = "atomic_energy_output_block.energy_shifts"
    if energy_learn_atomic_type_scales:
        mapping[
            "params/observables_0/atomic_scales"
        ] = "atomic_energy_output_block.energy_scales.weight"

    params_obs = "params/observables_0/"
    if use_zbl:
        mapping[f"{params_obs}zbl_repulsion/a1"] = "zbl_repulsion.a1_raw"
        mapping[f"{params_obs}zbl_repulsion/a2"] = "zbl_repulsion.a2_raw"
        mapping[f"{params_obs}zbl_repulsion/a3"] = "zbl_repulsion.a3_raw"
        mapping[f"{params_obs}zbl_repulsion/a4"] = "zbl_repulsion.a4_raw"
        mapping[f"{params_obs}zbl_repulsion/c1"] = "zbl_repulsion.c1_raw"
        mapping[f"{params_obs}zbl_repulsion/c2"] = "zbl_repulsion.c2_raw"
        mapping[f"{params_obs}zbl_repulsion/c3"] = "zbl_repulsion.c3_raw"
        mapping[f"{params_obs}zbl_repulsion/c4"] = "zbl_repulsion.c4_raw"
        mapping[f"{params_obs}zbl_repulsion/p"] = "zbl_repulsion.p_raw"
        mapping[f"{params_obs}zbl_repulsion/d"] = "zbl_repulsion.d_raw"

    # if use_electrostatic_energy:
    mapping[
        f"{params_obs}electrostatic_energy/partial_charges/Embed_0/embedding"
    ] = "partial_charges_output_block.atomic_embedding.weight"
    mapping[
        f"{params_obs}electrostatic_energy/partial_charges/charge_dense_regression_vec/kernel"
    ] = "partial_charges_output_block.transform_inv_features.0.weight"
    mapping[
        f"{params_obs}electrostatic_energy/partial_charges/charge_dense_regression_vec/bias"
    ] = "partial_charges_output_block.transform_inv_features.0.bias"
    mapping[
        f"{params_obs}electrostatic_energy/partial_charges/charge_dense_final_vec/kernel"
    ] = "partial_charges_output_block.transform_inv_features.2.weight"
    mapping[
        f"{params_obs}electrostatic_energy/partial_charges/charge_dense_final_vec/bias"
    ] = "partial_charges_output_block.transform_inv_features.2.bias"

    # if use_dispersion_energy:
    mapping[
        "params/observables_2/Embed_0/embedding"
    ] = "hirshfeld_output_block.v_shift_embedding.weight"
    mapping[
        "params/observables_2/Embed_1/embedding"
    ] = "hirshfeld_output_block.q_embedding.weight"
    mapping[
        "params/observables_2/hirshfeld_ratios_dense_regression/kernel"
    ] = "hirshfeld_output_block.transform_features.0.weight"
    mapping[
        "params/observables_2/hirshfeld_ratios_dense_regression/bias"
    ] = "hirshfeld_output_block.transform_features.0.bias"
    mapping[
        "params/observables_2/hirshfeld_ratios_dense_final/kernel"
    ] = "hirshfeld_output_block.transform_features.2.weight"
    mapping[
        "params/observables_2/hirshfeld_ratios_dense_final/bias"
    ] = "hirshfeld_output_block.transform_features.2.bias"

    return mapping


def get_model_settings_flax_to_torch(
    cfg: config_dict.ConfigDict,
    device: str,
    use_defined_shifts: bool,
    num_elements: int = 118,
    trainable_rbf: bool = True,
    dtype: torch.dtype = torch.float32,
):
    return dict(
        r_max=cfg.model.cutoff,
        r_max_lr=cfg.model.cutoff_lr,
        num_radial_basis_fn=cfg.model.num_radial_basis_fn,
        degrees=cfg.model.degrees,
        num_features=cfg.model.num_features,
        num_heads=cfg.model.num_heads,
        num_layers=cfg.model.num_layers,
        num_elements=num_elements,
        avg_num_neighbors=cfg.data.avg_num_neighbors,
        cutoff_fn=cfg.model.cutoff_fn,
        radial_basis_fn=cfg.model.radial_basis_fn,
        message_normalization=cfg.model.message_normalization,
        device=device,
        trainable_rbf=trainable_rbf,
        dtype=dtype,
        atomic_type_shifts=(
            cfg.data.energy_shifts.to_dict() if use_defined_shifts else None
        ),
        energy_learn_atomic_type_shifts=cfg.model.energy_learn_atomic_type_shifts,
        energy_learn_atomic_type_scales=cfg.model.energy_learn_atomic_type_scales,
        energy_regression_dim=cfg.model.energy_regression_dim,
        layer_normalization_1=cfg.model.layer_normalization_1,
        layer_normalization_2=cfg.model.layer_normalization_2,
        residual_mlp_1=cfg.model.residual_mlp_1,
        residual_mlp_2=cfg.model.residual_mlp_2,
        use_charge_embed=cfg.model.use_charge_embed,
        use_spin_embed=cfg.model.use_spin_embed,
        zbl_repulsion_bool=cfg.model.zbl_repulsion_bool,
        electrostatic_energy_bool=True,  # cfg.model.electrostatic_energy_bool, # always true, because its fuckting dumb how they handle the oberservable bools, turned false later
        electrostatic_energy_scale=cfg.model.electrostatic_energy_scale,
        dispersion_energy_bool=True,  # cfg.model.dispersion_energy_bool, # always true, because its fuckting dumb how they handle the oberservable bools, turned false later
        dispersion_energy_cutoff_lr_damping=cfg.model.dispersion_energy_cutoff_lr_damping,
        dispersion_energy_scale=cfg.model.dispersion_energy_scale,
        qk_non_linearity=cfg.model.qk_non_linearity,
        num_features_head=cfg.model.num_features // cfg.model.num_heads,
        activation_fn=cfg.model.activation_fn,
        layers_behave_like_identity_fn_at_init=cfg.model.layers_behave_like_identity_fn_at_init,
        output_is_zero_at_init=cfg.model.output_is_zero_at_init,
        input_convention=cfg.model.input_convention,
        energy_activation_fn=cfg.model.energy_activation_fn,
    )


def convert_flax_to_torch_params(
    torch_state_dict: Dict[str, torch.Tensor],
    flax_params: Dict[str, Any],
    mapping: Dict[str, str],
    dtype: torch.dtype = torch.float32,
):
    torch.set_default_dtype(dtype)
    if dtype == torch.float64:
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_enable_x64", False)
    flat_params = flatten_params(flax_params)
    embeddings = [
        "inv_feature_embedding.embedding.weight",
        "charge_embedding.Wq.weight",
        "spin_embedding.Wq.weight",
    ]
    special_embeddings = [
        "charge_embedding.Wk",
        "charge_embedding.Wv",
        "spin_embedding.Wk",
        "spin_embedding.Wv",
        "partial_charges_output_block.atomic_embedding.weight",
        "hirshfeld_output_block.v_shift_embedding.weight",
        "hirshfeld_output_block.q_embedding.weight",
    ]
    for flax_key, torch_key in mapping.items():
        flax_array = flat_params[flax_key]
        flax_array_np = np.array(flax_array)
        torched = torch.from_numpy(flax_array_np)

        expected_shape = torch_state_dict[torch_key].shape
        if flax_array.ndim == 2 and torch_key not in special_embeddings:
            torched = torched.T
        elif flax_array.ndim == 3:
            torched = torched.permute(0, 2, 1)

        if torch_key in embeddings:
            torched = torched[:, 1:]
        if (
            flax_key == "params/observables_0/energy_offset"
            or flax_key == "params/observables_0/atomic_scales"
        ):
            if torch_key == "atomic_energy_output_block.energy_shifts":
                torched = torched[1:]
            else:
                torched = torched[1:].unsqueeze(0)
        if torched.shape != expected_shape:
            print(
                f"Shape mismatch for {torch_key}: expected {expected_shape}, got {torched.shape}"
            )
        torch_state_dict[torch_key] = torched

    return torch_state_dict


def convert_flax_to_torch(
    path_to_flax_params: str,
    path_to_flax_hyperparams: str,
    so3lr: bool = True,
    torch_save_path: Optional[str] = None,
    device: str = "cpu",
    use_defined_shifts: bool = False,
    num_elements: int = 118,
    trainable_rbf: bool = False,
    dtype: torch.dtype = torch.float32,
    save_torch_settings: Optional[str] = None,
):
    with open(path_to_flax_params, "rb") as f:
        flax_params = pickle.load(f)
    with open(path_to_flax_hyperparams, "r") as f:
        cfg = json.load(f)
    cfg = config_dict.ConfigDict(cfg)
    torch_model_settings = get_model_settings_flax_to_torch(
        cfg=cfg,
        device=device,
        use_defined_shifts=use_defined_shifts,
        num_elements=num_elements,
        trainable_rbf=trainable_rbf,
        dtype=dtype,
    )
    if hasattr(cfg.data, "energy_shifts"):
        energy_shifts_dict = cfg.data.energy_shifts.to_dict()
        # turn keys in to float, sort and then back to str

        energy_shifts_dict = {int(k): v for k, v in energy_shifts_dict.items()}
        energy_shifts_dict = dict(
            sorted(energy_shifts_dict.items(), key=lambda item: item[0])
        )
        energy_shifts_dict.pop(0, None)
        energy_shifts_dict = {str(k): v for k, v in energy_shifts_dict.items()}

    if save_torch_settings:
        serializable_settings = torch_model_settings.copy()
        if cfg.model.electrostatic_energy_bool is False:
            serializable_settings["electrostatic_energy_bool"] = False
        if cfg.model.dispersion_energy_bool is False:
            serializable_settings["dispersion_energy_bool"] = False
        if cfg.model.zbl_repulsion_bool is False:
            serializable_settings["zbl_repulsion_bool"] = False
        serializable_settings["dtype"] = str(
            dtype
        )  # Convert torch.dtype to string

        if hasattr(cfg.data, "energy_shifts"):
            serializable_settings["atomic_type_shifts"] = energy_shifts_dict

        settings_to_save = {"ARCHITECTURE": serializable_settings}
        with open(save_torch_settings, "w") as f:
            yaml.dump(settings_to_save, f, default_flow_style=False)

    if so3lr:
        torch_model = SO3LR(**torch_model_settings)
        torch_model.electrostatic_energy_bool = (
            cfg.model.electrostatic_energy_bool
        )
        torch_model.dispersion_energy_bool = cfg.model.dispersion_energy_bool
        torch_model.zbl_repulsion_bool = cfg.model.zbl_repulsion_bool
        if (
            not cfg.model.electrostatic_energy_bool
            and not cfg.model.dispersion_energy_bool
        ):
            torch_model.use_lr = False
    else:
        torch_model = So3krates(**torch_model_settings)

    state_dict = torch_model.state_dict()

    torch_state_dict = convert_flax_to_torch_params(
        torch_state_dict=state_dict,
        flax_params=flax_params,
        mapping=get_flax_to_torch_mapping(
            cfg=cfg, trainable_rbf=trainable_rbf
        ),
        dtype=dtype,
    )

    if hasattr(cfg.data, "energy_shifts"):
        torch_state_dict[
            "atomic_energy_output_block.energy_shifts"
        ] = nn.Parameter(
            torch.tensor(
                list(energy_shifts_dict.values()),
                dtype=torch.get_default_dtype(),
                requires_grad=False,
            )
        )

    if torch_save_path:
        torch.save(torch_state_dict, torch_save_path)
    torch_model.load_state_dict(torch_state_dict)
    torch_model.to(device)

    if cfg.model.electrostatic_energy_bool is False:
        torch_model.electrostatic_energy_bool = False
    if cfg.model.dispersion_energy_bool is False:
        torch_model.dispersion_energy_bool = False
    if cfg.model.zbl_repulsion_bool is False:
        torch_model.zbl_repulsion_bool = False

    return torch_model


def get_torch_to_flax_mapping(cfg, trainable_rbf: bool):
    """Get mapping from PyTorch parameter names to Flax parameter names"""
    num_layers = cfg.model.num_layers
    layer_norm_1 = cfg.model.layer_normalization_1
    layer_norm_2 = cfg.model.layer_normalization_2
    residual_mlp_1 = cfg.model.residual_mlp_1
    residual_mlp_2 = cfg.model.residual_mlp_2
    energy_learn_atomic_type_shifts = cfg.model.energy_learn_atomic_type_shifts
    energy_learn_atomic_type_scales = cfg.model.energy_learn_atomic_type_scales
    use_charge_embed = cfg.model.use_charge_embed
    use_spin_embed = cfg.model.use_spin_embed
    use_zbl = cfg.model.zbl_repulsion_bool
    use_electrostatic_energy = cfg.model.electrostatic_energy_bool
    use_dispersion_energy = cfg.model.dispersion_energy_bool

    mapping = {}

    # Embedding layers
    mapping[
        "inv_feature_embedding.embedding.weight"
    ] = "params/feature_embeddings_0/Embed_0/embedding"
    if trainable_rbf:
        mapping[
            "radial_embedding.radial_basis_fn.centers"
        ] = "params/geometry_embeddings_0/rbf_fn/centers"
        mapping[
            "radial_embedding.radial_basis_fn.widths"
        ] = "params/geometry_embeddings_0/rbf_fn/widths"

    if use_charge_embed and not use_spin_embed:
        mapping[
            "charge_embedding.Wq.weight"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_0/embedding"
        mapping[
            "charge_embedding.Wk"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_1/embedding"
        mapping[
            "charge_embedding.Wv"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_2/embedding"
        mapping[
            "charge_embedding.mlp.1.weight"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_0/kernel"
        mapping[
            "charge_embedding.mlp.3.weight"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_1/kernel"
    elif use_spin_embed and not use_charge_embed:
        mapping[
            "spin_embedding.Wq.weight"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_0/embedding"
        mapping[
            "spin_embedding.Wk"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_1/embedding"
        mapping[
            "spin_embedding.Wv"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_2/embedding"
        mapping[
            "spin_embedding.mlp.1.weight"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_0/kernel"
        mapping[
            "spin_embedding.mlp.3.weight"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_1/kernel"
    elif use_charge_embed and use_spin_embed:
        mapping[
            "charge_embedding.Wq.weight"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_0/embedding"
        mapping[
            "charge_embedding.Wk"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_1/embedding"
        mapping[
            "charge_embedding.Wv"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Embed_2/embedding"
        mapping[
            "charge_embedding.mlp.1.weight"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_0/kernel"
        mapping[
            "charge_embedding.mlp.3.weight"
        ] = "params/feature_embeddings_1/ChargeSpinEmbedSparse_0/Residual_0/layers_1/kernel"
        mapping[
            "spin_embedding.Wq.weight"
        ] = "params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Embed_0/embedding"
        mapping[
            "spin_embedding.Wk"
        ] = "params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Embed_1/embedding"
        mapping[
            "spin_embedding.Wv"
        ] = "params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Embed_2/embedding"
        mapping[
            "spin_embedding.mlp.1.weight"
        ] = "params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Residual_0/layers_0/kernel"
        mapping[
            "spin_embedding.mlp.3.weight"
        ] = "params/feature_embeddings_2/ChargeSpinEmbedSparse_0/Residual_0/layers_1/kernel"

    # Per-layer transformer mappings
    for i in range(num_layers):
        flax_prefix = f"params/layers_{i}/attention_block"
        torch_prefix = f"euclidean_transformers.{i}"

        # Radial filters (inv)
        mapping[
            f"{torch_prefix}.filter_net_inv.mlp_rbf.0.weight"
        ] = f"{flax_prefix}/radial_filter1_layer_1/kernel"
        mapping[
            f"{torch_prefix}.filter_net_inv.mlp_rbf.0.bias"
        ] = f"{flax_prefix}/radial_filter1_layer_1/bias"
        mapping[
            f"{torch_prefix}.filter_net_inv.mlp_rbf.mlp_rbf_layer_1.0.weight"
        ] = f"{flax_prefix}/radial_filter1_layer_2/kernel"
        mapping[
            f"{torch_prefix}.filter_net_inv.mlp_rbf.mlp_rbf_layer_1.0.bias"
        ] = f"{flax_prefix}/radial_filter1_layer_2/bias"

        # Radial filters (ev)
        mapping[
            f"{torch_prefix}.filter_net_ev.mlp_rbf.0.weight"
        ] = f"{flax_prefix}/radial_filter2_layer_1/kernel"
        mapping[
            f"{torch_prefix}.filter_net_ev.mlp_rbf.0.bias"
        ] = f"{flax_prefix}/radial_filter2_layer_1/bias"
        mapping[
            f"{torch_prefix}.filter_net_ev.mlp_rbf.mlp_rbf_layer_1.0.weight"
        ] = f"{flax_prefix}/radial_filter2_layer_2/kernel"
        mapping[
            f"{torch_prefix}.filter_net_ev.mlp_rbf.mlp_rbf_layer_1.0.bias"
        ] = f"{flax_prefix}/radial_filter2_layer_2/bias"

        # Spherical filters (inv)
        mapping[
            f"{torch_prefix}.filter_net_inv.mlp_ev.0.weight"
        ] = f"{flax_prefix}/spherical_filter1_layer_1/kernel"
        mapping[
            f"{torch_prefix}.filter_net_inv.mlp_ev.0.bias"
        ] = f"{flax_prefix}/spherical_filter1_layer_1/bias"
        mapping[
            f"{torch_prefix}.filter_net_inv.mlp_ev.mlp_ev_layer_1.0.weight"
        ] = f"{flax_prefix}/spherical_filter1_layer_2/kernel"
        mapping[
            f"{torch_prefix}.filter_net_inv.mlp_ev.mlp_ev_layer_1.0.bias"
        ] = f"{flax_prefix}/spherical_filter1_layer_2/bias"

        # Spherical filters (ev)
        mapping[
            f"{torch_prefix}.filter_net_ev.mlp_ev.0.weight"
        ] = f"{flax_prefix}/spherical_filter2_layer_1/kernel"
        mapping[
            f"{torch_prefix}.filter_net_ev.mlp_ev.0.bias"
        ] = f"{flax_prefix}/spherical_filter2_layer_1/bias"
        mapping[
            f"{torch_prefix}.filter_net_ev.mlp_ev.mlp_ev_layer_1.0.weight"
        ] = f"{flax_prefix}/spherical_filter2_layer_2/kernel"
        mapping[
            f"{torch_prefix}.filter_net_ev.mlp_ev.mlp_ev_layer_1.0.bias"
        ] = f"{flax_prefix}/spherical_filter2_layer_2/bias"

        # Attention weights
        mapping[
            f"{torch_prefix}.euclidean_attention_block.W_q_inv"
        ] = f"{flax_prefix}/Wq1"
        mapping[
            f"{torch_prefix}.euclidean_attention_block.W_k_inv"
        ] = f"{flax_prefix}/Wk1"
        mapping[
            f"{torch_prefix}.euclidean_attention_block.W_v_inv"
        ] = f"{flax_prefix}/Wv1"
        mapping[
            f"{torch_prefix}.euclidean_attention_block.W_q_ev"
        ] = f"{flax_prefix}/Wq2"
        mapping[
            f"{torch_prefix}.euclidean_attention_block.W_k_ev"
        ] = f"{flax_prefix}/Wk2"

        # Exchange block
        mapping[
            f"{torch_prefix}.interaction_block.linear_layer.weight"
        ] = f"params/layers_{i}/exchange_block/mlp_layer_2/kernel"
        mapping[
            f"{torch_prefix}.interaction_block.linear_layer.bias"
        ] = f"params/layers_{i}/exchange_block/mlp_layer_2/bias"

        # Layer normalization
        if layer_norm_1:
            mapping[
                f"{torch_prefix}.layer_norm_inv_1.weight"
            ] = f"params/layers_{i}/layer_normalization_1/scale"
            mapping[
                f"{torch_prefix}.layer_norm_inv_1.bias"
            ] = f"params/layers_{i}/layer_normalization_1/bias"
        if layer_norm_2:
            mapping[
                f"{torch_prefix}.layer_norm_inv_2.weight"
            ] = f"params/layers_{i}/layer_normalization_2/scale"
            mapping[
                f"{torch_prefix}.layer_norm_inv_2.bias"
            ] = f"params/layers_{i}/layer_normalization_2/bias"

        # Residual MLPs
        if residual_mlp_1:
            mapping[
                f"{torch_prefix}.mlp_1.1.weight"
            ] = f"params/layers_{i}/res_mlp_1_layer_1/kernel"
            mapping[
                f"{torch_prefix}.mlp_1.1.bias"
            ] = f"params/layers_{i}/res_mlp_1_layer_1/bias"
            mapping[
                f"{torch_prefix}.mlp_1.3.weight"
            ] = f"params/layers_{i}/res_mlp_1_layer_2/kernel"
            mapping[
                f"{torch_prefix}.mlp_1.3.bias"
            ] = f"params/layers_{i}/res_mlp_1_layer_2/bias"
        if residual_mlp_2:
            mapping[
                f"{torch_prefix}.mlp_2.1.weight"
            ] = f"params/layers_{i}/res_mlp_2_layer_1/kernel"
            mapping[
                f"{torch_prefix}.mlp_2.1.bias"
            ] = f"params/layers_{i}/res_mlp_2_layer_1/bias"
            mapping[
                f"{torch_prefix}.mlp_2.3.weight"
            ] = f"params/layers_{i}/res_mlp_2_layer_2/kernel"
            mapping[
                f"{torch_prefix}.mlp_2.3.bias"
            ] = f"params/layers_{i}/res_mlp_2_layer_2/bias"

    # Output layers
    mapping[
        "atomic_energy_output_block.layers.0.weight"
    ] = "params/observables_0/energy_dense_regression/kernel"
    mapping[
        "atomic_energy_output_block.layers.0.bias"
    ] = "params/observables_0/energy_dense_regression/bias"
    mapping[
        "atomic_energy_output_block.final_layer.weight"
    ] = "params/observables_0/energy_dense_final/kernel"
    mapping[
        "atomic_energy_output_block.final_layer.bias"
    ] = "params/observables_0/energy_dense_final/bias"
    mapping[
        "atomic_energy_output_block.energy_shifts"
    ] = "params/observables_0/energy_offset"
    mapping[
        "atomic_energy_output_block.energy_scales.weight"
    ] = "params/observables_0/atomic_scales"

    params_obs = "params/observables_0/"
    # if use_zbl:
    mapping["zbl_repulsion.a1_raw"] = f"{params_obs}zbl_repulsion/a1"
    mapping["zbl_repulsion.a2_raw"] = f"{params_obs}zbl_repulsion/a2"
    mapping["zbl_repulsion.a3_raw"] = f"{params_obs}zbl_repulsion/a3"
    mapping["zbl_repulsion.a4_raw"] = f"{params_obs}zbl_repulsion/a4"
    mapping["zbl_repulsion.c1_raw"] = f"{params_obs}zbl_repulsion/c1"
    mapping["zbl_repulsion.c2_raw"] = f"{params_obs}zbl_repulsion/c2"
    mapping["zbl_repulsion.c3_raw"] = f"{params_obs}zbl_repulsion/c3"
    mapping["zbl_repulsion.c4_raw"] = f"{params_obs}zbl_repulsion/c4"
    mapping["zbl_repulsion.p_raw"] = f"{params_obs}zbl_repulsion/p"
    mapping["zbl_repulsion.d_raw"] = f"{params_obs}zbl_repulsion/d"

    # if use_electrostatic_energy:
    mapping[
        "partial_charges_output_block.atomic_embedding.weight"
    ] = f"{params_obs}electrostatic_energy/partial_charges/Embed_0/embedding"
    mapping[
        "partial_charges_output_block.transform_inv_features.0.weight"
    ] = f"{params_obs}electrostatic_energy/partial_charges/charge_dense_regression_vec/kernel"
    mapping[
        "partial_charges_output_block.transform_inv_features.0.bias"
    ] = f"{params_obs}electrostatic_energy/partial_charges/charge_dense_regression_vec/bias"
    mapping[
        "partial_charges_output_block.transform_inv_features.2.weight"
    ] = f"{params_obs}electrostatic_energy/partial_charges/charge_dense_final_vec/kernel"
    mapping[
        "partial_charges_output_block.transform_inv_features.2.bias"
    ] = f"{params_obs}electrostatic_energy/partial_charges/charge_dense_final_vec/bias"

    # if use_dispersion_energy:
    mapping[
        "hirshfeld_output_block.v_shift_embedding.weight"
    ] = "params/observables_2/Embed_0/embedding"
    mapping[
        "hirshfeld_output_block.q_embedding.weight"
    ] = "params/observables_2/Embed_1/embedding"
    mapping[
        "hirshfeld_output_block.transform_features.0.weight"
    ] = "params/observables_2/hirshfeld_ratios_dense_regression/kernel"
    mapping[
        "hirshfeld_output_block.transform_features.0.bias"
    ] = "params/observables_2/hirshfeld_ratios_dense_regression/bias"
    mapping[
        "hirshfeld_output_block.transform_features.2.weight"
    ] = "params/observables_2/hirshfeld_ratios_dense_final/kernel"
    mapping[
        "hirshfeld_output_block.transform_features.2.bias"
    ] = "params/observables_2/hirshfeld_ratios_dense_final/bias"

    return mapping


def get_model_settings_torch_to_flax(
    torch_settings: Dict[str, Any],
) -> config_dict.ConfigDict:
    """Convert PyTorch model settings to Flax ConfigDict format"""

    # Create the nested config structure
    cfg = config_dict.ConfigDict()

    # Model settings
    cfg.model = config_dict.ConfigDict()
    cfg.model.cutoff = torch_settings["r_max"]
    cfg.model.cutoff_lr = torch_settings.get(
        "r_max_lr", torch_settings["r_max"]
    )
    cfg.model.num_radial_basis_fn = torch_settings["num_radial_basis_fn"]
    cfg.model.degrees = torch_settings["degrees"]
    cfg.model.num_features = torch_settings["num_features"]
    cfg.model.num_heads = torch_settings["num_heads"]
    cfg.model.num_layers = torch_settings["num_layers"]
    cfg.model.cutoff_fn = torch_settings.get("cutoff_fn", "cosine")
    cfg.model.radial_basis_fn = torch_settings.get(
        "radial_basis_fn", "gaussian"
    )
    cfg.model.message_normalization = torch_settings.get(
        "message_normalization", "sqrt_num_features"
    )
    cfg.model.energy_learn_atomic_type_shifts = torch_settings.get(
        "energy_learn_atomic_type_shifts", False
    )
    cfg.model.energy_learn_atomic_type_scales = torch_settings.get(
        "energy_learn_atomic_type_scales", False
    )
    cfg.model.energy_regression_dim = torch_settings.get(
        "energy_regression_dim", None
    )
    cfg.model.layer_normalization_1 = torch_settings.get(
        "layer_normalization_1", False
    )
    cfg.model.layer_normalization_2 = torch_settings.get(
        "layer_normalization_2", False
    )
    cfg.model.residual_mlp_1 = torch_settings.get("residual_mlp_1", False)
    cfg.model.residual_mlp_2 = torch_settings.get("residual_mlp_2", False)
    cfg.model.use_charge_embed = torch_settings.get("use_charge_embed", False)
    cfg.model.use_spin_embed = torch_settings.get("use_spin_embed", False)
    cfg.model.zbl_repulsion_bool = torch_settings.get(
        "zbl_repulsion_bool", False
    )
    cfg.model.electrostatic_energy_bool = torch_settings.get(
        "electrostatic_energy_bool", False
    )
    cfg.model.electrostatic_energy_scale = torch_settings.get(
        "electrostatic_energy_scale", 1.0
    )
    cfg.model.dispersion_energy_bool = torch_settings.get(
        "dispersion_energy_bool", False
    )
    cfg.model.dispersion_energy_cutoff_lr_damping = torch_settings.get(
        "dispersion_energy_cutoff_lr_damping", None
    )
    cfg.model.dispersion_energy_scale = torch_settings.get(
        "dispersion_energy_scale", 1.0
    )
    cfg.model.num_features_head = torch_settings.get("num_features_head", None)
    cfg.model.qk_non_linearity = torch_settings.get(
        "qk_non_linearity", "identity"
    )
    cfg.model.activation_fn = torch_settings.get("activation_fn", None)
    cfg.model.layers_behave_like_identity_fn_at_init = torch_settings.get(
        "layers_behave_like_identity_fn_at_init", False
    )
    cfg.model.output_is_zero_at_init = torch_settings.get(
        "output_is_zero_at_init", False
    )
    cfg.model.input_convention = torch_settings.get(
        "input_convention", "positions"
    )
    cfg.model.energy_activation_fn = torch_settings.get(
        "energy_activation_fn", "silu"
    )
    cfg.neighborlist_format_lr = torch_settings.get(
        "neighborlist_format_lr", "sparse"
    )

    # Data settings
    cfg.data = config_dict.ConfigDict()
    cfg.data.avg_num_neighbors = torch_settings["avg_num_neighbors"]

    # Handle atomic type shifts if provided
    if torch_settings.get("atomic_type_shifts") is not None:
        cfg.data.energy_shifts = config_dict.ConfigDict(
            torch_settings["atomic_type_shifts"]
        )
    else:
        cfg.data.energy_shifts = config_dict.ConfigDict()

    return cfg


def convert_torch_to_flax_params(
    torch_params: Dict[str, Any],
    mapping: Dict[str, str],
    dtype: str = "float32",
):
    """Convert PyTorch parameters to Flax format"""
    import jax.numpy as jnp

    # Set JAX precision
    if dtype == "float64":
        jax.config.update("jax_enable_x64", True)
    else:
        jax.config.update("jax_enable_x64", False)

    flax_dtype = jnp.float64 if dtype == "float64" else jnp.float32

    # Define special parameter categories (same as in flax_to_torch)
    embeddings = [
        "inv_feature_embedding.embedding.weight",
        "charge_embedding.Wq.weight",
        "spin_embedding.Wq.weight",
    ]
    special_embeddings = [
        "charge_embedding.Wk",
        "charge_embedding.Wv",
        "spin_embedding.Wk",
        "spin_embedding.Wv",
        "partial_charges_output_block.atomic_embedding.weight",
        "hirshfeld_output_block.v_shift_embedding.weight",
        "hirshfeld_output_block.q_embedding.weight",
    ]

    flat_flax_params = {}

    for torch_key, flax_key in mapping.items():
        if torch_key not in torch_params:
            print(f"Warning: {torch_key} not found in torch_params")
            continue

        torch_tensor = torch_params[torch_key]

        # Convert to numpy
        if torch_tensor.is_cuda:
            torch_tensor = torch_tensor.cpu()
        numpy_array = torch_tensor.detach().numpy()

        # Handle special embedding cases - add padding dimension
        if torch_key in embeddings:
            # Add padding row at the beginning (index 0)
            pad_shape = list(numpy_array.shape)
            pad_shape[1] = 1  # Add one row
            padding = np.zeros(pad_shape, dtype=numpy_array.dtype)
            numpy_array = np.concatenate([padding, numpy_array], axis=1)

        # Handle atomic shifts/scales - add padding and reshape
        if (
            flax_key == "params/observables_0/energy_offset"
            or flax_key == "params/observables_0/atomic_scales"
        ):
            # Remove unsqueeze dimension and add padding at beginning
            if "energy_shifts" in torch_key:
                numpy_array = numpy_array.squeeze()  # Remove extra dimensions
            else:
                numpy_array = numpy_array.squeeze(
                    0
                )  # Remove the (1,) dimension
            padding = np.zeros((1,), dtype=numpy_array.dtype)
            numpy_array = np.concatenate([padding, numpy_array], axis=0)

        # Handle matrix transposes (reverse of flax_to_torch logic)
        if numpy_array.ndim == 2 and torch_key not in special_embeddings:
            numpy_array = numpy_array.T  # Transpose back
        elif numpy_array.ndim == 3:
            numpy_array = numpy_array.transpose(0, 2, 1)  # Reverse permutation

        # Convert to JAX array with correct dtype
        jax_array = jnp.array(numpy_array, dtype=flax_dtype)
        flat_flax_params[flax_key] = jax_array

    # Unflatten to nested structure
    flax_params = unflatten_params(flat_flax_params)

    return flax_params


def convert_torch_to_flax(
    torch_state_dict: Dict[str, Any],
    torch_settings: Dict[str, Any],
    trainable_rbf: bool = False,
    dtype: str = "float32",
):
    """
    Convert PyTorch model to Flax format

    Returns:
        tuple: (cfg, flax_params) where cfg is ConfigDict and flax_params is the parameter dict
    """
    # Convert torch settings to flax config
    cfg = get_model_settings_torch_to_flax(torch_settings)

    energy_shifts = torch_state_dict.get(
        "atomic_energy_output_block.energy_shifts", None
    )

    # turn energy_shifts into dict:
    energy_shifts_dict = {
        "0": 0.0,  # padding for atomic number 0
    }
    for i in range(1, 119):
        if energy_shifts is not None:
            energy_shifts_dict[str(i)] = energy_shifts[i - 1].item()
        else:
            energy_shifts_dict[str(i)] = 0.0
    cfg.data.energy_shifts = config_dict.ConfigDict(energy_shifts_dict)

    # Get the parameter mapping
    mapping = get_torch_to_flax_mapping(cfg, trainable_rbf)

    # Convert parameters
    flax_params = convert_torch_to_flax_params(
        torch_params=torch_state_dict, mapping=mapping, dtype=dtype
    )

    return cfg, flax_params
