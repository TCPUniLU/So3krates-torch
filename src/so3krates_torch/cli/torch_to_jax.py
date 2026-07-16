from so3krates_torch.config import Torch2JaxArgs
from so3krates_torch.tools.jax_torch_conversion import convert_torch_to_flax
import argparse
import yaml
import torch
import pickle
import json
from pathlib import Path


def main():
    argparser = argparse.ArgumentParser(
        description="Convert between Flax and PyTorch model formats"
    )
    argparser.add_argument(
        "--path_to_state_dict",
        type=str,
        required=True,
        help="Path to the model state dictionary file",
    )
    argparser.add_argument(
        "--path_to_hyperparams",
        type=str,
        required=True,
        help="Path to the model hyperparameters file",
    )
    argparser.add_argument(
        "--save_settings_path",
        type=str,
        default=None,
        help="Path to the directory (!) where the model settings will be saved."
        " In the JAX version the name of the hyperparameter file is hardcoded to"
        "'hyperparameters.json'. Thats why we need the directory not file name.",
    )
    argparser.add_argument(
        "--save_params_path",
        type=str,
        default=None,
        help="Path to the file where the params dictionary will be saved."
        " In the JAX version the name of the params file is hardcoded to"
        "'params.pkl'. Thats why we need the directory not file name.",
    )
    argparser.add_argument(
        "--so3lr",
        type=bool,
        default=True,
        help="Flag to indicate if the model is SO3LR",
    )
    argparser.add_argument(
        "--use_defined_shifts",
        action="store_true",
        help="Flag to indicate if defined shifts should be used",
    )
    argparser.add_argument(
        "--trainable_rbf",
        action="store_true",
        help="Flag to indicate if the RBF is trainable",
    )
    argparser.add_argument(
        "--dtype",
        default="float32",
        help="Data type to use for the model parameters",
    )
    argparser.add_argument(
        "--check_parity",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Verify JAX/Torch model outputs agree after conversion",
    )
    argparser.add_argument(
        "--parity_structure",
        type=str,
        default=None,
        help="Structure file to check parity on (default: bundled example)",
    )
    argparser.add_argument(
        "--parity_atol",
        type=float,
        default=1e-3,
        help="Absolute tolerance for the parity check",
    )
    argparser.add_argument(
        "--parity_rtol",
        type=float,
        default=1e-2,
        help="Relative tolerance for the parity check",
    )

    args = argparser.parse_args()
    validated = Torch2JaxArgs.model_validate(vars(args))

    path_to_settings_dir = Path(args.save_settings_path)
    path_to_params_dir = Path(args.save_params_path)
    # check if the directories exist, if not create them
    if args.save_settings_path:
        path_to_settings_dir.mkdir(parents=True, exist_ok=True)
    if args.save_params_path:
        path_to_params_dir.mkdir(parents=True, exist_ok=True)

    with open(args.path_to_hyperparams, "r") as f:
        torch_settings = yaml.safe_load(f)
    state_dict = torch.load(
        args.path_to_state_dict,
        weights_only=True,
        map_location=torch.device("cpu"),
    )
    cfg, params = convert_torch_to_flax(
        torch_state_dict=state_dict,
        torch_settings=torch_settings["ARCHITECTURE"],
        trainable_rbf=args.trainable_rbf,
        dtype=args.dtype,
    )

    cfg_dict = cfg.to_dict()
    if args.save_settings_path:
        with open(path_to_settings_dir / "hyperparameters.json", "w") as f:
            json.dump(cfg_dict, f)

    if args.save_params_path:
        with open(path_to_params_dir / "params.pkl", "wb") as f:
            pickle.dump(params, f)

    if args.check_parity:
        from so3krates_torch.modules.models import SO3LR, So3krates
        from so3krates_torch.tools.model_parity import check_model_parity

        arch_settings = dict(torch_settings["ARCHITECTURE"])
        # `arch_settings["dtype"]`, if present, may be a string like
        # "torch.float64" (as written by `convert_flax_to_torch`'s own
        # `save_torch_settings` option -- see jax_torch_conversion.py's
        # `serializable_settings["dtype"] = str(dtype)`), which
        # `So3krates.__init__`'s `getattr(torch, dtype)` string-handling
        # cannot parse (it expects a bare "float64", no "torch." prefix).
        # Override with the CLI's own already-validated --dtype instead
        # of trying to parse whatever string format happens to be in the
        # file.
        arch_settings["dtype"] = getattr(torch, args.dtype, torch.float32)

        check_model = (SO3LR if args.so3lr else So3krates)(**arch_settings)
        check_model.load_state_dict(state_dict)

        parity_ok = check_model_parity(
            cfg=cfg,
            flax_params=params,
            torch_model=check_model,
            r_max=arch_settings["r_max"],
            r_max_lr=arch_settings.get("r_max_lr"),
            structure_path=args.parity_structure,
            atol=args.parity_atol,
            rtol=args.parity_rtol,
        )
        if not parity_ok:
            print(
                "WARNING: JAX<->Torch parity check FAILED -- the converted "
                "params were still saved, but their outputs do not match "
                "the source model within tolerance. See the table above."
            )
            return 1

    return 0


if __name__ == "__main__":
    main()
