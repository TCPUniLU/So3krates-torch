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
        dtype=args.dtype,
    )

    cfg_dict = cfg.to_dict()
    if args.save_settings_path:
        with open(path_to_settings_dir / "hyperparameters.json", "w") as f:
            json.dump(cfg_dict, f)

    if args.save_params_path:
        with open(path_to_params_dir / "params.pkl", "wb") as f:
            pickle.dump(params, f)

    return 0


if __name__ == "__main__":
    main()
