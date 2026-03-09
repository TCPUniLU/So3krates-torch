from so3krates_torch.config import Jax2TorchArgs
from so3krates_torch.tools.jax_torch_conversion import convert_flax_to_torch
import argparse
import torch


def main():

    argparser = argparse.ArgumentParser(
        description="Convert between Flax and PyTorch model formats"
    )
    argparser.add_argument(
        "--path_to_params",
        type=str,
        required=True,
        help="Path to the model parameters file",
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
        help="Path to the file where the model settings will be saved",
    )
    argparser.add_argument(
        "--save_state_dict_path",
        type=str,
        default=None,
        help="Path to the file where the state dictionary will be saved",
    )
    argparser.add_argument(
        "--save_model_path",
        type=str,
        default=None,
        help="Path to the file where the model will be saved",
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
        "--device",
        type=str,
        default="cpu",
        help="Device to use for the model (e.g., 'cpu' or 'cuda')",
    )

    args = argparser.parse_args()
    validated = Jax2TorchArgs.model_validate(vars(args))

    dtype = getattr(torch, validated.dtype, torch.float32)
    model = convert_flax_to_torch(
        path_to_flax_hyperparams=args.path_to_hyperparams,
        path_to_flax_params=args.path_to_params,
        dtype=dtype,
        so3lr=args.so3lr,
        use_defined_shifts=args.use_defined_shifts,
        trainable_rbf=args.trainable_rbf,
        save_torch_settings=args.save_settings_path,
        device=args.device,
    )

    if args.save_state_dict_path is not None:
        torch.save(model.state_dict(), args.save_state_dict_path)

    if args.save_model_path is not None:
        torch.save(model, args.save_model_path)

    return 0


if __name__ == "__main__":
    main()
