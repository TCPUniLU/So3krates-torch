"""
CLI for converting between single-head and multi-head SO3LR models.

``--to-multihead N`` keeps the model's backbone and settings and adds N
freshly initialised, independent energy output heads (e.g. to fine-tune
an ensemble from a pretrained model). ``--select-head K`` turns a
multi-head model into a plain single-head model that keeps head K.

Usage::

    torchkrates-convert-heads --model v1 --output mh.model --to-multihead 4
    torchkrates-convert-heads --model tauri-m-2606 --output mh.model \\
        --to-multihead 4  # keeps theory level 5
    torchkrates-convert-heads --model ft.model --output head2.model \\
        --select-head 2
"""

import argparse
import os
from typing import List, Optional

import torch
from pydantic import ValidationError

from so3krates_torch.tools.eval import BUNDLED_LR_DAMPING, BUNDLED_R_MAX_LR
from so3krates_torch.calculator.so3 import (
    _PRETRAINED_SO3LR_MODELS,
    load_pretrained_so3lr,
)
from so3krates_torch.config import ConvertHeadsArgs
from so3krates_torch.modules.models import MultiHeadSO3LR
from so3krates_torch.tools.multihead_utils import (
    to_multihead_model,
    to_single_head_model,
)


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Convert an SO3LR model to a multi-head model with fresh "
            "energy heads, or keep one head of a multi-head model."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--model",
        required=True,
        help="Pickled model file (torch.save(model)) or a bundled model "
        f"name: {', '.join(sorted(_PRETRAINED_SO3LR_MODELS))}.",
    )
    parser.add_argument(
        "--output", required=True, help="Output .model file path."
    )
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument(
        "--to-multihead",
        type=int,
        metavar="N",
        help="Make a multi-head model with N freshly initialised heads.",
    )
    mode.add_argument(
        "--select-head",
        type=int,
        metavar="K",
        help="Keep head K (0-based) of a multi-head model.",
    )
    parser.add_argument(
        "--theory-level",
        type=int,
        metavar="T",
        default=None,
        help="Theory level to keep for multi-theory-level models such as "
        "tauri-* (only with --to-multihead). Unset means level 5, "
        "so3lr_dev's level.",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Seed for new head weights."
    )
    parser.add_argument("--device", default="cpu", help="Torch device.")
    parser.add_argument(
        "--force",
        action="store_true",
        help="Overwrite the output file if it exists.",
    )
    return parser


def _load_model(source: str, device: str) -> torch.nn.Module:
    if source in _PRETRAINED_SO3LR_MODELS:
        # As SO3LRCalculator: the checkpoints save no (v1) or a very
        # large (tauri: 1000 A) long-range cutoff.
        return load_pretrained_so3lr(
            source,
            device=device,
            r_max_lr=BUNDLED_R_MAX_LR,
            dispersion_energy_cutoff_lr_damping=BUNDLED_LR_DAMPING,
        )
    if not os.path.isfile(source):
        raise ValueError(
            f"Model file not found: {source} (and not a bundled model "
            f"name: {', '.join(sorted(_PRETRAINED_SO3LR_MODELS))})."
        )
    loaded = torch.load(source, map_location=device, weights_only=False)
    if not isinstance(loaded, torch.nn.Module):
        raise ValueError(
            f"{source} is not a pickled model (torch.save(model)); it "
            f"holds a {type(loaded).__name__}, e.g. a state_dict. Load "
            "it into a model first."
        )
    return loaded


def main(argv: Optional[List[str]] = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)
    try:
        cfg = ConvertHeadsArgs.model_validate(vars(args))
    except ValidationError as e:
        parser.error(
            "; ".join(
                err["msg"].removeprefix("Value error, ") for err in e.errors()
            )
        )

    out = os.path.abspath(cfg.output)
    if os.path.isfile(cfg.model) and os.path.abspath(cfg.model) == out:
        parser.error("--output must differ from --model.")
    if os.path.exists(out) and not cfg.force:
        parser.error(f"{cfg.output} exists; pass --force to overwrite.")

    try:
        model = _load_model(cfg.model, cfg.device)
        if cfg.to_multihead is not None:
            model = to_multihead_model(
                model,
                cfg.to_multihead,
                seed=cfg.seed,
                theory_level=cfg.theory_level,
            )
        else:
            model = to_single_head_model(model, cfg.select_head)
    except ValueError as e:
        parser.error(str(e))

    torch.save(model, out)
    heads = model.num_output_heads if isinstance(model, MultiHeadSO3LR) else 1
    dtype = next(model.parameters()).dtype
    print(
        f"Saved {type(model).__name__} ({heads} energy head(s), "
        f"r_max_lr={model.r_max_lr}, {dtype}) to {cfg.output}"
    )


if __name__ == "__main__":
    main()
