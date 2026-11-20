# Taken from MACE package: https://github.com/ACEsuit/mace

###########################################################################################
# Checkpointing
# Authors: Gregor Simm
# This program is distributed under the MIT License (see MIT.md)
###########################################################################################

import dataclasses
import logging
import os
import re
from typing import Dict, List, Optional, Tuple

import torch

from .torch_tools import TensorDict

Checkpoint = Dict[str, TensorDict]


class LossStateMismatchError(ValueError):
    """The loss module's learned parameters (adaptive robust loss) do not
    match the checkpoint."""


@dataclasses.dataclass
class CheckpointState:
    model: torch.nn.Module
    optimizer: torch.optim.Optimizer
    lr_scheduler: torch.optim.lr_scheduler.ExponentialLR
    # Loss module; only its parameters (adaptive robust loss alpha/scale)
    # are checkpointed, never its weight buffers (those come from config).
    loss_fn: Optional[torch.nn.Module] = None
    # Two-stage loss weights: whether the checkpoint was taken in stage
    # two (None: unknown / no two-stage training). Filled on load.
    stage_two_active: Optional[bool] = None


def _loss_params(
    loss_fn: Optional[torch.nn.Module],
) -> Dict[str, torch.Tensor]:
    if loss_fn is None:
        return {}
    return {n: p for n, p in loss_fn.named_parameters()}


class CheckpointBuilder:
    @staticmethod
    def create_checkpoint(
        state: CheckpointState,
        config: Optional[dict] = None,
    ) -> Checkpoint:
        ckpt = {
            "model": state.model.state_dict(),
            "optimizer": state.optimizer.state_dict(),
            "lr_scheduler": state.lr_scheduler.state_dict(),
        }
        if state.stage_two_active is not None:
            ckpt["stage_two_active"] = bool(state.stage_two_active)
        loss_params = _loss_params(state.loss_fn)
        if loss_params:
            ckpt["loss_fn"] = {
                n: p.detach().clone() for n, p in loss_params.items()
            }
        if config is not None:
            ckpt["config"] = config
        return ckpt

    @staticmethod
    def load_checkpoint(
        state: CheckpointState, checkpoint: Checkpoint, strict: bool
    ) -> None:
        # Validate the loss state before restoring anything, and copy the
        # loss parameters only after model/optimizer loaded successfully.
        loss_params = _loss_params(state.loss_fn)
        saved = checkpoint.get("loss_fn")
        optimizer_state = checkpoint["optimizer"]
        if loss_params:
            if saved is None or set(saved) != set(loss_params):
                raise LossStateMismatchError(
                    "This run learns loss parameters "
                    "(use_adaptive_robust_loss) for "
                    f"{sorted(loss_params)}, but the checkpoint holds "
                    f"{sorted(saved) if saved else 'none'}. Restart without "
                    "the checkpoint (MISC.restart_latest: false) or with the "
                    "same use_adaptive_robust_loss, loss weights and "
                    "stage_two_loss_weights."
                )
        elif saved is not None:
            optimizer_state = _drop_loss_param_group(
                optimizer_state, state.optimizer
            )
            logging.warning(
                "Checkpoint was trained with the adaptive robust loss but "
                "this run does not use it: resuming without its learned "
                "alpha/scale (their optimizer state is dropped)."
            )
        state.model.load_state_dict(checkpoint["model"], strict=strict)  # type: ignore
        state.optimizer.load_state_dict(optimizer_state)
        if loss_params:
            with torch.no_grad():
                for name, param in loss_params.items():
                    param.copy_(saved[name])
        state.lr_scheduler.load_state_dict(checkpoint["lr_scheduler"])
        state.stage_two_active = checkpoint.get("stage_two_active")


def _drop_loss_param_group(optimizer_state: dict, optimizer) -> dict:
    """Remove the trailing loss-parameter group (adaptive robust loss)
    from a saved optimizer state so it fits an optimizer without it."""
    groups = optimizer_state["param_groups"]
    if len(groups) != len(optimizer.param_groups) + 1:
        raise LossStateMismatchError(
            f"Checkpoint optimizer has {len(groups)} parameter groups; "
            f"expected {len(optimizer.param_groups)} plus one adaptive "
            "robust loss group. Restart without the checkpoint "
            "(MISC.restart_latest: false)."
        )
    dropped = set(groups[-1]["params"])
    return {
        "state": {
            k: v
            for k, v in optimizer_state["state"].items()
            if k not in dropped
        },
        "param_groups": groups[:-1],
    }


@dataclasses.dataclass
class CheckpointPathInfo:
    path: str
    tag: str
    epochs: int
    swa: bool


class CheckpointIO:
    def __init__(
        self,
        directory: str,
        tag: str,
        keep: bool = False,
        swa_start: int = None,
    ) -> None:
        self.directory = directory
        self.tag = tag
        self.keep = keep
        self.old_path: Optional[str] = None
        self.swa_start = swa_start

        self._epochs_string = "_epoch-"
        self._filename_extension = "pt"

    def _get_checkpoint_filename(self, epochs: int, swa_start=None) -> str:
        if swa_start is not None and epochs >= swa_start:
            return (
                self.tag
                + self._epochs_string
                + str(epochs)
                + "_swa"
                + "."
                + self._filename_extension
            )
        return (
            self.tag
            + self._epochs_string
            + str(epochs)
            + "."
            + self._filename_extension
        )

    def _list_file_paths(self) -> List[str]:
        if not os.path.isdir(self.directory):
            return []
        all_paths = [
            os.path.join(self.directory, f) for f in os.listdir(self.directory)
        ]
        return [path for path in all_paths if os.path.isfile(path)]

    def _parse_checkpoint_path(
        self, path: str
    ) -> Optional[CheckpointPathInfo]:
        filename = os.path.basename(path)
        regex = re.compile(
            rf"^(?P<tag>.+){self._epochs_string}(?P<epochs>\d+)\.{self._filename_extension}$"
        )
        regex2 = re.compile(
            rf"^(?P<tag>.+){self._epochs_string}(?P<epochs>\d+)_swa\.{self._filename_extension}$"
        )
        match = regex.match(filename)
        match2 = regex2.match(filename)
        swa = False
        if not match:
            if not match2:
                return None
            match = match2
            swa = True

        return CheckpointPathInfo(
            path=path,
            tag=match.group("tag"),
            epochs=int(match.group("epochs")),
            swa=swa,
        )

    def _get_latest_checkpoint_path(self, swa) -> Optional[str]:
        all_file_paths = self._list_file_paths()
        checkpoint_info_list = [
            self._parse_checkpoint_path(path) for path in all_file_paths
        ]
        selected_checkpoint_info_list = [
            info
            for info in checkpoint_info_list
            if info and info.tag == self.tag
        ]

        if len(selected_checkpoint_info_list) == 0:
            logging.warning(
                f"Cannot find checkpoint with tag '{self.tag}' in '{self.directory}'"
            )
            return None

        selected_checkpoint_info_list_swa = []
        selected_checkpoint_info_list_no_swa = []

        for ckp in selected_checkpoint_info_list:
            if ckp.swa:
                selected_checkpoint_info_list_swa.append(ckp)
            else:
                selected_checkpoint_info_list_no_swa.append(ckp)
        if swa:
            try:
                latest_checkpoint_info = max(
                    selected_checkpoint_info_list_swa,
                    key=lambda info: info.epochs,
                )
            except ValueError:
                logging.warning(
                    "No SWA checkpoint found, while SWA is enabled. Compare the swa_start parameter and the latest checkpoint."
                )
        else:
            latest_checkpoint_info = max(
                selected_checkpoint_info_list_no_swa,
                key=lambda info: info.epochs,
            )
        return latest_checkpoint_info.path

    def save(
        self, checkpoint: Checkpoint, epochs: int, keep_last: bool = False
    ) -> None:
        if not self.keep and self.old_path and not keep_last:
            logging.debug(f"Deleting old checkpoint file: {self.old_path}")
            os.remove(self.old_path)

        filename = self._get_checkpoint_filename(epochs, self.swa_start)
        path = os.path.join(self.directory, filename)
        logging.debug(f"Saving checkpoint: {path}")
        os.makedirs(self.directory, exist_ok=True)
        torch.save(obj=checkpoint, f=path)
        self.old_path = path

    def load_latest(
        self,
        swa: Optional[bool] = False,
        device: Optional[torch.device] = None,
    ) -> Optional[Tuple[Checkpoint, int]]:
        path = self._get_latest_checkpoint_path(swa=swa)
        if path is None:
            return None

        return self.load(path, device=device)

    def load(
        self, path: str, device: Optional[torch.device] = None
    ) -> Tuple[Checkpoint, int]:
        checkpoint_info = self._parse_checkpoint_path(path)

        if checkpoint_info is None:
            raise RuntimeError(f"Cannot find path '{path}'")

        logging.info(f"Loading checkpoint: {checkpoint_info.path}")
        return (
            torch.load(f=checkpoint_info.path, map_location=device),
            checkpoint_info.epochs,
        )


class CheckpointHandler:
    def __init__(self, *args, **kwargs) -> None:
        self.io = CheckpointIO(*args, **kwargs)
        self.builder = CheckpointBuilder()

    def save(
        self,
        state: CheckpointState,
        epochs: int,
        keep_last: bool = False,
        config: Optional[dict] = None,
    ) -> None:
        checkpoint = self.builder.create_checkpoint(state, config=config)
        self.io.save(checkpoint, epochs, keep_last)

    def load_latest(
        self,
        state: CheckpointState,
        swa: Optional[bool] = False,
        device: Optional[torch.device] = None,
        strict=False,
    ) -> Optional[int]:
        result = self.io.load_latest(swa=swa, device=device)
        if result is None:
            return None

        checkpoint, epochs = result
        self.builder.load_checkpoint(
            state=state, checkpoint=checkpoint, strict=strict
        )
        return epochs

    def load(
        self,
        state: CheckpointState,
        path: str,
        strict=False,
        device: Optional[torch.device] = None,
    ) -> int:
        checkpoint, epochs = self.io.load(path, device=device)
        self.builder.load_checkpoint(
            state=state, checkpoint=checkpoint, strict=strict
        )
        return epochs
