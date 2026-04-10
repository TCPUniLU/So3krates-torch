"""
Elastic Weight Consolidation (EWC) for forgetting-aware fine-tuning.

Reference:
    "An efficient forgetting-aware fine-tuning framework for pretrained
    universal machine-learning interatomic potentials"

Usage:
    ewc = EWC(ewc_lambda=1e6)
    ewc.compute_fisher(model, loss_fn, fisher_loader, output_args,
                       device, num_samples=1000)
    # Inside training loop:
    loss = task_loss + ewc.penalty(model)

The reEWC variant from the paper is obtained by additionally enabling
data replay (replay_datasets / replay_fractions / replay_total in the
training config). No extra code is required here.
"""

import logging
from typing import Any, Dict, Optional

import torch
from torch.utils.data import DataLoader

from so3krates_torch.tools import torch_geometric


class EWC:
    """Diagonal Fisher Information EWC regulariser.

    Call :meth:`compute_fisher` once before training begins to snapshot
    the pretrained parameter values and estimate their importance via the
    diagonal of the Fisher Information Matrix (FIM).  During training,
    add :meth:`penalty` to the task loss at every gradient step.

    Parameters
    ----------
    ewc_lambda:
        Regularisation strength λ.  The paper reports λ = 10⁶ for a
        SevenNet model with eV/atom energy and eV/Å force losses.
        So3krates uses comparable loss scales with float64 by default,
        so this is a reasonable starting point, but should be validated
        empirically.
    """

    def __init__(self, ewc_lambda: float = 1e6) -> None:
        self.ewc_lambda = ewc_lambda
        # Populated by compute_fisher():
        self._fisher: Dict[str, torch.Tensor] = {}
        self._baseline: Dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    # Fisher estimation
    # ------------------------------------------------------------------

    def compute_fisher(
        self,
        model: torch.nn.Module,
        loss_fn: torch.nn.Module,
        data_loader: DataLoader,
        output_args: Dict[str, Any],
        device: torch.device,
        num_samples: int = 1000,
    ) -> None:
        """Compute the diagonal FIM and snapshot the baseline parameters.

        Parameters
        ----------
        model:
            The pretrained model *before* any fine-tuning weight updates.
            Must already be on ``device``.
        loss_fn:
            The same loss function used for training (so Fisher weights
            reflect energy + force contributions).
        data_loader:
            DataLoader over the pretraining dataset subset (same format
            as replay_datasets).  Should NOT be the fine-tuning set.
        output_args:
            Dict with keys ``"forces"``, ``"virials"``, ``"stress"``
            (booleans), same as used in the training loop.
        device:
            Compute device.
        num_samples:
            Maximum number of individual **structures** to use.  Batches
            are drawn from ``data_loader`` until this count is reached.
        """
        logging.info(
            f"EWC: computing Fisher information over up to "
            f"{num_samples} structures ..."
        )

        # 1. Snapshot baseline parameters (no forward pass needed).
        self._baseline = {
            name: param.detach().clone()
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        # 2. Initialise Fisher accumulators.
        self._fisher = {
            name: torch.zeros_like(param)
            for name, param in model.named_parameters()
            if param.requires_grad
        }

        was_training = model.training
        model.train()
        total_structures = 0
        num_batches = 0
        try:
            for batch in data_loader:
                if total_structures >= num_samples:
                    break

                batch = batch.to(device)
                n_structures = int(batch.ptr.numel() - 1)

                model.zero_grad()

                # Full forward pass WITH force computation.
                # torch.no_grad() must NOT be used here: forces are
                # computed as -dE/dr via autograd, which requires an
                # active graph.  loss.backward() then propagates
                # dL/dθ through energy and force terms
                # (forces_weight = 1000 dominates by default).
                batch_dict = batch.to_dict()
                output = model(
                    batch_dict,
                    training=True,
                    compute_force=output_args.get("forces", True),
                    compute_virials=output_args.get("virials", False),
                    compute_stress=output_args.get("stress", False),
                )
                # Pass ddp=False to skip any distributed all_reduce
                # during Fisher estimation — it is always a local,
                # per-rank computation and does not need global sync.
                try:
                    loss = loss_fn(pred=output, ref=batch, ddp=False)
                except TypeError:
                    loss = loss_fn(pred=output, ref=batch)
                loss.backward()

                # Accumulate squared gradients (un-normalised).
                for name, param in model.named_parameters():
                    if param.requires_grad and param.grad is not None:
                        self._fisher[name] += param.grad.data ** 2

                total_structures += n_structures
                num_batches += 1

            if total_structures == 0:
                raise RuntimeError(
                    "EWC: ewc_fisher_data produced no batches. "
                    "Check that the dataset path is correct "
                    "and non-empty."
                )

            # 3. Normalise by the number of batches processed.
            # The loss functions return a batch-mean (not a sum),
            # so each gradient is already normalised per-batch.
            # Dividing the accumulated squared gradients by
            # num_batches gives the correct estimate of
            # E[(∂L_batch/∂θ)²] regardless of batch size.
            for name in self._fisher:
                self._fisher[name] /= num_batches
        finally:
            # Always restore model state even if an exception
            # (OOM, NaN, bad data) aborts the loop mid-way,
            # so stale gradients cannot corrupt later training.
            model.zero_grad()
            model.train(was_training)
        logging.info(
            f"EWC: Fisher information computed over "
            f"{total_structures} structures ({num_batches} batches)."
        )

    # ------------------------------------------------------------------
    # Regularisation penalty
    # ------------------------------------------------------------------

    def penalty(self, model: torch.nn.Module) -> torch.Tensor:
        """Return the EWC regularisation term.

        .. math::
            \\mathcal{L}_{\\text{EWC}} =
            \\frac{\\lambda}{2}
            \\sum_i F_i \\left(\\theta_i - \\theta^*_i\\right)^2

        Only parameters that were trainable at Fisher-computation time
        (i.e. present in ``self._fisher``) are included.  Parameters
        that were frozen during setup_finetuning() are excluded.

        Parameters
        ----------
        model:
            The model being fine-tuned (weights are the current θ).

        Returns
        -------
        torch.Tensor
            Scalar regularisation loss, on the same device as the model.
        """
        if not self._fisher:
            raise RuntimeError(
                "EWC.penalty() called before compute_fisher(). "
                "Call compute_fisher() once before training."
            )

        penalty = torch.tensor(
            0.0,
            dtype=next(model.parameters()).dtype,
            device=next(model.parameters()).device,
        )
        # Strip the "module." prefix added by DistributedDataParallel so
        # that parameter names match those stored during compute_fisher(),
        # which was run on the unwrapped model.
        params = {
            name.removeprefix("module."): p
            for name, p in model.named_parameters()
        }
        for name, fisher_val in self._fisher.items():
            if name not in params:
                logging.warning(
                    f"EWC: parameter '{name}' present in Fisher "
                    f"dict but not found in model (after stripping "
                    f"'module.' prefix). EWC penalty for this "
                    f"parameter will be skipped — check that the "
                    f"model architecture has not changed since "
                    f"compute_fisher() was called."
                )
                continue
            delta = params[name] - self._baseline[name]
            penalty = penalty + (fisher_val * delta ** 2).sum()

        return (self.ewc_lambda / 2.0) * penalty
