"""Long-range anchoring for fine-tuning (as ``training.
long_range_anchor_weights`` in so3lr 0.2.0).

A frozen copy of the starting model predicts the partial charges, a0
(Hirshfeld) ratios, C6 ratios and dipole of every training and validation
batch; the model being fine-tuned is penalised for drifting away from
them. This keeps its long-range behaviour when the fine-tuning data has
only energies and forces, which cannot tell short- from long-range
energy apart.
"""

import logging
from typing import Callable, Dict, Mapping, Optional

import torch

# Config key (so3lr/JAX names and torch aliases) -> internal name, i.e.
# the property name of the loss term in ``modules/loss.py``.
ANCHOR_KEY_ALIASES: Dict[str, str] = {
    "partial_charges": "charges",
    "charges": "charges",
    "a0_ratios": "hirshfeld_ratios",
    "hirshfeld_ratios": "hirshfeld_ratios",
    "c6_ratios": "c6_ratios",
    "dipole_vec": "dipole",
    "dipole": "dipole",
}
ANCHOR_QUANTITIES = ("charges", "hirshfeld_ratios", "c6_ratios", "dipole")


def normalize_anchor_weights(
    raw: Optional[Mapping[str, float]],
) -> Dict[str, float]:
    """Anchor weights by internal name, keeping only weights > 0, in the
    order of ``ANCHOR_QUANTITIES``.

    Accepts the so3lr keys (``partial_charges``, ``a0_ratios`` /
    ``hirshfeld_ratios``, ``c6_ratios``, ``dipole_vec``) and the torch
    aliases ``charges`` and ``dipole``.
    """
    if not raw:
        return {}
    unknown = sorted(set(raw) - set(ANCHOR_KEY_ALIASES))
    if unknown:
        raise ValueError(
            f"Unknown long_range_anchor_weights keys {unknown}; choose from "
            f"{sorted(ANCHOR_KEY_ALIASES)}."
        )
    by_name: Dict[str, float] = {}
    given: Dict[str, str] = {}
    for key, weight in raw.items():
        name = ANCHOR_KEY_ALIASES[key]
        if name in given:
            raise ValueError(
                f"long_range_anchor_weights has '{given[name]}' and "
                f"'{key}', which are the same quantity; give only one."
            )
        given[name] = key
        weight = float(weight)
        if weight < 0:
            raise ValueError(
                f"long_range_anchor_weights['{key}'] is negative "
                f"({weight}); weights must be >= 0."
            )
        by_name[name] = weight
    return {
        name: by_name[name]
        for name in ANCHOR_QUANTITIES
        if by_name.get(name, 0.0) > 0
    }


def make_anchor_teacher(model: torch.nn.Module, dtype: str) -> torch.nn.Module:
    """Frozen copy of ``model`` (eval mode, no gradients) in ``dtype``
    (``"float32"`` / ``"float64"``); ``model`` is not modified."""
    from so3krates_torch.tools.model_setup import set_dtype_model
    from so3krates_torch.tools.multihead_utils import _deepcopy_model

    teacher = _deepcopy_model(model)
    set_dtype_model(teacher, dtype)
    teacher.eval()
    for param in teacher.parameters():
        param.requires_grad_(False)
    return teacher


class _AnchorView:
    """Read-only view of a batch with some keys replaced; everything else
    is read from the real batch, which is never modified."""

    def __init__(self, ref, overrides: Dict[str, torch.Tensor]) -> None:
        self._ref = ref
        self._overrides = overrides

    def __getitem__(self, key):
        if key in self._overrides:
            return self._overrides[key]
        return self._ref[key]

    def __getattr__(self, name):
        overrides = self.__dict__["_overrides"]
        if name in overrides:
            return overrides[name]
        return getattr(self.__dict__["_ref"], name)


class AnchoredLoss(torch.nn.Module):
    """``base`` loss plus one anchor term per anchored quantity.

    The anchor term for a quantity is its usual loss term (same reduction
    and penalty as ``base``), with the frozen teacher's prediction as the
    label and every structure's per-property label weight taken as 1. The
    teacher is kept out of module registration, so it never shows up in
    ``parameters()``, ``state_dict()`` or ``modules()`` (and hence not in
    the optimizer or checkpoints). With the adaptive robust loss, the
    anchor terms use their own learned alpha/scale (``anchor_<name>``).
    """

    def __init__(
        self,
        base: torch.nn.Module,
        teacher: torch.nn.Module,
        weights: Dict[str, float],
        adaptive: Optional["AdaptiveRobustParams"] = None,
        warned: Optional[set] = None,
        label_weights: Optional[Dict[str, Dict[str, float]]] = None,
    ) -> None:
        super().__init__()
        from so3krates_torch.modules.loss import AdaptiveRobustParams

        unknown = sorted(set(weights) - set(ANCHOR_QUANTITIES))
        if unknown:
            raise ValueError(
                f"Unknown anchored quantities {unknown}; use internal names "
                f"{list(ANCHOR_QUANTITIES)} (see normalize_anchor_weights)."
            )
        self.base = base
        object.__setattr__(self, "_teacher", teacher)
        self.anchor_weights = {
            name: float(w) for name, w in weights.items() if w > 0
        }
        if base.robust == "adaptive" and adaptive is None:
            adaptive = AdaptiveRobustParams(
                [f"anchor_{name}" for name in self.anchor_weights], 1.0, 1.0
            )
        self.anchor_adaptive = adaptive if base.robust == "adaptive" else None
        self.last_anchor_targets: Dict[str, torch.Tensor] = {}
        # Quantities already warned about; shared between the losses of
        # one run so each warns once.
        self._warned = warned if warned is not None else set()
        # Loss weight of each anchored quantity's label term per training
        # stage ({"": w} for single-stage), for the hard-label warning.
        self._label_weights = label_weights or {
            name: {"": float(getattr(base, f"{name}_weight"))}
            for name in self.anchor_weights
        }

    # Attributes the training code reads from the loss.
    @property
    def adaptive(self):
        return self.base.adaptive

    @property
    def properties(self):
        return self.base.properties

    @property
    def reduction(self):
        return self.base.reduction

    @property
    def robust(self):
        return self.base.robust

    def _anchor_penalty(self) -> Callable:
        if self.anchor_adaptive is not None:
            adaptive = self.anchor_adaptive
            return lambda x, prop: adaptive.penalty(x, f"anchor_{prop}")
        return self.base.penalty

    def _teacher_targets(self, ref) -> Dict[str, torch.Tensor]:
        with torch.no_grad():
            out = self._teacher(
                ref.to_dict(), training=False, compute_force=False
            )
        targets = {}
        for name in self.anchor_weights:
            value = out.get(_TEACHER_OUTPUT_KEY[name])
            if value is None:
                raise ValueError(
                    f"Cannot anchor '{name}': the pretrained model does not "
                    f"predict '{_TEACHER_OUTPUT_KEY[name]}' (it predicts "
                    f"{sorted(k for k, v in out.items() if v is not None)})."
                )
            targets[name] = value.detach()
        return targets

    def check_teacher(self, batch) -> None:
        """Raise ValueError if the teacher cannot predict an anchored
        quantity."""
        self._teacher_targets(batch)

    def check_outputs(self, pred) -> None:
        from so3krates_torch.modules.loss import _PROPERTY_LOSS_TERMS

        self.base.check_outputs(pred)
        for name in self.anchor_weights:
            _term, pred_key, flag = _PROPERTY_LOSS_TERMS[name]
            if pred.get(pred_key) is None:
                hint = f" (enable ARCHITECTURE.{flag})" if flag else ""
                raise KeyError(
                    f"long_range_anchor_weights anchors '{name}' but the "
                    f"model output has no '{pred_key}'{hint}."
                )

    def _warn_if_hard_labels(self, ref) -> None:
        """Warn once per quantity when a batch has real labels for an
        anchored quantity. Absent labels have per-structure weight 0 in
        every data format (ratios are also NaN), so a structure has a
        label iff ``weight * <name>_weight != 0`` (and, for the ratios,
        the label is finite)."""
        from so3krates_torch.modules.loss import ratio_label_mask

        for name in self.anchor_weights:
            if name in self._warned:
                continue
            with torch.no_grad():
                if name in ("hirshfeld_ratios", "c6_ratios"):
                    labelled = ratio_label_mask(ref, name, f"{name}_weight")
                elif name == "dipole":
                    labelled = ref.weight.unsqueeze(-1) * ref.dipole_weight
                    labelled = labelled != 0
                else:
                    labelled = ref.weight * ref.charges_weight != 0
                if not bool(labelled.any()):
                    continue
            self._warned.add(name)
            logging.warning(self._hard_label_message(name))

    def _hard_label_message(self, name: str) -> str:
        weights = self._label_weights.get(name, {})
        trained = {stage: w for stage, w in weights.items() if w > 0}
        intro = (
            f"The data has hard labels for '{name}' while "
            "long_range_anchor_weights also anchors it to the pretrained "
            "model. "
        )
        if not trained:
            return intro + (
                "The labels are ignored (loss weight 0); only the anchor "
                "term is trained."
            )
        desc = ", ".join(
            f"{w:g} in {stage}" if stage else f"{w:g}"
            for stage, w in trained.items()
        )
        return intro + (
            f"Both the label term (loss weight {desc}) and the anchor term "
            f"are trained. To use only one, remove '{_CONFIG_KEY[name]}' "
            "from long_range_anchor_weights or set the label's loss weight "
            "to 0."
        )

    def forward(self, ref, pred, ddp: Optional[bool] = None) -> torch.Tensor:
        from so3krates_torch.modules.loss import _PROPERTY_LOSS_TERMS

        self._warn_if_hard_labels(ref)
        total = self.base(ref=ref, pred=pred, ddp=ddp)
        targets = self._teacher_targets(ref)
        self.last_anchor_targets = targets
        overrides = {}
        for name, value in targets.items():
            overrides[name] = value
            overrides[f"{name}_weight"] = torch.ones_like(
                ref[f"{name}_weight"]
            )
        view = _AnchorView(ref, overrides)
        penalty = self._anchor_penalty()
        for name, weight in self.anchor_weights.items():
            term = _PROPERTY_LOSS_TERMS[name][0]
            total = total + weight * term(
                view, pred, ddp, reduction=self.reduction, penalty=penalty
            )
        return total

    def __repr__(self):
        anchors = ", ".join(
            f"{name}={w:.3f}" for name, w in self.anchor_weights.items()
        )
        return f"AnchoredLoss({self.base!r}, anchors: {anchors})"


# so3lr config key per anchored quantity (for messages).
_CONFIG_KEY = {
    "charges": "partial_charges",
    "hirshfeld_ratios": "a0_ratios",
    "c6_ratios": "c6_ratios",
    "dipole": "dipole_vec",
}

# Teacher output key per anchored quantity.
_TEACHER_OUTPUT_KEY = {
    "charges": "partial_charges",
    "hirshfeld_ratios": "hirshfeld_ratios",
    "c6_ratios": "c6_ratios",
    "dipole": "dipole",
}
