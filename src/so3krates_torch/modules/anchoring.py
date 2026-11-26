"""Long-range anchoring for fine-tuning (as ``training.
long_range_anchor_weights`` in so3lr 0.2.0).

A frozen copy of the starting model predicts the partial charges, a0
(Hirshfeld) ratios, C6 ratios and dipole of every training and validation
batch; the model being fine-tuned is penalised for drifting away from
them. This keeps its long-range behaviour when the fine-tuning data has
only energies and forces, which cannot tell short- from long-range
energy apart.
"""

from typing import Dict, Mapping, Optional

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
