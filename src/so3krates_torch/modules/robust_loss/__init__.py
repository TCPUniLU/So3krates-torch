"""Robust loss (Barron, "A General and Adaptive Robust Loss Function").

PyTorch port of the JAX adaptation in SO3LR, itself a modified copy of
the Google Research implementation; Apache-2.0, see README.md here.
"""

from .distribution import Distribution
from .general import general_loss

__all__ = ["Distribution", "general_loss"]
