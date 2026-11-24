"""Row gathers whose backward pass is fast on GPUs."""

import torch


def gather_rows(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    """Return ``x[idx]`` for a 1-D integer ``idx`` (gather along dim 0).

    The values are identical to advanced indexing, but the backward pass is
    different: ``x[idx]`` backpropagates through ``index_put_`` with
    ``accumulate=True``, which on CUDA (``indexing_backward_kernel``) is
    extremely slow when every row is hit by hundreds or thousands of indices.
    That is the case for the long-range edge lists (about 8 million edges onto
    9650 atoms); there this kernel took 95 % of the GPU time of one force +
    stress evaluation.  The backward of ``index_select`` is ``index_add_``,
    which is far faster for such duplicate-heavy index tensors.

    Use this for gathers over edge lists whose result is differentiated; plain
    indexing is fine for integer-only gathers (no backward).
    """
    return x.index_select(0, idx)
