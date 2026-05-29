###########################################################################################

# Taken from MACE package: https://github.com/ACEsuit/mace

# Added long-range cutoff to the neighborhood function
###########################################################################################

from typing import Optional, Tuple

import numpy as np
from matscipy.neighbours import neighbour_list


def get_neighborhood(
    positions: np.ndarray,  # [num_positions, 3]
    cutoff: float,
    cutoff_lr: Optional[float] = None,
    pbc: Optional[Tuple[bool, bool, bool]] = None,
    cell: Optional[np.ndarray] = None,  # [3, 3]
    true_self_interaction=False,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pbc is None:
        pbc = (False, False, False)

    if cell is None or cell.any() == np.zeros((3, 3)).any():
        cell = np.identity(3, dtype=float)

    assert len(pbc) == 3 and all(isinstance(i, (bool, np.bool_)) for i in pbc)
    assert cell.shape == (3, 3)
    pbc_x = pbc[0]
    pbc_y = pbc[1]
    pbc_z = pbc[2]
    identity = np.identity(3, dtype=float)
    max_positions = np.max(np.absolute(positions)) + 1

    if cutoff_lr is not None:
        # When LR cutoff is provided, compute a single neighbor list at
        # cutoff_lr, then extract the SR subset via distance masking.
        # This avoids calling neighbour_list() twice since SR edges are
        # always a subset of LR edges.
        if not pbc_x:
            cell[0, :] = max_positions * 2 * cutoff_lr * identity[0, :]
        if not pbc_y:
            cell[1, :] = max_positions * 2 * cutoff_lr * identity[1, :]
        if not pbc_z:
            cell[2, :] = max_positions * 2 * cutoff_lr * identity[2, :]

        # !!! In MACE they don't follow the convention of naming
        # the senders as j and receivers as i. Because molecules are
        # undirected graphs in the context of GNNs for MLFFs it
        # doesn't matter.
        sender_lr, receiver_lr, unit_shifts_lr = neighbour_list(
            quantities="ijS",
            pbc=pbc,
            cell=cell,
            positions=positions,
            cutoff=cutoff_lr,
        )
        if not true_self_interaction:
            true_self_edge_lr = sender_lr == receiver_lr
            true_self_edge_lr &= np.all(unit_shifts_lr == 0, axis=1)
            keep_edge_lr = ~true_self_edge_lr
            sender_lr = sender_lr[keep_edge_lr]
            receiver_lr = receiver_lr[keep_edge_lr]
            unit_shifts_lr = unit_shifts_lr[keep_edge_lr]

        edge_index_lr = np.stack((sender_lr, receiver_lr))
        # From the docs: D = positions[j] - positions[i] + S.dot(cell)
        shifts_lr = np.dot(unit_shifts_lr, cell)  # [n_edges_lr, 3]

        # Extract SR subset by masking on squared distance (avoids sqrt)
        vectors = positions[receiver_lr] - positions[sender_lr] + shifts_lr
        dist_sq = np.sum(vectors**2, axis=1)
        sr_mask = dist_sq <= cutoff**2

        edge_index = np.stack((sender_lr[sr_mask], receiver_lr[sr_mask]))
        shifts = shifts_lr[sr_mask]
        unit_shifts = unit_shifts_lr[sr_mask]
    else:
        # No LR cutoff: single SR neighbor list call (unchanged path)
        # Extend cell in non-periodic directions
        # For models with more than 5 layers, the multiplicative
        # constant needs to be increased.
        if not pbc_x:
            cell[0, :] = max_positions * 5 * cutoff * identity[0, :]
        if not pbc_y:
            cell[1, :] = max_positions * 5 * cutoff * identity[1, :]
        if not pbc_z:
            cell[2, :] = max_positions * 5 * cutoff * identity[2, :]

        # !!! In MACE they don't follow the convention of naming
        # the senders as j and receivers as i. Because molecules are
        # undirected graphs in the context of GNNs for MLFFs it
        # doesn't matter.
        sender, receiver, unit_shifts = neighbour_list(
            quantities="ijS",
            pbc=pbc,
            cell=cell,
            positions=positions,
            cutoff=cutoff,
        )
        if not true_self_interaction:
            # Eliminate self-edges that don't cross periodic boundaries
            true_self_edge = sender == receiver
            true_self_edge &= np.all(unit_shifts == 0, axis=1)
            keep_edge = ~true_self_edge

            # Note: after eliminating self-edges, it can be that no
            # edges remain in this system
            sender = sender[keep_edge]
            receiver = receiver[keep_edge]
            unit_shifts = unit_shifts[keep_edge]

        # Build output
        edge_index = np.stack((sender, receiver))  # [2, n_edges]
        # From the docs: D = positions[j] - positions[i] + S.dot(cell)
        shifts = np.dot(unit_shifts, cell)  # [n_edges, 3]
        edge_index_lr = shifts_lr = unit_shifts_lr = None

    return (
        edge_index,
        shifts,
        unit_shifts,
        cell,
        edge_index_lr,
        shifts_lr,
        unit_shifts_lr,
    )
