"""Tests for the neighbor-list builder in data/neighborhood.py.

Wrong neighbor lists mean the model sees wrong local environments,
producing completely wrong predictions. Tests cover basic detection,
cutoff enforcement, PBC edges, long-range superset, and real molecules.
"""

import numpy as np
import pytest

from so3krates_torch.data.neighborhood import get_neighborhood


# ---------------------------------------------------------------------------
# Shared test data
# ---------------------------------------------------------------------------

_H2O_POSITIONS = np.array(
    [
        [0.0, 0.0, 0.0],  # O
        [0.96, 0.0, 0.0],  # H1
        [-0.24, 0.93, 0.0],  # H2
    ]
)


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_two_atoms_within_cutoff():
    """Two atoms closer than the cutoff produce two directed edges."""
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    edge_index, *_ = get_neighborhood(
        positions=positions,
        cutoff=3.0,
        pbc=(False, False, False),
    )

    assert edge_index.shape == (2, 2)
    edges = set(map(tuple, edge_index.T))
    assert edges == {(0, 1), (1, 0)}


def test_two_atoms_outside_cutoff():
    """Two atoms beyond the cutoff produce no edges."""
    positions = np.array([[0.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
    edge_index, *_ = get_neighborhood(
        positions=positions,
        cutoff=3.0,
        pbc=(False, False, False),
    )

    assert edge_index.shape[1] == 0


def test_no_self_edges():
    """A single atom in a non-periodic system has no edges."""
    positions = np.array([[0.0, 0.0, 0.0]])
    edge_index, *_ = get_neighborhood(
        positions=positions,
        cutoff=5.0,
        pbc=(False, False, False),
    )

    assert edge_index.shape[1] == 0


def test_pbc_creates_mirror_edges():
    """PBC connects atoms across the cell boundary; non-PBC does not."""
    # Atoms near opposite faces of a 5 Å box: non-PBC distance = 4.8 Å,
    # minimum-image distance = 0.2 Å.
    cell = np.eye(3) * 5.0
    positions = np.array([[0.1, 0.0, 0.0], [4.9, 0.0, 0.0]])
    cutoff = 3.0

    edge_nopbc, *_ = get_neighborhood(
        positions=positions,
        cutoff=cutoff,
        pbc=(False, False, False),
    )
    edge_pbc, *_ = get_neighborhood(
        positions=positions.copy(),
        cutoff=cutoff,
        pbc=(True, True, True),
        cell=cell.copy(),
    )

    assert edge_nopbc.shape[1] == 0, "no cross-boundary edge without PBC"
    assert edge_pbc.shape[1] > 0, "cross-boundary edge expected with PBC"


def test_pbc_more_edges_than_non_pbc():
    """PBC never produces fewer edges than the open-boundary equivalent."""
    cell = np.eye(3) * 6.0
    positions = np.array([[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]])
    cutoff = 4.0

    edge_nopbc, *_ = get_neighborhood(
        positions=positions.copy(),
        cutoff=cutoff,
        pbc=(False, False, False),
    )
    edge_pbc, *_ = get_neighborhood(
        positions=positions.copy(),
        cutoff=cutoff,
        pbc=(True, True, True),
        cell=cell.copy(),
    )

    assert edge_pbc.shape[1] >= edge_nopbc.shape[1]


def test_long_range_superset():
    """Every short-range edge appears in the long-range edge list."""
    positions = np.array(
        [[0.0, 0.0, 0.0], [3.0, 0.0, 0.0], [6.0, 0.0, 0.0]]
    )
    edge_sr, _, _, _, edge_lr, *_ = get_neighborhood(
        positions=positions,
        cutoff=4.0,
        cutoff_lr=7.0,
        pbc=(False, False, False),
    )

    sr_edges = set(map(tuple, edge_sr.T))
    lr_edges = set(map(tuple, edge_lr.T))
    assert sr_edges.issubset(lr_edges)


def test_long_range_none_when_not_requested():
    """Without cutoff_lr, long-range outputs are None."""
    positions = np.array([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    _, _, _, _, edge_index_lr, shifts_lr, unit_shifts_lr = (
        get_neighborhood(
            positions=positions,
            cutoff=3.0,
        )
    )

    assert edge_index_lr is None
    assert shifts_lr is None
    assert unit_shifts_lr is None


def test_h2o_fully_connected():
    """H2O with a 5 Å cutoff should produce 6 directed edges (3×2)."""
    edge_index, *_ = get_neighborhood(
        positions=_H2O_POSITIONS,
        cutoff=5.0,
        pbc=(False, False, False),
    )

    assert edge_index.shape[1] == 6
