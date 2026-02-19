"""Tests for individual component tensor shapes."""

import pytest
import torch

from so3krates_torch.modules.spherical_harmonics import (
    RealSphericalHarmonics,
)
from so3krates_torch.blocks.embedding import (
    InvariantEmbedding,
    EuclideanEmbedding,
)
from so3krates_torch.blocks.output_block import (
    AtomicEnergyOutputHead,
    MultiAtomicEnergyOutputHead,
)
from so3krates_torch.modules.cutoff import (
    CosineCutoff,
    PhysNetCutoff,
    PolynomialCutoff,
    ExponentialCutoff,
)


class TestSphericalHarmonicsShapes:

    @pytest.mark.parametrize(
        "degrees,expected_dim",
        [
            ([0], 1),
            ([1], 3),
            ([2], 5),
            ([1, 2], 8),
            ([1, 2, 3], 15),
            ([0, 1, 2, 3, 4], 25),
        ],
    )
    def test_output_dim(self, degrees, expected_dim):
        sh = RealSphericalHarmonics(degrees=degrees)
        vecs = torch.randn(10, 3)
        out = sh(vecs)
        assert out.shape == (10, expected_dim)

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_output_dtype_matches_input(self, dtype):
        sh = RealSphericalHarmonics(degrees=[1, 2])
        vecs = torch.randn(10, 3, dtype=dtype)
        out = sh(vecs)
        assert out.dtype == dtype


class TestEmbeddingShapes:

    def test_euclidean_embedding_zeros_shape(self):
        emb = EuclideanEmbedding(initialization_to_zeros=True)
        out = emb(
            sh_vectors=torch.randn(20, 8),
            cutoffs=torch.ones(20, 1),
            receivers=torch.randint(0, 5, (20,)),
            avg_num_neighbors=4.0,
            num_nodes=5,
        )
        assert out.shape == (5, 8)
        assert (out == 0).all()

    def test_euclidean_embedding_nonzero_shape(self):
        emb = EuclideanEmbedding(initialization_to_zeros=False)
        out = emb(
            sh_vectors=torch.randn(20, 8),
            cutoffs=torch.ones(20, 1),
            receivers=torch.randint(0, 5, (20,)),
            avg_num_neighbors=4.0,
            num_nodes=5,
        )
        assert out.shape == (5, 8)
        assert not (out == 0).all()


class TestOutputHeadShapes:

    @staticmethod
    def _make_data_dict(num_nodes, num_elements=118):
        one_hot = torch.zeros(num_nodes, num_elements)
        for i in range(num_nodes):
            one_hot[i, i % num_elements] = 1.0
        return {"node_attrs": one_hot}

    def test_atomic_energy_head_shape(self):
        head = AtomicEnergyOutputHead(
            num_features=16,
            num_elements=118,
            non_linearity=torch.nn.SiLU,
            use_non_linearity=True,
        )
        x = torch.randn(10, 16)
        data = self._make_data_dict(10)
        out = head(x, data)
        assert out.shape == (10, 1)

    def test_atomic_energy_head_layers(self):
        head = AtomicEnergyOutputHead(
            num_features=16,
            num_elements=118,
            layers=3,
            non_linearity=torch.nn.SiLU,
            use_non_linearity=True,
        )
        x = torch.randn(10, 16)
        data = self._make_data_dict(10)
        out = head(x, data)
        assert out.shape == (10, 1)

    @pytest.mark.parametrize("num_heads", [4])
    def test_multi_head_shape(self, num_heads):
        head = MultiAtomicEnergyOutputHead(
            num_output_heads=num_heads,
            num_features=16,
            num_elements=118,
            energy_regression_dim=16,
            non_linearity=torch.nn.SiLU,
            use_non_linearity=True,
        )
        x = torch.randn(10, 16)
        data = self._make_data_dict(10)
        out = head(x, data)
        assert out.shape == (10, num_heads, 1)

    def test_multi_head_single_node_squeeze(self):
        num_heads = 4
        head = MultiAtomicEnergyOutputHead(
            num_output_heads=num_heads,
            num_features=16,
            num_elements=118,
            energy_regression_dim=16,
            non_linearity=torch.nn.SiLU,
            use_non_linearity=True,
        )
        x = torch.randn(1, 16)
        data = self._make_data_dict(1)
        out = head(x, data)
        assert out.shape == (num_heads, 1)


_MASKING_CUTOFF_PARAMS = [
    (PhysNetCutoff, {"r_max": 5.0}),
    (PolynomialCutoff, {"r_max": 5.0, "p": 6}),
    (ExponentialCutoff, {"r_max": 5.0}),
]


class TestCutoffShapes:

    @pytest.mark.parametrize("cutoff_cls,kwargs", _MASKING_CUTOFF_PARAMS)
    def test_zero_beyond_cutoff(self, cutoff_cls, kwargs):
        cutoff = cutoff_cls(**kwargs)
        x = torch.tensor([5.0, 6.0, 10.0])
        out = cutoff(x)
        assert torch.allclose(out, torch.zeros_like(out))

    def test_cosine_no_masking(self):
        cutoff = CosineCutoff(r_max=5.0)
        x = torch.tensor([6.0, 10.0])
        out = cutoff(x)
        assert not torch.allclose(out, torch.zeros_like(out))
