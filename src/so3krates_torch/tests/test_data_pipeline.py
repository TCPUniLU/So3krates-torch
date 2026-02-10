"""Tests for data pipeline components."""
import pytest
import torch
import numpy as np
from ase import Atoms
from so3krates_torch.data.atomic_data import AtomicData
from so3krates_torch.data.neighborhood import get_neighborhood
from so3krates_torch.tools.torch_geometric import Batch


class TestAtomicDataConstruction:
    """Test AtomicData construction with various inputs."""

    def test_minimal_construction(self, h2o_atoms):
        """Test AtomicData with minimal required fields."""
        positions = h2o_atoms.get_positions()
        atomic_numbers = h2o_atoms.get_atomic_numbers()
        cell = h2o_atoms.get_cell()
        pbc = h2o_atoms.get_pbc()

        edge_index, shifts, unit_shifts, cell, *_ = get_neighborhood(
            positions=positions,
            cutoff=5.0,
            pbc=tuple(pbc),
            cell=np.array(cell),
        )

        # Create one-hot encoding for node_attrs
        num_atoms = len(atomic_numbers)
        node_attrs = torch.zeros(
            (num_atoms, 118), dtype=torch.get_default_dtype()
        )
        for i, z in enumerate(atomic_numbers):
            node_attrs[i, z - 1] = 1.0

        data = AtomicData(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            node_attrs=node_attrs,
            atomic_numbers=torch.tensor(
                atomic_numbers, dtype=torch.long
            ),
            positions=torch.tensor(
                positions, dtype=torch.get_default_dtype()
            ),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(
                unit_shifts, dtype=torch.get_default_dtype()
            ),
            cell=torch.tensor(cell, dtype=torch.get_default_dtype()),
            weight=None,
            head=None,
            energy_weight=None,
            forces_weight=None,
            stress_weight=None,
            virials_weight=None,
            dipole_weight=None,
            charges_weight=None,
            hirshfeld_ratios_weight=None,
            forces=None,
            energy=None,
            stress=None,
            virials=None,
            dipole=None,
            hirshfeld_ratios=None,
            charges=None,
            elec_temp=None,
        )

        assert data.num_nodes == 3
        assert data.edge_index.shape[0] == 2
        assert data.positions.shape == (3, 3)

    def test_with_target_properties(self, h2o_atoms):
        """Test AtomicData with energy, forces, stress targets."""
        positions = h2o_atoms.get_positions()
        atomic_numbers = h2o_atoms.get_atomic_numbers()
        cell = h2o_atoms.get_cell()
        pbc = h2o_atoms.get_pbc()

        edge_index, shifts, unit_shifts, cell, *_ = get_neighborhood(
            positions=positions,
            cutoff=5.0,
            pbc=tuple(pbc),
            cell=np.array(cell),
        )

        num_atoms = len(atomic_numbers)
        node_attrs = torch.zeros(
            (num_atoms, 118), dtype=torch.get_default_dtype()
        )
        for i, z in enumerate(atomic_numbers):
            node_attrs[i, z - 1] = 1.0

        # Add target properties
        energy = torch.tensor(1.5, dtype=torch.get_default_dtype())
        forces = torch.randn(
            (num_atoms, 3), dtype=torch.get_default_dtype()
        )
        stress = torch.randn(
            (1, 3, 3), dtype=torch.get_default_dtype()
        )

        data = AtomicData(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            node_attrs=node_attrs,
            atomic_numbers=torch.tensor(
                atomic_numbers, dtype=torch.long
            ),
            positions=torch.tensor(
                positions, dtype=torch.get_default_dtype()
            ),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(
                unit_shifts, dtype=torch.get_default_dtype()
            ),
            cell=torch.tensor(cell, dtype=torch.get_default_dtype()),
            weight=None,
            head=None,
            energy_weight=None,
            forces_weight=None,
            stress_weight=None,
            virials_weight=None,
            dipole_weight=None,
            charges_weight=None,
            hirshfeld_ratios_weight=None,
            forces=forces,
            energy=energy,
            stress=stress,
            virials=None,
            dipole=None,
            hirshfeld_ratios=None,
            charges=None,
            elec_temp=None,
        )

        assert data.energy.shape == ()
        assert data.forces.shape == (3, 3)
        assert data.stress.shape == (1, 3, 3)

    def test_long_range_edges(self, h2o_atoms):
        """Test AtomicData with dual cutoffs (short + long range)."""
        positions = h2o_atoms.get_positions()
        atomic_numbers = h2o_atoms.get_atomic_numbers()
        cell = h2o_atoms.get_cell()
        pbc = h2o_atoms.get_pbc()

        (
            edge_index,
            shifts,
            unit_shifts,
            cell,
            edge_index_lr,
            shifts_lr,
            unit_shifts_lr,
        ) = get_neighborhood(
            positions=positions,
            cutoff=5.0,
            cutoff_lr=10.0,
            pbc=tuple(pbc),
            cell=np.array(cell),
        )

        num_atoms = len(atomic_numbers)
        node_attrs = torch.zeros(
            (num_atoms, 118), dtype=torch.get_default_dtype()
        )
        for i, z in enumerate(atomic_numbers):
            node_attrs[i, z - 1] = 1.0

        data = AtomicData(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            node_attrs=node_attrs,
            atomic_numbers=torch.tensor(
                atomic_numbers, dtype=torch.long
            ),
            positions=torch.tensor(
                positions, dtype=torch.get_default_dtype()
            ),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(
                unit_shifts, dtype=torch.get_default_dtype()
            ),
            cell=torch.tensor(cell, dtype=torch.get_default_dtype()),
            edge_index_lr=torch.tensor(
                edge_index_lr, dtype=torch.long
            )
            if edge_index_lr is not None
            else None,
            shifts_lr=torch.tensor(
                shifts_lr, dtype=torch.get_default_dtype()
            )
            if shifts_lr is not None
            else None,
            unit_shifts_lr=torch.tensor(
                unit_shifts_lr, dtype=torch.get_default_dtype()
            )
            if unit_shifts_lr is not None
            else None,
            weight=None,
            head=None,
            energy_weight=None,
            forces_weight=None,
            stress_weight=None,
            virials_weight=None,
            dipole_weight=None,
            charges_weight=None,
            hirshfeld_ratios_weight=None,
            forces=None,
            energy=None,
            stress=None,
            virials=None,
            dipole=None,
            hirshfeld_ratios=None,
            charges=None,
            elec_temp=None,
        )

        assert data.edge_index is not None
        assert data.edge_index_lr is not None
        assert data.shifts_lr is not None
        assert data.edge_index.shape[1] <= data.edge_index_lr.shape[1]

    @pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
    def test_dtype_preservation(self, h2o_atoms, dtype):
        """Test AtomicData preserves dtype through construction."""
        torch.set_default_dtype(dtype)

        positions = h2o_atoms.get_positions()
        atomic_numbers = h2o_atoms.get_atomic_numbers()
        cell = h2o_atoms.get_cell()
        pbc = h2o_atoms.get_pbc()

        edge_index, shifts, unit_shifts, cell, *_ = get_neighborhood(
            positions=positions,
            cutoff=5.0,
            pbc=tuple(pbc),
            cell=np.array(cell),
        )

        num_atoms = len(atomic_numbers)
        node_attrs = torch.zeros((num_atoms, 118), dtype=dtype)
        for i, z in enumerate(atomic_numbers):
            node_attrs[i, z - 1] = 1.0

        data = AtomicData(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            node_attrs=node_attrs,
            atomic_numbers=torch.tensor(
                atomic_numbers, dtype=torch.long
            ),
            positions=torch.tensor(positions, dtype=dtype),
            shifts=torch.tensor(shifts, dtype=dtype),
            unit_shifts=torch.tensor(unit_shifts, dtype=dtype),
            cell=torch.tensor(cell, dtype=dtype),
            weight=None,
            head=None,
            energy_weight=None,
            forces_weight=None,
            stress_weight=None,
            virials_weight=None,
            dipole_weight=None,
            charges_weight=None,
            hirshfeld_ratios_weight=None,
            forces=None,
            energy=None,
            stress=None,
            virials=None,
            dipole=None,
            hirshfeld_ratios=None,
            charges=None,
            elec_temp=None,
        )

        assert data.positions.dtype == dtype
        assert data.shifts.dtype == dtype
        assert data.node_attrs.dtype == dtype


class TestAtomicDataBatching:
    """Test batching multiple AtomicData objects."""

    def _make_atomic_data(self, atoms, cutoff=5.0, cutoff_lr=None):
        """Helper to create AtomicData from ASE Atoms."""
        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        result = get_neighborhood(
            positions=positions,
            cutoff=cutoff,
            cutoff_lr=cutoff_lr,
            pbc=tuple(pbc),
            cell=np.array(cell),
        )
        edge_index, shifts, unit_shifts, cell = result[:4]
        edge_index_lr, shifts_lr, unit_shifts_lr = (
            result[4:] if len(result) > 4 else (None, None, None)
        )

        num_atoms = len(atomic_numbers)
        node_attrs = torch.zeros(
            (num_atoms, 118), dtype=torch.get_default_dtype()
        )
        for i, z in enumerate(atomic_numbers):
            node_attrs[i, z - 1] = 1.0

        return AtomicData(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            node_attrs=node_attrs,
            atomic_numbers=torch.tensor(
                atomic_numbers, dtype=torch.long
            ),
            positions=torch.tensor(
                positions, dtype=torch.get_default_dtype()
            ),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(
                unit_shifts, dtype=torch.get_default_dtype()
            ),
            cell=torch.tensor(cell, dtype=torch.get_default_dtype()),
            edge_index_lr=torch.tensor(
                edge_index_lr, dtype=torch.long
            )
            if edge_index_lr is not None
            else None,
            shifts_lr=torch.tensor(
                shifts_lr, dtype=torch.get_default_dtype()
            )
            if shifts_lr is not None
            else None,
            unit_shifts_lr=torch.tensor(
                unit_shifts_lr, dtype=torch.get_default_dtype()
            )
            if unit_shifts_lr is not None
            else None,
            weight=None,
            head=None,
            energy_weight=None,
            forces_weight=None,
            stress_weight=None,
            virials_weight=None,
            dipole_weight=None,
            charges_weight=None,
            hirshfeld_ratios_weight=None,
            forces=None,
            energy=None,
            stress=None,
            virials=None,
            dipole=None,
            hirshfeld_ratios=None,
            charges=None,
            elec_temp=None,
        )

    def test_batch_concatenation(self, random_batch_atoms):
        """Test batching multiple AtomicData objects."""
        data_list = [
            self._make_atomic_data(atoms) for atoms in random_batch_atoms
        ]
        batch = Batch.from_data_list(data_list)

        total_atoms = sum([len(atoms) for atoms in random_batch_atoms])
        assert batch.num_nodes == total_atoms
        assert batch.batch.shape[0] == batch.num_nodes
        assert batch.edge_index.max() < batch.num_nodes

        # Check batch tensor values (node→graph mapping)
        expected_batch = []
        for i, atoms in enumerate(random_batch_atoms):
            expected_batch.extend([i] * len(atoms))
        assert torch.allclose(
            batch.batch, torch.tensor(expected_batch, dtype=torch.long)
        )

    def test_heterogeneous_batch(self, h2o_atoms, si_bulk):
        """Test batching molecules with/without long-range edges."""
        # H2O without long-range
        data_h2o = self._make_atomic_data(h2o_atoms, cutoff=5.0)
        # Si bulk with long-range
        data_si = self._make_atomic_data(
            si_bulk, cutoff=3.0, cutoff_lr=6.0
        )

        # Batch them together
        batch = Batch.from_data_list([data_h2o, data_si])

        assert batch.num_nodes == len(h2o_atoms) + len(si_bulk)
        assert batch.edge_index is not None
        # Check that batching succeeded without errors



class TestAtomicDataDeviceTransfer:
    """Test device transfers and dtype handling."""

    def _make_atomic_data(self, atoms, cutoff=5.0):
        """Helper to create AtomicData from ASE Atoms."""
        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = atoms.get_pbc()

        edge_index, shifts, unit_shifts, cell, *_ = get_neighborhood(
            positions=positions,
            cutoff=cutoff,
            pbc=tuple(pbc),
            cell=np.array(cell),
        )

        num_atoms = len(atomic_numbers)
        node_attrs = torch.zeros(
            (num_atoms, 118), dtype=torch.get_default_dtype()
        )
        for i, z in enumerate(atomic_numbers):
            node_attrs[i, z - 1] = 1.0

        return AtomicData(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            node_attrs=node_attrs,
            atomic_numbers=torch.tensor(
                atomic_numbers, dtype=torch.long
            ),
            positions=torch.tensor(
                positions, dtype=torch.get_default_dtype()
            ),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(
                unit_shifts, dtype=torch.get_default_dtype()
            ),
            cell=torch.tensor(cell, dtype=torch.get_default_dtype()),
            weight=None,
            head=None,
            energy_weight=None,
            forces_weight=None,
            stress_weight=None,
            virials_weight=None,
            dipole_weight=None,
            charges_weight=None,
            hirshfeld_ratios_weight=None,
            forces=None,
            energy=None,
            stress=None,
            virials=None,
            dipole=None,
            hirshfeld_ratios=None,
            charges=None,
            elec_temp=None,
        )

    @pytest.mark.skipif(
        not torch.cuda.is_available(), reason="CUDA not available"
    )
    def test_to_device_cpu_to_cuda(self, h2o_atoms):
        """Test transferring AtomicData from CPU to CUDA."""
        data = self._make_atomic_data(h2o_atoms)

        # Verify it's on CPU initially
        assert data.positions.device.type == "cpu"

        # Transfer to CUDA
        data_cuda = data.to("cuda")

        # Check all tensors moved
        assert data_cuda.positions.device.type == "cuda"
        assert data_cuda.edge_index.device.type == "cuda"
        assert data_cuda.shifts.device.type == "cuda"
        assert data_cuda.node_attrs.device.type == "cuda"

        # Check shapes unchanged
        assert data_cuda.positions.shape == data.positions.shape
        assert data_cuda.num_nodes == data.num_nodes

    def test_to_device_preserves_none_fields(self, h2o_atoms, device):
        """Test device transfer with optional None fields."""
        positions = h2o_atoms.get_positions()
        atomic_numbers = h2o_atoms.get_atomic_numbers()
        cell = h2o_atoms.get_cell()
        pbc = h2o_atoms.get_pbc()

        edge_index, shifts, unit_shifts, cell, *_ = get_neighborhood(
            positions=positions,
            cutoff=5.0,
            pbc=tuple(pbc),
            cell=np.array(cell),
        )

        num_atoms = len(atomic_numbers)
        node_attrs = torch.zeros(
            (num_atoms, 118), dtype=torch.get_default_dtype()
        )
        for i, z in enumerate(atomic_numbers):
            node_attrs[i, z - 1] = 1.0

        # Create with some None fields
        data = AtomicData(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            node_attrs=node_attrs,
            atomic_numbers=torch.tensor(
                atomic_numbers, dtype=torch.long
            ),
            positions=torch.tensor(
                positions, dtype=torch.get_default_dtype()
            ),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(
                unit_shifts, dtype=torch.get_default_dtype()
            ),
            cell=torch.tensor(cell, dtype=torch.get_default_dtype()),
            weight=None,
            head=None,
            energy_weight=None,
            forces_weight=None,
            stress_weight=None,
            virials_weight=None,
            dipole_weight=None,
            charges_weight=None,
            hirshfeld_ratios_weight=None,
            forces=None,  # Explicitly None
            energy=None,  # Explicitly None
            stress=None,  # Explicitly None
            virials=None,
            dipole=None,
            hirshfeld_ratios=None,
            charges=None,
            elec_temp=None,
        )

        # Transfer to device
        data_device = data.to(device)

        # None fields should remain None
        assert data_device.forces is None
        assert data_device.energy is None
        assert data_device.stress is None

        # Non-None tensors should be on device
        assert data_device.positions.device.type == device.type


class TestAtomicDataEdgeCases:
    """Test edge cases in AtomicData."""

    def _make_atomic_data(self, atoms, cutoff=5.0, pbc_override=None):
        """Helper to create AtomicData from ASE Atoms."""
        positions = atoms.get_positions()
        atomic_numbers = atoms.get_atomic_numbers()
        cell = atoms.get_cell()
        pbc = pbc_override if pbc_override is not None else atoms.get_pbc()

        edge_index, shifts, unit_shifts, cell, *_ = get_neighborhood(
            positions=positions,
            cutoff=cutoff,
            pbc=tuple(pbc),
            cell=np.array(cell),
        )

        num_atoms = len(atomic_numbers)
        node_attrs = torch.zeros(
            (num_atoms, 118), dtype=torch.get_default_dtype()
        )
        for i, z in enumerate(atomic_numbers):
            node_attrs[i, z - 1] = 1.0

        return AtomicData(
            edge_index=torch.tensor(edge_index, dtype=torch.long),
            node_attrs=node_attrs,
            atomic_numbers=torch.tensor(
                atomic_numbers, dtype=torch.long
            ),
            positions=torch.tensor(
                positions, dtype=torch.get_default_dtype()
            ),
            shifts=torch.tensor(shifts, dtype=torch.get_default_dtype()),
            unit_shifts=torch.tensor(
                unit_shifts, dtype=torch.get_default_dtype()
            ),
            cell=torch.tensor(cell, dtype=torch.get_default_dtype()),
            weight=None,
            head=None,
            energy_weight=None,
            forces_weight=None,
            stress_weight=None,
            virials_weight=None,
            dipole_weight=None,
            charges_weight=None,
            hirshfeld_ratios_weight=None,
            forces=None,
            energy=None,
            stress=None,
            virials=None,
            dipole=None,
            hirshfeld_ratios=None,
            charges=None,
            elec_temp=None,
        )

    def test_single_atom_system(self):
        """Test AtomicData for single-atom molecule."""
        he_atom = Atoms("He", positions=[[0, 0, 0]])
        data = self._make_atomic_data(he_atom, cutoff=5.0)

        assert data.num_nodes == 1
        # Single atom should have no edges (no neighbors within cutoff)
        assert data.edge_index.shape[1] == 0
        assert data.positions.shape == (1, 3)

    def test_periodic_vs_nonperiodic(self, si_bulk):
        """Test PBC handling in AtomicData."""
        # Create with periodic boundaries
        data_periodic = self._make_atomic_data(
            si_bulk, cutoff=3.0, pbc_override=(True, True, True)
        )

        # Create with non-periodic boundaries
        data_nonperiodic = self._make_atomic_data(
            si_bulk, cutoff=3.0, pbc_override=(False, False, False)
        )

        assert data_periodic.num_nodes == data_nonperiodic.num_nodes
        # Periodic should have more edges due to periodic images
        # (In this case, since it's a small unit cell, the difference
        # may be significant)
        assert data_periodic.edge_index.shape[1] >= (
            data_nonperiodic.edge_index.shape[1]
        )
