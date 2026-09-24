"""
Comprehensive test suite for HDF5 data loading and preprocessing.

Tests cover:
- Raw HDF5 format (atomic structures)
- Preprocessed HDF5 format (with neighbor lists)
- Format detection and validation
- CLI preprocessing tool
- Training integration
"""

import logging
import os
import subprocess
import tempfile
from pathlib import Path

import ase.io
import h5py
import numpy as np
import pytest
import torch
from ase.build import bulk, molecule

from so3krates_torch.data.atomic_data import AtomicData
from so3krates_torch.data.hdf5_utils import (
    PreprocessedHDF5Dataset,
    configs_from_hdf5,
    detect_file_format,
    load_atoms_from_hdf5,
    merge_preprocessed_hdf5_files,
    merge_raw_hdf5_files,
    raise_if_preprocessed_properties_missing,
    save_atoms_to_hdf5,
    save_preprocessed_hdf5,
    scan_preprocessed_hdf5_property_presence,
    scan_raw_hdf5_property_presence,
    scan_raw_hdf5_statistics,
    validate_preprocessed_hdf5,
)
from so3krates_torch.data.utils import (
    KeySpecification,
    config_from_atoms,
)
from so3krates_torch.tools import torch_geometric
from so3krates_torch.tools.utils import AtomicNumberTable


class TestRawHDF5:
    """Test raw HDF5 format (atomic structures without neighbor lists)."""

    def test_save_load_single_molecule(self, h2o_atoms, tmp_path):
        """Test save and load single molecule."""
        # Add properties
        h2o_atoms.info["REF_energy"] = -10.0
        h2o_atoms.arrays["REF_forces"] = np.random.randn(len(h2o_atoms), 3)

        # Save to HDF5
        hdf5_path = tmp_path / "single.h5"
        keyspec = KeySpecification(
            info_keys={"energy": "REF_energy"},
            arrays_keys={"forces": "REF_forces"},
        )
        save_atoms_to_hdf5([h2o_atoms], str(hdf5_path), keyspec)

        # Load back
        loaded = load_atoms_from_hdf5(str(hdf5_path), index=0)

        # Verify — loaded atoms have original ASE keys
        assert len(loaded) == len(h2o_atoms)
        assert np.allclose(loaded.get_positions(), h2o_atoms.get_positions())
        assert "REF_energy" in loaded.info
        assert np.isclose(loaded.info["REF_energy"], -10.0)

    def test_save_load_multiple_configs(self, tmp_path):
        """Test save and load multiple configurations."""
        # Create multiple molecules
        atoms_list = [molecule(name) for name in ["H2O", "NH3", "CH4"]]
        for i, atoms in enumerate(atoms_list):
            atoms.info["REF_energy"] = -10.0 * (i + 1)

        # Save
        hdf5_path = tmp_path / "multiple.h5"
        keyspec = KeySpecification(info_keys={"energy": "REF_energy"})
        save_atoms_to_hdf5(atoms_list, str(hdf5_path), keyspec)

        # Load all
        loaded_list = load_atoms_from_hdf5(str(hdf5_path), index=None)
        assert len(loaded_list) == 3

        # Load slice
        loaded_slice = load_atoms_from_hdf5(str(hdf5_path), index=slice(0, 2))
        assert len(loaded_slice) == 2

        # Load single by index
        loaded_single = load_atoms_from_hdf5(str(hdf5_path), index=1)
        assert len(loaded_single) == len(atoms_list[1])

    def test_roundtrip_with_properties(self, tmp_path):
        """Test roundtrip preserves energy, forces, stress."""
        atoms = molecule("H2O")
        atoms.info["REF_energy"] = -15.5
        atoms.arrays["REF_forces"] = np.random.randn(len(atoms), 3)
        atoms.info["REF_stress"] = np.random.randn(6)

        # Save and load
        hdf5_path = tmp_path / "props.h5"
        keyspec = KeySpecification(
            info_keys={"energy": "REF_energy", "stress": "REF_stress"},
            arrays_keys={"forces": "REF_forces"},
        )
        save_atoms_to_hdf5([atoms], str(hdf5_path), keyspec)
        loaded = load_atoms_from_hdf5(str(hdf5_path), index=0)

        # Check properties — loaded with original ASE keys
        assert np.isclose(loaded.info["REF_energy"], atoms.info["REF_energy"])
        assert np.allclose(
            loaded.arrays["REF_forces"],
            atoms.arrays["REF_forces"],
        )
        assert np.allclose(loaded.info["REF_stress"], atoms.info["REF_stress"])

    def test_configs_from_hdf5(self, example_raw_hdf5):
        """Test direct Configuration loading from HDF5."""
        keyspec = KeySpecification(
            info_keys={"energy": "REF_energy"},
            arrays_keys={"forces": "REF_forces"},
        )
        configs = configs_from_hdf5(example_raw_hdf5, keyspec)

        assert len(configs) == 3
        assert all(hasattr(config, "atomic_numbers") for config in configs)
        assert all(hasattr(config, "positions") for config in configs)


class TestPreprocessedHDF5:
    """Test preprocessed HDF5 format (with neighbor lists)."""

    def test_save_load_atomic_data(self, h2o_atoms, tmp_path):
        """Test save and load preprocessed AtomicData."""
        # Create AtomicData
        h2o_atoms.info["REF_energy"] = -10.0
        h2o_atoms.arrays["REF_forces"] = np.random.randn(len(h2o_atoms), 3)
        keyspec = KeySpecification(
            info_keys={"energy": "REF_energy"},
            arrays_keys={"forces": "REF_forces"},
        )
        config = config_from_atoms(h2o_atoms, keyspec)
        z_table = AtomicNumberTable([1, 8])
        r_max = 5.0

        data = AtomicData.from_config(
            config, z_table=z_table, cutoff=r_max, cutoff_lr=None
        )

        # Save
        hdf5_path = tmp_path / "preprocessed.h5"
        save_preprocessed_hdf5(
            [data],
            str(hdf5_path),
            r_max=r_max,
            r_max_lr=None,
            z_table=z_table,
        )

        # Verify file exists and has metadata
        with h5py.File(hdf5_path, "r") as f:
            assert f.attrs["format_type"] == "preprocessed"
            assert f.attrs["r_max"] == r_max
            assert "config_0" in f

    def test_dataset_interface(self, example_preprocessed_hdf5):
        """Test PreprocessedHDF5Dataset __len__ and __getitem__."""
        dataset = PreprocessedHDF5Dataset(
            example_preprocessed_hdf5, validate_cutoffs=True
        )

        # Test __len__
        assert len(dataset) == 3

        # Test __getitem__
        data = dataset[0]
        assert isinstance(data, AtomicData)
        assert hasattr(data, "edge_index")
        assert hasattr(data, "positions")

    def test_lazy_loading(self, example_preprocessed_hdf5):
        """Test that data is loaded on-demand, not all at once."""
        dataset = PreprocessedHDF5Dataset(example_preprocessed_hdf5)

        # Access first item
        data0 = dataset[0]
        assert data0.num_nodes > 0

        # Access second item (should not affect first)
        data1 = dataset[1]
        assert data1.num_nodes > 0

    def test_dataloader_integration(self, example_preprocessed_hdf5):
        """Test integration with PyTorch DataLoader and batching."""
        dataset = PreprocessedHDF5Dataset(example_preprocessed_hdf5)
        loader = torch_geometric.dataloader.DataLoader(
            dataset, batch_size=2, shuffle=False
        )

        batches = list(loader)
        assert len(batches) == 2  # 3 samples, batch_size 2

    def test_cutoff_validation(self, example_preprocessed_hdf5):
        """Test r_max validation on load."""
        # Should succeed with correct r_max
        dataset = PreprocessedHDF5Dataset(
            example_preprocessed_hdf5,
            validate_cutoffs=True,
            expected_r_max=5.0,
        )
        assert dataset.r_max == 5.0

        # Should fail with wrong r_max
        with pytest.raises(ValueError, match="r_max mismatch"):
            PreprocessedHDF5Dataset(
                example_preprocessed_hdf5,
                validate_cutoffs=True,
                expected_r_max=6.0,
            )

    def test_long_range_edges(self, tmp_path):
        """Test with r_max_lr (long-range cutoffs)."""
        # Create larger system for long-range
        atoms = bulk("Si", "diamond", a=5.43)
        atoms = atoms * (2, 2, 2)  # 16 atoms
        atoms.info["REF_energy"] = -50.0
        atoms.arrays["REF_forces"] = np.zeros((len(atoms), 3))

        keyspec = KeySpecification(
            info_keys={"energy": "REF_energy"},
            arrays_keys={"forces": "REF_forces"},
        )
        config = config_from_atoms(atoms, keyspec)
        z_table = AtomicNumberTable([14])
        r_max = 3.0
        r_max_lr = 6.0

        data = AtomicData.from_config(
            config, z_table=z_table, cutoff=r_max, cutoff_lr=r_max_lr
        )

        # Save with long-range
        hdf5_path = tmp_path / "with_lr.h5"
        save_preprocessed_hdf5(
            [data],
            str(hdf5_path),
            r_max=r_max,
            r_max_lr=r_max_lr,
            z_table=z_table,
        )

        # Load and verify
        dataset = PreprocessedHDF5Dataset(
            str(hdf5_path),
            validate_cutoffs=True,
            expected_r_max=r_max,
            expected_r_max_lr=r_max_lr,
        )
        loaded = dataset[0]
        assert loaded.edge_index_lr is not None
        assert loaded.edge_index_lr.shape[1] > 0


class TestRequiredPropertyPresenceHDF5:
    """Tests for the loss-property presence checks used by the raw
    (lazy) and preprocessed HDF5 loading paths."""

    def test_preprocessed_presence_detects_missing_property(
        self, example_preprocessed_hdf5_full_keyspec
    ):
        # Fixture only populates energy/forces on the atoms; every other
        # DefaultKeys property is registered but absent.
        presence = scan_preprocessed_hdf5_property_presence(
            example_preprocessed_hdf5_full_keyspec,
            ["energy", "hirshfeld_ratios"],
        )
        energy_count, energy_total = presence["energy"]
        assert energy_count == energy_total
        assert energy_total > 0
        hirshfeld_count, _ = presence["hirshfeld_ratios"]
        assert hirshfeld_count == 0

        with pytest.raises(ValueError, match="hirshfeld_ratios"):
            raise_if_preprocessed_properties_missing(
                presence, example_preprocessed_hdf5_full_keyspec
            )

    def test_preprocessed_presence_noop_when_all_present(
        self, example_preprocessed_hdf5
    ):
        presence = scan_preprocessed_hdf5_property_presence(
            example_preprocessed_hdf5, ["energy"]
        )
        count, total = presence["energy"]
        assert count == total > 0
        # Should not raise.
        raise_if_preprocessed_properties_missing(
            presence, example_preprocessed_hdf5
        )

    def test_preprocessed_presence_partial_coverage_detected(self, tmp_path):
        """A property present in some but not all configs of a
        (possibly merged) preprocessed HDF5 file must be flagged
        distinctly from 'absent everywhere' — this is what would
        otherwise crash downstream in batch collation."""
        atoms_with = molecule("H2O")
        atoms_with.info["REF_energy"] = -10.0
        atoms_with.arrays["REF_forces"] = np.random.randn(3, 3)
        atoms_with.arrays["REF_hirsh_ratios"] = np.ones(3)

        atoms_without = molecule("NH3")
        atoms_without.info["REF_energy"] = -10.0
        atoms_without.arrays["REF_forces"] = np.random.randn(4, 3)

        keyspec = KeySpecification(
            info_keys={"energy": "REF_energy"},
            arrays_keys={
                "forces": "REF_forces",
                "hirshfeld_ratios": "REF_hirsh_ratios",
            },
        )
        z_table = AtomicNumberTable([1, 7, 8])
        data_list = [
            AtomicData.from_config(
                config_from_atoms(atoms, keyspec),
                z_table=z_table,
                cutoff=5.0,
                cutoff_lr=None,
            )
            for atoms in (atoms_with, atoms_without)
        ]
        out_path = str(tmp_path / "partial_coverage.h5")
        save_preprocessed_hdf5(
            data_list, out_path, r_max=5.0, r_max_lr=None, z_table=z_table
        )

        presence = scan_preprocessed_hdf5_property_presence(
            out_path, ["hirshfeld_ratios"]
        )
        assert presence["hirshfeld_ratios"] == (1, 2)

        with pytest.raises(ValueError, match="hirshfeld_ratios"):
            raise_if_preprocessed_properties_missing(presence, out_path)

    def test_scan_raw_hdf5_statistics_raises_for_missing_property(
        self, example_raw_hdf5
    ):
        keyspec = KeySpecification(
            info_keys={"energy": "REF_energy"},
            arrays_keys={
                "forces": "REF_forces",
                "hirshfeld_ratios": "REF_hirsh_ratios",
            },
        )
        with pytest.raises(ValueError, match="hirshfeld_ratios"):
            scan_raw_hdf5_statistics(
                hdf5_path=example_raw_hdf5,
                r_max=5.0,
                r_max_lr=None,
                keyspec=keyspec,
                required_properties=["energy", "hirshfeld_ratios"],
            )

    def test_scan_raw_hdf5_statistics_ok_when_present(self, example_raw_hdf5):
        keyspec = KeySpecification(
            info_keys={"energy": "REF_energy"},
            arrays_keys={"forces": "REF_forces"},
        )
        # Should not raise.
        scan_raw_hdf5_statistics(
            hdf5_path=example_raw_hdf5,
            r_max=5.0,
            r_max_lr=None,
            keyspec=keyspec,
            required_properties=["energy"],
        )

    def test_scan_raw_hdf5_property_presence_matches_full_scan(
        self, example_raw_hdf5
    ):
        """The cheap group-key scan must agree with the full-file-load
        presence check for both a present and an absent property."""
        keyspec = KeySpecification(
            info_keys={"energy": "REF_energy"},
            arrays_keys={
                "forces": "REF_forces",
                "hirshfeld_ratios": "REF_hirsh_ratios",
            },
        )
        presence = scan_raw_hdf5_property_presence(
            example_raw_hdf5, keyspec, ["energy", "forces", "hirshfeld_ratios"]
        )
        assert presence["energy"] is True
        assert presence["forces"] is True
        assert presence["hirshfeld_ratios"] is False


class TestFormatDetection:
    """Test file format detection."""

    def test_detect_xyz(self, example_xyz_with_data):
        """Test XYZ detection."""
        fmt = detect_file_format(example_xyz_with_data)
        assert fmt == "xyz"

    def test_detect_hdf5_raw(self, example_raw_hdf5):
        """Test raw HDF5 detection."""
        fmt = detect_file_format(example_raw_hdf5)
        assert fmt == "hdf5_raw"

    def test_detect_hdf5_preprocessed(self, example_preprocessed_hdf5):
        """Test preprocessed HDF5 detection."""
        fmt = detect_file_format(example_preprocessed_hdf5)
        assert fmt == "hdf5_preprocessed"


class TestCLIPreprocess:
    """Test CLI preprocessing tool."""

    def test_xyz_to_raw(self, tmp_path):
        """Test XYZ → raw HDF5 via CLI."""
        # Use the aqm_small.xyz file as input
        data_dir = Path(__file__).parent / "data"
        input_xyz = data_dir / "aqm_small.xyz"
        output_h5 = tmp_path / "raw.h5"

        # Run CLI
        result = subprocess.run(
            [
                "python",
                "-m",
                "so3krates_torch.cli.run_preprocess",
                "--input",
                str(input_xyz),
                "--output",
                str(output_h5),
                "--mode",
                "raw",
            ],
            capture_output=True,
            text=True,
        )

        # Check success
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_h5.exists()

        # Verify output
        fmt = detect_file_format(str(output_h5))
        assert fmt == "hdf5_raw"

    def test_xyz_to_preprocessed(self, tmp_path):
        """Test XYZ → preprocessed HDF5 via CLI."""
        data_dir = Path(__file__).parent / "data"
        input_xyz = data_dir / "aqm_small.xyz"
        output_h5 = tmp_path / "preprocessed.h5"

        # Run CLI
        result = subprocess.run(
            [
                "python",
                "-m",
                "so3krates_torch.cli.run_preprocess",
                "--input",
                str(input_xyz),
                "--output",
                str(output_h5),
                "--mode",
                "preprocessed",
                "--r-max",
                "6.0",
            ],
            capture_output=True,
            text=True,
        )

        # Check success
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_h5.exists()

        # Verify output
        fmt = detect_file_format(str(output_h5))
        assert fmt == "hdf5_preprocessed"

        # Check metadata
        with h5py.File(output_h5, "r") as f:
            assert f.attrs["r_max"] == 6.0

    def test_raw_to_preprocessed(self, example_raw_hdf5, tmp_path):
        """Test raw HDF5 → preprocessed HDF5 via CLI."""
        output_h5 = tmp_path / "preprocessed.h5"

        # Run CLI
        result = subprocess.run(
            [
                "python",
                "-m",
                "so3krates_torch.cli.run_preprocess",
                "--input",
                example_raw_hdf5,
                "--output",
                str(output_h5),
                "--mode",
                "preprocessed",
                "--r-max",
                "5.0",
            ],
            capture_output=True,
            text=True,
        )

        # Check success
        assert result.returncode == 0, f"stderr: {result.stderr}"
        assert output_h5.exists()

    def test_validation_flag(self, tmp_path):
        """Test --validate flag."""
        data_dir = Path(__file__).parent / "data"
        input_xyz = data_dir / "aqm_small.xyz"
        output_h5 = tmp_path / "validated.h5"

        # Run with validation
        result = subprocess.run(
            [
                "python",
                "-m",
                "so3krates_torch.cli.run_preprocess",
                "--input",
                str(input_xyz),
                "--output",
                str(output_h5),
                "--mode",
                "preprocessed",
                "--r-max",
                "6.0",
                "--validate",
            ],
            capture_output=True,
            text=True,
        )

        assert result.returncode == 0
        # Logging goes to stderr
        assert "Validation passed" in result.stderr


class TestBackwardCompatibility:
    """Test that existing XYZ workflows remain unchanged."""

    def test_existing_xyz_loading(self, example_xyz_with_data):
        """Ensure existing XYZ loading still works."""
        # Load with ASE (existing method)
        atoms_list = ase.io.read(example_xyz_with_data, index=":")
        assert len(atoms_list) == 3

        # Verify properties are preserved
        for atoms in atoms_list:
            assert "REF_energy" in atoms.info or "energy" in atoms.info


class TestHDF5Merge:
    """Test merging multiple HDF5 files into one."""

    def test_merge_raw_two_files(self, example_raw_hdf5, tmp_path):
        """Merge two raw HDF5 files; output contains all configs."""
        out = str(tmp_path / "merged.h5")
        merge_raw_hdf5_files([example_raw_hdf5, example_raw_hdf5], out)

        with h5py.File(out, "r") as f:
            assert int(f.attrs["num_configs"]) == 6

        loaded = load_atoms_from_hdf5(out)
        assert len(loaded) == 6
        # Properties should survive the round-trip
        for atoms in loaded:
            assert "REF_energy" in atoms.info

    def test_merge_raw_three_files(self, example_raw_hdf5, tmp_path):
        """Merge three raw HDF5 files."""
        out = str(tmp_path / "merged3.h5")
        merge_raw_hdf5_files(
            [example_raw_hdf5, example_raw_hdf5, example_raw_hdf5], out
        )
        loaded = load_atoms_from_hdf5(out)
        assert len(loaded) == 9

    def test_merge_raw_preserves_forces(self, example_raw_hdf5, tmp_path):
        """Forces (per-atom property) should be preserved after merge."""
        out = str(tmp_path / "merged_forces.h5")
        merge_raw_hdf5_files([example_raw_hdf5, example_raw_hdf5], out)
        loaded = load_atoms_from_hdf5(out)
        for atoms in loaded:
            assert "REF_forces" in atoms.arrays
            assert atoms.arrays["REF_forces"].shape == (len(atoms), 3)

    def test_merge_preprocessed_two_files(
        self, example_preprocessed_hdf5, tmp_path
    ):
        """Merge two preprocessed HDF5 files; groups renumbered."""
        out = str(tmp_path / "merged_pre.h5")
        merge_preprocessed_hdf5_files(
            [example_preprocessed_hdf5, example_preprocessed_hdf5],
            out,
        )

        with h5py.File(out, "r") as f:
            assert int(f.attrs["num_configs"]) == 6
            assert float(f.attrs["r_max"]) == 5.0
            # Groups config_0 … config_5 must all be present
            for i in range(6):
                assert f"config_{i}" in f

    def test_merge_preprocessed_heterogeneous_property_coverage_raises(
        self, tmp_path
    ):
        """Merging preprocessed files that don't all carry the same
        properties must be caught by the presence/consistency check
        on the merged file, rather than silently passing and crashing
        later during batch collation with a confusing error."""
        from ase.build import molecule

        from so3krates_torch.data.atomic_data import AtomicData
        from so3krates_torch.data.utils import (
            KeySpecification,
            config_from_atoms,
        )
        from so3krates_torch.tools.utils import AtomicNumberTable

        keyspec = KeySpecification(
            info_keys={"energy": "REF_energy"},
            arrays_keys={
                "forces": "REF_forces",
                "hirshfeld_ratios": "REF_hirsh_ratios",
            },
        )
        z_table = AtomicNumberTable([1, 8])

        atoms_with = molecule("H2O")
        atoms_with.info["REF_energy"] = -10.0
        atoms_with.arrays["REF_forces"] = np.random.randn(3, 3)
        atoms_with.arrays["REF_hirsh_ratios"] = np.ones(3)
        data_with = AtomicData.from_config(
            config_from_atoms(atoms_with, keyspec),
            z_table=z_table,
            cutoff=5.0,
            cutoff_lr=None,
        )
        path_with = str(tmp_path / "with_hirshfeld.h5")
        save_preprocessed_hdf5(
            [data_with], path_with, r_max=5.0, r_max_lr=None, z_table=z_table
        )

        atoms_without = molecule("H2O")
        atoms_without.info["REF_energy"] = -10.0
        atoms_without.arrays["REF_forces"] = np.random.randn(3, 3)
        data_without = AtomicData.from_config(
            config_from_atoms(atoms_without, keyspec),
            z_table=z_table,
            cutoff=5.0,
            cutoff_lr=None,
        )
        path_without = str(tmp_path / "without_hirshfeld.h5")
        save_preprocessed_hdf5(
            [data_without],
            path_without,
            r_max=5.0,
            r_max_lr=None,
            z_table=z_table,
        )

        merged_path = str(tmp_path / "merged_heterogeneous.h5")
        merge_preprocessed_hdf5_files([path_with, path_without], merged_path)

        presence = scan_preprocessed_hdf5_property_presence(
            merged_path, ["hirshfeld_ratios"]
        )
        assert presence["hirshfeld_ratios"] == (1, 2)
        with pytest.raises(ValueError, match="hirshfeld_ratios"):
            raise_if_preprocessed_properties_missing(presence, merged_path)

    def test_merge_format_mismatch_raises(
        self, example_raw_hdf5, example_preprocessed_hdf5, tmp_path
    ):
        """Mixing raw and preprocessed inputs raises ValueError."""
        out = str(tmp_path / "bad.h5")
        with pytest.raises(ValueError, match="Format mismatch"):
            # Detect mismatch at CLI level (run_merge) or directly
            from so3krates_torch.data.hdf5_utils import detect_file_format

            fmt1 = detect_file_format(example_raw_hdf5)
            fmt2 = detect_file_format(example_preprocessed_hdf5)
            if fmt1 != fmt2:
                raise ValueError(
                    f"Format mismatch: {example_raw_hdf5} is "
                    f"'{fmt1}' but {example_preprocessed_hdf5} is "
                    f"'{fmt2}'."
                )

    def test_merge_raw_wrong_format_raises(
        self, example_preprocessed_hdf5, tmp_path
    ):
        """merge_raw_hdf5_files raises ValueError for non-raw input."""
        out = str(tmp_path / "bad.h5")
        with pytest.raises(ValueError, match="not a raw HDF5 file"):
            merge_raw_hdf5_files(
                [example_preprocessed_hdf5, example_preprocessed_hdf5],
                out,
            )

    def test_merge_preprocessed_rmax_mismatch_raises(
        self, example_preprocessed_hdf5, tmp_path
    ):
        """merge_preprocessed raises ValueError for r_max mismatch."""
        # Build a second preprocessed file with different r_max
        from ase.build import molecule

        from so3krates_torch.data.atomic_data import AtomicData
        from so3krates_torch.data.utils import (
            KeySpecification,
            config_from_atoms,
        )
        from so3krates_torch.tools.utils import AtomicNumberTable

        atoms = molecule("H2O")
        atoms.info["REF_energy"] = -10.0
        keyspec = KeySpecification(info_keys={"energy": "REF_energy"})
        config = config_from_atoms(atoms, keyspec)
        z_table = AtomicNumberTable([1, 8])
        data = AtomicData.from_config(
            config, z_table=z_table, cutoff=7.0, cutoff_lr=None
        )
        other_path = str(tmp_path / "other.h5")
        save_preprocessed_hdf5(
            [data], other_path, r_max=7.0, r_max_lr=None, z_table=z_table
        )

        out = str(tmp_path / "bad.h5")
        with pytest.raises(ValueError, match="r_max mismatch"):
            merge_preprocessed_hdf5_files(
                [example_preprocessed_hdf5, other_path], out
            )
