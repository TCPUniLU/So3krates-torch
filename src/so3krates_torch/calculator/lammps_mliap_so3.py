import logging
import os
import sys
import time
from contextlib import contextmanager
from typing import Dict, List, Tuple

import torch
from ase.data import chemical_symbols

try:
    from lammps.mliap.mliap_unified_abc import MLIAPUnified
except ImportError:

    class MLIAPUnified:
        def __init__(self):
            pass


class So3LammpsConfig:
    """Configuration settings for So3krates-LAMMPS integration.

    Reads settings from environment variables:
        SO3_TIME: Enable per-step timing output
        SO3_PROFILE: Enable CUDA profiling
        SO3_PROFILE_START: Step to start CUDA profiling (default: 5)
        SO3_PROFILE_END: Step to stop CUDA profiling (default: 10)
        SO3_ALLOW_CPU: Allow CPU computation with Kokkos
        SO3_FORCE_CPU: Force CPU even with Kokkos
    """

    def __init__(self):
        self.debug_time = self._get_env_bool("SO3_TIME", False)
        self.debug_profile = self._get_env_bool("SO3_PROFILE", False)
        self.profile_start_step = int(os.environ.get("SO3_PROFILE_START", "5"))
        self.profile_end_step = int(os.environ.get("SO3_PROFILE_END", "10"))
        self.allow_cpu = self._get_env_bool("SO3_ALLOW_CPU", False)
        self.force_cpu = self._get_env_bool("SO3_FORCE_CPU", False)

    @staticmethod
    def _get_env_bool(var_name: str, default: bool) -> bool:
        return os.environ.get(var_name, str(default)).lower() in (
            "true",
            "1",
            "t",
            "yes",
        )


@contextmanager
def timer(name: str, enabled: bool = True):
    """Context manager for timing code blocks."""
    if not enabled:
        yield
        return

    start = time.perf_counter()
    try:
        yield
    finally:
        elapsed = time.perf_counter() - start
        logging.info(f"Timer - {name}: {elapsed*1000:.3f} ms")


class So3EdgeForcesWrapper(torch.nn.Module):
    """Wrapper that validates config and computes per-pair forces for LAMMPS.

    Validates that the model only uses short-range interactions (no electrostatics
    or dispersion), freezes parameters, and provides a forward method that returns
    (total_energy, node_energy, pair_forces).

    Args:
        model: SO3LR or MultiHeadSO3LR model instance.
        atomic_numbers: List of atomic numbers for the elements in the simulation
            (e.g. [14, 8] for Si and O). Order must match LAMMPS pair_coeff types.
        head: Head name for multi-head models (default: last head).
    """

    def __init__(
        self, model: torch.nn.Module, atomic_numbers: List[int], **kwargs
    ):
        super().__init__()

        # Validate model config: only short-range interactions allowed
        if getattr(model, "electrostatic_energy_bool", False):
            raise ValueError(
                "Model has electrostatic_energy_bool=True. "
                "LAMMPS MLIAP only supports short-range interactions. "
                "Retrain with electrostatic_energy_bool=False or use a "
                "different model."
            )
        if getattr(model, "dispersion_energy_bool", False):
            raise ValueError(
                "Model has dispersion_energy_bool=True. "
                "LAMMPS MLIAP only supports short-range interactions. "
                "Retrain with dispersion_energy_bool=False or use a "
                "different model."
            )

        self.model = model
        self.register_buffer(
            "atomic_numbers_map",
            torch.tensor(atomic_numbers, dtype=torch.long),
        )
        self.register_buffer("r_max", torch.tensor(model.r_max))
        self.register_buffer(
            "num_interactions", torch.tensor(model.num_layers)
        )
        self.num_elements = model.num_elements

        # Handle head selection for multi-head models
        if not hasattr(model, "heads"):
            model.heads = ["Default"]
        head_name = kwargs.get("head", model.heads[-1])
        head_idx = model.heads.index(head_name)
        self.register_buffer(
            "head", torch.tensor([head_idx], dtype=torch.long)
        )

        for p in self.model.parameters():
            p.requires_grad = False

    def forward(
        self, data: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute energies and per-pair forces.

        Returns:
            Tuple of (total_energy, node_energy, pair_forces):
                - total_energy: scalar
                - node_energy: (n_atoms,) per-atom energies
                - pair_forces: (n_edges, 3) per-pair force vectors
        """
        data["head"] = self.head

        out = self.model(
            data,
            training=False,
            compute_force=False,
            compute_virials=False,
            compute_stress=False,
            compute_displacement=False,
            compute_hessian=False,
            compute_edge_forces=True,
            lammps_mliap=True,
        )

        node_energy = out["node_energy"]
        pair_forces = out["edge_forces"]
        total_energy = out["energy"][0]

        if pair_forces is None:
            pair_forces = torch.zeros_like(data["vectors"])

        return total_energy, node_energy, pair_forces


class LAMMPS_MLIAP_SO3(MLIAPUnified):
    """So3krates integration for LAMMPS using the MLIAP interface.

    This class bridges between LAMMPS neighbor list data and the So3krates
    model. It handles device detection, batch preparation, and writing
    results back to LAMMPS data structures.

    Args:
        model: SO3LR or MultiHeadSO3LR model instance.
        atomic_numbers: List of atomic numbers for simulation elements
            (e.g. [14, 8] for Si and O). Order must match LAMMPS pair_coeff.
        **kwargs: Passed to So3EdgeForcesWrapper (e.g. head="head_name").

    LAMMPS usage::

        pair_style mliap unified so3lr model.pt-mliap_lammps.pt
        pair_coeff * * Si O
    """

    def __init__(self, model, atomic_numbers: List[int], **kwargs):
        super().__init__()
        self.config = So3LammpsConfig()
        self.model = So3EdgeForcesWrapper(
            model, atomic_numbers=atomic_numbers, **kwargs
        )
        self.atomic_numbers_list = atomic_numbers
        self.element_types = [chemical_symbols[z] for z in atomic_numbers]
        self.num_species = len(self.element_types)
        self.num_elements = model.num_elements
        self.rcutfac = 0.5 * float(model.r_max)
        self.ndescriptors = 1
        self.nparams = 1
        self.dtype = next(model.parameters()).dtype
        self.device = "cpu"
        self.initialized = False
        self.step = 0

    def _initialize_device(self, data):
        """Auto-detect device from LAMMPS data tensors (Kokkos GPU or CPU)."""
        using_kokkos = "kokkos" in data.__class__.__module__.lower()

        if using_kokkos and not self.config.force_cpu:
            device = torch.as_tensor(data.elems).device
            if device.type == "cpu" and not self.config.allow_cpu:
                raise ValueError(
                    "GPU requested but tensor is on CPU. "
                    "Set SO3_ALLOW_CPU=true to allow CPU computation."
                )
        else:
            device = torch.device("cpu")

        self.device = device
        self.model = self.model.to(device)
        logging.info(f"SO3LR model initialized on device: {device}")
        self.initialized = True

    def compute_forces(self, data):
        """Main entry point called by LAMMPS each timestep."""
        natoms = data.nlocal
        ntotal = data.ntotal
        nghosts = ntotal - natoms
        npairs = data.npairs
        species = torch.as_tensor(data.elems, dtype=torch.int64)

        if not self.initialized:
            self._initialize_device(data)

        self.step += 1
        self._manage_profiling()

        if natoms == 0 or npairs <= 1:
            return

        with timer("total_step", enabled=self.config.debug_time):
            with timer("prepare_batch", enabled=self.config.debug_time):
                batch = self._prepare_batch(
                    data, natoms, nghosts, species
                )

            with timer("model_forward", enabled=self.config.debug_time):
                _, atom_energies, pair_forces = self.model(batch)

                if (
                    isinstance(self.device, torch.device)
                    and self.device.type != "cpu"
                ):
                    torch.cuda.synchronize()

            with timer("update_lammps", enabled=self.config.debug_time):
                self._update_lammps_data(
                    data, atom_energies, pair_forces, natoms
                )

    def _prepare_batch(self, data, natoms, nghosts, species):
        """Build input dictionary from LAMMPS data structures.

        Maps LAMMPS element type indices to actual atomic numbers and
        creates the 118-class one-hot encoding expected by So3krates.
        """
        # Map LAMMPS type indices to actual atomic numbers
        atomic_numbers_map = self.model.atomic_numbers_map.to(self.device)
        actual_z = atomic_numbers_map[species.to(self.device)]

        # Create one-hot encoding over num_elements classes (Z-1 indexing)
        ntotal = natoms + nghosts
        node_attrs = torch.zeros(
            ntotal,
            self.num_elements,
            dtype=self.dtype,
            device=self.device,
        )
        node_attrs.scatter_(1, (actual_z - 1).unsqueeze(1), 1.0)

        return {
            "vectors": torch.as_tensor(data.rij)
            .to(self.dtype)
            .to(self.device),
            "node_attrs": node_attrs,
            "edge_index": torch.stack(
                [
                    torch.as_tensor(data.pair_j, dtype=torch.int64).to(
                        self.device
                    ),
                    torch.as_tensor(data.pair_i, dtype=torch.int64).to(
                        self.device
                    ),
                ],
                dim=0,
            ),
            "atomic_numbers": actual_z,
            "batch": torch.zeros(
                natoms, dtype=torch.int64, device=self.device
            ),
            "ptr": torch.tensor(
                [0, natoms], dtype=torch.long, device=self.device
            ),
            "total_charge": torch.zeros(
                1, dtype=self.dtype, device=self.device
            ),
            "total_spin": torch.zeros(
                1, dtype=self.dtype, device=self.device
            ),
            "lammps_class": data,
            "natoms": (natoms, ntotal),
        }

    def _update_lammps_data(self, data, atom_energies, pair_forces, natoms):
        """Write computed energies and forces back to LAMMPS data structures.

        Only sums energies over real atoms (first natoms), not ghost atoms.
        """
        if self.dtype == torch.float32:
            pair_forces = pair_forces.double()

        # Squeeze trailing dim from node_energy if present (e.g. from squeeze(-1) in scatter)
        if atom_energies.dim() > 1:
            atom_energies = atom_energies.squeeze(-1)

        eatoms = torch.as_tensor(data.eatoms)
        eatoms.copy_(atom_energies[:natoms])
        data.energy = torch.sum(atom_energies[:natoms])
        data.update_pair_forces_gpu(pair_forces)

    def _manage_profiling(self):
        """Start/stop CUDA profiler at specified steps."""
        if not self.config.debug_profile:
            return

        if self.step == self.config.profile_start_step:
            logging.info(f"Starting CUDA profiler at step {self.step}")
            torch.cuda.profiler.start()

        if self.step == self.config.profile_end_step:
            logging.info(f"Stopping CUDA profiler at step {self.step}")
            torch.cuda.profiler.stop()
            logging.info("Profiling complete. Exiting.")
            sys.exit()

    def compute_descriptors(self, data):
        pass

    def compute_gradients(self, data):
        pass
