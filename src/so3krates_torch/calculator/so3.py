###########################################################################################

# Based on the MACE package: https://github.com/ACEsuit/mace

###########################################################################################

from typing import Optional, Union, List
from pathlib import Path
from glob import glob
import torch
import yaml
from ase.calculators.calculator import Calculator, all_changes
from ase.stress import full_3x3_to_voigt_6_stress
import numpy as np
from so3krates_torch.data.atomic_data import AtomicData as So3Data
from so3krates_torch.tools import torch_geometric, torch_tools, utils
from so3krates_torch.data.utils import KeySpecification, config_from_atoms
from so3krates_torch.modules.models import SO3LR
import importlib.resources as resources


# Maps a pretrained-model keyword to its (subdirectory, file stem)
# under ``pretrained/so3lr/`` -- e.g. "v2-s" -> so3lr/v2/so3lr-s.pt +
# so3lr/v2/so3lr-s_settings.yaml. "v1" is the original architecture
# (single checkpoint, no size variants); "v2-{s,m,l}" are the three
# sizes of the newer architecture.
_PRETRAINED_SO3LR_MODELS = {
    "v1": ("v1", "so3lr"),
    "v2-s": ("v2", "so3lr-s"),
    "v2-m": ("v2", "so3lr-m"),
    "v2-l": ("v2", "so3lr-l"),
}


def load_pretrained_so3lr(
    model: str,
    device: str = "cpu",
    **architecture_overrides,
) -> torch.nn.Module:
    """Build a bundled pretrained SO3LR model from its converted
    ``state_dict`` + ``settings.yaml`` (``pretrained/so3lr/{v1,v2}/``).

    ``model`` selects which checkpoint to load: ``"v1"`` (the original,
    single checkpoint) or ``"v2-s"``/``"v2-m"``/``"v2-l"`` (the three
    sizes of the newer v2 architecture).

    ``architecture_overrides`` are merged into the checkpoint's saved
    ``ARCHITECTURE`` settings before construction -- e.g. pass
    ``use_pme=True, pme_smearing=..., pme_mesh_spacing=...`` to enable
    PME electrostatics/dispersion (every saved checkpoint has PME off
    by default). PME introduces zero new *learned* parameters (only
    ``torch-pme``'s own internal, non-learned buffers --
    ``smearing``/``prefactor``/``exponent``), so the state_dict is
    loaded with ``strict=False`` and only those specific buffer keys
    are allowed to be missing; anything else missing or any unexpected
    key raises.
    """
    if model not in _PRETRAINED_SO3LR_MODELS:
        raise ValueError(
            f"Unknown pretrained SO3LR model {model!r}. Must be one "
            f"of {sorted(_PRETRAINED_SO3LR_MODELS)}."
        )
    subdir, stem = _PRETRAINED_SO3LR_MODELS[model]
    base = resources.files("so3krates_torch.pretrained.so3lr") / subdir

    with resources.as_file(base / f"{stem}_settings.yaml") as yaml_path:
        with open(yaml_path) as f:
            settings = yaml.safe_load(f)["ARCHITECTURE"]
    settings = dict(settings)
    settings.pop("device", None)
    settings.update(architecture_overrides)

    torch_model = SO3LR(**settings)

    with resources.as_file(base / f"{stem}.pt") as pt_path:
        state_dict = torch.load(
            pt_path, map_location=device, weights_only=False
        )

    missing, unexpected = torch_model.load_state_dict(state_dict, strict=False)
    if unexpected:
        raise RuntimeError(
            f"Unexpected keys loading pretrained {model!r} checkpoint: "
            f"{unexpected}"
        )
    bad_missing = [k for k in missing if "pme_" not in k]
    if bad_missing:
        raise RuntimeError(
            f"Missing non-PME keys loading pretrained {model!r} "
            f"checkpoint: {bad_missing}"
        )
    torch_model.to(device)
    return torch_model


def get_model_dtype(model: torch.nn.Module) -> torch.dtype:
    """Get the dtype of the model"""
    model_dtype = next(model.parameters()).dtype
    if model_dtype == torch.float64:
        return "float64"
    if model_dtype == torch.float32:
        return "float32"
    raise ValueError(f"Unknown dtype {model_dtype}")


class TorchkratesCalculator(Calculator):
    """Calculator for Torchkrates models"""

    def __init__(
        self,
        model_paths: Union[list, str, None] = None,
        models: Union[List[torch.nn.Module], torch.nn.Module, None] = None,
        r_max_lr: Optional[float] = None,
        dispersion_energy_cutoff_lr_damping: Optional[float] = None,
        compute_stress: bool = False,
        device: str = "cpu",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="",
        charges_key="charges",
        key_specification: Optional[KeySpecification] = None,
        model_type="so3lr",
        fullgraph: Optional[bool] = None,
        compile: bool = False,
        theory_level: Optional[int] = None,
        **kwargs,
    ):
        Calculator.__init__(self, **kwargs)
        self.theory_level = theory_level
        if (model_paths is None) == (models is None):
            raise ValueError(
                "Exactly one of 'model_paths' or 'models' must be provided"
            )
        self.results = {}
        if key_specification is None:
            arrays_keys = {"forces": "REF_forces", "charges": "REF_charges"}

            info_keys = {
                "energy": "REF_energy",
                "stress": "REF_stress",
                "dipole": "REF_dipole",
                "polarizability": "REF_polarizability",
                "head": "REF_head",
                "total_charge": "charge",
                "total_spin": "total_spin",
            }

            self.key_specification = KeySpecification(
                info_keys=info_keys, arrays_keys=arrays_keys
            )
        else:
            self.key_specification = key_specification

        self.compute_stress = compute_stress

        self.model_type = model_type.lower()

        if self.model_type == "so3lr":
            self.implemented_properties = [
                "energy",
                "forces",
                "stress",
                "partial_charges",
                "hirshfeld_ratios",
                "dipole",
                "descriptors",
            ]
        else:
            self.implemented_properties = [
                "energy",
                "forces",
                "stress",
                "descriptors",
            ]

        if model_paths is not None:
            if isinstance(model_paths, str):
                # Find all models that satisfy the wildcard (e.g. model_*.pt)
                model_paths_glob = glob(model_paths)

                if len(model_paths_glob) == 0:
                    raise ValueError(
                        f"Couldn't find model files: {model_paths}"
                    )
                model_paths = model_paths_glob

            elif isinstance(model_paths, Path):
                model_paths = [model_paths]

            if len(model_paths) == 0:
                raise ValueError(f"No model files found in {model_paths}")

            self.num_models = len(model_paths)

            self.models = [
                torch.load(
                    f=model_path, map_location=device, weights_only=False
                )
                for model_path in model_paths
            ]

        elif models is not None:
            if not isinstance(models, list):
                models = [models]

            if len(models) == 0:
                raise ValueError("No models supplied")

            self.models = models
            self.num_models = len(models)

        if self.num_models > 1:
            print(f"Running committee with {self.num_models} models")

            if self.model_type in ["so3lr", "so3krates"]:
                self.implemented_properties.extend(
                    ["energies", "energy_var", "forces_comm", "stress_var"]
                )
            elif self.model_type == "so3lr":
                self.implemented_properties.extend(
                    ["dipole_var", "hirshfeld_var", "partial_charges_var"]
                )

        if self.model_type == "so3lr":
            for model in self.models:
                if dispersion_energy_cutoff_lr_damping is not None:
                    model.dispersion_energy_cutoff_lr_damping = (
                        dispersion_energy_cutoff_lr_damping
                    )
                # else: keep the value saved in the model checkpoint

        for model in self.models:
            model.to(device)
        r_maxs = [model.r_max for model in self.models]
        r_maxs = np.array(r_maxs)
        if not np.all(r_maxs == r_maxs[0]):
            raise ValueError(
                f"committee r_max are not all the same {' '.join(r_maxs)}"
            )
        self.r_max = float(r_maxs[0])
        self.r_max_lr = r_max_lr
        for model in self.models:
            if r_max_lr is not None and getattr(model, "use_lr", False):
                model.r_max_lr = r_max_lr

        self.device = torch_tools.init_device(device)
        self.energy_units_to_eV = energy_units_to_eV
        self.length_units_to_A = length_units_to_A
        self.z_table = utils.AtomicNumberTable([int(z) for z in range(1, 119)])
        self.charges_key = charges_key

        model_dtype = get_model_dtype(self.models[0])
        if default_dtype == "":
            print(
                f"No dtype selected, switching to {model_dtype} to match model dtype."
            )
            default_dtype = model_dtype
        if model_dtype != default_dtype:
            print(
                f"Default dtype {default_dtype} does not match model dtype {model_dtype}, converting models to {default_dtype}."
            )
            if default_dtype == "float64":
                self.models = [model.double() for model in self.models]
            elif default_dtype == "float32":
                self.models = [model.float() for model in self.models]
        torch_tools.set_default_dtype(default_dtype)

        for model in self.models:
            for param in model.parameters():
                param.requires_grad = False

        if fullgraph is None:
            # fullgraph=True is only safe on CUDA: on CPU, compiling the
            # long-range (ZBL/electrostatics/dispersion) scatter-sums
            # currently crashes Inductor's C++ backend regardless of
            # fullgraph, so keep the permissive default there.
            fullgraph = self.device.type == "cuda"

        if compile:
            if self.device.type != "cuda" and any(
                getattr(model, "use_lr", False) for model in self.models
            ):
                raise RuntimeError(
                    "torch.compile is not supported on this device "
                    f"({self.device}) for long-range SO3LR models: "
                    "PyTorch's Inductor CPU backend crashes while "
                    "compiling the ZBL/electrostatics/dispersion "
                    "scatter-reductions (a known upstream Inductor "
                    "limitation, not something so3krates-torch can fix). "
                    "This has only been confirmed working with "
                    "torch.compile on CUDA. Pass compile=False, or run "
                    "on a CUDA device."
                )
            self.models = [
                torch.compile(model, dynamic=True, fullgraph=fullgraph)
                for model in self.models
            ]

    def _create_result_tensors(
        self, model_type: str, num_models: int, num_atoms: int
    ) -> dict:
        dict_of_tensors = {}
        if model_type in ["so3lr", "so3krates"]:
            energies = torch.zeros(num_models, device=self.device)
            node_energy = torch.zeros(
                num_models, num_atoms, device=self.device
            )
            forces = torch.zeros(num_models, num_atoms, 3, device=self.device)
            stress = torch.zeros(num_models, 3, 3, device=self.device)
            dict_of_tensors.update(
                {
                    "energies": energies,
                    "node_energy": node_energy,
                    "forces": forces,
                    "stress": stress,
                }
            )
        if model_type in ["so3lr"]:
            dipole = torch.zeros(num_models, 3, device=self.device)
            partial_charges = torch.zeros(
                num_models, num_atoms, device=self.device
            )
            hirshfeld_ratios = torch.zeros(
                num_models, num_atoms, device=self.device
            )
            dict_of_tensors.update(
                {
                    "dipole": dipole,
                    "partial_charges": partial_charges,
                    "hirshfeld_ratios": hirshfeld_ratios,
                }
            )
        return dict_of_tensors

    def _atoms_to_batch(self, atoms, batch_size=1):
        self.key_specification.update(arrays_keys={self.charges_key: "Qs"})

        config = config_from_atoms(
            atoms, key_specification=self.key_specification
        )
        data_loader = torch_geometric.dataloader.DataLoader(
            dataset=[
                So3Data.from_config(
                    config,
                    z_table=self.z_table,
                    cutoff=self.r_max,
                    cutoff_lr=(self.r_max_lr),
                )
            ],
            batch_size=batch_size,
            shuffle=False,
            drop_last=False,
        )
        batch = next(iter(data_loader)).to(self.device)
        return batch

    def _clone_batch(self, batch):
        batch_clone = batch.clone()
        return batch_clone

    # pylint: disable=dangerous-default-value
    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes
    ):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        batch_base = self._atoms_to_batch(atoms)

        ret_tensors = self._create_result_tensors(
            self.model_type, self.num_models, len(atoms)
        )
        for i, model in enumerate(self.models):
            batch = self._clone_batch(batch_base)
            batch.positions.requires_grad_(True)
            batch_dict = batch.to_dict()
            if self.theory_level is not None:
                n_graphs = int(batch_dict["batch"].max().item()) + 1
                batch_dict["theory_level"] = torch.full(
                    (n_graphs,),
                    self.theory_level,
                    dtype=torch.long,
                    device=self.device,
                )
            out = model(
                batch_dict,
                compute_stress=self.compute_stress,
            )
            if self.model_type in ["so3lr", "so3krates"]:
                ret_tensors["energies"][i] = out["energy"].detach()
                ret_tensors["forces"][i] = out["forces"].detach()
                if out["stress"] is not None:
                    ret_tensors["stress"][i] = out["stress"].detach()

            if self.model_type in ["so3lr"]:
                if out["dipole"] is not None:
                    ret_tensors["dipole"][i] = out["dipole"].detach()
                if out["partial_charges"] is not None:
                    ret_tensors["partial_charges"][i] = out[
                        "partial_charges"
                    ].detach()
                if out["hirshfeld_ratios"] is not None:
                    ret_tensors["hirshfeld_ratios"][i] = out[
                        "hirshfeld_ratios"
                    ].detach()
        self._process_results(
            ret_tensors=ret_tensors,
            out=out,
            multi_output=self.num_models > 1,
        )

    def get_hessian(self, atoms=None):
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms

        batch = self._atoms_to_batch(atoms)
        hessians = []
        for model in self.models:
            batch_clone = self._clone_batch(batch)
            batch_clone.positions.requires_grad_(True)
            hessians.append(
                model(
                    batch_clone.to_dict(),
                    compute_hessian=True,
                    compute_stress=False,
                    training=False,
                )["hessian"]
            )
        hessians = [hessian.detach().cpu().numpy() for hessian in hessians]
        if self.num_models == 1:
            return hessians[0]
        return hessians

    def get_descriptors(self, atoms=None, invariants_only=True):
        if atoms is None and self.atoms is None:
            raise ValueError("atoms not set")
        if atoms is None:
            atoms = self.atoms

        batch = self._atoms_to_batch(atoms)
        descriptors = [
            model.get_representation(batch.to_dict()) for model in self.models
        ]
        invariants = [inv.detach().cpu().numpy() for (inv, ev) in descriptors]
        equivariants = [ev.detach().cpu().numpy() for (inv, ev) in descriptors]

        if invariants_only:
            return {
                "invariant_features": invariants,
            }
        else:
            return {
                "invariant_features": invariants,
                "equivariant_features": equivariants,
            }

    def _process_results(
        self, ret_tensors: dict, out: dict, multi_output: bool = False
    ):
        self.results = {}
        if self.model_type in ["so3lr", "so3krates"]:
            self.results["energy"] = (
                torch.mean(ret_tensors["energies"], dim=0).cpu().item()
                * self.energy_units_to_eV
            )
            self.results["free_energy"] = self.results["energy"]
            self.results["forces"] = (
                torch.mean(ret_tensors["forces"], dim=0).cpu().numpy()
                * self.energy_units_to_eV
                / self.length_units_to_A
            )
            if multi_output:
                self.results["energies"] = (
                    ret_tensors["energies"].cpu().numpy()
                    * self.energy_units_to_eV
                )
                self.results["energy_var"] = (
                    torch.var(ret_tensors["energies"], dim=0, unbiased=False)
                    .cpu()
                    .item()
                    * self.energy_units_to_eV
                )
                self.results["forces_comm"] = (
                    ret_tensors["forces"].cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A
                )
            if out["stress"] is not None:
                self.results["stress"] = full_3x3_to_voigt_6_stress(
                    torch.mean(ret_tensors["stress"], dim=0).cpu().numpy()
                    * self.energy_units_to_eV
                    / self.length_units_to_A**3
                )
                if multi_output:
                    self.results["stress_var"] = full_3x3_to_voigt_6_stress(
                        torch.var(ret_tensors["stress"], dim=0, unbiased=False)
                        .cpu()
                        .numpy()
                        * self.energy_units_to_eV
                        / self.length_units_to_A**3
                    )

        if self.model_type in ["so3lr"]:
            self.results["dipole"] = (
                torch.mean(ret_tensors["dipole"], dim=0).cpu().numpy()
            )
            self.results["partial_charges"] = (
                torch.mean(ret_tensors["partial_charges"], dim=0).cpu().numpy()
            )
            self.results["hirshfeld_ratios"] = (
                torch.mean(ret_tensors["hirshfeld_ratios"], dim=0)
                .cpu()
                .numpy()
            )

            if multi_output:
                self.results["dipole_var"] = (
                    torch.var(ret_tensors["dipole"], dim=0, unbiased=False)
                    .cpu()
                    .numpy()
                )
                self.results["partial_charges_var"] = (
                    torch.var(
                        ret_tensors["partial_charges"], dim=0, unbiased=False
                    )
                    .cpu()
                    .numpy()
                )
                self.results["hirshfeld_ratios_var"] = (
                    torch.var(
                        ret_tensors["hirshfeld_ratios"], dim=0, unbiased=False
                    )
                    .cpu()
                    .numpy()
                )


class SO3LRCalculator(TorchkratesCalculator):
    """Calculator for the bundled pretrained SO3LR models.

    ``model`` selects which of the bundled, converted checkpoints to
    load: ``"v1"`` (the original architecture, single checkpoint) or
    ``"v2-s"``/``"v2-m"``/``"v2-l"`` (the three sizes of the newer v2
    architecture). See ``load_pretrained_so3lr`` for how the
    checkpoint is resolved and constructed.
    """

    def __init__(
        self,
        model: str,
        r_max_lr: Optional[float] = 12.0,
        dispersion_energy_cutoff_lr_damping: Optional[float] = 2.0,
        compute_stress: bool = False,
        device: str = "cpu",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="",
        charges_key="Qs",
        key_specification=None,
        theory_level: int = 0,
        use_pme: Optional[bool] = None,
        pme_smearing: Optional[float] = None,
        pme_mesh_spacing: Optional[float] = None,
        use_pme_dispersion: Optional[bool] = None,
        pme_dispersion_smearing: Optional[float] = None,
        pme_dispersion_mesh_spacing: Optional[float] = None,
        **kwargs,
    ):
        architecture_overrides = {
            k: v
            for k, v in {
                "use_pme": use_pme,
                "pme_smearing": pme_smearing,
                "pme_mesh_spacing": pme_mesh_spacing,
                "use_pme_dispersion": use_pme_dispersion,
                "pme_dispersion_smearing": pme_dispersion_smearing,
                "pme_dispersion_mesh_spacing": pme_dispersion_mesh_spacing,
            }.items()
            if v is not None
        }
        models = [
            load_pretrained_so3lr(
                model, device=device, **architecture_overrides
            )
        ]
        super().__init__(
            model_paths=None,
            models=models,
            r_max_lr=r_max_lr,
            dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
            compute_stress=compute_stress,
            device=device,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
            default_dtype=default_dtype,
            charges_key=charges_key,
            key_specification=key_specification,
            model_type="so3lr",
            theory_level=theory_level,
            **kwargs,
        )


class MultiHeadSO3LRCalculator(TorchkratesCalculator):
    """Calculator for Multi-Head SO3LR models"""

    def __init__(
        self,
        model_path: Union[str, None] = None,
        model: Union[torch.nn.Module, None] = None,
        r_max_lr: Optional[float] = None,
        dispersion_energy_cutoff_lr_damping: Optional[float] = None,
        compute_stress: bool = False,
        device: str = "cpu",
        energy_units_to_eV: float = 1.0,
        length_units_to_A: float = 1.0,
        default_dtype="",
        charges_key="Qs",
        key_specification=None,
        **kwargs,
    ):
        super().__init__(
            model_paths=model_path,
            models=model,
            r_max_lr=r_max_lr,
            dispersion_energy_cutoff_lr_damping=dispersion_energy_cutoff_lr_damping,
            compute_stress=compute_stress,
            device=device,
            energy_units_to_eV=energy_units_to_eV,
            length_units_to_A=length_units_to_A,
            default_dtype=default_dtype,
            charges_key=charges_key,
            key_specification=key_specification,
            model_type="so3lr",
            **kwargs,
        )
        self.num_heads = self.models[0].num_heads
        self.models[0].select_heads = False

        assert (
            self.num_models == 1
        ), "MultiHeadSO3LRCalculator expects a single multi-head model."

    def calculate(
        self, atoms=None, properties=None, system_changes=all_changes
    ):
        """
        Calculate properties.
        :param atoms: ase.Atoms object
        :param properties: [str], properties to be computed, used by ASE internally
        :param system_changes: [str], system changes since last calculation, used by ASE internally
        :return:
        """
        # call to base-class to set atoms attribute
        Calculator.calculate(self, atoms)

        batch_base = self._atoms_to_batch(atoms)
        ret_tensors = self._create_result_tensors(
            self.model_type, self.num_heads, len(atoms)
        )
        model = self.models[0]
        batch = self._clone_batch(batch_base)
        batch.positions.requires_grad_(True)
        out = model(
            batch.to_dict(),
            compute_stress=self.compute_stress,
        )
        if self.model_type in ["so3lr", "so3krates"]:
            ret_tensors["energies"] = out["energy"].detach()
            ret_tensors["forces"] = out["forces"].detach()
            if out["stress"] is not None:
                ret_tensors["stress"] = out["stress"].detach()

        if self.model_type in ["so3lr"]:
            if out["dipole"] is not None:
                ret_tensors["dipole"] = out["dipole"].detach()
            if out["partial_charges"] is not None:
                ret_tensors["partial_charges"] = out[
                    "partial_charges"
                ].detach()
            if out["hirshfeld_ratios"] is not None:
                ret_tensors["hirshfeld_ratios"] = out[
                    "hirshfeld_ratios"
                ].detach()

        self._process_results(ret_tensors, out, multi_output=True)

    def _create_result_tensors(
        self, model_type: str, num_heads: int, num_atoms: int
    ) -> dict:
        dict_of_tensors = {}
        if model_type in ["so3lr", "so3krates"]:
            energies = torch.zeros(num_heads, device=self.device)
            node_energy = torch.zeros(num_heads, num_atoms, device=self.device)
            forces = torch.zeros(num_heads, num_atoms, 3, device=self.device)
            stress = torch.zeros(num_heads, 3, 3, device=self.device)
            dict_of_tensors.update(
                {
                    "energies": energies,
                    "node_energy": node_energy,
                    "forces": forces,
                    "stress": stress,
                }
            )
        if model_type in ["so3lr"]:
            dipole = torch.zeros(3, device=self.device)
            partial_charges = torch.zeros(num_atoms, device=self.device)
            hirshfeld_ratios = torch.zeros(num_atoms, device=self.device)
            dict_of_tensors.update(
                {
                    "dipole": dipole,
                    "partial_charges": partial_charges,
                    "hirshfeld_ratios": hirshfeld_ratios,
                }
            )
        return dict_of_tensors

    def get_hessian(self, atoms=None):
        raise NotImplementedError(
            "Hessian calculation is not implemented for Multi-Head SO3LR models."
        )
