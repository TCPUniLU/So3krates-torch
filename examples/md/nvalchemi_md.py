"""GPU-native NVT/NPT MD with NVAlchemi — programmatic example.

Requires: pip install ".[nvalchemi]"

For most runs the ``torchkrates-md-nvalchemi`` CLI is enough; this script shows
the underlying API and, crucially, the TWO ways to handle long-range
electrostatics. Pick the one that matches how your model was TRAINED:

  electrostatics = "model"  (default)
      The model computes its own electrostatics, exactly as trained:
        * damped erf-Coulomb  if trained with use_pme: false
        * torch-pme           if trained with use_pme: true
      PME parameters are baked into the model (training YAML: pme_smearing [Å],
      pme_mesh_spacing [Å]) and are NOT set here. Use this for pre-trained SO3LR.

  electrostatics = "nvalchemi"
      The model emits partial charges only; NVAlchemi's GPU-native PME
      (full Ewald) supplies the Coulomb energy. Configured here via
      build_nvalchemi_pme_model(...). This is a DIFFERENT backend than the
      model's own term, so only use it with a model trained for full Ewald
      electrostatics (use_pme: true). NOTE: NVAlchemi parameterizes the smearing
      as the Ewald split `alpha` [1/Å] (alpha ~= 1/(sqrt(2)*pme_smearing)), not
      a length like torch-pme's pme_smearing.

Dispersion is always computed by the model (NVAlchemi has no native dispersion
PME). NPT requires a periodic cell.
"""

import argparse

import torch
from ase.io import read as ase_read

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.hooks.neighbor_list import NeighborListHook

from so3krates_torch.calculator.nvalchemi_so3 import (
    NVAlchemiSO3LR,
    build_nvalchemi_pme_model,
)


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help="trained model .pt")
    p.add_argument("--atoms", required=True, help="periodic structure file")
    p.add_argument("--ensemble", choices=["nvt", "npt"], default="nvt")
    p.add_argument(
        "--electrostatics",
        choices=["model", "nvalchemi", "none"],
        default="model",
    )
    p.add_argument("--steps", type=int, default=1000)
    p.add_argument("--device", default="cuda")
    args = p.parse_args()

    dtype = torch.float64

    # --- Load and wrap the model -------------------------------------------
    raw = torch.load(args.model, map_location=args.device, weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        raw = raw["model"]

    if args.electrostatics == "nvalchemi":
        # SR model emits charges -> NVAlchemi-native PME supplies Coulomb.
        # Tune PME here (NVAlchemi convention):
        model = build_nvalchemi_pme_model(
            raw,
            mesh_spacing=0.8,  # FFT mesh spacing [Å]
            alpha=None,  # Ewald split [1/Å]; None = auto-estimate
            spline_order=4,
            accuracy=1e-6,
        )
    else:
        if args.electrostatics == "none":
            raw.electrostatic_energy_bool = False
        model = NVAlchemiSO3LR(raw, electrostatics="model")
    model = model.to(device=args.device, dtype=dtype).eval()

    # --- Build a (single-system) batch -------------------------------------
    atoms = ase_read(args.atoms)
    n = len(atoms)
    data = AtomicData.from_atoms(
        atoms, device=args.device, dtype=dtype
    ).model_copy(
        update={
            "forces": torch.zeros(n, 3, dtype=dtype, device=args.device),
            "energy": torch.zeros(1, 1, dtype=dtype, device=args.device),
            "stress": torch.zeros(1, 3, 3, dtype=dtype, device=args.device),
        }
    )
    batch = Batch.from_data_list([data])

    # --- Neighbor list rebuilt each step on the GPU ------------------------
    hooks = [
        NeighborListHook(
            config=model.model_config.neighbor_config,
            skin=0.5,
            stage=DynamicsStage.BEFORE_COMPUTE,
        )
    ]

    # --- Integrator (dt / time constants in fs) ----------------------------
    if args.ensemble == "nvt":
        integrator = NVTLangevin(
            model=model,
            dt=0.5,
            temperature=300.0,
            friction=1e-2,
            n_steps=args.steps,
            hooks=hooks,
        )
    else:
        integrator = NPT(
            model=model,
            dt=0.5,
            temperature=300.0,
            pressure=1.0 * 6.2415e-7,  # 1 bar -> eV/Å³
            barostat_time=2000.0,
            thermostat_time=500.0,
            n_steps=args.steps,
            hooks=hooks,
        )

    batch = integrator.run(batch)
    print("final energy [eV]:", float(batch.energy.reshape(-1)[0]))


if __name__ == "__main__":
    main()
