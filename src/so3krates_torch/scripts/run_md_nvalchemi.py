"""NVT / NPT molecular dynamics using NVAlchemi + So3krates-torch.

Runs GPU-native MD with a trained So3krates / SO3LR model.  The neighbor
list is managed automatically by NVAlchemi's NeighborListHook.

Usage (NVT)::

    python run_md_nvalchemi.py \\
        --model  model.pt \\
        --atoms  NaCl.xyz \\
        --ensemble nvt \\
        --temperature 300 \\
        --steps 10000

Usage (NPT)::

    python run_md_nvalchemi.py \\
        --model  model.pt \\
        --atoms  NaCl.xyz \\
        --ensemble npt \\
        --temperature 300 \\
        --pressure 1.0 \\
        --steps 10000

CSV log columns:
    step, graph_idx, status, energy (eV), fmax (eV/Å), temperature (K),
    volume_A3 (Å³), density_g_cm3 (g/cm³), elapsed_s (s), s_per_step (s)
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from ase.io import read as ase_read

from nvalchemi.data.atomic_data import AtomicData
from nvalchemi.data.batch import Batch
from nvalchemi.dynamics.base import DynamicsStage
from nvalchemi.dynamics import initialize_velocities
from nvalchemi.dynamics.hooks.logging import LoggingHook, temperature_per_graph
from nvalchemi.dynamics.integrators.npt import NPT
from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
from nvalchemi.hooks.neighbor_list import NeighborListHook

from so3krates_torch.calculator.nvalchemi_so3 import (
    NVAlchemiSO3LR,
    build_nvalchemi_pme_model,
)

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------
AMU_TO_G_PER_CM3 = 1.66054  # (amu/Å³) → g/cm³
BAR_TO_EV_PER_A3 = 6.2415e-7  # bar → eV/Å³


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="NVT / NPT MD with So3krates-torch + NVAlchemi"
    )
    p.add_argument(
        "--model", required=True, help="Path to trained model (.pt)"
    )
    p.add_argument(
        "--atoms", required=True, help="Input structure (ASE-readable)"
    )
    p.add_argument(
        "--index",
        default="-1",
        help="ASE frame selection; e.g. '-1' (last frame, default) or ':' "
        "to run every frame as an independent system in one batch.",
    )
    p.add_argument(
        "--ensemble",
        choices=["nvt", "npt"],
        default="nvt",
        help="Ensemble: nvt (Langevin) or npt (MTK barostat, default: nvt)",
    )
    p.add_argument(
        "--temperature", type=float, default=300.0, help="Temperature [K]"
    )
    p.add_argument(
        "--pressure",
        type=float,
        default=1.0,
        help="Pressure [bar], NPT only (default: 1.0 bar)",
    )
    p.add_argument(
        "--dt",
        type=float,
        default=0.5,
        help="Time step [fs] (default: 0.5 fs)",
    )
    p.add_argument(
        "--steps",
        type=int,
        default=10_000,
        help="Number of MD steps (default: 10000)",
    )
    p.add_argument(
        "--log-freq",
        type=int,
        default=100,
        help="Log every N steps (default: 100)",
    )
    p.add_argument(
        "--log-file",
        default="md_log.csv",
        help="Output CSV file (default: md_log.csv)",
    )
    p.add_argument(
        "--device",
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device: cuda or cpu (default: cuda if available)",
    )
    p.add_argument(
        "--dtype",
        choices=["float32", "float64"],
        default="float64",
        help="Floating-point precision (default: float64)",
    )
    p.add_argument(
        "--skin",
        type=float,
        default=0.5,
        help="Verlet skin distance [Å] (default: 0.5 Å)",
    )
    p.add_argument(
        "--friction",
        type=float,
        default=1e-2,
        help="Langevin friction coefficient [1/fs] (default: 1e-2, NVT only)",
    )
    p.add_argument(
        "--barostat-time",
        type=float,
        default=2000.0,
        help="NPT barostat time constant [fs] (default: 2000 fs)",
    )
    p.add_argument(
        "--thermostat-time",
        type=float,
        default=500.0,
        help="NPT thermostat time constant [fs] (default: 500 fs)",
    )
    p.add_argument(
        "--pressure-coupling",
        choices=["isotropic", "anisotropic", "triclinic"],
        default="isotropic",
        help="NPT pressure coupling mode (default: isotropic)",
    )
    p.add_argument(
        "--electrostatics",
        choices=["model", "nvalchemi", "none"],
        default="model",
        # "model"     -> the model's own electrostatics: real-space
        #                erf-Coulomb (use_pme=False) or torch-pme
        #                (use_pme=True), per how the model was configured.
        # "nvalchemi" -> NVAlchemi-native GPU PME (full Ewald) from the
        #                model's partial charges.
        # "none"      -> long-range electrostatics disabled.
        help="Long-range electrostatics backend (default: the model's own)",
    )
    p.add_argument(
        "--pme-mesh-spacing",
        type=float,
        default=1.0,
        help="NVAlchemi PME target mesh spacing [Å] "
        "(--electrostatics nvalchemi only; default: 1.0)",
    )
    p.add_argument(
        "--pme-alpha",
        type=float,
        default=None,
        help="NVAlchemi PME Ewald splitting / smearing [1/Å] "
        "(--electrostatics nvalchemi only; default: auto-estimated from "
        "--pme-accuracy)",
    )
    p.add_argument(
        "--pme-spline-order",
        type=int,
        default=4,
        help="NVAlchemi PME B-spline interpolation order "
        "(--electrostatics nvalchemi only; default: 4)",
    )
    p.add_argument(
        "--pme-accuracy",
        type=float,
        default=1e-6,
        help="NVAlchemi PME target accuracy for auto parameter estimation "
        "(--electrostatics nvalchemi only; default: 1e-6)",
    )
    p.add_argument(
        "--skip-vel-init",
        action="store_true",
        help="Skip Maxwell-Boltzmann velocity initialization. Use when the "
        "input structure already has velocities (e.g. from a restart file). "
        "Default: initialize velocities from MB at --temperature.",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Compile the inner model with torch.compile(dynamic=True). "
        "Adds ~30-60 s first-step overhead, then typically 2-4× speedup "
        "for large systems. Remaining intended graph-breaks: "
        "torch.autograd.grad (forces/stress) and the model's own "
        "torch-pme per-graph loop when use_pme=True. "
        "The default erf-Coulomb path and NVAlchemi-native PME "
        "(--electrostatics nvalchemi) are unaffected.",
    )
    p.add_argument(
        "--electrostatic-energy-scale",
        type=float,
        default=None,
        help="Override the electrostatic energy scale factor stored in the "
        "model (e.g. 4.0). Default: use the value saved with the model.",
    )
    p.add_argument(
        "--dispersion-energy-scale",
        type=float,
        default=None,
        help="Override the dispersion energy scale factor stored in the "
        "model (e.g. 1.2). Default: use the value saved with the model.",
    )
    p.add_argument(
        "--dispersion-energy-cutoff-lr-damping",
        type=float,
        default=None,
        help="Override the dispersion LR cutoff damping (Å) stored in the "
        "model (e.g. 2.0). Default: use the value saved with the model.",
    )
    return p.parse_args()


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------


def load_model(
    path: str,
    device: str,
    dtype: torch.dtype,
    electrostatics: str = "model",
    pme_mesh_spacing: float = 1.0,
    pme_alpha: float = None,
    pme_spline_order: int = 4,
    pme_accuracy: float = 1e-6,
    compile_model: bool = False,
    electrostatic_energy_scale: float = None,
    dispersion_energy_scale: float = None,
    dispersion_energy_cutoff_lr_damping: float = None,
):
    """Load a trained model and wrap it for NVAlchemi MD.

    Returns a NVAlchemi ``BaseModelMixin`` — either a bare ``NVAlchemiSO3LR``
    (``electrostatics`` in {"model", "none"}) or a ``PipelineModelWrapper``
    composing the short-range model with NVAlchemi-native PME
    (``electrostatics="nvalchemi"``).
    """
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        raw = raw["model"]

    # Apply scale overrides BEFORE torch.compile so Inductor sees the
    # final float values (it bakes Python-float attributes as constants).
    if electrostatic_energy_scale is not None:
        raw.electrostatic_energy_scale = electrostatic_energy_scale
    if dispersion_energy_scale is not None:
        raw.dispersion_energy_scale = dispersion_energy_scale
    if dispersion_energy_cutoff_lr_damping is not None:
        raw.dispersion_energy_cutoff_lr_damping = (
            dispersion_energy_cutoff_lr_damping
        )

    # Print a diagnostic so there's no ambiguity about what is active.
    _print_model_diagnostics(raw)

    if compile_model:
        # Compile the inner So3krates/SO3LR forward pass before NVAlchemi
        # wrapping.  dynamic=True handles varying neighbor-list sizes.
        raw = torch.compile(raw, dynamic=True)

    if electrostatics == "nvalchemi":
        model = build_nvalchemi_pme_model(
            raw,
            mesh_spacing=pme_mesh_spacing,
            alpha=pme_alpha,
            spline_order=pme_spline_order,
            accuracy=pme_accuracy,
        )
    else:
        if electrostatics == "none":
            raw.electrostatic_energy_bool = False
        model = NVAlchemiSO3LR(raw, electrostatics="model")

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def _print_model_diagnostics(raw) -> None:
    """Print which long-range terms are active and their scale factors."""
    r_sr = getattr(raw, "r_max", None)
    r_lr = getattr(raw, "r_max_lr", None)
    elec_bool = getattr(raw, "electrostatic_energy_bool", False)
    elec_scale = getattr(raw, "electrostatic_energy_scale", None)
    disp_bool = getattr(raw, "dispersion_energy_bool", False)
    disp_scale = getattr(raw, "dispersion_energy_scale", None)
    disp_damp = getattr(raw, "dispersion_energy_cutoff_lr_damping", None)
    zbl_bool = getattr(raw, "zbl_repulsion_bool", False)

    def _yn(b):
        return "yes" if b else "NO"

    print(
        f"  SR cutoff: {r_sr} Å   LR cutoff: {r_lr} Å\n"
        f"  ZBL repulsion:  {_yn(zbl_bool)}\n"
        f"  Electrostatics: {_yn(elec_bool)}"
        + (f"  scale={elec_scale}" if elec_bool else "")
        + "\n"
        f"  Dispersion:     {_yn(disp_bool)}"
        + (
            f"  scale={disp_scale}  lr_damping={disp_damp}"
            if disp_bool
            else ""
        )
    )


# ---------------------------------------------------------------------------
# Hooks
# ---------------------------------------------------------------------------

_HDR_NVT = (
    f"{'step':>8}  {'sys':>4}  {'time/ps':>8}  {'energy/eV':>12}"
    f"  {'T/K':>8}  {'elapsed/s':>10}  {'ms/step':>8}"
)
_HDR_NPT = (
    f"{'step':>8}  {'sys':>4}  {'time/ps':>8}  {'energy/eV':>12}  {'T/K':>8}"
    f"  {'vol/Å³':>10}  {'dens/gcc':>9}  {'elapsed/s':>10}  {'ms/step':>8}"
)
_DIV_NVT = "-" * 78
_DIV_NPT = "-" * 96


class _PrintHook:
    """Prints a live table row to stdout at each log step."""

    def __init__(
        self,
        frequency: int,
        dt_fs: float,
        ensemble: str,
        start_time_ref: list[float],
    ) -> None:
        self.frequency = frequency
        self.stage = DynamicsStage.AFTER_STEP
        self._dt_fs = dt_fs
        self._npt = ensemble == "npt"
        self._start = start_time_ref
        self._header_done = False
        # start_time_ref[1] is set to the elapsed time at step-0 completion
        # (compile + first forward overhead) so ms/step excludes it.

    def _header(self) -> None:
        div = _DIV_NPT if self._npt else _DIV_NVT
        print(div)
        print(_HDR_NPT if self._npt else _HDR_NVT)
        print(div)
        self._header_done = True

    def __call__(self, ctx, stage) -> None:  # noqa: ARG002
        batch = ctx.batch
        step = int(ctx.step_count)
        elapsed = time.perf_counter() - self._start[0]

        # Step 0 includes compilation overhead — record it and skip the row.
        if step == 0:
            self._start[1] = elapsed
            return

        if not self._header_done:
            self._header()

        time_ps = step * self._dt_fs * 1e-3
        md_elapsed = elapsed - self._start[1]
        ms = md_elapsed / step * 1000
        B = int(batch.num_graphs)

        # Per-graph quantities (one row per system in the batch).
        energy = batch.energy.reshape(-1)  # [B]
        temps = temperature_per_graph(
            batch.velocities,
            batch.atomic_masses,
            batch.batch_idx,
            batch.num_graphs,
            batch.num_nodes_per_graph,
        )  # [B]

        if self._npt:
            cell = batch.cell
            vols = torch.abs(torch.linalg.det(cell))  # [B]
            mass = torch.zeros(B, dtype=cell.dtype, device=cell.device)
            mass.index_add_(0, batch.batch_idx, batch.atomic_masses)
            denss = mass / vols * AMU_TO_G_PER_CM3  # [B]

        for g in range(B):
            e = float(energy[g])
            t = float(temps[g])
            if self._npt:
                row = (
                    f"{step:>8d}  {g:>4d}  {time_ps:>8.3f}  {e:>12.4f}"
                    f"  {t:>8.2f}  {float(vols[g]):>10.2f}"
                    f"  {float(denss[g]):>9.4f}"
                    f"  {md_elapsed:>10.1f}  {ms:>8.1f}"
                )
            else:
                row = (
                    f"{step:>8d}  {g:>4d}  {time_ps:>8.3f}  {e:>12.4f}"
                    f"  {t:>8.2f}  {md_elapsed:>10.1f}  {ms:>8.1f}"
                )
            print(row, flush=True)


def _make_hooks(
    model: NVAlchemiSO3LR,
    skin: float,
    log_file: str,
    log_freq: int,
    dt_fs: float,
    ensemble: str,
    start_time_ref: list[float],
) -> list:
    nl_hook = NeighborListHook(
        config=model.model_config.neighbor_config,
        skin=skin,
        stage=DynamicsStage.BEFORE_COMPUTE,
    )

    def _volume(ctx):
        return torch.abs(torch.linalg.det(ctx.batch.cell))  # [B] Å³

    def _density(ctx):
        cell = ctx.batch.cell
        vol = torch.abs(torch.linalg.det(cell))  # [B] Å³
        mass = torch.zeros(
            ctx.batch.num_graphs,
            dtype=cell.dtype,
            device=cell.device,
        )
        mass.index_add_(0, ctx.batch.batch_idx, ctx.batch.atomic_masses)
        return mass / vol * AMU_TO_G_PER_CM3  # [B] g/cm³

    def _elapsed(ctx) -> float:
        return time.perf_counter() - start_time_ref[0]

    def _s_per_step(ctx) -> float:
        step = int(ctx.step_count)
        if step == 0:
            return 0.0
        md_elapsed = _elapsed(ctx) - start_time_ref[1]
        return md_elapsed / step

    log_hook = LoggingHook(
        backend="csv",
        log_path=log_file,
        frequency=log_freq,
        custom_scalars={
            "volume_A3": _volume,
            "density_g_cm3": _density,
            "elapsed_s": _elapsed,
            "s_per_step": _s_per_step,
        },
    )
    print_hook = _PrintHook(log_freq, dt_fs, ensemble, start_time_ref)
    return [nl_hook, log_hook, print_hook]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def main() -> None:
    args = _parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    print(f"Loading model from {args.model} …")
    model = load_model(
        args.model,
        args.device,
        dtype,
        electrostatics=args.electrostatics,
        pme_mesh_spacing=args.pme_mesh_spacing,
        pme_alpha=args.pme_alpha,
        pme_spline_order=args.pme_spline_order,
        pme_accuracy=args.pme_accuracy,
        compile_model=args.compile,
        electrostatic_energy_scale=args.electrostatic_energy_scale,
        dispersion_energy_scale=args.dispersion_energy_scale,
        dispersion_energy_cutoff_lr_damping=(
            args.dispersion_energy_cutoff_lr_damping
        ),
    )
    cutoff = model.model_config.neighbor_config.cutoff
    compile_note = "  compiled=yes (first step slow)" if args.compile else ""
    print(
        f"  electrostatics={args.electrostatics}  "
        f"neighbor cutoff={cutoff:.2f} Å{compile_note}"
    )

    print(f"Reading structure(s) from {args.atoms} (index={args.index}) …")
    frames = ase_read(args.atoms, index=args.index)
    if not isinstance(frames, list):  # single-frame selection
        frames = [frames]
    print(
        f"  {len(frames)} system(s), "
        f"{sum(len(a) for a in frames)} atoms total"
    )

    # Build one AtomicData per system; initialise forces/energy/stress to
    # zeros so compute() can write back in-place (filled at BEFORE_COMPUTE).
    if args.ensemble == "npt":
        non_periodic = [
            i for i, a in enumerate(frames) if not a.get_pbc().any()
        ]
        if non_periodic:
            sys.exit(
                f"Error: NPT requires a periodic cell, but structure(s) at "
                f"index {non_periodic} have pbc=False.\n"
                "Ensure your structure file uses extended-XYZ format with a "
                "'Lattice=' header, or use a POSCAR/CIF file so ASE reads "
                "the cell and sets pbc=True."
            )

    data_list = []
    for atoms in frames:
        data = AtomicData.from_atoms(atoms, device=args.device, dtype=dtype)
        data = data.model_copy(
            update={
                "forces": torch.zeros(
                    len(atoms), 3, dtype=dtype, device=args.device
                ),
                "energy": torch.zeros(1, 1, dtype=dtype, device=args.device),
                "stress": torch.zeros(
                    1, 3, 3, dtype=dtype, device=args.device
                ),
            }
        )
        # Zero-initialize velocities so Batch.from_data_list can aggregate
        # them even if the input file has no velocity data.
        data.use_default_velocities()
        data_list.append(data)
    batch = Batch.from_data_list(data_list)

    if not args.skip_vel_init:
        T_init = torch.full(
            (batch.num_graphs,),
            args.temperature,
            dtype=dtype,
            device=args.device,
        )
        initialize_velocities(
            velocities=batch.velocities,
            masses=batch.atomic_masses,
            temperature=T_init,
            batch_idx=batch.batch_idx.int(),
        )
        print(
            f"  Velocities initialized from MB distribution "
            f"at {args.temperature} K (use --skip-vel-init to disable)"
        )

    # start_time_ref is a mutable list so the closure in _make_hooks can
    # share the value that is set just before integrator.run().
    # [start_wall_time, step0_overhead] — second element set by _PrintHook
    # after step 0 so that ms/step excludes compilation overhead.
    start_time_ref: list[float] = [0.0, 0.0]
    hooks = _make_hooks(
        model,
        args.skin,
        args.log_file,
        args.log_freq,
        args.dt,
        args.ensemble,
        start_time_ref,
    )

    # dt, friction, barostat_time, thermostat_time are all in fs
    if args.ensemble == "nvt":
        integrator = NVTLangevin(
            model=model,
            dt=args.dt,
            temperature=args.temperature,
            friction=args.friction,
            n_steps=args.steps,
            hooks=hooks,
        )
    else:
        integrator = NPT(
            model=model,
            dt=args.dt,
            temperature=args.temperature,
            pressure=args.pressure * BAR_TO_EV_PER_A3,
            barostat_time=args.barostat_time,
            thermostat_time=args.thermostat_time,
            pressure_coupling=args.pressure_coupling,
            n_steps=args.steps,
            hooks=hooks,
        )

    print(
        f"\nRunning {args.ensemble.upper()}: {args.steps} steps  "
        f"T={args.temperature} K  dt={args.dt} fs"
        + (f"  P={args.pressure} bar" if args.ensemble == "npt" else "")
    )
    print(f"Logging every {args.log_freq} steps → {args.log_file}\n")

    start_time_ref[0] = time.perf_counter()
    batch = integrator.run(batch)
    total = time.perf_counter() - start_time_ref[0]
    md_total = total - start_time_ref[1]

    print(
        f"\nDone in {total:.1f} s total  "
        f"({start_time_ref[1]:.1f} s compile/step-0 overhead, "
        f"{md_total:.1f} s MD, "
        f"{md_total / args.steps * 1000:.2f} ms/step).\n"
        f"Log written to {args.log_file}"
    )


if __name__ == "__main__":
    main()
