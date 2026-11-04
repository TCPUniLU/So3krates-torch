"""NVT / NPT molecular dynamics using NVAlchemi + So3krates-torch.

Runs GPU-native MD with a trained So3krates / SO3LR model.  The neighbor
list is managed automatically by the hooks the model's own
``make_neighbor_hooks()`` builds (NVAlchemi's ``NeighborListHook``).

Usage (NVT)::

    torchkrates-md-nvalchemi \\
        --model  model.pt \\
        --atoms  NaCl.xyz \\
        --ensemble nvt \\
        --temperature 300 \\
        --steps 10000

Usage (NPT)::

    torchkrates-md-nvalchemi \\
        --model  model.pt \\
        --atoms  NaCl.xyz \\
        --ensemble npt \\
        --temperature 300 \\
        --pressure 1.0 \\
        --steps 10000

CSV log columns:
    step, graph_idx, status, energy (eV), fmax (eV/Å), temperature (K),
    elapsed_s (s), s_per_step (s); plus volume_A3 (Å³) and density_g_cm3
    (g/cm³) for periodic structures, and pressure_bar (bar) for NPT.

Total charge and spin are read from ``atoms.info["charge"]`` and
``atoms.info["total_spin"]`` (the keys ``TorchkratesCalculator`` uses).
"""

from __future__ import annotations

import argparse
import sys
import time

import torch
from ase.io import read as ase_read

try:
    from nvalchemi.data.atomic_data import AtomicData
    from nvalchemi.data.batch import Batch
    from nvalchemi.dynamics.base import DynamicsStage
    from nvalchemi.dynamics import initialize_velocities
    from nvalchemi.dynamics.hooks.logging import LoggingHook
    from nvalchemi.dynamics.integrators.npt import NPT
    from nvalchemi.dynamics.integrators.nvt_langevin import NVTLangevin
    from nvalchemi.models.base import BaseModelMixin

    from so3krates_torch.calculator.nvalchemi_so3 import (
        NVAlchemiSO3LR,
        build_nvalchemi_pme_model,
        set_active_outputs,
    )
except ImportError as _err:  # e.g. Python 3.10: the extra installs nothing
    _NVALCHEMI_IMPORT_ERROR: ImportError | None = _err
else:
    _NVALCHEMI_IMPORT_ERROR = None

_NVALCHEMI_MISSING_MSG = (
    "Error: torchkrates-md-nvalchemi needs nvalchemi-toolkit, which could "
    "not be imported ({err}).\n"
    "nvalchemi-toolkit requires Python >= 3.11; on Python >= 3.11 install "
    'it with:  pip install ".[nvalchemi]"'
)

# Outputs NVT needs.  nvalchemi-toolkit 0.2 integrators do not narrow
# model_config.active_outputs themselves, so without this NVT would compute
# (unused) stress every step.  NPT keeps the wrapper's defaults.
_NVT_OUTPUTS = frozenset({"energy", "forces"})

# ---------------------------------------------------------------------------
# Unit conversion constants
# ---------------------------------------------------------------------------
AMU_TO_G_PER_CM3 = 1.66054  # (amu/Å³) → g/cm³
BAR_TO_EV_PER_A3 = 6.2415e-7  # bar → eV/Å³
KB_EV_PER_K = 8.617333262e-5  # Boltzmann constant [eV/K]


def _temperature_per_graph(batch) -> torch.Tensor:
    """Instantaneous kinetic temperature per graph ``[B]`` in K.

    Equipartition with 3N degrees of freedom (no constraint correction).
    Computed locally (no nvalchemi private-module import): per-atom KE
    ``0.5 * m_i |v_i|^2`` is scattered into a ``[B]`` tensor via
    ``index_add_`` over ``batch.batch_idx``.
    """
    masses = batch.atomic_masses
    if masses.dim() > 1:
        masses = masses.squeeze(-1)
    per_atom_ke = 0.5 * masses * (batch.velocities**2).sum(-1)  # [V]
    ke = torch.zeros(
        int(batch.num_graphs),
        dtype=per_atom_ke.dtype,
        device=per_atom_ke.device,
    )
    ke.index_add_(0, batch.batch_idx, per_atom_ke)  # [B]
    dof = 3.0 * batch.num_nodes_per_graph.to(ke.dtype)  # [B]
    return 2.0 * ke / (dof * KB_EV_PER_K)


def _pressure_bar_per_graph(batch) -> torch.Tensor:
    """Instantaneous scalar pressure per graph ``[B]`` in bar.

    ``P = (2/3) KE / V - (1/3) tr(σ)`` where ``σ = batch.stress`` is the
    tensile-positive Cauchy stress (ASE convention, the sign convention the
    adapter already emits).  The kinetic term equals ``N kB T / V`` exactly,
    because ``_temperature_per_graph`` defines T with 3N degrees of freedom.

    This matches the total pressure LAMMPS reports (``thermo press``), so the
    two engines can be compared directly at the same configuration.
    """
    cell = batch.cell
    vols = torch.abs(torch.linalg.det(cell))  # [B] Å³
    temps = _temperature_per_graph(batch)  # [B] K
    n_atoms = batch.num_nodes_per_graph.to(vols.dtype)  # [B]
    p_kin = n_atoms * KB_EV_PER_K * temps / vols  # [B] eV/Å³
    # batch.stress is tensile-positive Cauchy stress (ASE convention);
    # P_vir = -tr(sigma)/3, as in ASE and LAMMPS `thermo press`.
    p_vir = -batch.stress.diagonal(dim1=-2, dim2=-1).sum(-1) / 3.0  # [B] eV/Å³
    return (p_kin + p_vir) / BAR_TO_EV_PER_A3  # [B] bar


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def _build_parser() -> argparse.ArgumentParser:
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
        help="Ensemble: nvt (Langevin) or npt (MTK barostat, default: nvt). "
        "Note: NPT in nvalchemi-toolkit 0.2.0 computes the isotropic "
        "barostat mass 3x smaller than the hand-patched 0.1.0 runs this "
        "CLI originally targeted, so NPT trajectories are not directly "
        "comparable to those earlier runs.",
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
        "--seed",
        type=int,
        default=42,
        help="Langevin RNG seed (NVT only; default: 42)",
    )
    p.add_argument(
        "--barostat-time",
        type=float,
        default=None,
        help="NPT barostat time constant [fs] "
        "(default: 1000 * dt, matching the LAMMPS pdamp convention)",
    )
    p.add_argument(
        "--thermostat-time",
        type=float,
        default=None,
        help="NPT thermostat time constant [fs] "
        "(default: 100 * dt, matching the LAMMPS tdamp convention)",
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
        help="Skip Maxwell-Boltzmann velocity initialization and keep the "
        "velocities stored in the structure file. Default: initialize "
        "velocities from MB at --temperature (overwriting any velocities "
        "already present in the input file).",
    )
    p.add_argument(
        "--compile",
        action="store_true",
        help="Compile the inner model with torch.compile(dynamic=True). "
        "Adds ~30-60 s first-step overhead, then typically 2-4× speedup "
        "for large systems. On CUDA the model is compiled with "
        "fullgraph=True (no graph breaks allowed; compilation fails "
        "loudly if one occurs). On CPU, long-range SO3LR models are "
        "refused (Inductor CPU backend crash). NVAlchemi-native PME "
        "(--electrostatics nvalchemi) runs outside the compiled module.",
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
    return p


def _parse_args() -> argparse.Namespace:
    args = _build_parser().parse_args()

    # Resolve NPT coupling time constants as multiples of the timestep,
    # mirroring the LAMMPS convention (tdamp ~ 100 steps, pdamp ~ 1000 steps).
    if args.thermostat_time is None:
        args.thermostat_time = 100.0 * args.dt
    if args.barostat_time is None:
        args.barostat_time = 1000.0 * args.dt

    return args


# ---------------------------------------------------------------------------
# Structure -> AtomicData
# ---------------------------------------------------------------------------


def atoms_to_data(atoms, device: str, dtype: torch.dtype) -> AtomicData:
    """ASE ``Atoms`` -> ``AtomicData`` with zeroed outputs and ASE velocities.

    ``forces``/``energy``/``stress`` are zero-filled so ``compute()`` can
    write back in-place (filled at ``BEFORE_COMPUTE``).  nvalchemi's
    internal time unit equals ASE's (``sqrt(amu*Å^2/eV)``), so
    ``atoms.get_velocities()`` is copied unchanged (zeros when the
    structure file carries no velocity data).

    The total charge comes from ``atoms.info["charge"]`` (via
    ``AtomicData.from_atoms``) and the total spin *S* (not the
    multiplicity) from ``atoms.info["total_spin"]`` — the same keys
    ``TorchkratesCalculator`` uses.  Spin is stored as ``data.spin``
    ``[1, 1]``; absent means 0.
    """
    n = len(atoms)
    data = AtomicData.from_atoms(atoms, device=device, dtype=dtype)
    update = {
        "forces": torch.zeros(n, 3, dtype=dtype, device=device),
        "energy": torch.zeros(1, 1, dtype=dtype, device=device),
        "stress": torch.zeros(1, 3, 3, dtype=dtype, device=device),
        "velocities": torch.as_tensor(
            atoms.get_velocities(), dtype=dtype, device=device
        ),
    }
    total_spin = atoms.info.get("total_spin")
    if total_spin is not None:
        update["spin"] = torch.tensor(
            [[float(total_spin)]], dtype=dtype, device=device
        )
    return data.model_copy(update=update)


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
    skin: float = 0.0,
    compile_model: bool = False,
    fullgraph: bool | None = None,
    electrostatic_energy_scale: float = None,
    dispersion_energy_scale: float = None,
    dispersion_energy_cutoff_lr_damping: float = None,
):
    """Load a trained model and wrap it for NVAlchemi MD.

    Returns a NVAlchemi ``BaseModelMixin`` — either a bare ``NVAlchemiSO3LR``
    (``electrostatics`` in {"model", "none"}) or a ``PipelineModelWrapper``
    composing the short-range model with NVAlchemi-native PME
    (``electrostatics="nvalchemi"``).  ``skin`` (Verlet skin, Å) is
    forwarded to the wrapper's ``model_config.neighbor_config.skin`` so
    ``model.make_neighbor_hooks()`` builds the neighbor list with the
    requested skin.
    """
    raw = torch.load(path, map_location=device, weights_only=False)
    if isinstance(raw, dict) and "model" in raw:
        raw = raw["model"]

    # Convert to the requested device/dtype BEFORE wrapping or compiling
    # (as TorchkratesCalculator does), and freeze the weights: MD only
    # needs gradients w.r.t. positions/strain.
    raw = raw.to(device=device, dtype=dtype)
    for param in raw.parameters():
        param.requires_grad = False

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

    # Apply the electrostatics backend switch on the raw module BEFORE
    # compiling (attribute writes on a compiled OptimizedModule would not
    # reach the inner model) and before printing the diagnostics.
    if electrostatics == "none":
        raw.electrostatic_energy_bool = False
    elif electrostatics == "nvalchemi":
        # build_nvalchemi_pme_model sets these again; harmless.
        raw.electrostatic_energy_bool = False
        raw.output_partial_charges = True

    # Print a diagnostic so there's no ambiguity about what is active.
    _print_model_diagnostics(raw, electrostatics)

    if compile_model:
        if torch.device(device).type != "cuda" and getattr(
            raw, "use_lr", False
        ):
            raise RuntimeError(
                "torch.compile is not supported on this device "
                f"({device}) for long-range SO3LR models: PyTorch's "
                "Inductor CPU backend crashes on the "
                "ZBL/electrostatics/dispersion scatter-reductions. Run "
                "without --compile or on CUDA."
            )
        if fullgraph is None:
            fullgraph = torch.device(device).type == "cuda"
        # Compile the inner So3krates/SO3LR forward pass before NVAlchemi
        # wrapping.  dynamic=True handles varying neighbor-list sizes.
        raw = torch.compile(raw, dynamic=True, fullgraph=fullgraph)

    if electrostatics == "nvalchemi":
        model = build_nvalchemi_pme_model(
            raw,
            mesh_spacing=pme_mesh_spacing,
            alpha=pme_alpha,
            spline_order=pme_spline_order,
            accuracy=pme_accuracy,
            skin=skin,
        )
    else:
        model = NVAlchemiSO3LR(raw, electrostatics="model", skin=skin)

    model = model.to(device=device, dtype=dtype)
    model.eval()
    return model


def _print_model_diagnostics(raw, electrostatics: str = "model") -> None:
    """Print which long-range terms are active and their scale factors.

    Called after the ``electrostatics`` backend switch has been applied to
    ``raw``, so the report reflects what will actually run.
    """
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

    if electrostatics == "nvalchemi":
        elec = "NVAlchemi PME (from model partial charges)"
    else:
        elec = _yn(elec_bool) + (f"  scale={elec_scale}" if elec_bool else "")
    print(
        f"  SR cutoff: {r_sr} Å   LR cutoff: {r_lr} Å\n"
        f"  ZBL repulsion:  {_yn(zbl_bool)}\n"
        f"  Electrostatics: {elec}\n"
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
    f"  {'vol/Å³':>10}  {'dens/gcc':>9}  {'press/bar':>12}"
    f"  {'elapsed/s':>10}  {'ms/step':>8}"
)
_DIV_NVT = "-" * 78
_DIV_NPT = "-" * 110


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
        temps = _temperature_per_graph(batch)  # [B]

        if self._npt:
            cell = batch.cell
            vols = torch.abs(torch.linalg.det(cell))  # [B]
            mass = torch.zeros(B, dtype=cell.dtype, device=cell.device)
            mass.index_add_(0, batch.batch_idx, batch.atomic_masses)
            denss = mass / vols * AMU_TO_G_PER_CM3  # [B]
            press = _pressure_bar_per_graph(batch)  # [B] bar

        for g in range(B):
            e = float(energy[g])
            t = float(temps[g])
            if self._npt:
                row = (
                    f"{step:>8d}  {g:>4d}  {time_ps:>8.3f}  {e:>12.4f}"
                    f"  {t:>8.2f}  {float(vols[g]):>10.2f}"
                    f"  {float(denss[g]):>9.4f}  {float(press[g]):>12.1f}"
                    f"  {md_elapsed:>10.1f}  {ms:>8.1f}"
                )
            else:
                row = (
                    f"{step:>8d}  {g:>4d}  {time_ps:>8.3f}  {e:>12.4f}"
                    f"  {t:>8.2f}  {md_elapsed:>10.1f}  {ms:>8.1f}"
                )
            print(row, flush=True)


def _make_hooks(
    model: BaseModelMixin,
    log_file: str,
    log_freq: int,
    dt_fs: float,
    ensemble: str,
    start_time_ref: list[float],
    has_cell: bool = True,
) -> list:
    """Neighbor, CSV-logging and stdout hooks.

    Cell-derived CSV columns (``volume_A3``, ``density_g_cm3``) are only
    registered when the batch has a cell (periodic input), and
    ``pressure_bar`` only for NPT (NVT does not compute stress); a
    non-periodic ``Batch`` has no ``cell`` attribute at all.
    """

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

    def _pressure(ctx):
        return _pressure_bar_per_graph(ctx.batch)  # [B] bar

    def _elapsed(ctx) -> float:
        return time.perf_counter() - start_time_ref[0]

    def _s_per_step(ctx) -> float:
        step = int(ctx.step_count)
        if step == 0:
            return 0.0
        md_elapsed = _elapsed(ctx) - start_time_ref[1]
        return md_elapsed / step

    custom_scalars = {}
    if has_cell:
        custom_scalars["volume_A3"] = _volume
        custom_scalars["density_g_cm3"] = _density
    if ensemble == "npt":
        custom_scalars["pressure_bar"] = _pressure
    custom_scalars["elapsed_s"] = _elapsed
    custom_scalars["s_per_step"] = _s_per_step

    log_hook = LoggingHook(
        backend="csv",
        log_path=log_file,
        frequency=log_freq,
        custom_scalars=custom_scalars,
    )
    print_hook = _PrintHook(log_freq, dt_fs, ensemble, start_time_ref)
    # The neighbor list is managed by the model's own make_neighbor_hooks()
    # (NVAlchemi's BaseModelMixin), which already encodes the model's skin
    # (model_config.neighbor_config.skin, set via load_model(skin=...)).
    return [*model.make_neighbor_hooks(), log_hook, print_hook]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _configure_active_outputs(model, ensemble: str) -> None:
    """Narrow the model's ``active_outputs`` for the chosen ensemble."""
    if ensemble == "nvt":
        set_active_outputs(model, _NVT_OUTPUTS)


def main() -> None:
    if _NVALCHEMI_IMPORT_ERROR is not None:
        sys.exit(_NVALCHEMI_MISSING_MSG.format(err=_NVALCHEMI_IMPORT_ERROR))
    args = _parse_args()
    dtype = torch.float64 if args.dtype == "float64" else torch.float32

    # Read and validate the structures before the (slow) model load.
    print(f"Reading structure(s) from {args.atoms} (index={args.index}) …")
    frames = ase_read(args.atoms, index=args.index)
    if not isinstance(frames, list):  # single-frame selection
        frames = [frames]
    print(
        f"  {len(frames)} system(s), "
        f"{sum(len(a) for a in frames)} atoms total"
    )
    periodic = [bool(a.get_pbc().any()) for a in frames]
    if len(set(periodic)) > 1:
        sys.exit(
            "Error: the selected frames are a mix of periodic and "
            "non-periodic structures (periodic: "
            f"{[i for i, p in enumerate(periodic) if p]}, non-periodic: "
            f"{[i for i, p in enumerate(periodic) if not p]}); they cannot "
            "share one batch. Select only one kind with --index or run "
            "them separately."
        )
    if args.ensemble == "npt" and not all(periodic):
        sys.exit(
            "Error: NPT requires a periodic cell, but the structure(s) have "
            "pbc=False.\n"
            "Ensure your structure file uses extended-XYZ format with a "
            "'Lattice=' header, or use a POSCAR/CIF file so ASE reads "
            "the cell and sets pbc=True."
        )

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
        skin=args.skin,
        compile_model=args.compile,
        electrostatic_energy_scale=args.electrostatic_energy_scale,
        dispersion_energy_scale=args.dispersion_energy_scale,
        dispersion_energy_cutoff_lr_damping=(
            args.dispersion_energy_cutoff_lr_damping
        ),
    )
    _configure_active_outputs(model, args.ensemble)
    cutoff = model.model_config.neighbor_config.cutoff
    compile_note = "  compiled=yes (first step slow)" if args.compile else ""
    print(
        f"  electrostatics={args.electrostatics}  "
        f"neighbor cutoff={cutoff:.2f} Å{compile_note}"
    )

    # Build one AtomicData per system; initialise forces/energy/stress to
    # zeros so compute() can write back in-place (filled at BEFORE_COMPUTE).
    data_list = [atoms_to_data(atoms, args.device, dtype) for atoms in frames]
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
        args.log_file,
        args.log_freq,
        args.dt,
        args.ensemble,
        start_time_ref,
        has_cell=getattr(batch, "cell", None) is not None,
    )

    # dt, friction, barostat_time, thermostat_time are all in fs
    if args.ensemble == "nvt":
        integrator = NVTLangevin(
            model=model,
            dt=args.dt,
            temperature=args.temperature,
            friction=args.friction,
            random_seed=args.seed,
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
        + (
            f"  P={args.pressure} bar  "
            f"tdamp={args.thermostat_time:g} fs ({args.thermostat_time / args.dt:g} steps)  "
            f"pdamp={args.barostat_time:g} fs ({args.barostat_time / args.dt:g} steps)"
            if args.ensemble == "npt"
            else ""
        )
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
