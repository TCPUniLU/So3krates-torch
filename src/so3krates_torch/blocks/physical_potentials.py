import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, Optional
import math
import numpy as np
from so3krates_torch.tools.scatter import scatter_sum
from so3krates_torch.blocks.dispersion_ref_data import NHL_AA, NHL_BB


# Constants for dispersion energy calculations
# From ase.units
BOHR = 0.5291772105638411  # Bohr radius in Angstrom
HARTREE = 27.211386245988  # Hartree in eV
FINE_STRUCTURE = 0.0072973525693  # fine structure constant

# Reference data for dispersion calculations (from mlff dispersion_ref_data)
ALPHAS = torch.tensor(
    [
        4.5,
        1.38,
        164.2,
        38.0,
        21.0,
        12.0,
        7.4,
        5.4,
        3.8,
        2.67,
        162.7,
        71.0,
        60.0,
        37.0,
        25.0,
        19.6,
        15.0,
        11.1,
        292.9,
        160.0,
        120.0,
        98.0,
        84.0,
        78.0,
        63.0,
        56.0,
        50.0,
        48.0,
        42.0,
        40.0,
        60.0,
        41.0,
        29.0,
        25.0,
        20.0,
        16.8,
        319.2,
        199.0,
        126.74,
        119.97,
        101.6,
        88.42,
        80.08,
        65.89,
        56.1,
        23.68,
        50.6,
        39.7,
        70.22,
        55.95,
        43.67,
        37.65,
        35.0,
        27.3,
        399.9,
        275.0,
        213.7,
        204.7,
        215.8,
        208.4,
        200.2,
        192.1,
        184.2,
        158.3,
        169.5,
        164.64,
        156.3,
        150.2,
        144.3,
        138.9,
        137.2,
        99.52,
        82.53,
        71.04,
        63.04,
        55.06,
        42.51,
        39.68,
        36.5,
        33.9,
        69.92,
        61.8,
        49.02,
        45.01,
        38.93,
        33.54,
        317.8,
        246.2,
        203.3,
        217.0,
        154.4,
        127.8,
        150.5,
        132.2,
        131.2,
        143.6,
        125.3,
        121.5,
        117.5,
        113.4,
        109.4,
        105.4,
    ],
    dtype=torch.float32,
)

C6_COEF = torch.tensor(
    [
        6.50000e00,
        1.46000e00,
        1.38700e03,
        2.14000e02,
        9.95000e01,
        4.66000e01,
        2.42000e01,
        1.56000e01,
        9.52000e00,
        6.38000e00,
        1.55600e03,
        6.27000e02,
        5.28000e02,
        3.05000e02,
        1.85000e02,
        1.34000e02,
        9.46000e01,
        6.43000e01,
        3.89700e03,
        2.22100e03,
        1.38300e03,
        1.04400e03,
        8.32000e02,
        6.02000e02,
        5.52000e02,
        4.82000e02,
        4.08000e02,
        3.73000e02,
        2.53000e02,
        2.84000e02,
        4.98000e02,
        3.54000e02,
        2.46000e02,
        2.10000e02,
        1.62000e02,
        1.29600e02,
        4.69100e03,
        3.17000e03,
        1.96858e03,
        1.67791e03,
        1.26361e03,
        1.02873e03,
        1.39087e03,
        6.09750e02,
        4.69000e02,
        1.57500e02,
        3.39000e02,
        4.52000e02,
        7.07050e02,
        5.87420e02,
        4.59320e02,
        3.96000e02,
        3.85000e02,
        2.85900e02,
        6.84600e03,
        5.72700e03,
        3.88450e03,
        3.70833e03,
        3.91184e03,
        3.90875e03,
        3.84768e03,
        3.70869e03,
        3.51171e03,
        2.78153e03,
        3.12441e03,
        2.98429e03,
        2.83995e03,
        2.72412e03,
        2.57678e03,
        2.38753e03,
        2.37180e03,
        1.27480e03,
        1.01992e03,
        8.47930e02,
        7.10200e02,
        5.96670e02,
        3.59100e02,
        3.47100e02,
        2.98000e02,
        3.92000e02,
        7.17440e02,
        6.97000e02,
        5.71000e02,
        5.30920e02,
        4.57530e02,
        3.90630e02,
        4.22444e03,
        4.85132e03,
        3.60441e03,
        4.04754e03,
        2.87677e03,
        2.37589e03,
        3.10212e03,
        2.82047e03,
        2.79400e03,
        3.15095e03,
        2.75600e03,
        2.70257e03,
        2.62659e03,
        2.54862e03,
        2.46869e03,
        2.38680e03,
    ],
    dtype=torch.float32,
)


def softplus_inverse(x):
    """Inverse of softplus function"""
    return x + torch.log(-torch.expm1(-x))


def sigma(x):
    """Sigma function used in switching function"""
    return torch.where(
        x > 0, torch.exp(-1.0 / x.clamp_min(1e-12)), torch.zeros_like(x)
    )


def switching_fn(x, x_on, x_off):
    """Switching function for smooth cutoff"""
    c = (x - x_on) / (x_off - x_on)
    sigma_1_c = sigma(1 - c)
    sigma_c = sigma(c)
    return sigma_1_c / (sigma_1_c + sigma_c + 1e-12)


def segment_sum(data, segment_ids, num_segments):
    """Sum data according to segment_ids"""
    result = torch.zeros(num_segments, dtype=data.dtype, device=data.device)
    result.scatter_add_(0, segment_ids, data)
    return result


def _dispersion_tables(
    legacy_dispersion_bool: bool,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Free-atom alpha/C6 reference tables (indexed by Z-1).

    Returns the legacy (JAX v1) module-level ``ALPHAS``/``C6_COEF`` tables
    when ``legacy_dispersion_bool`` is True, otherwise the refitted tables
    from ``dispersion_ref_data`` (so3lr-s/m/l). Both tables share the
    identical Z-1 indexing convention and units. Returned tables are the
    raw (undtyped/undeviced) tensors — callers apply
    ``.to(device=..., dtype=...)`` themselves, as ``mixing_rules`` already
    does for the legacy case.
    """
    if legacy_dispersion_bool:
        return ALPHAS, C6_COEF

    from so3krates_torch.blocks.dispersion_ref_data import (
        alphas as _ALPHAS_REFITTED,
        C6_coef as _C6_COEF_REFITTED,
    )

    return (
        torch.tensor(_ALPHAS_REFITTED, dtype=torch.get_default_dtype()),
        torch.tensor(_C6_COEF_REFITTED, dtype=torch.get_default_dtype()),
    )


def mixing_rules(
    atomic_numbers: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    hirshfeld_ratios: torch.Tensor,
    c6_ratios: Optional[torch.Tensor] = None,
    legacy_dispersion_bool: bool = True,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply mixing rules to compute alpha_ij and C6_ij for dispersion.

    When c6_ratios is provided, alpha and C6 are scaled independently:
      alpha_i = alpha_free[Z_i] * hirshfeld_ratio_i
      C6_i    = C6_free[Z_i]   * c6_ratio_i
    Otherwise (backward compat) C6 is scaled by hirshfeld_ratio²:
      C6_i    = C6_free[Z_i]   * hirshfeld_ratio_i²

    legacy_dispersion_bool selects which free-atom alpha/C6 reference
    table is used (see `_dispersion_tables`): True (default, backward
    compatible) uses the legacy JAX v1 table; False uses the refitted
    table required by so3lr-s/m/l.
    """
    dtype = hirshfeld_ratios.dtype
    device = hirshfeld_ratios.device

    alphas_table, c6_table = _dispersion_tables(legacy_dispersion_bool)
    alphas = alphas_table.to(device=device, dtype=dtype)
    c6_coef = c6_table.to(device=device, dtype=dtype)

    atomic_number_i = atomic_numbers[idx_i] - 1
    atomic_number_j = atomic_numbers[idx_j] - 1
    hirshfeld_ratio_i = hirshfeld_ratios[idx_i]
    hirshfeld_ratio_j = hirshfeld_ratios[idx_j]

    alpha_i = alphas[atomic_number_i] * hirshfeld_ratio_i
    alpha_j = alphas[atomic_number_j] * hirshfeld_ratio_j

    if c6_ratios is not None:
        C6_i = c6_coef[atomic_number_i] * c6_ratios[idx_i]
        C6_j = c6_coef[atomic_number_j] * c6_ratios[idx_j]
    else:
        C6_i = c6_coef[atomic_number_i] * torch.square(hirshfeld_ratio_i)
        C6_j = c6_coef[atomic_number_j] * torch.square(hirshfeld_ratio_j)

    alpha_ij = (alpha_i + alpha_j) / 2
    C6_ij = (
        2
        * C6_i
        * C6_j
        * alpha_j
        * alpha_i
        / (alpha_i**2 * C6_j + alpha_j**2 * C6_i)
    )

    return alpha_ij, C6_ij


def atomic_c6_pseudo_charges(
    atomic_numbers: torch.Tensor,  # (N,)
    hirshfeld_ratios: torch.Tensor,  # (N,)
    c6_ratios: Optional[torch.Tensor] = None,  # (N,), optional
    legacy_dispersion_bool: bool = True,
) -> torch.Tensor:
    """Per-atom free-atom-scaled C6 coefficient, C6_i = C6_free[Z_i-1] * ratio_i.

    ratio_i = c6_ratios[i] if provided, else hirshfeld_ratios[i]**2 (legacy
    fallback — matches so3lr_dev's `legacy` branch: C6 derived from Hirshfeld
    ratios when there's no dedicated C6 head).

    Returns:
        (N,) tensor of per-atom C6_i values (may be used directly, or via
        q_i = sqrt(clamp(C6_i, min=0)) for PME pseudo-charges).
    """
    dtype = hirshfeld_ratios.dtype
    device = hirshfeld_ratios.device

    _, c6_table = _dispersion_tables(legacy_dispersion_bool)
    c6_coef = c6_table.to(device=device, dtype=dtype)

    atomic_number_idx = atomic_numbers - 1

    if c6_ratios is not None:
        C6_i = c6_coef[atomic_number_idx] * c6_ratios
    else:
        C6_i = c6_coef[atomic_number_idx] * torch.square(hirshfeld_ratios)

    return C6_i


def gamma_cubic_fit(alpha: torch.Tensor) -> torch.Tensor:
    """Compute gamma parameter using cubic fit"""
    input_dtype = alpha.dtype

    vdW_radius = torch.tensor(
        FINE_STRUCTURE, dtype=input_dtype, device=alpha.device
    ) ** (-4.0 / 21) * alpha ** (1.0 / 7)

    # Cubic fit coefficients
    b0 = torch.tensor(-0.00433008, dtype=input_dtype, device=alpha.device)
    b1 = torch.tensor(0.24428889, dtype=input_dtype, device=alpha.device)
    b2 = torch.tensor(0.04125273, dtype=input_dtype, device=alpha.device)
    b3 = torch.tensor(-0.00078893, dtype=input_dtype, device=alpha.device)

    sigma = (
        b3 * torch.pow(vdW_radius, 3)
        + b2 * torch.square(vdW_radius)
        + b1 * vdW_radius
        + b0
    )
    gamma = torch.tensor(
        0.5, dtype=input_dtype, device=alpha.device
    ) / torch.square(sigma)
    return gamma


def _vdw_qdo_disp_damp_terms(
    R: torch.Tensor,
    gamma: torch.Tensor,
    C6: torch.Tensor,
    alpha_ij: torch.Tensor,
    gamma_scale: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Shared per-term computation behind ``vdw_qdo_disp_damp``.

    Returns the individual (undamped-sum, un-Hartree-scaled) C6/C8/C10
    damped terms plus the damping length ``p``, so that PME-dispersion's
    real-space residual branch (`DispersionInteraction.forward`) can
    replace only the C6 term while reusing the exact same C8/C10/p
    computation as the non-PME path — no duplicated math, no risk of the
    two paths silently diverging.
    """
    # Compute higher-order dispersion coefficients
    C8 = 5 / gamma * C6
    C10 = 245 / 8 / gamma**2 * C6
    p = gamma_scale * 2 * 2.54 * alpha_ij ** (1 / 7)

    C6_term = -C6 / (R**6 + p**6)
    C8_term = -C8 / (R**8 + p**8)
    C10_term = -C10 / (R**10 + p**10)

    return C6_term, C8_term, C10_term, p


def vdw_qdo_disp_damp(
    R: torch.Tensor,
    gamma: torch.Tensor,
    C6: torch.Tensor,
    alpha_ij: torch.Tensor,
    gamma_scale: float,
    c: float,
) -> torch.Tensor:
    """Compute vdW-QDO dispersion energy with damping"""
    input_dtype = R.dtype
    device = R.device

    # Compute potential — inline to avoid 6 simultaneous [N_pairs] tensors
    C6_term, C8_term, C10_term, _p = _vdw_qdo_disp_damp_terms(
        R, gamma, C6, alpha_ij, gamma_scale
    )
    V3 = C6_term + C8_term + C10_term

    hartree_factor = torch.tensor(HARTREE, dtype=input_dtype, device=device)
    return c * V3 * hartree_factor


def _coulomb_erf_energy(
    q: torch.Tensor,
    rij: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    ke,
    sigma,
    c,
) -> torch.Tensor:
    """Pairwise Coulomb energy with erf damping (no cutoff smoothing).

    `ke`/`sigma`/`c` may be plain floats or 0-dim tensors.
    """
    if q.dim() == 2 and q.size(-1) == 1:
        q = q.squeeze(-1)
    if rij.dim() == 2 and rij.size(-1) == 1:
        rij = rij.squeeze(-1)
    qi = q[receivers]
    qj = q[senders]

    r = rij.clamp_min(1e-12)
    pairwise = torch.erf(r / sigma) / r
    return c * ke * qi * qj * pairwise


class CoulombErf(nn.Module):
    """Pairwise Coulomb with erf damping (no cutoff smoothing).

    All constants are buffers initialized in __init__.
    """

    def __init__(
        self,
        *,
        ke: float,
        sigma: float,
        neighborlist_format_lr: str = "sparse",
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        if neighborlist_format_lr not in ("sparse", "ordered_sparse"):
            raise ValueError(
                "neighborlist_format_lr must be 'sparse' or 'ordered_sparse'"
            )
        c_val = 0.5 if neighborlist_format_lr == "sparse" else 1.0
        dt = dtype if dtype is not None else torch.get_default_dtype()
        self.register_buffer("ke", torch.tensor(ke, dtype=dt))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=dt))
        self.register_buffer("c", torch.tensor(c_val, dtype=dt))
        self.neighborlist_format_lr = neighborlist_format_lr

    def forward(
        self,
        q: torch.Tensor,
        rij: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> torch.Tensor:
        return _coulomb_erf_energy(
            q, rij, senders, receivers, self.ke, self.sigma, self.c
        )


def _coulomb_erf_shift_and_force_shift(cutoff: float, sigma: float):
    """Potential and its (negated) derivative evaluated at the cutoff,
    used to shift-and-force-shift the erf-damped Coulomb energy smoothly
    to zero at the cutoff.

    Pure Python scalar math: `cutoff`/`sigma` are fixed, non-trainable
    floats, so no gradient ever flows through this computation.
    """
    r = max(cutoff, 1e-12)
    shift = math.erf(r / sigma) / r
    force_shift = (
        2 * r * math.exp(-((r / sigma) ** 2)) / (math.sqrt(math.pi) * sigma)
        - math.erf(r / sigma)
    ) / (r**2)
    return shift, force_shift


def _coulomb_erf_shifted_force_smooth_energy(
    q: torch.Tensor,
    rij: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    ke,
    sigma,
    cutoff,
    cuton,
    c,
) -> torch.Tensor:
    """Coulomb erf energy with smooth shifted-force cutoff in
    [cuton, cutoff]. `ke`/`sigma`/`cutoff`/`cuton`/`c` may be plain
    floats or 0-dim tensors.
    """
    if q.dim() == 2 and q.size(-1) == 1:
        q = q.squeeze(-1)
    if rij.dim() == 2 and rij.size(-1) == 1:
        rij = rij.squeeze(-1)

    # smooth switching
    f = switching_fn(rij, cuton, cutoff)

    r_safe = rij.clamp_min(1e-12)
    pairwise = torch.erf(r_safe / sigma) / r_safe
    shift, force_shift = _coulomb_erf_shift_and_force_shift(
        float(cutoff), float(sigma)
    )
    shifted_potential = pairwise - shift - force_shift * (rij - cutoff)
    qi = q[receivers]
    qj = q[senders]

    inside = rij < cutoff
    energy = (
        c
        * ke
        * qi
        * qj
        * (f * (pairwise - shift) + (1 - f) * shifted_potential)
    )
    return torch.where(inside, energy, torch.zeros_like(energy))


def _coulomb_erf_shift_and_force_shift_pme(
    cutoff: float, sigma: float, smearing: float
):
    """Potential and its (negated) derivative evaluated at the cutoff for
    the PME-aware erf-damped Coulomb potential -- the model's own
    `sigma`-damped term minus the Ewald long-range complement
    `erf(r/(smearing*sqrt(2)))/r` (which the separate k-space
    contribution adds back) -- used to shift-and-force-shift smoothly to
    zero at the cutoff. Mirrors so3lr_dev's
    `coulomb_erf_shifted_force_smooth_pme`.

    Pure Python scalar math: `cutoff`/`sigma`/`smearing` are fixed,
    non-trainable floats, so no gradient ever flows through this
    computation.
    """
    r = max(cutoff, 1e-12)
    _smearing = smearing * math.sqrt(2.0)

    def _term_and_deriv(width: float):
        val = math.erf(r / width) / r
        deriv = (
            2
            * r
            * math.exp(-((r / width) ** 2))
            / (math.sqrt(math.pi) * width)
            - math.erf(r / width)
        ) / (r**2)
        return val, deriv

    val_sigma, deriv_sigma = _term_and_deriv(sigma)
    val_smear, deriv_smear = _term_and_deriv(_smearing)
    shift = val_sigma - val_smear
    force_shift = deriv_sigma - deriv_smear
    return shift, force_shift


def _coulomb_erf_shifted_force_smooth_pme_energy(
    q: torch.Tensor,
    rij: torch.Tensor,
    senders: torch.Tensor,
    receivers: torch.Tensor,
    ke,
    sigma,
    cutoff,
    cuton,
    smearing,
    c,
) -> torch.Tensor:
    """PME-aware Coulomb erf energy: the model's own `sigma`-damped erf
    term minus the Ewald long-range complement (which the separate
    k-space contribution -- see `PMEElectrostaticInteraction` -- adds
    back), with the same smooth shifted-force cutoff in [cuton, cutoff]
    as `_coulomb_erf_shifted_force_smooth_energy`. Mirrors so3lr_dev's
    `coulomb_erf_shifted_force_smooth_pme` exactly.

    `ke`/`sigma`/`cutoff`/`cuton`/`smearing`/`c` may be plain floats or
    0-dim tensors.
    """
    if q.dim() == 2 and q.size(-1) == 1:
        q = q.squeeze(-1)
    if rij.dim() == 2 and rij.size(-1) == 1:
        rij = rij.squeeze(-1)

    f = switching_fn(rij, cuton, cutoff)

    r_safe = rij.clamp_min(1e-12)
    _smearing = smearing * (2.0**0.5)
    pairwise = (
        torch.erf(r_safe / sigma) / r_safe
        - torch.erf(r_safe / _smearing) / r_safe
    )
    shift, force_shift = _coulomb_erf_shift_and_force_shift_pme(
        float(cutoff), float(sigma), float(smearing)
    )
    shifted_potential = pairwise - shift - force_shift * (rij - cutoff)
    qi = q[receivers]
    qj = q[senders]

    inside = rij < cutoff
    energy = (
        c
        * ke
        * qi
        * qj
        * (f * (pairwise - shift) + (1 - f) * shifted_potential)
    )
    return torch.where(inside, energy, torch.zeros_like(energy))


class CoulombErfShiftedForceSmooth(nn.Module):
    """Coulomb erf with smooth shifted-force cutoff in [cuton, cutoff].

    All constants are buffers initialized in __init__.
    """

    def __init__(
        self,
        *,
        ke: float,
        sigma: float,
        cutoff: float,
        cuton: float,
        neighborlist_format_lr: str = "sparse",
        dtype: torch.dtype = None,
    ) -> None:
        super().__init__()
        if neighborlist_format_lr not in ("sparse", "ordered_sparse"):
            raise ValueError(
                "neighborlist_format_lr must be 'sparse' or 'ordered_sparse'"
            )
        c_val = 0.5 if neighborlist_format_lr == "sparse" else 1.0
        dt = dtype if dtype is not None else torch.get_default_dtype()
        self.register_buffer("ke", torch.tensor(ke, dtype=dt))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=dt))
        self.register_buffer("cutoff", torch.tensor(cutoff, dtype=dt))
        self.register_buffer("cuton", torch.tensor(cuton, dtype=dt))
        self.register_buffer("c", torch.tensor(c_val, dtype=dt))
        self.neighborlist_format_lr = neighborlist_format_lr

    def forward(
        self,
        q: torch.Tensor,
        rij: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> torch.Tensor:
        return _coulomb_erf_shifted_force_smooth_energy(
            q,
            rij,
            senders,
            receivers,
            self.ke,
            self.sigma,
            self.cutoff,
            self.cuton,
            self.c,
        )


class ZBLRepulsion(nn.Module):
    """
    Ziegler-Biersack-Littmark repulsion.
    """

    def __init__(self):
        super().__init__()
        self.module_name = "zbl_repulsion"
        self.a0 = 0.5291772105638411
        self.ke = 14.399645351950548

        # Init learnable params with softplus_inverse defaults
        self.a1_raw = nn.Parameter(
            softplus_inverse(torch.tensor(3.20000)).unsqueeze(0)
        )
        self.a2_raw = nn.Parameter(
            softplus_inverse(torch.tensor(0.94230)).unsqueeze(0)
        )
        self.a3_raw = nn.Parameter(
            softplus_inverse(torch.tensor(0.40280)).unsqueeze(0)
        )
        self.a4_raw = nn.Parameter(
            softplus_inverse(torch.tensor(0.20160)).unsqueeze(0)
        )
        self.c1_raw = nn.Parameter(
            softplus_inverse(torch.tensor(0.18180)).unsqueeze(0)
        )
        self.c2_raw = nn.Parameter(
            softplus_inverse(torch.tensor(0.50990)).unsqueeze(0)
        )
        self.c3_raw = nn.Parameter(
            softplus_inverse(torch.tensor(0.28020)).unsqueeze(0)
        )
        self.c4_raw = nn.Parameter(
            softplus_inverse(torch.tensor(0.02817)).unsqueeze(0)
        )
        self.p_raw = nn.Parameter(
            softplus_inverse(torch.tensor(0.23)).unsqueeze(0)
        )
        self.d_raw = nn.Parameter(
            softplus_inverse(torch.tensor(1 / (0.8854 * self.a0))).unsqueeze(0)
        )

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        cutoffs: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        lengths: torch.Tensor,
        num_nodes: int,
    ) -> Dict[str, torch.Tensor]:
        cutoffs = cutoffs.squeeze(1)
        lengths = lengths.squeeze(1)

        # Apply softplus to get positive parameters
        a1 = F.softplus(self.a1_raw)
        a2 = F.softplus(self.a2_raw)
        a3 = F.softplus(self.a3_raw)
        a4 = F.softplus(self.a4_raw)
        c1 = F.softplus(self.c1_raw)
        c2 = F.softplus(self.c2_raw)
        c3 = F.softplus(self.c3_raw)
        c4 = F.softplus(self.c4_raw)
        p = F.softplus(self.p_raw)
        d = F.softplus(self.d_raw)

        # Normalize c coefficients
        c_sum = c1 + c2 + c3 + c4
        c1 = c1 / c_sum
        c2 = c2 / c_sum
        c3 = c3 / c_sum
        c4 = c4 / c_sum

        # Get atomic numbers for pairs
        z_i = atomic_numbers[receivers]
        z_j = atomic_numbers[senders]
        zz_ij = z_i * z_j

        # Compute z_lengths with safe division
        z_lengths = zz_ij / lengths.clamp(min=1e-6)

        # Compute x term
        x = self.ke * cutoffs * z_lengths

        # Compute rzd term
        rzd = lengths * (torch.pow(z_i, p) + torch.pow(z_j, p)) * d

        # Compute y term (exponential sum)
        y = (
            c1 * torch.exp(-a1 * rzd)
            + c2 * torch.exp(-a2 * rzd)
            + c3 * torch.exp(-a3 * rzd)
            + c4 * torch.exp(-a4 * rzd)
        )

        # Apply switching function
        w = switching_fn(lengths, x_on=0, x_off=1.5)

        # Compute edge repulsion energies
        e_rep_edge = w * x * y / 2.0

        # Sum over edges for each node
        # segment_sum(e_rep_edge, segment_ids=idx_i, num_segments=num_nodes)
        e_rep_edge = scatter_sum(
            src=e_rep_edge,
            index=receivers,
            dim=0,
            dim_size=num_nodes,
        ).unsqueeze(1)
        return e_rep_edge

    def reset_output_convention(self, output_convention):
        """Compatibility method - does nothing in this implementation"""
        pass


class NHLRepulsion(nn.Module):
    """Pair-specific NHL (nuclear-nuclear Hartree-like) short-range repulsion.

    Uses pre-tabulated element-pair screening coefficients instead of the
    learnable universal ZBL screening.  No trainable parameters.

    φ(r) = Σ_k  a[zi,zj,k] * exp(-b[zi,zj,k] * r)
    E_rep = Σ_{ij} w(r) * ke * (Zi*Zj/r) * cutoff * φ(r) / 2
    """

    def __init__(self) -> None:
        super().__init__()
        self.module_name = "nhl_repulsion"
        self.ke = 14.399645351950548

        # Fixed lookup tables (92×92×3), indexed by (Z-1) with zi≤zj
        nhl_aa = torch.tensor(np.array(NHL_AA), dtype=torch.float64)
        nhl_bb = torch.tensor(np.array(NHL_BB), dtype=torch.float64)
        self.register_buffer("nhl_aa", nhl_aa)
        self.register_buffer("nhl_bb", nhl_bb)

    def forward(
        self,
        atomic_numbers: torch.Tensor,
        cutoffs: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        lengths: torch.Tensor,
        num_nodes: int,
    ) -> torch.Tensor:
        cutoffs = cutoffs.squeeze(1)
        lengths = lengths.squeeze(1)

        z_i = atomic_numbers[receivers] - 1  # 0-based
        z_j = atomic_numbers[senders] - 1

        zi = torch.minimum(z_i, z_j)
        zj = torch.maximum(z_i, z_j)

        dtype = lengths.dtype
        aa = self.nhl_aa.to(dtype=dtype)[zi, zj]  # (E, 3)
        bb = self.nhl_bb.to(dtype=dtype)[zi, zj]  # (E, 3)

        phi = (aa * torch.exp(-bb * lengths.unsqueeze(-1))).sum(-1)

        z_over_r = (z_i + 1) * (z_j + 1) / lengths.clamp(min=1e-6)
        x = self.ke * cutoffs * z_over_r

        w = switching_fn(lengths, x_on=0.0, x_off=1.5)
        e_rep_edge = w * x * phi / 2.0

        return scatter_sum(
            src=e_rep_edge,
            index=receivers,
            dim=0,
            dim_size=num_nodes,
        ).unsqueeze(1)

    def reset_output_convention(self, output_convention):
        pass


class ElectrostaticInteraction(nn.Module):
    """
    Electrostatic energy with erf damping and optional smooth cutoff.
    """

    def __init__(
        self,
        *,
        ke: float = 14.399645351950548,
        neighborlist_format_lr: str = "sparse",
    ) -> None:
        super().__init__()
        if neighborlist_format_lr not in ("sparse", "ordered_sparse"):
            raise ValueError(
                "neighborlist_format_lr must be 'sparse' or 'ordered_sparse'"
            )
        self.ke = ke
        self.neighborlist_format_lr = neighborlist_format_lr

    def forward(
        self,
        partial_charges: torch.Tensor,
        senders_lr: torch.Tensor,
        receivers_lr: torch.Tensor,
        lengths_lr: torch.Tensor,
        num_nodes: int,
        cutoff_lr: float | None = None,
        electrostatic_energy_scale: float = 1.0,
        use_pme: bool = False,
        pme_smearing: Optional[float] = None,
    ) -> torch.Tensor:
        """Compute per-node electrostatic energy.

        Args:
            partial_charges: (N,) or (N,1) charges.
            senders_lr: (E,) source indices (j).
            receivers_lr: (E,) target indices (i).
            lengths_lr: (E,) or (E,1) edge distances.
            num_nodes: number of nodes to scatter to.
            use_pme: when True, this is the real-space *residual* of a
                combined PME calculation -- the model's own
                `electrostatic_energy_scale`-damped erf term minus the
                Ewald long-range complement, so that adding
                `PMEElectrostaticInteraction`'s k-space contribution on
                top recovers the correct total (mirrors
                `DispersionInteraction`'s PME-aware real-space residual).
                Requires `cutoff_lr`/`pme_smearing` to be set. Defaults to
                `False`, which preserves this function's original
                (non-PME) behavior exactly.
            pme_smearing: the PME/Ewald smearing width. Required when
                `use_pme=True`, ignored otherwise.

        Returns:
            (N,1) atomic electrostatic energies.
        """
        # Computed inline (not stored on self in __init__) so that
        # pretrained checkpoints pickled before this attribute existed
        # keep working — unpickling restores __dict__ directly without
        # re-running __init__.
        c = 0.5 if self.neighborlist_format_lr == "sparse" else 1.0

        if lengths_lr.dim() == 2 and lengths_lr.size(-1) == 1:
            lengths_lr = lengths_lr.squeeze(-1)

        if use_pme:
            if cutoff_lr is None:
                raise ValueError("use_pme=True requires cutoff_lr to be set.")
            if pme_smearing is None:
                raise ValueError(
                    "use_pme=True requires pme_smearing to be set "
                    "(got None)."
                )
            cutoff = float(cutoff_lr)
            cuton = 0.45 * cutoff
            edge_e = _coulomb_erf_shifted_force_smooth_pme_energy(
                partial_charges,
                lengths_lr,
                senders_lr,
                receivers_lr,
                self.ke,
                electrostatic_energy_scale,
                cutoff,
                cuton,
                pme_smearing,
                c,
            )
        elif cutoff_lr is not None:
            cutoff = float(cutoff_lr)
            cuton = 0.45 * cutoff
            edge_e = _coulomb_erf_shifted_force_smooth_energy(
                partial_charges,
                lengths_lr,
                senders_lr,
                receivers_lr,
                self.ke,
                electrostatic_energy_scale,
                cutoff,
                cuton,
                c,
            )
        else:
            edge_e = _coulomb_erf_energy(
                partial_charges,
                lengths_lr,
                senders_lr,
                receivers_lr,
                self.ke,
                electrostatic_energy_scale,
                c,
            )

        atomic_e = scatter_sum(
            src=edge_e,
            index=receivers_lr,
            dim=0,
            dim_size=num_nodes,
        ).unsqueeze(1)

        return atomic_e

    def reset_output_convention(self, output_convention):
        # No-op to keep interface compatibility
        pass


class PMEElectrostaticInteraction(nn.Module):
    """K-space-ONLY electrostatic energy via Particle Mesh Ewald (torch-pme).

    The model's own `electrostatic_energy_scale`-damped real-space term
    is computed separately by `ElectrostaticInteraction` (called with
    `use_pme=True`, over the LR neighbor list) -- its PME-aware residual
    formula already subtracts the Ewald long-range complement that this
    class's k-space contribution adds back, so the two must always be
    added together (mirrors `DispersionInteraction`/
    `PMEDispersionInteraction`'s real-space-residual + k-space split).
    This class alone is NOT the full PME electrostatic energy.

    Calls `torchpme.PMECalculator._compute_kspace(...)` directly rather
    than `.forward(...)` -- `_compute_kspace` already returns the
    reciprocal-space potential together with the standard Ewald
    self-energy and neutralizing-background corrections (halved, per
    torch-pme's own double-counting convention), exactly matching the
    JAX reference's k-space-only contribution. No additional self-energy
    term or `0.5` factor should be applied on top of its output.

    Parameters follow the torch-pme ML-potential convention:
        smearing     = r_max / 5
        mesh_spacing = smearing / 2

    For optimally tuned parameters use torchpme.tuning.tune_pme() on a
    representative batch and pass the results as smearing / mesh_spacing.
    """

    def __init__(
        self,
        *,
        smearing: float,
        mesh_spacing: float,
        interpolation_nodes: int = 4,
        ke: float = 14.399645351950548,
    ) -> None:
        super().__init__()
        import torchpme  # lazy import — only required when use_pme=True

        self.ke = ke
        potential = torchpme.CoulombPotential(smearing=smearing)
        self.calculator = torchpme.PMECalculator(
            potential=potential,
            mesh_spacing=mesh_spacing,
            interpolation_nodes=interpolation_nodes,
        )

    def forward(
        self,
        partial_charges: torch.Tensor,  # (N,)
        positions: torch.Tensor,  # (N, 3)
        cell: torch.Tensor,  # (N_graphs, 3, 3)
        batch_segments: torch.Tensor,  # (N,)
        num_graphs: int,
        num_nodes: int,
    ) -> torch.Tensor:  # (N, 1)
        # PyG Batch concatenates [3,3] cells along dim 0 → [N*3,3].
        # Restore to [N,3,3] so cell[g] yields the correct [3,3] matrix.
        cell = cell.view(-1, 3, 3)
        atomic_e = torch.zeros(
            num_nodes,
            1,
            dtype=positions.dtype,
            device=positions.device,
        )
        for g in range(num_graphs):
            atom_mask = batch_segments == g
            pos_g = positions[atom_mask]  # (N_g, 3)
            q_g = partial_charges[atom_mask]  # (N_g,)
            cell_g = cell[g]  # (3, 3)
            if cell_g.abs().sum() == 0:
                raise ValueError(
                    "PME electrostatics requires periodic boundary "
                    "conditions, but the cell for graph "
                    f"{g} is all zeros (non-periodic system). "
                    "Either set pbc=True and provide a cell, or "
                    "use use_pme=False for non-periodic systems."
                )

            # K-space-only contribution (self-energy + background/
            # neutrality correction already included, halved -- see the
            # class docstring). Do NOT apply an extra external 0.5 here.
            kspace_potentials_g = self.calculator._compute_kspace(
                charges=q_g.unsqueeze(1),
                cell=cell_g,
                positions=pos_g,
                periodic=None,
                node_mask=None,
                kvectors=None,
            )

            e_g = self.ke * q_g.unsqueeze(1) * kspace_potentials_g
            atomic_e[atom_mask] = e_g

        return atomic_e

    def reset_output_convention(self, output_convention):
        pass  # interface compatibility


class PMEDispersionInteraction(nn.Module):
    """K-space-ONLY C6 dispersion energy via Particle Mesh Ewald / Ewald
    (torch-pme).

    Uses per-atom pseudo-charges q_i = sqrt(C6_i) (geometric-mean
    factorization, required for Ewald/PME additivity), matching
    so3lr_dev's DispersionEnergyKspace exactly.

    Uses a stock, unmodified ``torchpme.InversePowerLawPotential(exponent=6)``.
    No background-correction patch is needed here: torch-pme's own
    ``background_correction``/``lr_from_k_sq`` already handle exponent > 3
    correctly — for exponent > 3 the reciprocal-space k=0 term is finite and
    is computed directly by ``lr_from_k_sq`` (its ``k0_limit`` branch), so
    ``background_correction()`` correctly returns zero to avoid double
    counting that k=0 contribution. (torch-pme only needs a nonzero
    ``background_correction`` patch for exponent <= 3 — the Coulomb case —
    where the k=0 term is manually zeroed in ``lr_from_k_sq`` and must be
    patched back in for non-neutral systems; that case does not apply here.)

    This class computes the reciprocal-space (k-space) contribution ONLY —
    it passes an empty short-range neighbor list to torch-pme's
    ``Calculator.forward`` so that the real-space term it would otherwise
    add is trivially zero. This mirrors so3lr_dev's
    ``DispersionEnergyKspace``, whose docstring states it handles "the
    Casimir-Polder mixing-rule residual handled in real space by
    DispersionEnergySparse" — i.e. the real-space Casimir-Polder residual
    and the cancellation of the near-range portion of the geometric-mean
    long-range tail are handled entirely elsewhere, by
    ``DispersionInteraction`` (the residual formula
    ``V_C6_residual = V_C6_full + C6_geom_ij * lr_r_ij``), not by this
    class.
    """

    def __init__(
        self,
        *,
        smearing: float,
        mesh_spacing: float,
        interpolation_nodes: int = 4,
        do_ewald: bool = False,
    ) -> None:
        super().__init__()
        import torchpme  # lazy import — only required when use_pme_dispersion=True

        # torchpme itself operates in whatever length unit `positions`/
        # `cell` are given in at `forward` call time; since `forward`
        # converts positions/cell/lengths to Bohr (to match the Hartree
        # atomic-unit convention used for the C6 pseudo-charges, mirroring
        # so3lr_dev's DispersionEnergyKspace), `smearing`/`mesh_spacing`
        # must be baked into the (Bohr-space) potential/calculator here at
        # construction time — this is the "torchpme call boundary" for
        # these two parameters, since they are fixed at construction and
        # not passed again on every `forward` call.
        smearing_bohr = smearing / BOHR
        mesh_spacing_bohr = mesh_spacing / BOHR

        potential = torchpme.InversePowerLawPotential(
            exponent=6, smearing=smearing_bohr
        )
        self.calculator = (
            torchpme.EwaldCalculator(
                potential=potential, lr_wavelength=mesh_spacing_bohr
            )
            if do_ewald
            else torchpme.PMECalculator(
                potential=potential,
                mesh_spacing=mesh_spacing_bohr,
                interpolation_nodes=interpolation_nodes,
            )
        )

    def forward(
        self,
        c6_pseudo_charges: torch.Tensor,  # (N,) — sqrt(clamp(C6_i, min=0)), already computed by caller
        positions: torch.Tensor,  # (N, 3), Angstrom
        cell: torch.Tensor,  # (N_graphs, 3, 3) or (N_graphs*3, 3) — PyG-batched, same convention as PMEElectrostaticInteraction
        batch_segments: torch.Tensor,  # (N,)
        num_graphs: int,
        num_nodes: int,
    ) -> torch.Tensor:  # (N, 1) per-atom k-space-only dispersion energy in eV
        # PyG Batch concatenates [3,3] cells along dim 0 → [N*3,3].
        # Restore to [N,3,3] so cell[g] yields the correct [3,3] matrix.
        cell = cell.view(-1, 3, 3)

        input_dtype = positions.dtype
        device = positions.device
        bohr_factor = torch.tensor(BOHR, dtype=input_dtype, device=device)

        atomic_e = torch.zeros(
            num_nodes,
            1,
            dtype=input_dtype,
            device=device,
        )
        for g in range(num_graphs):
            atom_mask = batch_segments == g
            pos_g = positions[atom_mask] / bohr_factor  # (N_g, 3), Bohr
            c_g = c6_pseudo_charges[atom_mask]  # (N_g,)
            cell_g = cell[g] / bohr_factor  # (3, 3), Bohr
            if cell_g.abs().sum() == 0:
                raise ValueError(
                    "PME dispersion requires periodic boundary "
                    "conditions, but the cell for graph "
                    f"{g} is all zeros (non-periodic system). "
                    "Either set pbc=True and provide a cell, or "
                    "use use_pme_dispersion=False for non-periodic "
                    "systems."
                )

            # This class is k-space-ONLY (see class docstring): the
            # real-space term is made trivially zero by handing
            # torch-pme's Calculator.forward an empty neighbor list, so
            # only the reciprocal-space contribution is computed.
            empty_idx = torch.zeros((0, 2), dtype=torch.long, device=device)
            empty_dist = torch.zeros((0,), dtype=input_dtype, device=device)

            # charges: (N_g, 1); output: (N_g, 1)
            potentials_g = self.calculator.forward(
                charges=c_g.unsqueeze(1),
                cell=cell_g,
                positions=pos_g,
                neighbor_indices=empty_idx,
                neighbor_distances=empty_dist,
            )

            # E_i = -q_i * phi_i * Hartree (no 0.5, no ke prefactor)
            e_g = -c_g.unsqueeze(1) * potentials_g * HARTREE
            atomic_e[atom_mask] = e_g

        return atomic_e

    def reset_output_convention(self, output_convention):
        pass  # interface compatibility


class DispersionInteraction(nn.Module):
    """
    Dispersion energy calculation using vdW-QDO method with Hirshfeld ratios.
    """

    def __init__(
        self,
        *,
        neighborlist_format_lr: str = "sparse",
    ) -> None:
        super().__init__()

        self.c = 0.5 if neighborlist_format_lr == "sparse" else 1.0

    def forward(
        self,
        hirshfeld_ratios: torch.Tensor,
        atomic_numbers: torch.Tensor,
        senders_lr: torch.Tensor,
        receivers_lr: torch.Tensor,
        lengths_lr: torch.Tensor,
        num_nodes: int,
        cutoff_lr: Optional[float] = None,
        cutoff_lr_damping: Optional[float] = None,
        dispersion_energy_scale: float = 1.0,
        c6_ratios: Optional[torch.Tensor] = None,
        use_pme_dispersion: bool = False,
        pme_dispersion_smearing: Optional[float] = None,
        legacy_dispersion_bool: bool = True,
    ) -> torch.Tensor:
        """
        Compute per-node dispersion energy using potential described
        DOI: 10.1021/acs.jctc.3c00797.

        Args:
            hirshfeld_ratios: (N,) Hirshfeld volume ratios for each atom.
            atomic_numbers: (N,) atomic numbers.
            senders_lr: (E,) source indices (j) for long-range edges.
            receivers_lr: (E,) target indices (i) for long-range edges.
            lengths_lr: (E,) or (E,1) edge distances in Angstrom.
            num_nodes: number of nodes to scatter to.
            c6_ratios: (N,) optional per-atom C6 scaling ratios. When
                provided, C6 is scaled independently from alpha (new JAX
                behaviour). When None, C6 is scaled by hirshfeld_ratio²
                (backward-compatible).
            use_pme_dispersion: when True, the C6 term is replaced by its
                real-space residual (full damped C6 minus the geometric-
                mean long-range part that `PMEDispersionInteraction` adds
                back in k-space). C8/C10 and the switching-function cutoff
                are unaffected. Default False reproduces today's exact
                real-space-only behaviour byte-for-byte.
            pme_dispersion_smearing: PME-dispersion smearing width, in
                Angstrom (same convention as `PMEDispersionInteraction`'s
                `smearing`). Required (raises if None) when
                `use_pme_dispersion=True`; ignored otherwise.
            legacy_dispersion_bool: selects the free-atom alpha/C6
                reference table forwarded to `mixing_rules` (and, when
                `use_pme_dispersion=True`, to `atomic_c6_pseudo_charges`).
                True (default) reproduces today's exact table selection.

        Returns:
            (N,1) atomic dispersion energies.
        """

        if cutoff_lr is not None and cutoff_lr_damping is None:
            raise ValueError(
                f"cutoff_lr is set but cutoff_lr_damping is not. "
                f"Got cutoff_lr={cutoff_lr} and "
                f"cutoff_lr_damping={cutoff_lr_damping}"
            )

        if lengths_lr.dim() == 2 and lengths_lr.size(-1) == 1:
            lengths_lr = lengths_lr.squeeze(-1)
        if hirshfeld_ratios.dim() == 2 and hirshfeld_ratios.size(-1) == 1:
            hirshfeld_ratios = hirshfeld_ratios.squeeze(-1)
        if c6_ratios is not None and c6_ratios.dim() == 2:
            c6_ratios = c6_ratios.squeeze(-1)

        input_dtype = lengths_lr.dtype
        device = lengths_lr.device

        # Calculate alpha_ij and C6_ij using mixing rules
        alpha_ij, C6_ij = mixing_rules(
            atomic_numbers,
            receivers_lr,
            senders_lr,
            hirshfeld_ratios,
            c6_ratios=c6_ratios,
            legacy_dispersion_bool=legacy_dispersion_bool,
        )

        # Use cubic fit for gamma
        gamma_ij = gamma_cubic_fit(alpha_ij)

        # Convert distances to atomic units for dispersion calculation
        bohr_factor = torch.tensor(BOHR, dtype=input_dtype, device=device)
        distances_au = lengths_lr / bohr_factor

        if use_pme_dispersion:
            if pme_dispersion_smearing is None:
                raise ValueError(
                    "use_pme_dispersion=True requires "
                    "pme_dispersion_smearing to be set (got None)."
                )

            # --- PME-aware real-space splitting ---
            # C6 becomes a real-space residual: the full damped
            # Casimir-Polder C6 term minus the geometric-mean long-range
            # part that PMEDispersionInteraction adds back in k-space.
            # C8/C10 keep their existing form and cutoff treatment.
            C6_term, C8_term, C10_term, _p = _vdw_qdo_disp_damp_terms(
                distances_au,
                gamma_ij,
                C6_ij,
                alpha_ij,
                dispersion_energy_scale,
            )

            # Per-atom pseudo-charges (Task 1) gathered per edge — NOT the
            # mixing-rule C6_ij combination used in C6_term above.
            C6_atomic = atomic_c6_pseudo_charges(
                atomic_numbers,
                hirshfeld_ratios,
                c6_ratios=c6_ratios,
                legacy_dispersion_bool=legacy_dispersion_bool,
            )
            C6_i = C6_atomic[receivers_lr]
            C6_j = C6_atomic[senders_lr]
            C6_geom_ij = torch.sqrt(C6_i * C6_j)

            smearing_bohr = (
                torch.tensor(
                    pme_dispersion_smearing,
                    dtype=input_dtype,
                    device=device,
                )
                / bohr_factor
            )
            two_sigma2 = 2 * smearing_bohr**2
            x = distances_au**2 / two_sigma2
            exp_neg_x = torch.exp(-x)
            distances_au_safe = torch.where(
                distances_au > 0,
                distances_au,
                torch.ones_like(distances_au),
            )
            lr_r = (1 - exp_neg_x * (1 + x + 0.5 * x**2)) / (
                distances_au_safe**6
            )

            V_C6_residual = C6_term + C6_geom_ij * lr_r
            V_C8C10 = C8_term + C10_term

            # Apply smooth cutoff to C8/C10 only — the C6 residual is
            # deliberately left uncut (the Ewald split already localizes
            # it near zero well before cutoff_lr).
            if cutoff_lr is not None:
                w = switching_fn(
                    lengths_lr,
                    x_on=cutoff_lr - cutoff_lr_damping,
                    x_off=cutoff_lr,
                )
                mask = lengths_lr > 0
                w = torch.where(mask, w, torch.zeros_like(w))
                V_C8C10 = V_C8C10 * w

            hartree_factor = torch.tensor(
                HARTREE, dtype=input_dtype, device=device
            )
            dispersion_energy_ij = (
                self.c * (V_C6_residual + V_C8C10) * hartree_factor
            )
        else:
            # Get dispersion energy per edge
            dispersion_energy_ij = vdw_qdo_disp_damp(
                distances_au,
                gamma_ij,
                C6_ij,
                alpha_ij,
                dispersion_energy_scale,
                self.c,
            )

            # Apply smooth cutoff if specified
            if cutoff_lr is not None:
                # Apply switching function for smooth cutoff
                w = switching_fn(
                    lengths_lr,
                    x_on=cutoff_lr - cutoff_lr_damping,
                    x_off=cutoff_lr,
                )
                # Apply mask where distances > 0
                mask = lengths_lr > 0
                w = torch.where(mask, w, torch.zeros_like(w))
                dispersion_energy_ij = dispersion_energy_ij * w

        # Sum over edges for each node
        atomic_dispersion_energy = scatter_sum(
            src=dispersion_energy_ij,
            index=receivers_lr,
            dim=0,
            dim_size=num_nodes,
        )

        return atomic_dispersion_energy.unsqueeze(1)
