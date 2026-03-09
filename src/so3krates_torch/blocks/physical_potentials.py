import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any
import math
from so3krates_torch.tools.scatter import scatter_sum


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


def mixing_rules(
    atomic_numbers: torch.Tensor,
    idx_i: torch.Tensor,
    idx_j: torch.Tensor,
    hirshfeld_ratios: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Apply mixing rules to compute alpha_ij and C6_ij for dispersion"""
    dtype = hirshfeld_ratios.dtype
    device = hirshfeld_ratios.device

    # Move reference data to correct device and dtype
    alphas = ALPHAS.to(device=device, dtype=dtype)
    c6_coef = C6_COEF.to(device=device, dtype=dtype)

    atomic_number_i = atomic_numbers[idx_i] - 1  # Convert to 0-based indexing
    atomic_number_j = atomic_numbers[idx_j] - 1
    hirshfeld_ratio_i = hirshfeld_ratios[idx_i]
    hirshfeld_ratio_j = hirshfeld_ratios[idx_j]

    alpha_i = alphas[atomic_number_i] * hirshfeld_ratio_i
    C6_i = c6_coef[atomic_number_i] * torch.square(hirshfeld_ratio_i)
    alpha_j = alphas[atomic_number_j] * hirshfeld_ratio_j
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

    # Compute higher-order dispersion coefficients
    C8 = 5 / gamma * C6
    C10 = 245 / 8 / gamma**2 * C6
    p = gamma_scale * 2 * 2.54 * alpha_ij ** (1 / 7)

    # Compute potential
    R6 = torch.pow(R, 6)
    R8 = torch.pow(R, 8)
    R10 = torch.pow(R, 10)
    p6 = torch.pow(p, 6)
    p8 = torch.pow(p, 8)
    p10 = torch.pow(p, 10)

    V3 = -C6 / (R6 + p6) - C8 / (R8 + p8) - C10 / (R10 + p10)

    hartree_factor = torch.tensor(HARTREE, dtype=input_dtype, device=device)
    return c * V3 * hartree_factor


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
    ) -> None:
        super().__init__()
        if neighborlist_format_lr not in ("sparse", "ordered_sparse"):
            raise ValueError(
                "neighborlist_format_lr must be 'sparse' or 'ordered_sparse'"
            )
        c_val = 0.5 if neighborlist_format_lr == "sparse" else 1.0
        default_dtype = torch.get_default_dtype()
        self.register_buffer("ke", torch.tensor(ke, dtype=default_dtype))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=default_dtype))
        self.register_buffer("c", torch.tensor(c_val, dtype=default_dtype))
        self.neighborlist_format_lr = neighborlist_format_lr

    def forward(
        self,
        q: torch.Tensor,
        rij: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> torch.Tensor:
        # normalize shapes
        if q.dim() == 2 and q.size(-1) == 1:
            q = q.squeeze(-1)
        if rij.dim() == 2 and rij.size(-1) == 1:
            rij = rij.squeeze(-1)
        qi = q[receivers]
        qj = q[senders]

        r = rij.clamp_min(1e-12)
        pairwise = torch.erf(r / self.sigma) / r
        return self.c * self.ke * qi * qj * pairwise


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
    ) -> None:
        super().__init__()
        if neighborlist_format_lr not in ("sparse", "ordered_sparse"):
            raise ValueError(
                "neighborlist_format_lr must be 'sparse' or 'ordered_sparse'"
            )
        c_val = 0.5 if neighborlist_format_lr == "sparse" else 1.0
        default_dtype = torch.get_default_dtype()
        self.register_buffer("ke", torch.tensor(ke, dtype=default_dtype))
        self.register_buffer("sigma", torch.tensor(sigma, dtype=default_dtype))
        self.register_buffer(
            "cutoff", torch.tensor(cutoff, dtype=default_dtype)
        )
        self.register_buffer("cuton", torch.tensor(cuton, dtype=default_dtype))
        self.register_buffer("c", torch.tensor(c_val, dtype=default_dtype))
        self.neighborlist_format_lr = neighborlist_format_lr

    @torch.no_grad()
    def _potential(self, r: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        r = r.clamp_min(1e-12)
        return torch.erf(r / sigma) / r

    @torch.no_grad()
    def _force(self, r: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        # derivative of potential w.r.t r, used for force shift at cutoff
        r = r.clamp_min(1e-12)
        return (
            2
            * r
            * torch.exp(-((r / sigma) ** 2))
            / (math.sqrt(math.pi) * sigma)
            - torch.erf(r / sigma)
        ) / (r**2)

    def forward(
        self,
        q: torch.Tensor,
        rij: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
    ) -> torch.Tensor:
        if q.dim() == 2 and q.size(-1) == 1:
            q = q.squeeze(-1)
        if rij.dim() == 2 and rij.size(-1) == 1:
            rij = rij.squeeze(-1)

        # smooth switching
        f = switching_fn(rij, self.cuton, self.cutoff)

        r_safe = rij.clamp_min(1e-12)
        pairwise = torch.erf(r_safe / self.sigma) / r_safe
        shift = self._potential(self.cutoff, self.sigma)
        force_shift = self._force(self.cutoff, self.sigma)
        shifted_potential = (
            pairwise - shift - force_shift * (rij - self.cutoff)
        )
        qi = q[receivers]
        qj = q[senders]

        inside = rij < self.cutoff
        energy = (
            self.c
            * self.ke
            * qi
            * qj
            * (f * (pairwise - shift) + (1 - f) * shifted_potential)
        )
        return torch.where(inside, energy, torch.zeros_like(energy))


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
    ) -> torch.Tensor:
        """Compute per-node electrostatic energy.

        Args:
            partial_charges: (N,) or (N,1) charges.
            senders_lr: (E,) source indices (j).
            receivers_lr: (E,) target indices (i).
            lengths_lr: (E,) or (E,1) edge distances.
            num_nodes: number of nodes to scatter to.

        Returns:
            (N,1) atomic electrostatic energies.
        """
        if cutoff_lr is not None:
            cuton = 0.45 * float(cutoff_lr)
            self.coulomb = CoulombErfShiftedForceSmooth(
                ke=self.ke,
                sigma=electrostatic_energy_scale,
                cutoff=float(cutoff_lr),
                cuton=cuton,
                neighborlist_format_lr=self.neighborlist_format_lr,
            )
        else:
            self.coulomb = CoulombErf(
                ke=self.ke,
                sigma=electrostatic_energy_scale,
                neighborlist_format_lr=self.neighborlist_format_lr,
            )
        if lengths_lr.dim() == 2 and lengths_lr.size(-1) == 1:
            lengths_lr = lengths_lr.squeeze(-1)

        edge_e = self.coulomb(
            partial_charges, lengths_lr, senders_lr, receivers_lr
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
        cutoff_lr: float | None = None,
        cutoff_lr_damping: float | None = None,
        dispersion_energy_scale: float = 1.0,
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
            node_mask: (N,) optional node mask for padded nodes.

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

        input_dtype = lengths_lr.dtype
        device = lengths_lr.device

        # Calculate alpha_ij and C6_ij using mixing rules
        alpha_ij, C6_ij = mixing_rules(
            atomic_numbers, receivers_lr, senders_lr, hirshfeld_ratios
        )

        # Use cubic fit for gamma
        gamma_ij = gamma_cubic_fit(alpha_ij)

        # Convert distances to atomic units for dispersion calculation
        bohr_factor = torch.tensor(BOHR, dtype=input_dtype, device=device)
        distances_au = lengths_lr / bohr_factor

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
