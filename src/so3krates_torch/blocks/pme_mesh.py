"""PME mesh-size rule shared with the JAX reference (so3lr_dev).

so3lr_dev's jax-pme fork (``jaxpme.kspace.get_kgrid_mesh_shape``) picks the
number of mesh points along each cell axis as

    n_i = 2 ** ceil(log2(2 * ceil(L_i / h) + 1))

whereas stock torch-pme (``torchpme.lib.kvectors.get_ns_mesh``) omits the
inner ``ceil``. Near power-of-two boundaries the two rules choose different
meshes (7.849 A at h=0.25 A: 128 vs 64 points), which changes energies by
~0.2 eV. The JAX mesh is never smaller than the stock one.

``JaxMeshPMECalculator`` uses the JAX rule; ``make_pme_calculator`` selects
between it (``"jax"``, the default everywhere) and stock torch-pme
(``"torchpme"``).
"""

import torch
import torchpme

PME_MESH_RULES = ("jax", "torchpme")


def jax_ns_mesh(cell: torch.Tensor, mesh_spacing: float) -> torch.Tensor:
    """Mesh size per cell axis using the jax-pme rule.

    :param cell: ``(3, 3)`` tensor, rows are the lattice vectors
    :param mesh_spacing: target mesh spacing, in the unit of ``cell``
    :return: ``(3,)`` ``long`` tensor
    """
    basis_norms = torch.linalg.norm(cell.to(torch.float64), dim=1)
    ewald_shape = torch.ceil(basis_norms / mesh_spacing)
    actual = 2 * ewald_shape + 1
    return torch.pow(2, torch.ceil(torch.log2(actual)).long())


class JaxMeshPMECalculator(torchpme.PMECalculator):
    """``torchpme.PMECalculator`` with the jax-pme mesh-size rule.

    ``_compute_kspace`` is a copy of torch-pme 0.5.0's with the single
    ``get_ns_mesh`` call replaced by ``jax_ns_mesh``; re-verify it when
    upgrading torch-pme (``test_torchpme_version_pinned``).
    """

    def _compute_kspace(
        self,
        charges: torch.Tensor,
        cell: torch.Tensor,
        positions: torch.Tensor,
        periodic: torch.Tensor | None = None,
        node_mask: torch.Tensor | None = None,
        kvectors: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if node_mask is not None or kvectors is not None:
            raise NotImplementedError(
                "Batching not implemented for mesh-based calculators"
            )
        ns = jax_ns_mesh(cell, self.mesh_spacing)

        self.mesh_interpolator.update(cell, ns)

        self.kspace_filter.update(cell, ns)

        self.mesh_interpolator.compute_weights(positions)
        rho_mesh = self.mesh_interpolator.points_to_mesh(
            particle_weights=charges
        )

        potential_mesh = self.kspace_filter.forward(rho_mesh)

        ivolume = torch.abs(cell.det()).pow(-1)
        interpolated_potential = (
            self.mesh_interpolator.mesh_to_points(potential_mesh) * ivolume
        )

        interpolated_potential -= charges * self.potential.self_contribution()

        charge_tot = torch.sum(charges, dim=0)
        prefac = self.potential.background_correction()
        interpolated_potential -= 2 * prefac * charge_tot * ivolume

        interpolated_potential += self.potential.pbc_correction(
            periodic, positions, cell, charges
        )

        return interpolated_potential / 2


def make_pme_calculator(mesh_rule: str, **kwargs) -> torchpme.PMECalculator:
    """``JaxMeshPMECalculator`` for ``"jax"``, stock
    ``torchpme.PMECalculator`` for ``"torchpme"``."""
    if mesh_rule == "jax":
        return JaxMeshPMECalculator(**kwargs)
    if mesh_rule == "torchpme":
        return torchpme.PMECalculator(**kwargs)
    raise ValueError(
        f"pme_mesh_rule must be one of {PME_MESH_RULES}, got {mesh_rule!r}"
    )
