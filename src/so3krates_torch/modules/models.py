import torch
from typing import Any, Callable, Dict, List, Optional, Type, Union
from so3krates_torch.data.utils import prepare_graph
from so3krates_torch.modules.cutoff import cutoff_fn_dict
from so3krates_torch.modules.spherical_harmonics import RealSphericalHarmonics
from so3krates_torch.blocks import (
    embedding,
    euclidean_transformer,
)
from so3krates_torch.blocks.output_block import (
    AtomicEnergyOutputHead,
    MultiAtomicEnergyOutputHead,
    PartialChargesOutputHead,
    DipoleVecOutputHead,
    HirshfeldOutputHead,
)
from so3krates_torch.blocks import radial_basis
import math
from so3krates_torch.blocks.physical_potentials import (
    ZBLRepulsion,
    ElectrostaticInteraction,
    DispersionInteraction,
)
from so3krates_torch.tools import scatter
from so3krates_torch.tools import utils


class So3krates(torch.nn.Module):
    def __init__(
        self,
        r_max: float,
        num_radial_basis_fn: int,
        degrees: List[int],
        num_features: int,
        num_heads: int,
        num_layers: int,
        num_elements: int = 118,
        avg_num_neighbors: Optional[float] = None,
        final_mlp_layers: int = 2,
        energy_regression_dim: Optional[int] = None,
        message_normalization: str = "avg_num_neighbors",
        initialize_ev_to_zeros: bool = True,
        radial_basis_fn: str = "gaussian",
        trainable_rbf: bool = False,
        atomic_type_shifts: Optional[dict[str, float]] = None,
        energy_learn_atomic_type_shifts: bool = False,
        energy_learn_atomic_type_scales: bool = False,
        layer_normalization_1: bool = False,
        layer_normalization_2: bool = False,
        residual_mlp_1: bool = False,
        residual_mlp_2: bool = False,
        use_charge_embed: bool = False,
        use_spin_embed: bool = False,
        interaction_bias: bool = True,
        qk_non_linearity: str = "identity",
        cutoff_fn: str = "cosine",
        cutoff_p: int = 5,
        activation_fn: str = "silu",
        energy_activation_fn: str = "silu",
        seed: int = 42,
        device: Union[str, torch.device] = "cpu",
        dtype: Union[str, torch.dtype] = torch.float32,
        layers_behave_like_identity_fn_at_init: bool = False,
        output_is_zero_at_init: bool = False,
        input_convention: str = "positions",
        num_features_head: Optional[
            int
        ] = None,  # not used; just for compatibility with jax version
    ):
        super().__init__()

        if layers_behave_like_identity_fn_at_init:
            raise NotImplementedError(
                "Layers behaving like identity functions at initialization is not implemented."
            )

        if output_is_zero_at_init:
            raise NotImplementedError(
                "Output being zero at initialization is not implemented."
            )

        if input_convention not in ["positions"]:
            raise ValueError(
                f"Unknown input convention: {input_convention}"
                "Only 'positions' is supported at the moment."
            )
        self.device = device
        if isinstance(dtype, str):
            dtype = getattr(torch, dtype)
        torch.set_default_dtype(dtype)
        torch.manual_seed(seed)

        # Store all constructor arguments as model attributes
        self.r_max = r_max
        self.num_radial_basis_fn = num_radial_basis_fn
        self.degrees = degrees
        self.num_features = num_features
        self.num_heads = num_heads
        self.num_layers = num_layers
        self.num_elements = num_elements
        self.avg_num_neighbors_param = (
            avg_num_neighbors  # Store original parameter
        )
        self.final_mlp_layers = final_mlp_layers
        self.energy_regression_dim = energy_regression_dim
        self.message_normalization = message_normalization
        self.initialize_ev_to_zeros = initialize_ev_to_zeros
        self.radial_basis_fn = radial_basis_fn
        self.trainable_rbf = trainable_rbf
        self.atomic_type_shifts = atomic_type_shifts
        self.energy_learn_atomic_type_shifts = energy_learn_atomic_type_shifts
        self.energy_learn_atomic_type_scales = energy_learn_atomic_type_scales
        self.layer_normalization_1 = layer_normalization_1
        self.layer_normalization_2 = layer_normalization_2
        self.residual_mlp_1 = residual_mlp_1
        self.residual_mlp_2 = residual_mlp_2
        self.use_charge_embed = use_charge_embed
        self.use_spin_embed = use_spin_embed
        self.interaction_bias = interaction_bias
        self.qk_non_linearity = qk_non_linearity
        self.cutoff_fn_name = cutoff_fn
        self.cutoff_p = cutoff_p
        self.activation_fn_name = activation_fn
        self.energy_activation_fn_name = energy_activation_fn
        self.seed = seed
        self.dtype = dtype
        self.layers_behave_like_identity_fn_at_init = (
            layers_behave_like_identity_fn_at_init
        )
        self.output_is_zero_at_init = output_is_zero_at_init
        self.input_convention = input_convention
        self.num_features_head = num_features_head

        if cutoff_fn == "polynomial":
            self.cutoff_fn = cutoff_fn_dict[cutoff_fn](r_max, p=cutoff_p)
        else:
            self.cutoff_fn = cutoff_fn_dict[cutoff_fn](r_max)

        self.spherical_harmonics = RealSphericalHarmonics(
            degrees=self.degrees,
        )

        self.activation_fn = utils.activation_fn_dict.get(
            activation_fn, torch.nn.SiLU
        )
        self.energy_activation_fn = utils.activation_fn_dict.get(
            energy_activation_fn, torch.nn.SiLU
        )
        qk_non_linearity = utils.activation_fn_dict.get(
            qk_non_linearity, torch.nn.Identity
        )

        self.inv_feature_embedding = embedding.InvariantEmbedding(
            num_elements=self.num_elements,
            out_features=self.num_features,
            bias=False,
        )
        self.num_embeddings = 1
        if self.use_charge_embed:
            self.charge_embedding = embedding.ChargeSpinEmbedding(
                num_features=self.num_features,
                activation_fn=self.activation_fn,
                num_elements=self.num_elements,
            )
            self.num_embeddings += 1
        if self.use_spin_embed:
            self.spin_embedding = embedding.ChargeSpinEmbedding(
                num_features=self.num_features,
                activation_fn=self.activation_fn,
                num_elements=self.num_elements,
            )
            self.num_embeddings += 1
        self.embedding_scale = math.sqrt(self.num_embeddings)
        self.ev_embedding = embedding.EuclideanEmbedding(
            initialization_to_zeros=self.initialize_ev_to_zeros,
        )
        self.avg_num_neighbors = (
            1.0 if avg_num_neighbors is None else avg_num_neighbors
        )

        self.radial_embedding = radial_basis.ComputeRBF(
            r_max=r_max,
            num_radial_basis_fn=self.num_radial_basis_fn,
            radial_basis_fn=self.radial_basis_fn,
            trainable=self.trainable_rbf,
        )

        self.euclidean_transformers = torch.nn.ModuleList(
            [
                euclidean_transformer.EuclideanTransformer(
                    degrees=self.degrees,
                    num_heads=self.num_heads,
                    num_features=self.num_features,
                    num_radial_basis_fn=self.num_radial_basis_fn,
                    interaction_bias=self.interaction_bias,
                    message_normalization=self.message_normalization,
                    avg_num_neighbors=self.avg_num_neighbors,
                    layer_normalization_1=self.layer_normalization_1,
                    layer_normalization_2=self.layer_normalization_2,
                    residual_mlp_1=self.residual_mlp_1,
                    residual_mlp_2=self.residual_mlp_2,
                    activation_fn=self.activation_fn,
                    qk_non_linearity=qk_non_linearity,
                    device=self.device,
                )
                for _ in range(self.num_layers)
            ]
        )
        self.atomic_energy_output_block = AtomicEnergyOutputHead(
            num_features=self.num_features,
            energy_regression_dim=self.energy_regression_dim,
            final_output_features=1,  # TODO: remove hardcoded value
            layers=self.final_mlp_layers,
            bias=True,
            non_linearity=self.energy_activation_fn,
            final_non_linearity=False,
            use_non_linearity=True,
            atomic_type_shifts=self.atomic_type_shifts,
            energy_learn_atomic_type_shifts=self.energy_learn_atomic_type_shifts,
            energy_learn_atomic_type_scales=self.energy_learn_atomic_type_scales,
            num_elements=self.num_elements,
            device=self.device,
        )

    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict with backward compatibility for models saved before
        degree_repeats and vera_* matrices were converted to buffers.
        """
        # Check if state dict is missing the new buffer keys
        missing_buffer_keys = [
            key
            for key in self.state_dict().keys()
            if "degree_repeats" in key or "vera_" in key
        ]

        has_missing_buffers = any(
            key not in state_dict for key in missing_buffer_keys
        )

        if has_missing_buffers and strict:
            # Old state dict detected - load with strict=False
            # The missing buffers are already initialized correctly in __init__
            print(
                "Loading old state dict format (missing degree_repeats/vera buffers). "
                "Using strict=False to skip non-trainable computed buffers."
            )
            return super().load_state_dict(state_dict, strict=False)
        else:
            # New state dict or strict=False explicitly requested
            return super().load_state_dict(state_dict, strict=strict)

    def _get_graph(
        self,
        data: Dict[str, torch.Tensor],
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        lammps_mliap: bool = False,
    ):
        ######### PROCESSING DATA #########
        self.ctx = prepare_graph(
            data=data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )

    def get_representation(
        self,
        data: Dict[str, torch.Tensor],
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        return_att: bool = False,
    ):

        self._get_graph(
            data=data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
        )
        self.is_lammps = self.ctx.is_lammps
        self.num_atoms_arange = self.ctx.num_atoms_arange
        self.num_graphs = self.ctx.num_graphs
        self.displacement = self.ctx.displacement
        self.positions = self.ctx.positions
        self.vectors = -1.0 * self.ctx.vectors
        self.lengths = self.ctx.lengths
        self.cell = self.ctx.cell
        self.node_heads = self.ctx.node_heads
        self.interaction_kwargs = self.ctx.interaction_kwargs
        self.lammps_natoms = self.interaction_kwargs.lammps_natoms
        self.lammps_class = self.interaction_kwargs.lammps_class
        self.receivers, self.senders = (
            data["edge_index"][0],
            data["edge_index"][1],
        )
        self.head_idxs = data["head"]
        # normalize the vectors to unit length
        self.vectors_unit = self.vectors / (
            self.vectors.norm(dim=-1, keepdim=True) + 1e-8
        )
        sh_vectors = self.spherical_harmonics(self.vectors_unit)
        self.cutoffs = self.cutoff_fn(self.lengths)

        ######### EMBEDDING #########
        inv_features = self.inv_feature_embedding(data["node_attrs"])

        if self.use_charge_embed:
            inv_features += self.charge_embedding(
                elements_one_hot=data["node_attrs"],
                psi=data["total_charge"],
                batch_segments=data["batch"],
                num_graphs=self.num_graphs,
            )
        if self.use_spin_embed:
            #  We use number of unpaired electrons = 2*total_spin.
            spin_embedding = self.spin_embedding(
                elements_one_hot=data["node_attrs"],
                psi=data["total_spin"] * 2,
                batch_segments=data["batch"],
                num_graphs=self.num_graphs,
            )
            inv_features += spin_embedding
        # never mentionend in the paper, but done in the JAX code ...
        inv_features /= self.embedding_scale
        ev_features = self.ev_embedding(
            sh_vectors=sh_vectors,
            cutoffs=self.cutoffs,
            receivers=self.receivers,
            avg_num_neighbors=self.avg_num_neighbors,
            num_nodes=inv_features.shape[0],
        )

        rbf = self.radial_embedding(self.lengths)

        ######### TRANSFORMER #########
        n_real = self.lammps_natoms[0] if self.is_lammps else 0
        n_total = self.lammps_natoms[1] if self.is_lammps else 0
        has_ghosts = self.is_lammps and n_total > n_real

        if return_att:
            att_scores = {"inv": {}, "ev": {}}
        for layer_idx, transformer in enumerate(self.euclidean_transformers):
            # Sync ghost atom features before each layer (except the first,
            # where all n_total features are already correct from embedding).
            # After each layer we truncate to n_real; here we pad back to
            # n_total and use LAMMPS reverse-communication to copy the real
            # atom features into their ghost copies.
            if has_ghosts and layer_idx > 0:
                inv_features = self._lammps_pad_and_sync(inv_features)
                ev_features = self._lammps_pad_and_sync(ev_features)

            transformer_output = transformer(
                inv_features=inv_features,
                ev_features=ev_features,
                rbf=rbf,
                senders=self.senders,
                receivers=self.receivers,
                sh_vectors=sh_vectors,
                cutoffs=self.cutoffs,
                return_att=return_att,
            )
            if return_att:
                (inv_features, ev_features, (alpha_inv, alpha_ev)) = (
                    transformer_output
                )
                att_scores["inv"][layer_idx] = alpha_inv
                att_scores["ev"][layer_idx] = alpha_ev

            else:
                inv_features, ev_features = transformer_output

            # After each layer, truncate to real atoms only.  Ghost atom
            # features are stale (they never receive messages because they
            # only appear as senders, never receivers).  Keeping them would
            # pollute the next layer.
            if has_ghosts:
                inv_features = inv_features[:n_real]
                ev_features = ev_features[:n_real]

        if has_ghosts:
            # Final pad+sync so downstream code (output block, energy sum)
            # sees n_total features.  The output block only reads [:n_real],
            # but the autograd graph for edge_forces needs features at ghost
            # indices to be connected to the real atom features.
            inv_features = self._lammps_pad_and_sync(inv_features)
            ev_features = self._lammps_pad_and_sync(ev_features)

        if return_att:
            return inv_features, ev_features, att_scores
        else:
            return inv_features, ev_features

    def _lammps_pad_and_sync(self, features: torch.Tensor) -> torch.Tensor:
        """Pad real-atom features with zeros for ghosts, then sync via LAMMPS.

        After truncation to n_real atoms, ghost atoms have no features.
        This pads the tensor back to n_total and calls LAMMPS
        forward_exchange to copy real atom features to their ghost copies.
        The backward pass (reverse_exchange) accumulates ghost gradients
        back onto the real atoms, which is essential for correct edge forces.
        """
        n_real, n_total = self.lammps_natoms
        n_ghost = n_total - n_real
        pad = torch.zeros(
            (n_ghost, features.shape[1]),
            dtype=features.dtype,
            device=features.device,
        )
        features = torch.cat((features, pad), dim=0)
        features = utils.LAMMPS_MP.apply(features, self.lammps_class)
        return features

    def _create_output_dict(
        self,
        total_energy: torch.Tensor,
        forces: torch.Tensor,
        virials: torch.Tensor,
        stress: torch.Tensor,
        hessian: torch.Tensor,
        edge_forces: torch.Tensor,
        inv_features: Optional[torch.Tensor],
        att_scores: Optional[torch.Tensor],
        training: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:
        return {
            "energy": total_energy,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "hessian": hessian,
            "edge_forces": edge_forces,
            "inv_features": inv_features,
            "att_scores": att_scores,
        }

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        return_descriptors: bool = False,
        return_att: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:

        self.batch_segments = data["batch"]
        self.data_ptr = data["ptr"]

        repr_output = self.get_representation(
            data=data,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
            lammps_mliap=lammps_mliap,
            return_att=return_att,
        )
        if return_att:
            inv_features, ev_features, att_scores = repr_output
        else:
            inv_features, ev_features = repr_output

        ######### OUTPUT #########
        atomic_energies = self.atomic_energy_output_block(
            inv_features,
            data,
        )

        if self.is_lammps:
            n_real = self.lammps_natoms[0]
            total_energy = scatter.scatter_sum(
                src=atomic_energies[:n_real],
                index=self.batch_segments[:n_real],
                dim=0,
                dim_size=1,
            ).squeeze(-1)
        else:
            total_energy = scatter.scatter_sum(
                src=atomic_energies,
                index=self.batch_segments,
                dim=0,
                dim_size=self.data_ptr.shape[0] - 1,
            ).squeeze(-1)

        forces, virials, stress, hessian, edge_forces = utils.get_outputs(
            energy=total_energy,
            positions=self.positions,
            displacement=self.displacement,
            vectors=self.ctx.vectors,
            cell=self.cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
        )
        return self._create_output_dict(
            total_energy=total_energy,
            forces=forces,
            virials=virials,
            stress=stress,
            hessian=hessian,
            edge_forces=edge_forces,
            att_scores=att_scores if return_att else None,
            inv_features=inv_features if return_descriptors else None,
            training=training,
        )


class SO3LR(So3krates):
    def __init__(
        self,
        zbl_repulsion_bool: bool = True,
        electrostatic_energy_bool: bool = True,
        electrostatic_energy_scale: float = 4.0,
        dispersion_energy_bool: bool = True,
        dispersion_energy_scale: float = 1.2,
        dispersion_energy_cutoff_lr_damping: float = None,
        r_max_lr: float = 12.0,
        neighborlist_format_lr: str = "sparse",
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Store SO3LR-specific constructor arguments as attributes
        self.zbl_repulsion_bool = zbl_repulsion_bool
        self.electrostatic_energy_bool = electrostatic_energy_bool
        self.electrostatic_energy_scale = electrostatic_energy_scale
        self.dispersion_energy_bool = dispersion_energy_bool
        self.dispersion_energy_scale = dispersion_energy_scale
        self.dispersion_energy_cutoff_lr_damping = (
            dispersion_energy_cutoff_lr_damping
        )
        self.neighborlist_format_lr = neighborlist_format_lr

        if r_max_lr is not None:
            self.r_max_lr = r_max_lr
        else:
            self.r_max_lr = None

        self.use_lr = False

        # Short-range repulsion
        self.zbl_repulsion = ZBLRepulsion()

        # i'm sorry for this. but in the jax version they always initiate
        # the long range modules even though the booleans are set to false.
        # if didn't do this there would be problems when converting models
        # to jax back and forth ... its ugly and wasteful, i know.

        # Electrostatics
        if self.electrostatic_energy_bool:
            self.use_lr = True
        self.partial_charges_output_block = PartialChargesOutputHead(
            num_features=self.num_features,
            regression_dim=self.energy_regression_dim,
            activation_fn=self.energy_activation_fn,
        )
        self.dipole_output_head = DipoleVecOutputHead()
        self.electrostatic_potential = ElectrostaticInteraction(
            neighborlist_format_lr=self.neighborlist_format_lr
        )

        # Dispersion
        if self.dispersion_energy_bool:
            self.use_lr = True
        self.hirshfeld_output_block = HirshfeldOutputHead(
            num_features=self.num_features,
            regression_dim=self.energy_regression_dim,
            activation_fn=self.energy_activation_fn,
        )
        self.dispersion_potential = DispersionInteraction(
            neighborlist_format_lr=self.neighborlist_format_lr
        )

    def load_state_dict(self, state_dict, strict=True):
        """
        Load state dict with backward compatibility for models saved before
        degree_repeats and vera_* matrices were converted to buffers.
        """
        # Check if state dict is missing the new buffer keys
        missing_buffer_keys = [
            key
            for key in self.state_dict().keys()
            if "degree_repeats" in key or "vera_" in key
        ]

        has_missing_buffers = any(
            key not in state_dict for key in missing_buffer_keys
        )

        if has_missing_buffers and strict:
            # Old state dict detected - load with strict=False
            # The missing buffers are already initialized correctly in __init__
            print(
                "Loading old state dict format (missing degree_repeats/vera buffers). "
                "Using strict=False to skip non-trainable computed buffers."
            )
            return super().load_state_dict(state_dict, strict=False)
        else:
            # New state dict or strict=False explicitly requested
            return super().load_state_dict(state_dict, strict=strict)

    def _get_graph(
        self,
        data: Dict[str, torch.Tensor],
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        lammps_mliap: bool = False,
    ):
        ######### PROCESSING DATA #########
        self.ctx = prepare_graph(
            data=data,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            lammps_mliap=lammps_mliap,
            lr=self.use_lr,
        )

    def _combine_energies(
        self,
        atomic_energies: torch.Tensor,
        zbl_atomic_energies: Optional[torch.Tensor] = None,
        electrostatic_energies: Optional[torch.Tensor] = None,
        dispersion_energies: Optional[torch.Tensor] = None,
    ):

        torch.set_printoptions(precision=8)
        if self.zbl_repulsion_bool and zbl_atomic_energies is not None:
            atomic_energies += zbl_atomic_energies
        if (
            self.electrostatic_energy_bool
            and electrostatic_energies is not None
        ):
            atomic_energies += electrostatic_energies
        if self.dispersion_energy_bool and dispersion_energies is not None:
            atomic_energies += dispersion_energies
        return atomic_energies

    def _get_outputs(
        self,
        energy: torch.Tensor,
        positions: torch.Tensor,
        displacement: torch.Tensor,
        vectors: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        batch: Optional[torch.tensor] = None,
    ):
        return utils.get_outputs(
            energy=energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
        )

    def _create_output_dict(
        self,
        total_energy: torch.Tensor,
        forces: torch.Tensor,
        virials: torch.Tensor,
        stress: torch.Tensor,
        hessian: torch.Tensor,
        edge_forces: torch.Tensor,
        zbl_atomic_energies: torch.Tensor,
        partial_charges: torch.Tensor,
        dipole: torch.Tensor,
        hirshfeld_ratios: torch.Tensor,
        inv_features: torch.Tensor,
        att_scores: torch.Tensor,
        node_energy: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:

        return {
            "energy": total_energy,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "hessian": hessian,
            "edge_forces": edge_forces,
            "node_energy": node_energy,
            "zbl_repulsion": zbl_atomic_energies,
            "partial_charges": partial_charges,
            "dipole": dipole,
            "hirshfeld_ratios": hirshfeld_ratios,
            "inv_features": inv_features,
            "att_scores": att_scores,
        }

    def forward(
        self,
        data: Dict[str, torch.Tensor],
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_displacement: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        return_descriptors: bool = False,
        compute_atomic_stresses: bool = False,
        lammps_mliap: bool = False,
        return_att: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:

        self.batch_segments = data["batch"]
        self.data_ptr = data["ptr"]
        repr_output = self.get_representation(
            data=data,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_displacement=compute_displacement,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
            lammps_mliap=lammps_mliap,
            return_att=return_att,
        )

        if return_att:
            inv_features, ev_features, att_scores = repr_output
        else:
            inv_features, ev_features = repr_output
        ######### OUTPUT #########
        atomic_energies = self.atomic_energy_output_block(
            inv_features,
            data,
            atomic_numbers=data["atomic_numbers"],
        )

        if self.use_lr:
            self.receivers_lr, self.senders_lr = (
                data["edge_index_lr"][0],
                data["edge_index_lr"][1],
            )
            self.lengths_lr = self.ctx.lengths_lr

        zbl_atomic_energies = None
        if self.zbl_repulsion_bool:
            zbl_atomic_energies = self.zbl_repulsion(
                atomic_numbers=data["atomic_numbers"],
                cutoffs=self.cutoffs,
                senders=self.senders,
                receivers=self.receivers,
                lengths=self.lengths,
                num_nodes=inv_features.shape[0],
            )
        electrostatic_energies = None
        if self.electrostatic_energy_bool:
            partial_charges = self.partial_charges_output_block(
                inv_features=inv_features,
                atomic_numbers=data["atomic_numbers"],
                total_charge=data["total_charge"],
                batch_segments=self.batch_segments,
                num_graphs=self.num_graphs,
            )

            dipole = self.dipole_output_head(
                partial_charges=partial_charges,
                positions=self.positions,
                batch_segments=self.batch_segments,
                num_graphs=self.num_graphs,
            )
            electrostatic_energies = self.electrostatic_potential(
                partial_charges=partial_charges,
                senders_lr=self.senders_lr,
                receivers_lr=self.receivers_lr,
                lengths_lr=self.lengths_lr,
                num_nodes=inv_features.shape[0],
                cutoff_lr=self.r_max_lr,
                electrostatic_energy_scale=self.electrostatic_energy_scale,
            )
        dispersion_energies = None
        if self.dispersion_energy_bool:
            hirshfeld_ratios = self.hirshfeld_output_block(
                inv_features=inv_features,
                atomic_numbers=data["atomic_numbers"],
            )
            dispersion_energies = self.dispersion_potential(
                hirshfeld_ratios=hirshfeld_ratios,
                atomic_numbers=data["atomic_numbers"],
                senders_lr=self.senders_lr,
                receivers_lr=self.receivers_lr,
                lengths_lr=self.lengths_lr,
                num_nodes=inv_features.shape[0],
                cutoff_lr=self.r_max_lr,
                cutoff_lr_damping=self.dispersion_energy_cutoff_lr_damping,
                dispersion_energy_scale=self.dispersion_energy_scale,
            )

        atomic_energies = self._combine_energies(
            atomic_energies=atomic_energies,
            zbl_atomic_energies=zbl_atomic_energies,
            electrostatic_energies=electrostatic_energies,
            dispersion_energies=dispersion_energies,
        )
        # In LAMMPS mode, only sum real (non-ghost) atom energies so that
        # edge forces derived via autograd do not include ghost contributions.
        if self.is_lammps:
            n_real = self.lammps_natoms[0]
            total_energy = scatter.scatter_sum(
                src=atomic_energies[:n_real],
                index=self.batch_segments[:n_real],
                dim=0,
                dim_size=1,
            ).squeeze(-1)
        else:
            total_energy = scatter.scatter_sum(
                src=atomic_energies,
                index=self.batch_segments,
                dim=0,
                dim_size=self.data_ptr.shape[0] - 1,
            ).squeeze(-1)
        forces, virials, stress, hessian, edge_forces = self._get_outputs(
            energy=total_energy,
            positions=self.positions,
            displacement=self.displacement,
            # Use original (non-negated) vectors for correct edge force sign.
            # self.vectors is negated (-1 * ctx.vectors) for So3krates' internal
            # convention, but edge forces must be w.r.t. the original LAMMPS vectors.
            vectors=self.ctx.vectors,
            cell=self.cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
            batch=data["batch"],
        )

        return self._create_output_dict(
            total_energy=total_energy,
            forces=forces,
            virials=virials,
            stress=stress,
            hessian=hessian,
            edge_forces=edge_forces,
            zbl_atomic_energies=zbl_atomic_energies,
            partial_charges=(
                partial_charges if self.electrostatic_energy_bool else None
            ),
            dipole=dipole if self.electrostatic_energy_bool else None,
            hirshfeld_ratios=(
                hirshfeld_ratios if self.dispersion_energy_bool else None
            ),
            inv_features=inv_features if return_descriptors else None,
            att_scores=att_scores if return_att else None,
            node_energy=atomic_energies,
            training=training,
        )


class MultiHeadSO3LR(SO3LR):
    def __init__(
        self,
        num_output_heads: int = 8,
        *args,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.num_output_heads = num_output_heads
        self.atomic_energy_output_block = MultiAtomicEnergyOutputHead(
            num_features=self.num_features,
            energy_regression_dim=self.energy_regression_dim,
            final_output_features=1,  # TODO: remove hardcoded value
            layers=self.final_mlp_layers,
            bias=True,
            non_linearity=self.energy_activation_fn,
            final_non_linearity=False,
            use_non_linearity=True,
            atomic_type_shifts=self.atomic_type_shifts,
            energy_learn_atomic_type_shifts=self.energy_learn_atomic_type_shifts,
            energy_learn_atomic_type_scales=self.energy_learn_atomic_type_scales,
            num_elements=self.num_elements,
            num_output_heads=self.num_output_heads,
            device=self.device,
        )
        self.select_heads = False
        self.return_mean = False

    def _get_outputs(
        self,
        energy: torch.Tensor,
        positions: torch.Tensor,
        displacement: torch.Tensor,
        vectors: torch.Tensor,
        cell: Optional[torch.Tensor] = None,
        training: bool = False,
        compute_force: bool = True,
        compute_virials: bool = False,
        compute_stress: bool = False,
        compute_hessian: bool = False,
        compute_edge_forces: bool = False,
        batch: Optional[torch.tensor] = None,
    ):

        forces, virials, stress, hessian, edge_forces = utils.get_outputs(
            energy=energy,
            positions=positions,
            displacement=displacement,
            vectors=vectors,
            cell=cell,
            training=training,
            compute_force=compute_force,
            compute_virials=compute_virials,
            compute_stress=compute_stress,
            compute_hessian=compute_hessian,
            compute_edge_forces=compute_edge_forces,
            is_multihead=True,
            batch=batch,
        )
        return forces, virials, stress, hessian, edge_forces

    def _combine_energies(
        self,
        atomic_energies: torch.Tensor,
        zbl_atomic_energies: Optional[torch.Tensor] = None,
        electrostatic_energies: Optional[torch.Tensor] = None,
        dispersion_energies: Optional[torch.Tensor] = None,
    ):
        # atomic_energies has shape (num_atoms, num_output_heads)
        if self.zbl_repulsion_bool and zbl_atomic_energies is not None:
            atomic_energies += zbl_atomic_energies.unsqueeze(-1)
        if (
            self.electrostatic_energy_bool
            and electrostatic_energies is not None
        ):
            atomic_energies += electrostatic_energies.unsqueeze(-1)
        if self.dispersion_energy_bool and dispersion_energies is not None:
            atomic_energies += dispersion_energies.unsqueeze(-1)
        return atomic_energies

    def _select_heads(
        self,
        output_dict: dict,
    ) -> dict:

        row_indices = torch.arange(
            output_dict["energy"].shape[1],
            device=self.device,
        )
        output_dict["energy"] = output_dict["energy"][
            self.head_idxs, row_indices
        ]
        graph_splits = self.data_ptr
        sizes = (graph_splits[1:] - graph_splits[:-1]).long()
        full_node_index = torch.arange(graph_splits[-1], device=self.device)
        head_index_repeated = torch.repeat_interleave(
            self.head_idxs,
            sizes,
        )
        output_dict["forces"] = output_dict["forces"][
            head_index_repeated, full_node_index
        ]

        if output_dict["stress"] is not None:
            output_dict["stress"] = output_dict["stress"][
                self.head_idxs, row_indices
            ]
        if output_dict["virials"] is not None:
            output_dict["virials"] = output_dict["virials"][
                self.head_idxs, row_indices
            ]

        return output_dict

    def _create_output_dict(
        self,
        total_energy: torch.Tensor,
        forces: torch.Tensor,
        virials: torch.Tensor,
        stress: torch.Tensor,
        hessian: torch.Tensor,
        edge_forces: torch.Tensor,
        zbl_atomic_energies: torch.Tensor,
        partial_charges: torch.Tensor,
        dipole: torch.Tensor,
        hirshfeld_ratios: torch.Tensor,
        inv_features: torch.Tensor,
        att_scores: torch.Tensor,
        node_energy: Optional[torch.Tensor] = None,
        training: bool = False,
    ) -> Dict[str, Optional[torch.Tensor]]:

        # total_energy has shape (num_graphs, num_output_heads)
        # permute to shape (num_output_heads, num_graphs)
        total_energy = total_energy.permute(1, 0)
        output_dict = {
            "energy": total_energy,
            "forces": forces,
            "virials": virials,
            "stress": stress,
            "hessian": hessian,
            "edge_forces": edge_forces,
            "node_energy": node_energy,
            "zbl_repulsion": zbl_atomic_energies,
            "partial_charges": partial_charges,
            "dipole": dipole,
            "hirshfeld_ratios": hirshfeld_ratios,
            "inv_features": inv_features,
            "att_scores": att_scores,
        }
        if self.select_heads:
            assert (
                not self.return_mean
            ), "Cannot both select heads and return mean."
            output_dict = self._select_heads(output_dict)

        if self.return_mean:
            mean_properties = [
                "energy",
                "forces",
                "virials",
                "stress",
                "hessian",
                "edge_forces",
            ]
            for k, v in output_dict.items():
                if v is not None and k in mean_properties:
                    output_dict[k] = v.mean(dim=0)

        return output_dict
