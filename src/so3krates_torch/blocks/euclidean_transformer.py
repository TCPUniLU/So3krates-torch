import torch
from typing import Any, Callable, Dict, List, Optional, Type, Union, Tuple
from so3krates_torch.blocks.so3_conv_invariants import L0Contraction
import math
from so3krates_torch.tools import scatter

torch.set_printoptions(precision=8)


class FilterNet(torch.nn.Module):
    """
    Fig. 3e) in https://doi.org/10.1038/s41467-024-50620-6
    """

    def __init__(
        self,
        degrees: List[int],
        num_radial_basis_fn: int,
        num_features: int,
        num_layers: int = 2,
        non_linearity: Type[torch.nn.Module] = torch.nn.SiLU,
    ):
        super().__init__()

        assert (
            num_features % 4 == 0
        ), f"num_features {num_features} must be divisible by 4 for the EuclideanTransformer."

        self.mlp_rbf = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=num_radial_basis_fn,
                out_features=num_features,
            ),
            non_linearity(),
        )
        self.mlp_ev = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=len(degrees),
                out_features=num_features // 4,
            ),
            non_linearity(),
        )
        for i in range(num_layers - 1):
            # last layer does not have a non-linearity
            if i == num_layers - 2:
                non_linearity = torch.nn.Identity
            self.mlp_rbf.add_module(
                f"mlp_rbf_layer_{i+1}",
                torch.nn.Sequential(
                    torch.nn.Linear(
                        in_features=num_features,
                        out_features=num_features,
                    ),
                    non_linearity(),
                ),
            )
            if i == 0:
                self.mlp_ev.add_module(
                    f"mlp_ev_layer_{i+1}",
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            in_features=num_features // 4,
                            out_features=num_features,
                        ),
                        non_linearity(),
                    ),
                )
            else:
                self.mlp_ev.add_module(
                    f"mlp_ev_layer_{i+1}",
                    torch.nn.Sequential(
                        torch.nn.Linear(
                            in_features=num_features,
                            out_features=num_features,
                        ),
                        non_linearity(),
                    ),
                )

    def reset_parameters(self):
        # JAX init (lecun normal)
        for m in self.mlp_rbf:
            if isinstance(m, torch.nn.Linear):
                std = 1.0 / (m.in_features**0.5)
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)
        for m in self.mlp_ev:
            if isinstance(m, torch.nn.Linear):
                std = 1.0 / (m.in_features**0.5)
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        rbf: torch.Tensor,
        ev_difference_invariants: torch.Tensor,
    ) -> torch.Tensor:
        # Eq. 20 in https://doi.org/10.1038/s41467-024-50620-6
        rbf_features = self.mlp_rbf(rbf)
        ev_features = self.mlp_ev(ev_difference_invariants)

        filter_weights = rbf_features + ev_features
        return filter_weights


class EuclideanTransformer(torch.nn.Module):
    def __init__(
        self,
        degrees: List[int],
        num_heads: int,
        num_features: int,
        num_radial_basis_fn: int,
        activation_fn: Type[torch.nn.Module] = torch.nn.SiLU,
        interaction_bias: bool = True,
        message_normalization: str = "sqrt_num_features",
        avg_num_neighbors: Optional[int] = None,
        device: Union[str, torch.device] = "cpu",
        filter_net_inv_layers: int = 2,
        filter_net_ev_layers: int = 2,
        layer_normalization_1: bool = False,
        layer_normalization_2: bool = False,
        residual_mlp_1: bool = False,
        residual_mlp_2: bool = False,
        qk_non_linearity: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
    ):
        super().__init__()

        self.filter_net_inv = FilterNet(
            degrees=degrees,
            num_features=num_features,
            num_radial_basis_fn=num_radial_basis_fn,
            num_layers=filter_net_inv_layers,
            non_linearity=activation_fn,
        )

        self.filter_net_ev = FilterNet(
            degrees=degrees,
            num_features=num_features,
            num_radial_basis_fn=num_radial_basis_fn,
            num_layers=filter_net_ev_layers,
            non_linearity=activation_fn,
        )

        self.euclidean_attention_block = EuclideanAttentionBlock(
            degrees=degrees,
            num_heads=num_heads,
            num_features=num_features,
            filter_net_inv=self.filter_net_inv,
            filter_net_ev=self.filter_net_ev,
            device=device,
            message_normalization=message_normalization,
            avg_num_neighbors=avg_num_neighbors,
            qk_non_linearity=qk_non_linearity,
        )
        self.interaction_block = InteractionBlock(
            degrees=degrees,
            num_features=num_features,
            bias=interaction_bias,
            device=device,
        )

        # The following blocks have been mentioned in the SO3LR paper
        # 10.26434/chemrxiv-2024-bdfr0-v3
        self.layer_normalization_1 = layer_normalization_1
        if layer_normalization_1:
            self.layer_norm_inv_1 = torch.nn.LayerNorm(
                normalized_shape=num_features,
                eps=1e-6,  # flax default is 1e-6 while pytorch default is 1e-5
            )
        self.layer_normalization_2 = layer_normalization_2
        if layer_normalization_2:
            self.layer_norm_inv_2 = torch.nn.LayerNorm(
                normalized_shape=num_features,
                eps=1e-6,  # flax default is 1e-6 while pytorch default is 1e-5
            )

        self.residual_mlp_1 = residual_mlp_1
        if residual_mlp_1:
            self.mlp_1 = torch.nn.Sequential(
                activation_fn(),
                torch.nn.Linear(
                    in_features=num_features,
                    out_features=num_features,
                ),
                activation_fn(),
                torch.nn.Linear(
                    in_features=num_features,
                    out_features=num_features,
                ),
            )

        self.residual_mlp_2 = residual_mlp_2
        if residual_mlp_2:
            self.mlp_2 = torch.nn.Sequential(
                activation_fn(),
                torch.nn.Linear(
                    in_features=num_features,
                    out_features=num_features,
                ),
                activation_fn(),
                torch.nn.Linear(
                    in_features=num_features,
                    out_features=num_features,
                ),
            )

    def reset_parameters(self):
        # JAX init (lecun normal) for MLPs
        if self.residual_mlp_1:
            for m in self.mlp_1:
                if isinstance(m, torch.nn.Linear):
                    std = 1.0 / (m.in_features**0.5)
                    torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)
        if self.residual_mlp_2:
            for m in self.mlp_2:
                if isinstance(m, torch.nn.Linear):
                    std = 1.0 / (m.in_features**0.5)
                    torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                    if m.bias is not None:
                        torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        inv_features: torch.Tensor,
        ev_features: torch.Tensor,
        rbf: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        sh_vectors: torch.Tensor,
        cutoffs: torch.Tensor,
        return_att: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        att_output = self.euclidean_attention_block(
            inv_features,
            ev_features,
            rbf=rbf,
            senders=senders,
            receivers=receivers,
            sh_vectors=sh_vectors,
            cutoffs=cutoffs,
            return_att=return_att,
        )
        if return_att:
            (d_att_inv_features, d_att_ev_features, (alpha_inv, alpha_ev)) = (
                att_output
            )

        else:
            d_att_inv_features, d_att_ev_features = att_output
            alpha_inv, alpha_ev = None, None

        att_inv_features = inv_features + d_att_inv_features
        att_ev_features = ev_features + d_att_ev_features

        if self.layer_normalization_1:
            att_inv_features = self.layer_norm_inv_1(att_inv_features)

        if self.residual_mlp_1:
            att_inv_features_temp = att_inv_features.clone()
            att_inv_features = att_inv_features + self.mlp_1(
                att_inv_features_temp
            )

        d_inv_features, d_ev_features = self.interaction_block(
            att_inv_features, att_ev_features
        )
        # Eq. 26, 27 in https://doi.org/10.1038/s41467-024-50620-6
        new_inv_features = att_inv_features + d_inv_features
        new_ev_features = att_ev_features + d_ev_features

        if self.residual_mlp_2:
            new_inv_features_temp = new_inv_features.clone()
            new_inv_features = new_inv_features + self.mlp_2(
                new_inv_features_temp
            )

        if self.layer_normalization_2:
            new_inv_features = self.layer_norm_inv_2(new_inv_features)

        if return_att:
            return new_inv_features, new_ev_features, (alpha_inv, alpha_ev)
        else:
            return new_inv_features, new_ev_features


class EuclideanAttentionBlock(torch.nn.Module):
    """
    Fig. 3c) in https://doi.org/10.1038/s41467-024-50620-6
    """

    def __init__(
        self,
        degrees: List[int],
        num_heads: int,
        num_features: int,
        filter_net_inv: FilterNet,
        filter_net_ev: FilterNet,
        message_normalization: str = "sqrt_num_features",
        qk_non_linearity: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        avg_num_neighbors: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__()
        self.degrees = degrees
        self.num_heads = num_heads
        self.num_features = num_features
        self.message_normalization = message_normalization
        self.avg_num_neighbors = avg_num_neighbors

        self.ev_features_dim = torch.sum(
            torch.tensor([2 * y + 1 for y in degrees])
        ).item()
        self.inv_features_dim = num_features
        self.so3_conv_invariants = L0Contraction(
            degrees=degrees, device=device
        )

        self.filter_net_inv = filter_net_inv
        self.filter_net_ev = filter_net_ev

        # query, key, value weights for invariants
        self.inv_heads = num_heads
        self.inv_head_dim = num_features // num_heads
        self.W_q_inv = torch.nn.Parameter(
            torch.empty(
                self.inv_heads,
                self.inv_head_dim,
                self.inv_head_dim,
                device=device,
            )
        )
        self.W_k_inv = torch.nn.Parameter(
            torch.empty(
                self.inv_heads,
                self.inv_head_dim,
                self.inv_head_dim,
                device=device,
            )
        )
        self.W_v_inv = torch.nn.Parameter(
            torch.empty(
                self.inv_heads,
                self.inv_head_dim,
                self.inv_head_dim,
                device=device,
            )
        )
        # query, key weights for ev features
        # no value weights, as it uses spherical harmonics as values
        self.ev_heads = len(degrees)
        self.ev_head_dim = num_features // len(degrees)
        self.W_q_ev = torch.nn.Parameter(
            torch.empty(
                self.ev_heads,
                self.ev_head_dim,
                self.ev_head_dim,
                device=device,
            )
        )
        self.W_k_ev = torch.nn.Parameter(
            torch.empty(
                self.ev_heads,
                self.ev_head_dim,
                self.ev_head_dim,
                device=device,
            )
        )
        # JAX weight init (lecun normal)
        std_inv = 1.0 / (self.W_q_inv.size(-1) ** 0.5)
        torch.nn.init.normal_(self.W_q_inv, mean=0.0, std=std_inv)
        torch.nn.init.normal_(self.W_k_inv, mean=0.0, std=std_inv)
        torch.nn.init.normal_(self.W_v_inv, mean=0.0, std=std_inv)

        std_ev = 1.0 / (self.W_q_ev.size(-1) ** 0.5)
        torch.nn.init.normal_(self.W_q_ev, mean=0.0, std=std_ev)
        torch.nn.init.normal_(self.W_k_ev, mean=0.0, std=std_ev)

        # normalization for attention weights
        # Eq. 21 https://doi.org/10.1038/s41467-024-50620-6
        assert message_normalization in [
            "sqrt_num_features",
            "identity",
            "avg_num_neighbors",
        ]
        if message_normalization == "sqrt_num_features":
            self.att_norm_inv = math.sqrt(self.inv_head_dim)
            self.att_norm_ev = math.sqrt(self.ev_head_dim)
        elif message_normalization == "identity":
            self.att_norm_inv = 1.0
            self.att_norm_ev = 1.0
        elif message_normalization == "avg_num_neighbors":
            assert avg_num_neighbors is not None
            self.att_norm_inv = avg_num_neighbors
            self.att_norm_ev = avg_num_neighbors

        # non-linearity for query and key weights
        try:
            self.qk_non_linearity = qk_non_linearity()
        except TypeError:
            self.qk_non_linearity = qk_non_linearity

        self.register_buffer(
            "degree_repeats",
            torch.tensor([2 * y + 1 for y in degrees]),
        )

    def _get_qkv(
        self,
        inv_features_inv: torch.Tensor,
        inv_features_ev: torch.Tensor,
        receivers: torch.Tensor,
        senders: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:
        # computing the queries, keys, and values and immediately
        # selecting receivers (i) and senders (j) (Eq. 21 https://doi.org/10.1038/s41467-024-50620-6)
        q_inv = self.qk_non_linearity(
            torch.matmul(
                inv_features_inv.transpose(0, 1), self.W_q_inv
            ).transpose(0, 1)
        )[receivers]
        k_inv = self.qk_non_linearity(
            torch.matmul(
                inv_features_inv.transpose(0, 1), self.W_k_inv
            ).transpose(0, 1)
        )[senders]
        v_inv = torch.matmul(
            inv_features_inv.transpose(0, 1), self.W_v_inv
        ).transpose(0, 1)[senders]
        q_ev = self.qk_non_linearity(
            torch.matmul(
                inv_features_ev.transpose(0, 1), self.W_q_ev
            ).transpose(0, 1)
        )[receivers]
        k_ev = self.qk_non_linearity(
            torch.matmul(
                inv_features_ev.transpose(0, 1), self.W_k_ev
            ).transpose(0, 1)
        )[senders]
        return q_inv, k_inv, v_inv, q_ev, k_ev

    def forward(
        self,
        inv_features: torch.Tensor,
        ev_features: torch.Tensor,
        rbf: torch.Tensor,
        senders: torch.Tensor,
        receivers: torch.Tensor,
        sh_vectors: torch.Tensor,
        cutoffs: torch.Tensor,
        return_att: bool = False,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        inv_features = inv_features.contiguous()
        ev_features = ev_features.contiguous()
        rbf = rbf.contiguous()
        ev_differences = ev_features[senders] - ev_features[receivers]
        ev_differences_invariants = self.so3_conv_invariants(ev_differences)
        filter_w_inv = self.filter_net_inv(rbf, ev_differences_invariants)
        filter_w_ev = self.filter_net_ev(rbf, ev_differences_invariants)

        # split filter weights into heads
        # now has shape [neighbors, num_heads, inv_head_dim]
        filter_w_inv = filter_w_inv.contiguous().view(
            -1, self.inv_heads, self.inv_head_dim
        )
        # now has shape [neighbors, len(degrees), ev_head_dim]
        filter_w_ev = filter_w_ev.contiguous().view(
            -1, self.ev_heads, self.ev_head_dim
        )
        # split features into heads
        # first for invariants; has shape [nodes, num_heads, inv_head_dim] now
        inv_features_inv = inv_features.view(
            -1, self.inv_heads, self.inv_head_dim
        )
        # now for the invariants used for updating the ev features
        # has shape [nodes, len(degrees), ev_head_dim] now
        inv_features_ev = inv_features.view(
            -1, self.ev_heads, self.ev_head_dim
        )

        q_inv, k_inv, v_inv, q_ev, k_ev = self._get_qkv(
            inv_features_inv=inv_features_inv,
            inv_features_ev=inv_features_ev,
            receivers=receivers,
            senders=senders,
        )
        # Eq. 21 https://doi.org/10.1038/s41467-024-50620-6
        filtered_k_inv = k_inv * filter_w_inv
        filtered_k_ev = k_ev * filter_w_ev

        # attention coefficients
        alpha_inv = (q_inv * filtered_k_inv).sum(
            -1, keepdim=True
        ) / self.att_norm_inv
        alpha_ev = (q_ev * filtered_k_ev).sum(-1) / self.att_norm_ev

        # Eq. 15 in https://doi.org/10.1038/s41467-024-50620-6
        scaled_neighbors_inv = cutoffs[:, None] * alpha_inv * v_inv

        d_h_att_inv_features = scatter.scatter_sum(
            src=scaled_neighbors_inv,
            index=receivers,
            dim=0,
            dim_size=inv_features.shape[0],  # number of nodes
        )
        d_att_inv_features = d_h_att_inv_features.view(
            -1, self.inv_features_dim
        )

        # already combining heads for ev here
        alpha_ev = torch.repeat_interleave(
            alpha_ev,
            self.degree_repeats,
            dim=-1,
            output_size=self.ev_features_dim,
        )
        # Eq. 26 https://doi.org/10.1038/s41467-024-50620-6
        scaled_neighbors_ev = cutoffs * alpha_ev * sh_vectors
        d_att_ev_features = scatter.scatter_sum(
            src=scaled_neighbors_ev,
            index=receivers,
            dim=0,
            dim_size=ev_features.shape[0],  # number of nodes
        )
        if return_att:
            return (
                d_att_inv_features,
                d_att_ev_features,
                (alpha_inv.clone().detach(), alpha_ev.clone().detach()),
            )
        else:
            return d_att_inv_features, d_att_ev_features


class InteractionBlock(torch.nn.Module):
    """
    Fig. 3d) in https://doi.org/10.1038/s41467-024-50620-6
    """

    def __init__(
        self,
        degrees: List[int],
        num_features: int,
        bias: bool = True,
        device: Union[str, torch.device] = "cpu",
    ):
        super().__init__()
        len_degrees = len(degrees)
        self.linear_layer = torch.nn.Linear(
            in_features=num_features + len_degrees,
            out_features=num_features + len_degrees,
            bias=bias,
        )
        self.so3_conv_invariants = L0Contraction(
            degrees=degrees, device=device
        )
        # repeat the b_ev_features for each degree
        # e.g. for degrees=[0,1,2], we have repeats = [1, 3, 5]
        self.register_buffer(
            "degree_repeats",
            torch.tensor([2 * y + 1 for y in degrees]),
        )

    def reset_parameters(self):
        # JAX init (lecun normal)
        for m in self.linear_layer:
            if isinstance(m, torch.nn.Linear):
                std = 1.0 / (m.in_features**0.5)
                torch.nn.init.normal_(m.weight, mean=0.0, std=std)
                if m.bias is not None:
                    torch.nn.init.zeros_(m.bias)

    def forward(
        self,
        inv_features: torch.Tensor,
        ev_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:

        ev_invariants = self.so3_conv_invariants(ev_features)

        # Eq. 25 in https://doi.org/10.1038/s41467-024-50620-6
        cat_features = torch.concatenate([inv_features, ev_invariants], dim=-1)
        # combine features
        transformed_features = self.linear_layer(cat_features)
        # split the features back
        d_inv_features, b_ev_features = torch.split(
            transformed_features,
            [inv_features.shape[-1], ev_invariants.shape[-1]],
            dim=-1,
        )
        b_ev_features = torch.repeat_interleave(
            b_ev_features,
            self.degree_repeats,
            dim=-1,
            output_size=ev_features.shape[-1],
        )
        # Eq. 24 in https://doi.org/10.1038/s41467-024-50620-6
        d_ev_features = b_ev_features * ev_features

        return d_inv_features, d_ev_features


class EuclideanAttentionBlockLORA(EuclideanAttentionBlock):
    def __init__(
        self,
        degrees: List[int],
        num_heads: int,
        num_features: int,
        filter_net_inv: callable,
        filter_net_ev: callable,
        lora_rank: int = 4,
        lora_alpha: int = 1,
        freeze_A: bool = False,
        message_normalization: str = "sqrt_num_features",
        qk_non_linearity: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        avg_num_neighbors: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__(
            degrees=degrees,
            num_heads=num_heads,
            num_features=num_features,
            filter_net_inv=filter_net_inv,
            filter_net_ev=filter_net_ev,
            message_normalization=message_normalization,
            qk_non_linearity=qk_non_linearity,
            avg_num_neighbors=avg_num_neighbors,
            device=device,
        )
        self.weights_fused = False
        # LoRA parameters for invariants
        self.rank = lora_rank
        self.alpha = lora_alpha
        self.scaling = self.alpha / math.sqrt(self.rank)
        # LoRA for query weights
        self.lora_A_q_inv = torch.nn.Parameter(
            torch.empty(
                self.inv_heads, self.inv_head_dim, lora_rank, device=device
            )
        )
        torch.nn.init.kaiming_uniform_(self.lora_A_q_inv, a=math.sqrt(5))
        self.lora_B_q_inv = torch.nn.Parameter(
            torch.zeros(
                self.inv_heads, lora_rank, self.inv_head_dim, device=device
            )
        )
        # LoRA for key weights
        self.lora_A_k_inv = torch.nn.Parameter(
            torch.empty(
                self.inv_heads, self.inv_head_dim, lora_rank, device=device
            )
        )
        torch.nn.init.kaiming_uniform_(self.lora_A_k_inv, a=math.sqrt(5))
        self.lora_B_k_inv = torch.nn.Parameter(
            torch.zeros(
                self.inv_heads, lora_rank, self.inv_head_dim, device=device
            )
        )

        # LoRA for value weights
        self.lora_A_v_inv = torch.nn.Parameter(
            torch.empty(
                self.inv_heads, self.inv_head_dim, lora_rank, device=device
            )
        )
        torch.nn.init.kaiming_uniform_(self.lora_A_v_inv, a=math.sqrt(5))

        self.lora_B_v_inv = torch.nn.Parameter(
            torch.zeros(
                self.inv_heads, lora_rank, self.inv_head_dim, device=device
            )
        )

        # LoRA for ev features
        # LoRA for query weights
        self.lora_A_q_ev = torch.nn.Parameter(
            torch.empty(
                self.ev_heads, self.ev_head_dim, lora_rank, device=device
            )
        )
        torch.nn.init.kaiming_uniform_(self.lora_A_q_ev, a=math.sqrt(5))

        self.lora_B_q_ev = torch.nn.Parameter(
            torch.zeros(
                self.ev_heads, lora_rank, self.ev_head_dim, device=device
            )
        )
        # LoRA for key weights
        self.lora_A_k_ev = torch.nn.Parameter(
            torch.empty(
                self.ev_heads, self.ev_head_dim, lora_rank, device=device
            )
        )
        torch.nn.init.kaiming_uniform_(self.lora_A_k_ev, a=math.sqrt(5))

        self.lora_B_k_ev = torch.nn.Parameter(
            torch.zeros(
                self.ev_heads, lora_rank, self.ev_head_dim, device=device
            )
        )
        # No LoRA for value weights, as it uses spherical harmonics as values

        if freeze_A:
            self.lora_A_q_inv.requires_grad = False
            self.lora_A_k_inv.requires_grad = False
            self.lora_A_v_inv.requires_grad = False
            self.lora_A_q_ev.requires_grad = False
            self.lora_A_k_ev.requires_grad = False

    def _use_lora(self, features, W, lora_A, lora_B):

        # lora in two steps to avoid large intermediate tensors
        return torch.matmul(features.transpose(0, 1), W).transpose(
            0, 1
        ) + self.scaling * torch.matmul(
            torch.matmul(features.transpose(0, 1), lora_A), lora_B
        ).transpose(
            0, 1
        )

        # This is the original implementation which creates a large intermediate tensor

        # return (
        #    torch.matmul(
        #        features.transpose(0, 1), W + self.scaling * torch.matmul(lora_A, lora_B)
        #    ).transpose(0, 1)
        # )

    def fuse_lora_weights(self):
        if self.weights_fused:
            return
        # Fuse the LoRA weights into the original weights for inference
        with torch.no_grad():
            self.W_q_inv += self.scaling * torch.matmul(
                self.lora_A_q_inv, self.lora_B_q_inv
            )
            self.W_k_inv += self.scaling * torch.matmul(
                self.lora_A_k_inv, self.lora_B_k_inv
            )
            self.W_v_inv += self.scaling * torch.matmul(
                self.lora_A_v_inv, self.lora_B_v_inv
            )
            self.W_q_ev += self.scaling * torch.matmul(
                self.lora_A_q_ev, self.lora_B_q_ev
            )
            self.W_k_ev += self.scaling * torch.matmul(
                self.lora_A_k_ev, self.lora_B_k_ev
            )
            # After fusing, delete the LoRA parameters to save memory
            del self.lora_A_q_inv
            del self.lora_B_q_inv
            del self.lora_A_k_inv
            del self.lora_B_k_inv
            del self.lora_A_v_inv
            del self.lora_B_v_inv
            del self.lora_A_q_ev
            del self.lora_B_q_ev
            del self.lora_A_k_ev
            del self.lora_B_k_ev

        self.weights_fused = True

    def _get_qkv(
        self,
        inv_features_inv: torch.Tensor,
        inv_features_ev: torch.Tensor,
        receivers: torch.Tensor,
        senders: torch.Tensor,
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:

        if self.weights_fused:
            # If weights are fused, use the original forward method
            return super()._get_qkv(
                inv_features_inv=inv_features_inv,
                inv_features_ev=inv_features_ev,
                senders=senders,
                receivers=receivers,
            )
        else:
            q_inv = self.qk_non_linearity(
                self._use_lora(
                    inv_features_inv,
                    self.W_q_inv,
                    self.lora_A_q_inv,
                    self.lora_B_q_inv,
                )
            )[receivers]
            k_inv = self.qk_non_linearity(
                self._use_lora(
                    inv_features_inv,
                    self.W_k_inv,
                    self.lora_A_k_inv,
                    self.lora_B_k_inv,
                )
            )[senders]
            v_inv = self._use_lora(
                inv_features_inv,
                self.W_v_inv,
                self.lora_A_v_inv,
                self.lora_B_v_inv,
            )[senders]
            q_ev = self.qk_non_linearity(
                self._use_lora(
                    inv_features_ev,
                    self.W_q_ev,
                    self.lora_A_q_ev,
                    self.lora_B_q_ev,
                )
            )[receivers]
            k_ev = self.qk_non_linearity(
                self._use_lora(
                    inv_features_ev,
                    self.W_k_ev,
                    self.lora_A_k_ev,
                    self.lora_B_k_ev,
                )
            )[senders]

            return q_inv, k_inv, v_inv, q_ev, k_ev


class EuclideanAttentionBlockDoRA(EuclideanAttentionBlockLORA):
    def __init__(
        self,
        degrees: List[int],
        num_heads: int,
        num_features: int,
        filter_net_inv: callable,
        filter_net_ev: callable,
        lora_rank: int = 4,
        lora_alpha: int = 4,
        scaling_to_one: bool = True,
        message_normalization: str = "sqrt_num_features",
        qk_non_linearity: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        avg_num_neighbors: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__(
            degrees=degrees,
            num_heads=num_heads,
            num_features=num_features,
            filter_net_inv=filter_net_inv,
            filter_net_ev=filter_net_ev,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            message_normalization=message_normalization,
            qk_non_linearity=qk_non_linearity,
            avg_num_neighbors=avg_num_neighbors,
            device=device,
        )
        self.weights_fused = False
        # LoRA parameters for invariants
        self.rank = lora_rank
        self.alpha = lora_alpha

        if scaling_to_one:
            self.scaling = 1.0
        else:
            self.scaling = self.alpha / math.sqrt(self.rank)

        self.get_magnitude_vectors()

    def get_magnitude_vectors(
        self,
    ):
        self.dora_m_q_inv = torch.nn.Parameter(torch.norm(self.W_q_inv, dim=2))
        self.dora_m_k_inv = torch.nn.Parameter(torch.norm(self.W_k_inv, dim=2))
        self.dora_m_v_inv = torch.nn.Parameter(torch.norm(self.W_v_inv, dim=2))
        self.dora_m_q_ev = torch.nn.Parameter(torch.norm(self.W_q_ev, dim=2))
        self.dora_m_k_ev = torch.nn.Parameter(torch.norm(self.W_k_ev, dim=2))

    def _get_constant_norms(
        self,
    ):
        """
        Compute normalization constants for DoRA and detach them from the computation graph.
        """
        self.norm_q_inv = torch.norm(
            self.W_q_inv
            + self.scaling
            * torch.matmul(self.lora_A_q_inv, self.lora_B_q_inv),
            dim=2,
        ).detach()
        self.norm_k_inv = torch.norm(
            self.W_k_inv
            + self.scaling
            * torch.matmul(self.lora_A_k_inv, self.lora_B_k_inv),
            dim=2,
        ).detach()
        self.norm_v_inv = torch.norm(
            self.W_v_inv
            + self.scaling
            * torch.matmul(self.lora_A_v_inv, self.lora_B_v_inv),
            dim=2,
        ).detach()

        self.norm_q_ev = torch.norm(
            self.W_q_ev
            + self.scaling * torch.matmul(self.lora_A_q_ev, self.lora_B_q_ev),
            dim=2,
        ).detach()
        self.norm_k_ev = torch.norm(
            self.W_k_ev
            + self.scaling * torch.matmul(self.lora_A_k_ev, self.lora_B_k_ev),
            dim=2,
        ).detach()

    def _get_qkv(
        self,
        inv_features_inv: torch.Tensor,
        inv_features_ev: torch.Tensor,
        receivers: torch.Tensor,
        senders: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:

        if self.weights_fused:
            # If weights are fused, use the original forward method
            return super()._get_qkv(
                inv_features_inv=inv_features_inv,
                inv_features_ev=inv_features_ev,
                senders=senders,
                receivers=receivers,
            )
        else:
            self._get_constant_norms()
            lora_q_inv = self._use_lora(
                inv_features_inv,
                self.W_q_inv,
                self.lora_A_q_inv,
                self.lora_B_q_inv,
            )
            q_inv = self.qk_non_linearity(
                lora_q_inv * (self.dora_m_q_inv / self.norm_q_inv)[None, :, :]
            )[receivers]
            lora_k_inv = self._use_lora(
                inv_features_inv,
                self.W_k_inv,
                self.lora_A_k_inv,
                self.lora_B_k_inv,
            )
            k_inv = self.qk_non_linearity(
                lora_k_inv * (self.dora_m_k_inv / self.norm_k_inv)[None, :, :]
            )[senders]

            lora_v_inv = self._use_lora(
                inv_features_inv,
                self.W_v_inv,
                self.lora_A_v_inv,
                self.lora_B_v_inv,
            )
            v_inv = (
                lora_v_inv * (self.dora_m_v_inv / self.norm_v_inv)[None, :, :]
            )[senders]

            lora_q_ev = self._use_lora(
                inv_features_ev,
                self.W_q_ev,
                self.lora_A_q_ev,
                self.lora_B_q_ev,
            )
            q_ev = self.qk_non_linearity(
                lora_q_ev * (self.dora_m_q_ev / self.norm_q_ev)[None, :, :]
            )[receivers]

            lora_k_ev = self._use_lora(
                inv_features_ev,
                self.W_k_ev,
                self.lora_A_k_ev,
                self.lora_B_k_ev,
            )
            k_ev = self.qk_non_linearity(
                lora_k_ev * (self.dora_m_k_ev / self.norm_k_ev)[None, :, :]
            )[senders]

            return q_inv, k_inv, v_inv, q_ev, k_ev

    def fuse_lora_weights(self):
        if self.weights_fused:
            return
        # Fuse the LoRA weights into the original weights for inference
        self._get_constant_norms()
        with torch.no_grad():

            self.W_q_inv = torch.nn.Parameter(
                (self.dora_m_q_inv / self.norm_q_inv)[:, :, None]
                * (
                    self.W_q_inv
                    + self.scaling
                    * torch.matmul(self.lora_A_q_inv, self.lora_B_q_inv)
                )
            )
            self.W_k_inv = torch.nn.Parameter(
                (self.dora_m_k_inv / self.norm_k_inv)[:, :, None]
                * (
                    self.W_k_inv
                    + self.scaling
                    * torch.matmul(self.lora_A_k_inv, self.lora_B_k_inv)
                )
            )
            self.W_v_inv = torch.nn.Parameter(
                (self.dora_m_v_inv / self.norm_v_inv)[:, :, None]
                * (
                    self.W_v_inv
                    + self.scaling
                    * torch.matmul(self.lora_A_v_inv, self.lora_B_v_inv)
                )
            )
            self.W_q_ev = torch.nn.Parameter(
                (self.dora_m_q_ev / self.norm_q_ev)[:, :, None]
                * (
                    self.W_q_ev
                    + self.scaling
                    * torch.matmul(self.lora_A_q_ev, self.lora_B_q_ev)
                )
            )
            self.W_k_ev = torch.nn.Parameter(
                (self.dora_m_k_ev / self.norm_k_ev)[:, :, None]
                * (
                    self.W_k_ev
                    + self.scaling
                    * torch.matmul(self.lora_A_k_ev, self.lora_B_k_ev)
                )
            )
            # After fusing, delete the LoRA parameters to save memory
            del self.lora_A_q_inv
            del self.lora_B_q_inv
            del self.lora_A_k_inv
            del self.lora_B_k_inv
            del self.lora_A_v_inv
            del self.lora_B_v_inv
            del self.lora_A_q_ev
            del self.lora_B_q_ev
            del self.lora_A_k_ev
            del self.lora_B_k_ev
            del self.dora_m_q_inv
            del self.dora_m_k_inv
            del self.dora_m_v_inv
            del self.dora_m_q_ev
            del self.dora_m_k_ev
            del self.norm_q_inv
            del self.norm_k_inv
            del self.norm_v_inv
            del self.norm_q_ev
            del self.norm_k_ev

        self.weights_fused = True


class EuclideanAttentionBlockVeRA(EuclideanAttentionBlockLORA):
    def __init__(
        self,
        degrees: List[int],
        num_heads: int,
        num_features: int,
        vera_A_matrix_inv: torch.Tensor,
        vera_B_matrix_inv: torch.Tensor,
        vera_A_matrix_ev: torch.Tensor,
        vera_B_matrix_ev: torch.Tensor,
        filter_net_inv: callable,
        filter_net_ev: callable,
        lora_rank: int = 4,
        lora_alpha: int = 8,
        scaling_to_one: bool = True,
        message_normalization: str = "sqrt_num_features",
        qk_non_linearity: Optional[
            Callable[[torch.Tensor], torch.Tensor]
        ] = None,
        avg_num_neighbors: Optional[int] = None,
        device: str = "cpu",
    ):
        super().__init__(
            degrees=degrees,
            num_heads=num_heads,
            num_features=num_features,
            filter_net_inv=filter_net_inv,
            filter_net_ev=filter_net_ev,
            lora_rank=lora_rank,
            lora_alpha=lora_alpha,
            message_normalization=message_normalization,
            qk_non_linearity=qk_non_linearity,
            avg_num_neighbors=avg_num_neighbors,
            device=device,
        )
        self.weights_fused = False
        # VeRA parameters for invariants
        self.rank = lora_rank
        self.alpha = lora_alpha
        if scaling_to_one:
            self.scaling = 1.0
        else:
            self.scaling = self.alpha / math.sqrt(self.rank)

        # A, B are passed as arguments so that they can be shared across layers
        assert vera_A_matrix_inv.shape == (
            self.inv_heads,
            self.inv_head_dim,
            lora_rank,
        )
        assert vera_B_matrix_inv.shape == (
            self.inv_heads,
            lora_rank,
            self.inv_head_dim,
        )
        assert vera_A_matrix_ev.shape == (
            self.ev_heads,
            self.ev_head_dim,
            lora_rank,
        )
        assert vera_B_matrix_ev.shape == (
            self.ev_heads,
            lora_rank,
            self.ev_head_dim,
        )
        self.register_buffer("vera_A_matrix_inv", vera_A_matrix_inv)
        self.register_buffer("vera_B_matrix_inv", vera_B_matrix_inv)
        self.register_buffer("vera_A_matrix_ev", vera_A_matrix_ev)
        self.register_buffer("vera_B_matrix_ev", vera_B_matrix_ev)

        self.d_k_inv = torch.nn.Parameter(
            torch.ones(self.inv_heads, self.rank, device=device)
        )
        self.b_k_inv = torch.nn.Parameter(
            torch.zeros(self.inv_heads, self.rank, device=device)
        )
        self.d_v_inv = torch.nn.Parameter(
            torch.ones(self.inv_heads, self.rank, device=device)
        )
        self.b_v_inv = torch.nn.Parameter(
            torch.zeros(self.inv_heads, self.rank, device=device)
        )
        self.d_q_inv = torch.nn.Parameter(
            torch.ones(self.inv_heads, self.rank, device=device)
        )
        self.b_q_inv = torch.nn.Parameter(
            torch.zeros(self.inv_heads, self.rank, device=device)
        )

        self.b_k_ev = torch.nn.Parameter(
            torch.zeros(self.ev_heads, self.rank, device=device)
        )
        self.d_k_ev = torch.nn.Parameter(
            torch.ones(self.ev_heads, self.rank, device=device)
        )
        self.b_q_ev = torch.nn.Parameter(
            torch.zeros(self.ev_heads, self.rank, device=device)
        )
        self.d_q_ev = torch.nn.Parameter(
            torch.ones(self.ev_heads, self.rank, device=device)
        )

    def _use_lora(self, features, W, lora_A, lora_B, vera_b, vera_d):
        features_lora = torch.matmul(
            features.transpose(0, 1), lora_A
        ).transpose(0, 1)

        features_lora = features_lora * vera_d[None, :, :]
        B_scaled = lora_B * vera_b[:, :, None]

        features_lora = torch.matmul(
            features_lora.transpose(0, 1), B_scaled
        ).transpose(0, 1)

        return (
            torch.matmul(features.transpose(0, 1), W).transpose(0, 1)
            + self.scaling * features_lora
        )

    def fuse_lora_weights(self):
        if self.weights_fused:
            return
        # Fuse the VeRA weights into the original weights for inference
        with torch.no_grad():
            self.W_q_inv.add_(
                self.scaling
                * torch.matmul(
                    self.vera_A_matrix_inv * self.d_q_inv[:, None, :],
                    self.vera_B_matrix_inv * self.b_q_inv[:, :, None],
                )
            )
            self.W_k_inv.add_(
                self.scaling
                * torch.matmul(
                    self.vera_A_matrix_inv * self.d_k_inv[:, None, :],
                    self.vera_B_matrix_inv * self.b_k_inv[:, :, None],
                )
            )
            self.W_v_inv.add_(
                self.scaling
                * torch.matmul(
                    self.vera_A_matrix_inv * self.d_v_inv[:, None, :],
                    self.vera_B_matrix_inv * self.b_v_inv[:, :, None],
                )
            )
            self.W_q_ev.add_(
                self.scaling
                * torch.matmul(
                    self.vera_A_matrix_ev * self.d_q_ev[:, None, :],
                    self.vera_B_matrix_ev * self.b_q_ev[:, :, None],
                )
            )
            self.W_k_ev.add_(
                self.scaling
                * torch.matmul(
                    self.vera_A_matrix_ev * self.d_k_ev[:, None, :],
                    self.vera_B_matrix_ev * self.b_k_ev[:, :, None],
                )
            )
            # After fusing, delete the VeRA parameters to save memory
            del self.vera_A_matrix_ev
            del self.vera_B_matrix_ev
            del self.vera_A_matrix_inv
            del self.vera_B_matrix_inv
            del self.d_q_inv
            del self.b_q_inv
            del self.d_k_inv
            del self.b_k_inv
            del self.d_v_inv
            del self.b_v_inv
            del self.d_q_ev
            del self.b_q_ev
            del self.d_k_ev
            del self.b_k_ev

        self.weights_fused = True

    def _get_qkv(
        self,
        inv_features_inv: torch.Tensor,
        inv_features_ev: torch.Tensor,
        receivers: torch.Tensor,
        senders: torch.Tensor,
    ) -> Tuple[
        torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor
    ]:

        if self.weights_fused:
            # If weights are fused, use the original forward method
            return super()._get_qkv(
                inv_features_inv=inv_features_inv,
                inv_features_ev=inv_features_ev,
                senders=senders,
                receivers=receivers,
            )
        else:
            q_inv = self.qk_non_linearity(
                self._use_lora(
                    features=inv_features_inv,
                    W=self.W_q_inv,
                    lora_A=self.vera_A_matrix_inv,
                    lora_B=self.vera_B_matrix_inv,
                    vera_b=self.b_q_inv,
                    vera_d=self.d_q_inv,
                )
            )[receivers]
            k_inv = self.qk_non_linearity(
                self._use_lora(
                    features=inv_features_inv,
                    W=self.W_k_inv,
                    lora_A=self.vera_A_matrix_inv,
                    lora_B=self.vera_B_matrix_inv,
                    vera_b=self.b_k_inv,
                    vera_d=self.d_k_inv,
                )
            )[senders]

            v_inv = self._use_lora(
                features=inv_features_inv,
                W=self.W_v_inv,
                lora_A=self.vera_A_matrix_inv,
                lora_B=self.vera_B_matrix_inv,
                vera_b=self.b_v_inv,
                vera_d=self.d_v_inv,
            )[senders]

            q_ev = self.qk_non_linearity(
                self._use_lora(
                    features=inv_features_ev,
                    W=self.W_q_ev,
                    lora_A=self.vera_A_matrix_ev,
                    lora_B=self.vera_B_matrix_ev,
                    vera_b=self.b_q_ev,
                    vera_d=self.d_q_ev,
                )
            )[receivers]

            k_ev = self.qk_non_linearity(
                self._use_lora(
                    features=inv_features_ev,
                    W=self.W_k_ev,
                    lora_A=self.vera_A_matrix_ev,
                    lora_B=self.vera_B_matrix_ev,
                    vera_b=self.b_k_ev,
                    vera_d=self.d_k_ev,
                )
            )[senders]

            return q_inv, k_inv, v_inv, q_ev, k_ev
