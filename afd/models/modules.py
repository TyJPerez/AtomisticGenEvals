"""
Neural-network building blocks for the Conditional Equivariant Transformer (CT).

Consolidates what used to live in three files:
  - ``conditioning.py``    — conditioning / normalization layers
  - ``output_modules.py``  — scalar / vector output heads
  - ``priors.py``          — the (identity) prior base class

Sections below, in dependency order:

  1. Prior base class
       - BasePrior         : identity prior; kept for the CET state-dict contract
  2. Conditioning & normalization
       - JointDropPath     : joint stochastic depth on (x, v)
       - DyT               : Dynamic Tanh layer norm (Transformers w/o Normalization)
       - adaDyT2           : conditional Dynamic Tanh
       - adaLN2            : conditional layer norm with scale / shift / gate from cond
       - DropCond          : training-time conditioning dropout w/ learned mask token
       - ProjHead2         : graph-level embedding projection head
  3. Output heads
       - MLP               : Linear -> act -> [Linear -> act] x n -> Linear
       - GatedEquivariantBlock
       - OutputModel       : abstract reduce/post-reduce base
       - EquivariantScalar : per-atom scalar head (the `y` prediction)
       - EquivariantVector : per-atom vector head (the noise prediction)
"""

from abc import ABCMeta, abstractmethod
from typing import Dict, Optional
import warnings
from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from .utils import act_class_mapping, scatter


# ---------------------------------------------------------------------------
# 1. Prior base class
# ---------------------------------------------------------------------------
class BasePrior(nn.Module):
    """Base class for prior models. Identity by default.

    Kept only so the `prior_model` ModuleList type-check in CET's BaseModel
    works. The pretrained checkpoints store an empty prior_model, so no concrete
    prior subclass is needed. To attach an Atomref or similar, copy the relevant
    class out of `SelfConditionedDenoisingAtoms/models/ET_models/priors.py`.
    """

    def __init__(self, dataset=None):
        super().__init__()

    def get_init_args(self):
        return {}

    def pre_reduce(self, x, z, pos, batch, extra_args: Optional[Dict[str, Tensor]]):
        return x

    def post_reduce(
        self,
        y,
        z,
        pos,
        batch,
        box: Optional[Tensor],
        extra_args: Optional[Dict[str, Tensor]],
    ):
        return y


# ---------------------------------------------------------------------------
# 2. Conditioning & normalization
# ---------------------------------------------------------------------------
class JointDropPath(nn.Module):
    """Joint drop-path on invariant (x) and equivariant (v) features."""

    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x, v):
        if self.drop_prob == 0.0 or not self.training:
            return x, v
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        binary_mask = torch.floor(random_tensor)
        return (
            (x / keep_prob) * binary_mask,
            (v / keep_prob) * binary_mask.unsqueeze(-1),
        )


class DyT(nn.Module):
    """Dynamic Tanh — drop-in for LayerNorm, faster (Zhu et al. 2025)."""

    def __init__(self, dim: int, init_alpha: float = 0.2):
        super().__init__()
        self.norm_alpha = nn.Parameter(torch.tensor(init_alpha))
        self.norm_gamma = nn.Parameter(torch.ones(dim) / init_alpha)
        self.norm_beta = nn.Parameter(torch.zeros(dim))

    def reset_parameters(self, init_alpha: float = 0.2):
        nn.init.constant_(self.norm_alpha, init_alpha)
        nn.init.constant_(self.norm_gamma, 1.0 / init_alpha)
        nn.init.constant_(self.norm_beta, 0.0)

    def forward(self, x):
        return F.tanh(self.norm_alpha * x) * self.norm_gamma + self.norm_beta


class adaDyT2(nn.Module):
    """Conditional Dynamic Tanh — scale/shift from external `c`."""

    def __init__(self, dim: int, init_alpha: float = 0.2):
        super().__init__()
        self.alpha = nn.Parameter(torch.tensor(init_alpha))

    def reset_parameters(self, init_alpha: float = 0.2):
        nn.init.constant_(self.alpha, init_alpha)

    def forward(self, x, scale, shift):
        alpha = torch.clamp(self.alpha.abs(), min=1e-5)
        gamma = (1 + scale) / alpha
        return F.tanh(alpha * x) * gamma + shift


class adaLN2(nn.Module):
    """Conditional pre-norm: produces (norm(x) * gamma + shift, gate) from `c`."""

    def __init__(self, dim: int, bias: bool = True, init_alpha: float = 0.2, norm_type: str = "LN"):
        super().__init__()
        norm_class = DyT if norm_type == "DyT" else nn.LayerNorm
        self.dim = dim
        self.x_norm = norm_class(dim)
        self.fc = nn.Sequential(
            nn.Linear(2 * dim, dim, bias=bias),
            nn.SiLU(),
            norm_class(dim),
            nn.Linear(dim, 3 * dim, bias=bias),
        )
        self.x_modulate = adaDyT2(dim, init_alpha=init_alpha)
        self.alpha_x = nn.Parameter(torch.tensor(init_alpha))
        self.init_params()

    def init_params(self, zero: bool = False):
        if zero:
            nn.init.constant_(self.fc[-1].weight, 0)
            nn.init.constant_(self.fc[-1].bias, 0)
        else:
            nn.init.xavier_uniform_(self.fc[-1].weight)
            nn.init.constant_(self.fc[-1].bias, 0)

    def reset_parameters(self, norm_type: str = "LN"):
        norm_class = DyT if norm_type == "DyT" else nn.LayerNorm
        dim = self.dim
        self.x_norm = norm_class(dim)

        self.fc[0].reset_parameters()
        if isinstance(self.fc[2], (DyT, nn.LayerNorm)):
            self.fc[2] = norm_class(dim)
        else:
            raise ValueError("Unexpected layer type in fc")
        self.fc[3].reset_parameters()
        self.x_modulate.reset_parameters()
        nn.init.constant_(self.alpha_x, 0.2)
        self.init_params()

    def forward(self, x, c):
        if c is None:
            return self.x_norm(x), 1.0

        x = self.x_norm(x)
        c = torch.cat([c, x.detach()], dim=-1)
        shift_x, scale_x, gate_x = self.fc(c).chunk(3, dim=1)

        x = self.x_modulate(x, scale_x, shift_x)
        gate_x = F.tanh(gate_x * self.alpha_x)
        return x, gate_x


class DropCond(nn.Module):
    """Replace conditioning with a learned mask token with probability `p_drop`."""

    def __init__(self, dim: int, p_drop: float = 0.25):
        super().__init__()
        self.p_drop_cond = p_drop
        self.mask_token = nn.Parameter(torch.zeros(1, dim))
        self.norm = nn.LayerNorm(dim)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.mask_token)
        self.norm.reset_parameters()

    def forward(self, c, batch_idx):
        num_graphs = batch_idx.max().item() + 1 if batch_idx.numel() > 0 else 0

        if c is None:
            c = self.mask_token.repeat(num_graphs, 1)
        else:
            if self.training:
                drop_mask = torch.rand(c.size(0), device=c.device) < self.p_drop_cond
                drop_mask = drop_mask.float().unsqueeze(1)
                c_mask = self.mask_token.repeat(num_graphs, 1)
                c = c * (1 - drop_mask) + c_mask * drop_mask

        c = self.norm(c)
        return c[batch_idx]


class ProjHead2(nn.Module):
    """Projection head for the graph-level embedding (`mol_emb`)."""

    def __init__(self, emb_dim: int, agg: str = "sum"):
        super().__init__()
        self.emb_dim = emb_dim
        self.agg = agg

        self.pre_proj = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.LayerNorm(emb_dim),
        )
        self.post_agg_mlp = nn.Sequential(
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
            nn.SiLU(),
            nn.LayerNorm(emb_dim),
            nn.Linear(emb_dim, emb_dim),
        )
        self.init_params()

    def init_params(self):
        nn.init.xavier_uniform_(self.pre_proj[1].weight)
        self.pre_proj[1].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.post_agg_mlp[1].weight)
        self.post_agg_mlp[1].bias.data.fill_(0)
        nn.init.xavier_uniform_(self.post_agg_mlp[4].weight)
        self.post_agg_mlp[4].bias.data.fill_(0)

    def forward(self, x, batch):
        x = self.pre_proj(x)
        x = scatter(x, batch, dim=0, reduce=self.agg)
        return self.post_agg_mlp(x)


# ---------------------------------------------------------------------------
# 3. Output heads
# ---------------------------------------------------------------------------
class MLP(nn.Module):
    """Linear -> act -> [Linear -> act] x num_hidden_layers -> Linear."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        hidden_channels: int,
        activation: str,
        num_hidden_layers: int = 0,
        dtype: torch.dtype = torch.float32,
    ):
        super().__init__()
        act_class = act_class_mapping[activation]
        self.act = act_class()
        self.layers = nn.Sequential()
        self.layers.append(nn.Linear(in_channels, hidden_channels, dtype=dtype))
        self.layers.append(self.act)
        for _ in range(num_hidden_layers):
            self.layers.append(nn.Linear(hidden_channels, hidden_channels, dtype=dtype))
            self.layers.append(self.act)
        self.layers.append(nn.Linear(hidden_channels, out_channels, dtype=dtype))

    def reset_parameters(self):
        for layer in self.layers:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                layer.bias.data.fill_(0)

    def forward(self, x):
        return self.layers(x)


class GatedEquivariantBlock(nn.Module):
    """Gated Equivariant Block (Schütt et al. 2021)."""

    def __init__(
        self,
        hidden_channels: int,
        out_channels: int,
        intermediate_channels: Optional[int] = None,
        activation: str = "silu",
        scalar_activation: bool = False,
        dtype: torch.dtype = torch.float,
    ):
        super().__init__()
        self.out_channels = out_channels

        if intermediate_channels is None:
            intermediate_channels = hidden_channels

        self.vec1_proj = nn.Linear(
            hidden_channels, hidden_channels, bias=False, dtype=dtype
        )
        self.vec2_proj = nn.Linear(
            hidden_channels, out_channels, bias=False, dtype=dtype
        )

        act_class = act_class_mapping[activation]
        self.update_net = MLP(
            in_channels=hidden_channels * 2,
            out_channels=out_channels * 2,
            hidden_channels=intermediate_channels,
            activation=activation,
            num_hidden_layers=0,
            dtype=dtype,
        )
        self.act = act_class() if scalar_activation else None

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.vec1_proj.weight)
        nn.init.xavier_uniform_(self.vec2_proj.weight)
        self.update_net.reset_parameters()

    def forward(self, x, v):
        vec1_buffer = self.vec1_proj(v)

        vec1 = torch.zeros(
            vec1_buffer.size(0),
            vec1_buffer.size(2),
            device=vec1_buffer.device,
            dtype=vec1_buffer.dtype,
        )
        mask = (vec1_buffer != 0).view(vec1_buffer.size(0), -1).any(dim=1)
        if not mask.all():
            warnings.warn(
                f"Skipping gradients for {(~mask).sum()} atoms due to vector "
                "features being zero. This usually means an atom has no "
                "neighbours inside the cutoff radius."
            )
        vec1[mask] = torch.norm(vec1_buffer[mask], dim=-2)

        vec2 = self.vec2_proj(v)
        x = torch.cat([x, vec1], dim=-1)

        x, v = torch.split(self.update_net(x), self.out_channels, dim=-1)
        v = v.unsqueeze(1) * vec2

        if self.act is not None:
            x = self.act(x)
        return x, v


class OutputModel(nn.Module, metaclass=ABCMeta):
    def __init__(self, allow_prior_model: bool, reduce_op: str):
        super().__init__()
        self.allow_prior_model = allow_prior_model
        self.reduce_op = reduce_op
        self.dim_size = 0

    def reset_parameters(self):
        pass

    @abstractmethod
    def pre_reduce(self, x, v, z, pos, batch):
        ...

    def reduce(self, x, batch):
        # Upstream branched here on whether the current CUDA stream was
        # being captured into a CUDA graph; in that mode the dim_size had
        # to be locked so the captured graph would have a fixed shape. We
        # don't use CUDA graphs in AFD, so always recompute dim_size
        # from the batch index.
        self.dim_size = int(batch.max().item() + 1)
        return scatter(x, batch, dim=0, dim_size=self.dim_size, reduce=self.reduce_op)

    def post_reduce(self, x):
        return x


class EquivariantScalar(OutputModel):
    def __init__(
        self,
        hidden_channels: int,
        activation: str = "silu",
        allow_prior_model: bool = True,
        reduce_op: str = "sum",
        dtype: torch.dtype = torch.float,
        **kwargs,
    ):
        super().__init__(allow_prior_model=allow_prior_model, reduce_op=reduce_op)
        if kwargs.get("num_layers", 0) > 0:
            warn("num_layers is not used in EquivariantScalar")
        self.output_network = nn.ModuleList(
            [
                GatedEquivariantBlock(
                    hidden_channels,
                    hidden_channels // 2,
                    activation=activation,
                    scalar_activation=True,
                    dtype=dtype,
                ),
                GatedEquivariantBlock(
                    hidden_channels // 2,
                    1,
                    activation=activation,
                    dtype=dtype,
                ),
            ]
        )
        self.reset_parameters()

    def reset_parameters(self):
        for layer in self.output_network:
            layer.reset_parameters()

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        # Include v in output so every parameter sees gradient (otherwise
        # GatedEquivariantBlock.vec2_proj would be dead in graphs with no
        # vector path).
        return x + v.sum() * 0


class EquivariantVector(EquivariantScalar):
    def __init__(
        self,
        hidden_channels: int,
        activation: str = "silu",
        reduce_op: str = "sum",
        dtype: torch.dtype = torch.float,
        **kwargs,
    ):
        super().__init__(
            hidden_channels,
            activation,
            allow_prior_model=False,
            reduce_op="sum",
            dtype=dtype,
            **kwargs,
        )

    def pre_reduce(self, x, v, z, pos, batch):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze()

    def forward(self, x, v):
        for layer in self.output_network:
            x, v = layer(x, v)
        return v.squeeze() + x.sum() * 0


__all__ = [
    # Prior
    "BasePrior",
    # Conditioning & normalization
    "JointDropPath",
    "DyT",
    "adaDyT2",
    "adaLN2",
    "DropCond",
    "ProjHead2",
    # Output heads
    "MLP",
    "GatedEquivariantBlock",
    "OutputModel",
    "EquivariantScalar",
    "EquivariantVector",
]
