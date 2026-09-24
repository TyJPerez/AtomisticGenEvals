"""
AFD — minimal copy of the Conditional Equivariant Transformer (CT)
extracted from `SelfConditionedDenoisingAtoms`. See `afd.models.scd_model`
for the top-level `CET` class.

Standard usage:

    from afd.models import CET, load_pretrained_ct, AddStandardKeys

    model = load_pretrained_ct("ct-scd-pcq")  # downloads from HF
    model.eval()

    # Wrap your torch_geometric Data with AddStandardKeys() so the CT graph
    # generator sees the pbc / natoms / cell attributes it requires:
    transform = AddStandardKeys()
    data = transform(my_data)

    with torch.no_grad():
        out = model(z=data.z, pos=data.pos, batch=data.batch, graph_batch=data,
                    return_atom_embs=True)
    # out has keys "y", "noise_pred", "mol_emb", "atom_embs".
"""

from .graph_utils import GraphGenerator
from .loader import load_ct_from_ckpt, load_pretrained_ct
from .modules import (
    BasePrior,
    DropCond,
    DyT,
    EquivariantScalar,
    EquivariantVector,
    GatedEquivariantBlock,
    JointDropPath,
    MLP,
    ProjHead2,
    adaDyT2,
    adaLN2,
)
from .scd_model import CET, AccumulatedNormalization, BaseModel
from .scet import ConditionalET, CondEquivMultiHeadAttention
from .transforms import AddStandardKeys

__all__ = [
    # Top-level CET API
    "CET",
    "load_pretrained_ct",
    "load_ct_from_ckpt",
    # Data prep
    "AddStandardKeys",
    "GraphGenerator",
    # Building blocks (re-exported for advanced use)
    "BaseModel",
    "ConditionalET",
    "CondEquivMultiHeadAttention",
    "AccumulatedNormalization",
    "ProjHead2",
    "DropCond",
    "DyT",
    "JointDropPath",
    "adaLN2",
    "adaDyT2",
    "GatedEquivariantBlock",
    "MLP",
    "EquivariantScalar",
    "EquivariantVector",
    "BasePrior",
]
