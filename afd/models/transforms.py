"""
Data transforms — minimal subset needed to prepare QM9-like samples for the CT
forward pass. The upstream `AddStandardKeys` from
`SelfConditionedDenoisingAtoms/data/datasets/transforms.py` carried optional
hooks into `StructureCloud` and a `remove_edge_index` flag we don't need; this
copy strips both.
"""

from typing import Optional

import torch
from torch_geometric.transforms import BaseTransform


class AddStandardKeys(BaseTransform):
    """Ensure each Data has `pbc`, `natoms`, and `cell` populated.

    QM9 entries normally only carry `z` and `pos`. The CT graph generator wants
    a bounding-box `cell`, a per-graph `pbc` flag, and an `natoms` count. This
    transform supplies sensible defaults for non-periodic molecules: zero pbc,
    a diagonal bounding-box cell padded by `cell_buffer` Å, and natoms = N.

    Args:
        cell_buffer: padding added to the bounding box in each direction
            so that all atoms strictly fit inside the cell. Default 0.5 Å.
        alt_cell_key: alternative attribute to use as the cell if `cell` is
            absent (e.g. "box" used by some upstream code). Default "box".
        remove_edge_index: if True, strip any pre-existing `edge_index` on the
            Data so the CT regenerates the graph fresh. Default True.
    """

    def __init__(
        self,
        cell_buffer: float = 0.5,
        alt_cell_key: str = "box",
        remove_edge_index: bool = True,
    ):
        self.cell_buffer = cell_buffer
        self.alt_cell_key = alt_cell_key
        self.remove_edge_index = remove_edge_index

    def forward(self, data, depth: int = 0):
        if isinstance(data, tuple) and depth == 0:
            return tuple(self.forward(d, depth=depth + 1) for d in data)

        #ensure pos and z are tensors
        if not isinstance(data.pos, torch.Tensor):
            data.pos = torch.as_tensor(data.pos, dtype=torch.float32)
        if not isinstance(data.z, torch.Tensor):
            data.z = torch.as_tensor(data.z, dtype=torch.long)

        # pbc
        if not hasattr(data, "pbc"):
            data.pbc = torch.zeros((1, 3), dtype=torch.bool, device=data.pos.device)
        elif data.pbc.dim() == 1:
            data.pbc = data.pbc.reshape(1, 3)

        # natoms
        if not hasattr(data, "natoms"):
            data.natoms = torch.tensor(
                [data.pos.shape[0]], dtype=torch.long, device=data.pos.device
            )

        # cell
        if not hasattr(data, "cell"):
            if hasattr(data, self.alt_cell_key):
                data.cell = getattr(data, self.alt_cell_key)
            else:
                min_coords = data.pos.min(dim=0).values
                max_coords = data.pos.max(dim=0).values
                lengths = (max_coords - min_coords) + self.cell_buffer
                data.cell = torch.diag(lengths).unsqueeze(0).to(data.pos.device)

        if hasattr(data, "edge_index") and self.remove_edge_index:
            del data.edge_index

        return data

    def __repr__(self):
        return (
            f"{self.__class__.__name__}(cell_buffer={self.cell_buffer}, "
            f"alt_cell_key={self.alt_cell_key}, "
            f"remove_edge_index={self.remove_edge_index})"
        )


__all__ = ["AddStandardKeys"]
