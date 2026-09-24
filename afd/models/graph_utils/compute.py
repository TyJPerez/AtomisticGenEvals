"""
Graph generation helpers — periodic and non-periodic — lifted from
`SelfConditionedDenoisingAtoms/models/ET_models/graph_utils/compute.py` and
rewritten to use relative imports only.

`GraphGenerator` builds a radius graph with the pure-PyTorch brute-force
`radius_graph_pbc`. It is the numerical baseline the fast `torch_cluster`
builder is validated against, and the fallback `FastGraphGenerator` uses when
torch_cluster is unavailable. It works for both periodic crystals and
non-periodic molecules: when `pbc` is False on every axis, the cell is just a
bounding box and the periodic offsets all evaluate to zero, so you get a
standard radius graph back.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from .radius_graph_pbc import radius_graph_pbc


def get_pbc_distances(
    pos,
    edge_index,
    cell,
    cell_offsets,
    neighbors,
    return_offsets: bool = False,
    return_distance_vec: bool = False,
):
    row, col = edge_index

    distance_vectors = pos[row] - pos[col]

    # PBC correction
    neighbors = neighbors.to(cell.device)
    cell = torch.repeat_interleave(cell, neighbors, dim=0)
    offsets = cell_offsets.float().view(-1, 1, 3).bmm(cell.float()).view(-1, 3)
    distance_vectors += offsets

    distances = distance_vectors.norm(dim=-1)

    nonzero_idx = torch.arange(len(distances), device=distances.device)[distances != 0]
    edge_index = edge_index[:, nonzero_idx]
    distances = distances[nonzero_idx]

    out = {"edge_index": edge_index, "distances": distances}
    if return_distance_vec:
        out["distance_vec"] = distance_vectors[nonzero_idx]
    if return_offsets:
        out["offsets"] = offsets[nonzero_idx]
    return out


@torch.compiler.disable()
def generate_graph(
    data,
    cutoff: float,
    max_neighbors: int,
    enforce_max_neighbors_strictly: bool,
    pbc: torch.Tensor,
    self_loops: bool = False,
) -> dict:
    (
        edge_index_per_system,
        cell_offsets_per_system,
        neighbors_per_system,
    ) = list(
        zip(
            *[
                radius_graph_pbc(
                    data[idx],
                    cutoff,
                    max_neighbors,
                    enforce_max_neighbors_strictly,
                    pbc=pbc[idx],
                )
                for idx in range(len(data))
            ]
        )
    )

    atom_index_offset = data.natoms.cumsum(dim=0).roll(1)
    atom_index_offset[0] = 0
    edge_index = torch.hstack(
        [
            edge_index_per_system[idx] + atom_index_offset[idx]
            for idx in range(len(data))
        ]
    )
    cell_offsets = torch.vstack(cell_offsets_per_system)
    neighbors = torch.hstack(neighbors_per_system)

    out = get_pbc_distances(
        data.pos,
        edge_index,
        data.cell,
        cell_offsets,
        neighbors,
        return_offsets=True,
        return_distance_vec=True,
    )

    edge_index = out["edge_index"]
    edge_dist = out["distances"]
    cell_offset_distances = out["offsets"]
    distance_vec = out["distance_vec"]

    if self_loops:
        num_atoms = data.pos.size(0)
        self_loop_index = (
            torch.arange(num_atoms, device=edge_index.device).unsqueeze(0).repeat(2, 1)
        )
        edge_index = torch.cat([edge_index, self_loop_index], dim=1)
        zero_distances = torch.zeros(num_atoms, device=edge_dist.device)
        zero_vectors = torch.zeros(num_atoms, 3, device=distance_vec.device)
        zero_offsets = torch.zeros(num_atoms, 3, device=cell_offset_distances.device)
        edge_dist = torch.cat([edge_dist, zero_distances])
        distance_vec = torch.cat([distance_vec, zero_vectors])
        cell_offset_distances = torch.cat([cell_offset_distances, zero_offsets])

    return {
        "edge_index": edge_index,
        "edge_distance": edge_dist,
        "edge_distance_vec": distance_vec,
        "cell_offsets": cell_offsets,
        "offset_distances": cell_offset_distances,
        "neighbors": neighbors,
    }


class GraphGenerator:
    """Periodic / non-periodic radius graph generator on a torch_geometric Batch.

    Uses the pure-PyTorch brute-force `radius_graph_pbc`. Requires the input to
    carry `pos`, `cell`, `natoms`, and `pbc` keys. The `AddStandardKeys`
    transform in `afd.models.transforms` populates those keys for QM9-like
    (non-periodic) data.
    """

    required_keys = ["pos", "cell", "natoms", "pbc"]

    def __init__(
        self,
        cutoff: float,
        max_neighbors: int,
        self_loops: bool = False,
        enforce_max_neighbors_strictly: bool = False,
        alt_cell_key: str = "box",
    ):
        self.cutoff = cutoff
        self.max_neighbors = max_neighbors
        self.enforce_max_neighbors_strictly = enforce_max_neighbors_strictly
        self.alt_cell_key = alt_cell_key
        self.self_loops = self_loops

    def __call__(self, batch) -> dict:
        if isinstance(batch, Data):
            pass
        elif isinstance(batch, dict):
            batch = Data.from_dict(batch)
        else:
            raise ValueError(
                f"Batch must be a torch_geometric Data object or a dict, "
                f"got {type(batch)}"
            )

        if not hasattr(batch, "cell"):
            assert hasattr(batch, self.alt_cell_key), (
                f"Batch is missing both 'cell' and alternative cell key "
                f"'{self.alt_cell_key}' for graph generation"
            )
            setattr(batch, "cell", getattr(batch, self.alt_cell_key))

        if not hasattr(batch, "natoms"):
            setattr(batch, "natoms", torch.bincount(batch.batch))

        for key in self.required_keys:
            if not hasattr(batch, key):
                raise ValueError(
                    f"Batch is missing required key '{key}' for graph generation"
                )

        return generate_graph(
            batch,
            pbc=batch.pbc,
            cutoff=self.cutoff,
            max_neighbors=self.max_neighbors,
            enforce_max_neighbors_strictly=self.enforce_max_neighbors_strictly,
            self_loops=self.self_loops,
        )
