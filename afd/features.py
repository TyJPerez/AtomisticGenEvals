"""
Feature extraction — run a set of atomistic structures through the pretrained
CT model and collect graph-level embeddings (`mol_emb`).

The contract:

    feats = compute_features(records, model, transform=AddStandardKeys())
    feats  # torch.Tensor of shape (len(records), 256), on CPU
"""

from __future__ import annotations
from typing import Iterable, Optional
import warnings

import torch
from torch_geometric.loader import DataLoader
from tqdm.auto import tqdm

from torch.utils.data import Dataset
from torch_geometric.data import Data
from .models import AddStandardKeys, load_pretrained_ct

def _get_model_name(data_keyword: str):
    '''
    Helper to select a model name based on given data keyword.
    args:
        data_keyword : str, keyword indicating the type of data (e.g. 'qm9', 'mp20', 'materials', 'molecules')
    returns:
        model_name : str, name of the pretrained CT model to use for feature extraction

    '''
    data_keyword = data_keyword.lower()
    molecules_keys = ['qm9', 'molecules', 'small_molecules']
    geom_keys = ['geom', 'geom1', 'geom10']
    materials_keys = ['mp20', 'mp_20', 'materials', 'crystals', 'periodic_crystals']

    if data_keyword in geom_keys:
        return 'ct-scd-geom10'
    elif data_keyword in molecules_keys:
        return 'ct-scd-pcq'
    elif data_keyword in materials_keys:
        return 'ct-scd-amp20'
    else:
        raise ValueError(f"Unrecognized data keyword '{data_keyword}'. Please use one of {geom_keys + molecules_keys + materials_keys}.")

def _build_model(model_name: str, cache_dir: Optional[str] = None, device: str = "cpu"):
    # default HF cache dir: ~/.cache/huggingface/hub
    if cache_dir is None:
        from pathlib import Path
        cache_dir = str(Path.home() / ".cache" / "huggingface" / "hub")
    model = load_pretrained_ct(model_name, cache_dir=cache_dir, device=device).eval()
    return model #, AddStandardKeys()


def _get_field(sample, key):
    if isinstance(sample, dict):
        return sample.get(key)
    if hasattr(sample, key):
        return getattr(sample, key)
    try:
        return sample[key]
    except Exception:
        return None


def _leading_dim(value) -> Optional[int]:
    if value is None:
        return None
    if hasattr(value, "shape") and len(value.shape) > 0:
        return int(value.shape[0])
    try:
        return len(value)
    except TypeError:
        return None


def _is_featurizable(sample) -> bool:
    pos_atoms = _leading_dim(_get_field(sample, "pos"))
    z_atoms = _leading_dim(_get_field(sample, "z"))
    return pos_atoms is not None and z_atoms is not None and pos_atoms > 0 and z_atoms > 0


class GenerationDataset(Dataset):
    """In-memory wrapper for dataset (or list of `Data`) with an optional per-item transform.
    
    input dataset must have have keys `z` and `pos` (e.g. QM9, MP20) so that the CT can featurize it.  
    
    """

    def __init__(self, data: Iterable[Data], transform=None, drop_invalid: bool = True):

        if not hasattr(data, "__getitem__") or not hasattr(data, "__len__"):
            raise ValueError("data must be an iterable dataset (e.g. list of Data objects, or a PyG Dataset)")

        self.data = data
        self.transform = transform
        self.indices = list(range(len(data)))

        sample = data[0]
        self.dict_data = False
        required_keys = {"z", "pos"}
        # Sanity check: ensure that data items are either Data objects or dicts
        if not isinstance(sample, Data):
            if not isinstance(sample, dict):
                raise ValueError("Each item in data must be a torch_geometric.data.Data object or a dictionary")
            else:
                self.dict_data = True
        if self.dict_data:
            if not all(isinstance(sample, dict) for sample in data):
                raise ValueError("If the first item in data is a dict, all items must be dicts")
        else:
            if not all(hasattr(sample, key) for key in required_keys):
                raise ValueError(f"Each Data object must have keys: {required_keys}")

        if drop_invalid:
            valid_indices = [idx for idx in self.indices if _is_featurizable(data[idx])]
            dropped = len(self.indices) - len(valid_indices)
            if dropped:
                warnings.warn(
                    f"Skipping {dropped} non-featurizable samples with empty/missing z or pos.",
                    RuntimeWarning,
                )
            self.indices = valid_indices

        if not self.indices:
            raise RuntimeError("No featurizable records remain after filtering empty/missing z or pos")

    def __len__(self) -> int:
        return len(self.indices)

    def __getitem__(self, idx: int) -> Data:
        data = self.data[self.indices[idx]]
        if self.dict_data:
            # Convert dict to Data object on the fly
            data = Data(**data)
        return self.transform(data) if self.transform is not None else data

@torch.no_grad()
def compute_features(
    dataset,
    model,
    transform=None,
    batch_size: int = 512,
    device: Optional[str] = None,
    show_progress: bool = True,
    feature_key: str = "mol_emb",
    desc: str = "features",
    drop_invalid: bool = True,
) -> torch.Tensor:
    """Run `dataset` through `model` and stack the per-graph embeddings.

    Args:
        dataset: iterable of `torch_geometric.data.Data` objects.
        model: a `CET` (or compatible) on the desired device, in eval mode.
        transform: optional per-record transform applied at __getitem__ time
            (e.g. `AddStandardKeys` to populate pbc/natoms/cell).
        batch_size: graphs per forward pass.
        device: device string ("cpu" / "cuda" / "cuda:0"). If None, uses
            whatever device the model's parameters are already on.
        show_progress: render a tqdm progress bar over the batches.
        feature_key: which key from the model's forward dict to harvest.
            "mol_emb" (default) gives the 256-d projection head output.
        desc: prefix shown on the tqdm progress bar.
        drop_invalid: if True, skip records with empty/missing `z` or `pos`.

    Returns:
        torch.Tensor of shape (len(dataset), feature_dim), on CPU.
    """
    if device is None:
        # check if cuda available and use it, otherwise default to cpu
        device = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(device)
    model = model.to(device).eval()

    dataset = GenerationDataset(dataset, transform=transform, drop_invalid=drop_invalid)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    chunks = []
    iterator = tqdm(
        loader,
        total=len(loader),
        desc=desc,
        unit="batch",
        disable=not show_progress,
        leave=False,
    )
    try:
        for batch in iterator:
            batch = batch.to(device)
            out = model(
                z=batch.z,
                pos=batch.pos,
                batch=batch.batch,
                graph_batch=batch,
            )
            # Detach + move features to CPU immediately so nothing accumulates
            # on the GPU across batches; drop references to the batch / outputs.
            chunks.append(out[feature_key].detach().to("cpu"))
            del out, batch
    finally:
        # Always release the GPU — move the model back to CPU and free the
        # caching allocator's reserved blocks — even if a batch raised (e.g. an
        # OOM on a pathological structure), so the next call starts clean.
        model.to("cpu")
        if device.type == "cuda":
            torch.cuda.empty_cache()

    if not chunks:
        raise RuntimeError("compute_features got zero records")
    return torch.cat(chunks, dim=0)


class FeatureExtractor():
    """Wrapper to hold a pretrained CT model and extract features from a dataset.

    This is not strictly necessary, but it bundles the model loading and
    feature extraction into one place, which may be convenient for users.

    Usage:
        extractor = FeatureExtractor(data_keyword="qm9")
        feats = extractor.compute_features(qm9_subset)
    """

    def __init__(self, data_keyword: str = None,
                    model_name: Optional[str] = None, 
                    batch_size: int = 512,
                    device: Optional[str] = None,
                    model_cache_dir: Optional[str] = None,
                    verbose: bool = False
                   ):
        if model_name is None:
            model_name = _get_model_name(data_keyword)
        self.device = device
        self.batch_size = batch_size
        self.verbose = verbose
        if self.verbose:
            print(f"Using model for feature extraction: {model_name}")
        self.model_name = model_name
        self.model = _build_model(model_name, cache_dir=model_cache_dir, device=device)
        self.standard_transform = AddStandardKeys()
    
    def _compute_features(self, dataset, **kwargs) -> torch.Tensor:
        #set default kwargs for compute_features if not provided
        kwargs.setdefault("batch_size", self.batch_size)
        kwargs.setdefault("device", self.device)

        return compute_features(dataset, self.model, transform=self.standard_transform, **kwargs)
    
    def __call__(self, dataset, **kwargs) -> torch.Tensor:
        
        return self._compute_features(dataset, **kwargs)



__all__ = ["compute_features"]
