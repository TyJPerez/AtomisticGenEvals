"""
Fréchet distance on CT feature distributions — the AFD (Atomistic Fréchet
Distance) metric.

Mirrors the closed-form Wasserstein-2-between-Gaussians used by image FID
(Heusel et al. 2017, eq. 6), but with `mol_emb` from a chemistry-pretrained
Conditional Equivariant Transformer instead of Inception-v3.

    AFD²  =  ||μ_ref - μ_gen||²  +  Tr(Σ_ref + Σ_gen - 2·(Σ_ref·Σ_gen)^{1/2})

Implementation notes / caveats inherited from image FID:
  - The estimator is biased upward; the bias depends on N. **Comparing AFD
    scores across different sample sizes is invalid** — fix N on both sides.
  - The Gaussian assumption on CT `mol_emb` features is unverified. Treat
    AFD as a Gaussian *projection* of W2, not the true W2.
  - For numerical stability we add `eps · I` to the covariance product before
    taking its matrix square root; any imaginary part of the output is discarded
    (it is bounded by float-precision noise).
"""

from __future__ import annotations

from typing import Optional, Tuple
import numpy as np
import torch
from huggingface_hub import hf_hub_download
from .features import FeatureExtractor, _get_model_name 


def torch_sqrtm(matrix: np.ndarray | torch.Tensor) -> np.ndarray | torch.Tensor:
    """Return the principal matrix square root using Torch operations.

    Torch does not currently expose ``torch.linalg.sqrtm``. The stabilized
    covariance products used by AFD are diagonalizable with non-negative
    eigenvalues, so their principal square root can be computed from the
    eigendecomposition.
    """
    if isinstance(matrix, torch.Tensor):
        if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
            raise ValueError(f"matrix must be square, got shape {tuple(matrix.shape)}")

        if matrix.dtype in (torch.complex64, torch.float16, torch.float32, torch.bfloat16):
            work = matrix.to(dtype=torch.complex64)
        else:
            work = matrix.to(dtype=torch.complex128)

        eigvals, eigvecs = torch.linalg.eig(work)
        sqrt_eigvals = torch.sqrt(eigvals)
        sqrt_diag = torch.diag_embed(sqrt_eigvals)
        return eigvecs @ sqrt_diag @ torch.linalg.inv(eigvecs)

    matrix = np.asarray(matrix)
    if matrix.ndim < 2 or matrix.shape[-1] != matrix.shape[-2]:
        raise ValueError(f"matrix must be square, got shape {matrix.shape}")

    work = torch.as_tensor(matrix, dtype=torch.complex128)
    eigvals, eigvecs = torch.linalg.eig(work)
    sqrt_eigvals = torch.sqrt(eigvals)
    sqrt_diag = torch.diag_embed(sqrt_eigvals)
    return (eigvecs @ sqrt_diag @ torch.linalg.inv(eigvecs)).cpu().numpy()


def gaussian_stats(features) -> Tuple[np.ndarray, np.ndarray]:
    """Return (mean, covariance) of a (N, D) feature tensor / array."""
    if isinstance(features, torch.Tensor):
        features = features.detach().cpu().numpy()
    features = np.asarray(features, dtype=np.float64)
    assert features.ndim == 2, f"features must be 2D, got shape {features.shape}"
    mu = features.mean(axis=0)
    # ddof=1 is the unbiased sample covariance. We use it on both sides so the
    # bias contributions are matched.
    cov = np.cov(features, rowvar=False, ddof=1)
    return mu, cov


def frechet_distance(
    mu_a: np.ndarray,
    cov_a: np.ndarray,
    mu_b: np.ndarray,
    cov_b: np.ndarray,
    eps: float = 1e-6,
) -> float:
    """Closed-form squared Fréchet distance between two Gaussians."""
    mu_a = np.asarray(mu_a, dtype=np.float64)
    mu_b = np.asarray(mu_b, dtype=np.float64)
    cov_a = np.asarray(cov_a, dtype=np.float64)
    cov_b = np.asarray(cov_b, dtype=np.float64)

    diff = mu_a - mu_b

    # Stabilise the matrix sqrt of (cov_a @ cov_b) by adding eps*I.
    offset = eps * np.eye(cov_a.shape[0], dtype=np.float64)
    covmean = torch_sqrtm((cov_a + offset).dot(cov_b + offset))

    if np.iscomplexobj(covmean):
        # Imag part is float-noise; drop it. If it is large, surface that.
        max_imag = np.max(np.abs(covmean.imag))
        if max_imag > 1e-3:
            import warnings
            warnings.warn(
                f"sqrtm returned non-trivial imaginary component "
                f"(max |imag| = {max_imag:.3e}); using real part only.",
                RuntimeWarning,
            )
        covmean = covmean.real

    fid = diff.dot(diff) + np.trace(cov_a) + np.trace(cov_b) - 2.0 * np.trace(covmean)
    return float(fid)


def afd_score(features_ref, features_gen, eps: float = 1e-6) -> float:
    """End-to-end: features in, AFD score out."""
    mu_r, cov_r = gaussian_stats(features_ref)
    mu_g, cov_g = gaussian_stats(features_gen)
    return frechet_distance(mu_r, cov_r, mu_g, cov_g, eps=eps)


def _features_to_cpu(features):
    """Keep reference features off GPU; GEOM refs are large enough to OOM."""
    if isinstance(features, torch.Tensor):
        return features.detach().to("cpu")
    return torch.as_tensor(features, device="cpu")


def _get_ref_features(name: str):
    hf_repo = 'Ty-Perez/AtomisticEval'
    avail_datasets = {
        "qm9": 'qm9_features.pt',
        "mp20": 'mp20_features.pt',
        "geom": 'geom_features.pt',
    }
    ds_keys = set(list(avail_datasets.keys()))
    name = name.lower()
    if name not in ds_keys:
        raise ValueError(f"Unrecognized reference dataset name '{name}'. "
                         f"Available options are: {ds_keys}")
    
    #download the target file from HuggingFace and load the features
    file_name = avail_datasets[name]
    local_path = hf_hub_download(repo_id=hf_repo, 
                                 filename=file_name,
                                 repo_type="dataset")
    features, ids = torch.load(local_path, map_location="cpu")
    return _features_to_cpu(features), ids


class AFDScore():

    '''
    AFD scorer class. Initializes with a pretrained CT and featurizer, then can be called on a dataset to compute the AFD score against the reference QM9 distribution.
    
    args:
        ref_dataset (str or dataset object): 
            this can be either the reference dataset
            name (QM9 or mp20) or a dataset that will be loaded and used as the reference distribution for AFD scoring.
        
        data_modality (str : None): 
            if ref_dataset is given as a dataset object, 
            then a data_modality keyword must be provided to specify the type of 
            data being scored (e.g. "molecules", "materials"). This is used to 
            select the appropriate model checkpoint used to create features and standard transform.
         
        model_name [Optional] (str : None): 
            A specific model checkpoint may be chosen by name if provided. This is only appropriate if using a unique reference dataset that is not QM9 or MP20 - eg datasest with a larger range of elements or non-groundstate datasets.

        feature_cache_dir [Optional] (str : None):
            Optional directory to cache computed reference features. If None, new reference features will be computed on each initialization and not cached. 
            This is only relevent when ref_dataset is given as a dataset object, rather than a string key.
            
        batch_size (int : 512): batch size to use when computing features with the CT; passed to `compute_features`. Default 512.
        device (str : "cuda"): device to run CT featurization on; passed to `compute_features`. Default "cuda:0".
    
    '''
    def __init__(self, 
        ref_dataset,
        data_modality : str = None,
        model_name : str = None,
        batch_size=256,  
        device="cuda:0",
        feature_cache_dir : Optional[str] = None,
        verbose : bool = False,
        seed = 42 # determins the shuffle of the reference dataset
        ):
        self.verbose = verbose
        self.device = device
        self.seed = seed

        if isinstance(ref_dataset, str):
            #Using standard precomputed reference dataset features
            ref_dataset = ref_dataset.lower()
            self.ref_features, self.ref_ids = _get_ref_features(ref_dataset)
            model_name = _get_model_name(ref_dataset) if model_name is None else model_name
            
            self.featurizer = FeatureExtractor(model_name=model_name, batch_size=batch_size, device=device, verbose=verbose)

        else:
            if data_modality is None and model_name is None:
                raise ValueError("If ref_dataset is given as a dataset object, either data_modality or model_name must be provided to select the appropriate CT checkpoint for featurization.")
            if model_name is None:
                #if no model name is provided, select based on data_modality
                model_name = _get_model_name(data_modality)
            print(f"Initializing AFDScore with model '{model_name}' ")
            self.ref_ids = None
            self.featurizer = FeatureExtractor(model_name=model_name, batch_size=batch_size, device=device, verbose=verbose)
            
            #check for cached features
            if feature_cache_dir is not None:
                #if a cache file exists, load it. Otherwise compute and cache the features
                pass
            else:
                # compute reference features
                self.ref_features = self.featurizer(ref_dataset)
        self.ref_features = _features_to_cpu(self.ref_features)
        
        # shuffle the reference features
        torch.manual_seed(self.seed)
        indices = torch.randperm(len(self.ref_features), device="cpu")
        self.ref_features = self.ref_features.index_select(0, indices)
        if self.ref_ids is not None:
            if isinstance(self.ref_ids, torch.Tensor):
                self.ref_ids = self.ref_ids.index_select(0, indices.to(self.ref_ids.device))
            elif isinstance(self.ref_ids, np.ndarray):
                self.ref_ids = self.ref_ids[indices.cpu().numpy()]
            else:
                self.ref_ids = [self.ref_ids[i] for i in indices.cpu().tolist()]
        
        # Reference dataset is shuffled to ensure reproducible random sampling 
        # when the refernece set needs to be trimmed to match the generation sample size

    def sample_ref(self, size, seed=None):
        '''
        Sample a subset of the reference features for use in AFD scoring. 
        This is necessary to ensure that the same number of samples are used from both distributions, since AFD is not comparable across different sample sizes.
        
        args:
            size (int): number of reference samples to use in AFD scoring. Must be less than or equal to the total number of reference features.
            seed (int : None): random seed for reproducible sampling. Default None.\
        returns:
            sampled_ref_features (Tensor): a (size, D) tensor of reference features sampled from the full reference distribution.
        '''
        if size > len(self.ref_features):
            raise ValueError(f"Requested sample size {size} exceeds total number of reference features {len(self.ref_features)}.")
        if seed is not None:
            torch.manual_seed(seed)
        indices = torch.randperm(len(self.ref_features), device="cpu")[:size]
        return self.ref_features[indices]
    
    def embeddings(self, dataset, **kwargs):
        '''
        Compute CT features for a given dataset. This is a separate method from `score` to allow users to compute and save features for their own datasets without needing to compute AFD scores.
        
        args:
            dataset: a dataset object containing the data to be featurized. The expected format will depend on the featurizer model being used - e.g. for the standard QM9/MP20 featurizer, the dataset should contain a "molecules" column with RDKit molecule objects.
            **kwargs: additional keyword arguments to pass to the featurizer's `compute_features` method. This may include things like batch_size and device, which will override the defaults set at initialization.
        
        returns:
            features (Tensor): a (N, D) tensor of CT features computed for the input dataset.
        '''
        return self.featurizer(dataset, **kwargs)
        
    def score(self, dataset, trim_samples=True, **kwargs) -> float:

        #choose ref features
        ref_sample_size = kwargs.pop("ref_sample_size", None)
        if ref_sample_size is not None:
            ref_feats = self.sample_ref(ref_sample_size, seed=kwargs.pop("seed", None))
        else:             
            ref_feats = self.ref_features

        #compute features for dataset
        target_feats = self.featurizer(dataset, **kwargs)
        
        if trim_samples:
            min_size = min(ref_feats.shape[0], target_feats.shape[0])
            if self.verbose:
                print(f"Trimming samples to size: {min_size}")
            ref_feats = ref_feats[:min_size]
            target_feats = target_feats[:min_size]
                
        #compute AFD score
        return afd_score(ref_feats, target_feats)
    
    def __call__(self, dataset, **kwargs) -> float:
        return self.score(dataset, **kwargs)




__all__ = ["AFDScore", "gaussian_stats", "torch_sqrtm", "frechet_distance", "afd_score"]
