"""
AFD — Atomistic Fréchet Distance.

Fréchet-distance scoring for atomistic generative models. Same recipe as
image FID (Heusel et al. 2017) but the Inception feature extractor is
replaced with a Conditional Equivariant Transformer (CT) pretrained on
atomistic data — so the score lives in a chemistry-aware feature space.

Standard usage (the high-level `AFDScore` scorer):

    from afd import AFDScore

    # Canonical reference (features fetched from HuggingFace, no GPU cost):
    #   "qm9"  -> molecule checkpoint  ct-scd-pcq
    #   "mp20" -> materials checkpoint ct-scd-amp20
    scorer = AFDScore("qm9", device="cuda:0")

    # Score any candidate set (list of Data / dicts with z, pos[, cell, pbc]).
    print("AFD:", scorer(gen_records))
    # Match sample size on both sides (AFD is biased in N — never compare across N):
    print("AFD@5000:", scorer(gen_records, ref_sample_size=5000, seed=0))

    # A custom reference distribution + explicit modality/checkpoint:
    scorer = AFDScore(my_ref_records, data_modality="materials")

Lower-level building blocks:

    from afd import FeatureExtractor, compute_features, afd_score
    feats_a = FeatureExtractor(data_keyword="qm9")(records_a)
    feats_b = FeatureExtractor(data_keyword="qm9")(records_b)
    afd_score(feats_a, feats_b)

Or via the CLI:

    python -m afd score --gen qm9_gens/ADiT_qm9.pt --ref qm9 --device cuda:0
"""

from .features import FeatureExtractor, compute_features
from .score import (
    AFDScore,
    afd_score,
    frechet_distance,
    gaussian_stats,
    torch_sqrtm,
)

__all__ = [
    # High-level scorer
    "AFDScore",
    # Feature extraction
    "FeatureExtractor",
    "compute_features",
    # Distance math
    "afd_score",
    "frechet_distance",
    "gaussian_stats",
    "torch_sqrtm",
]
