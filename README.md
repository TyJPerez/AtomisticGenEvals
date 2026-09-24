# Distributional Evaluations for Atomistic Generations 

This is the public development repo for better evaluation methods of atomistic generative models. Here we will be sharing our methods and technical reports to help push the research community towards better model evaluations. If you use anything from this repo please cite it with the BibTeX below.

## Citing this repo

Please cite the repository itself, and mention the specific method (e.g. AFD) and the version or commit you used.

```bibtex
@misc{perez2026atomisticgenevals,
  author       = {Perez, Tynan and G{\'o}mez-Bombarelli,Rafael},
  title        = {Distributional Evaluations for Atomistic Generations: a set of methods and technical reports for evaluating atomistic generative models},
  year         = {2026},
  howpublished = {\url{https://github.com/TyJPerez/AtomisticGenEvals}},
}
```

## Methods

| Method | Status | Technical report | Example notebook |
|---|---|---|---|
| [AFD — Atomistic Fréchet Distance](#afd--atomistic-fréchet-distance) | available | [docs/AFD_report.pdf](docs/AFD_report.pdf) | [afd_examples.ipynb](afd_examples.ipynb) |
| [AGD](#agd) | coming soon | — | — |

### AFD — Atomistic Fréchet Distance

AFD is "FID for atoms": the Fréchet Inception Distance recipe with the Inception image encoder replaced by a Conditional Equivariant Transformer (CT) pretrained on atomistic data. Every reference and generated structure is embedded with the frozen CT, a Gaussian is fit to each set of 256-d graph embeddings, and AFD is the closed-form Fréchet (Wasserstein-2) distance between the two Gaussians, so it is a single set-level score that needs no labels, matching, or likelihood and registers drift in geometry, composition, or both. The technical report shows it is monotone in controlled degradations (coordinate noise, atom mutation, vacancies, bond and lattice strain), calibrates raw values to physical perturbations, and scores a panel of published molecule and materials generators; lower is better, and it should be paired with validity, stability, and novelty checks rather than replace them.

- Technical report: [docs/AFD_report.pdf](docs/AFD_report.pdf) (LaTeX source in [docs/afd_report/](docs/afd_report/))
- Example notebook: [afd_examples.ipynb](afd_examples.ipynb)
- Code: [afd/](afd/) (`AFDScore`, `FeatureExtractor`, `afd_score`, and the `python -m afd` CLI)

Practical rules from the report: always compare scores at the same sample size N (AFD is biased upward as N shrinks), use N ≥ 1,000 to rank models and N ≥ 5,000 to resolve small gaps, and always use the domain-matched checkpoint (`ct-scd-pcq` for QM9-like molecules, `ct-scd-geom10` for drug-like molecules, `ct-scd-amp20` for periodic crystals).

### AGD

Coming soon.

## Installation

The repo is self-contained; pretrained checkpoints and reference features are downloaded from the HuggingFace Hub on first use.

```bash
git clone https://github.com/TyJPerez/AtomisticGenEvals.git
cd AtomisticGenEvals
pip install torch torch_geometric huggingface_hub tqdm numpy
pip install torch_cluster   # optional: faster neighbour-graph construction
```

Run scripts and notebooks from the repository root so that `afd` is importable.

## Quick start

Score a set of generated structures against the canonical QM9 reference:

```python
import torch
from afd import AFDScore

records = torch.load("my_gens.pt", weights_only=False)  # list of {"z": ..., "pos": ...[, "cell", "pbc"]}
scorer = AFDScore("qm9", device="cuda:0")               # "mp20" and "geom" are also available
print(scorer(records, ref_sample_size=len(records), seed=0))
```

Or from the command line:

```bash
python -m afd score --ref qm9 --gen my_gens.pt --ref-sample-size 5000 --seed 0 --device cuda:0
```

See [afd_examples.ipynb](afd_examples.ipynb) for a full walkthrough, including the low-level building blocks and a demonstration of the sample-size bias.
