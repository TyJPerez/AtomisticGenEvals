"""
Command-line driver for AFD (Atomistic Fréchet Distance).

Thin wrapper over the high-level :class:`afd.AFDScore` scorer. One subcommand:

  - `score` — score one or more candidate ("generation") sets against a
    reference distribution.

A `--ref` spec is one of:
  - `qm9` / `mp20`         — a canonical reference whose CT features are fetched
                             from HuggingFace (no embedding pass; picks the
                             matching checkpoint automatically), or
  - a path to a `.pt` file — a list of records (dicts or `Data` with `z`/`pos`
                             [+ `cell`/`pbc`]) used as a custom reference; pair
                             with `--data-modality` (molecules/materials) or an
                             explicit `--model-name`.

Each `--gen` spec is a `.pt` file holding a list of the same record dicts.

Because AFD is biased in the sample size N, use `--ref-sample-size` to subsample
the reference to match each candidate set (never compare scores across N).
"""

from __future__ import annotations

import argparse
import os
import time
from typing import List, Optional

import torch

from .score import AFDScore


def _load_records(path: str) -> list:
    """Load a `.pt` file holding a list of structure records (dicts or Data)."""
    raw = torch.load(path, map_location="cpu", weights_only=False)
    if not isinstance(raw, list):
        raise TypeError(f"Expected a list of records in {path}, got {type(raw).__name__}")
    return raw


def _build_scorer(args) -> AFDScore:
    """Construct the AFDScore reference scorer from the CLI args."""
    is_key = args.ref.lower() in ("qm9", "mp20") and not os.path.exists(args.ref)
    if is_key:
        # Canonical reference: features pulled from HF, checkpoint auto-selected
        # (overridable with --model-name).
        print(f"[ref] canonical reference {args.ref!r}")
        return AFDScore(
            args.ref,
            model_name=args.model_name,
            batch_size=args.batch_size,
            device=args.device,
        )

    # Custom reference from a .pt file of records.
    print(f"[ref] custom reference from {args.ref!r}")
    records = _load_records(args.ref)
    if args.ref_n is not None:
        records = records[: args.ref_n]
    print(f"[ref] {len(records)} records; embedding on {args.device}")
    return AFDScore(
        records,
        data_modality=args.data_modality,
        model_name=args.model_name,
        batch_size=args.batch_size,
        device=args.device,
    )


def cmd_score(args):
    scorer = _build_scorer(args)
    n_ref = len(scorer.ref_features)
    feat_dim = scorer.ref_features.shape[1]

    print("\n=== Scores ===")
    print(f"reference: {args.ref!r}, N_ref={n_ref}, feat_dim={feat_dim}"
          + (f", ref_sample_size={args.ref_sample_size}" if args.ref_sample_size else ""))

    rows = []
    for path in args.gen:
        print(f"\n[gen] {path!r}")
        records = _load_records(path)
        if args.gen_n is not None:
            records = records[: args.gen_n]
        t0 = time.time()
        score_value = scorer(
            records,
            ref_sample_size=args.ref_sample_size,
            seed=args.seed,
            show_progress=not args.quiet,
        )
        dt = time.time() - t0
        label = os.path.basename(path) if path.endswith(".pt") else path
        print(f"[gen] {label}: AFD = {score_value:.4f}  (N_gen={len(records)}, dt={dt:.1f}s)")
        rows.append((label, len(records), score_value))

    print("\n=== Summary ===")
    print(f"{'source':<60}{'N_gen':>8}{'AFD':>14}")
    for name, n, val in rows:
        print(f"{name:<60}{n:>8}{val:>14.4f}")


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m afd",
        description="Fréchet-distance scoring for atomistic generative models.",
    )
    sub = p.add_subparsers(dest="cmd", required=True)

    p_score = sub.add_parser("score", help="Score one or more candidate sets.")
    p_score.add_argument(
        "--ref", default="qm9",
        help="Reference spec: 'qm9' / 'mp20' (canonical HF features) or a path "
             "to a .pt list of records. Default: qm9.",
    )
    p_score.add_argument("--gen", nargs="+", required=True,
                         help="One or more candidate specs (.pt list of records).")
    p_score.add_argument("--ref-n", type=int, default=None,
                         help="Cap a custom (path) reference at the first N records.")
    p_score.add_argument("--gen-n", type=int, default=None,
                         help="Cap each candidate set at the first N records.")
    p_score.add_argument("--ref-sample-size", type=int, default=None,
                         help="Subsample the reference to this N (match candidate N; "
                              "AFD is biased in N, so never compare across N).")
    p_score.add_argument("--seed", type=int, default=None,
                         help="Seed for reference subsampling (reproducibility).")
    p_score.add_argument("--data-modality", default=None,
                         help="For a custom (path) reference: 'molecules' or "
                              "'materials' — selects the CT checkpoint.")
    p_score.add_argument("--model-name", default=None,
                         help="Explicit CT checkpoint (e.g. ct-scd-pcq / ct-scd-amp20). "
                              "Overrides the modality/canonical default.")
    p_score.add_argument("--batch-size", type=int, default=256,
                         help="Batch size for the CT forward pass.")
    p_score.add_argument("--device", default="cuda:0",
                         help="Device for the model (cpu / cuda / cuda:0 / ...).")
    p_score.add_argument("--quiet", action="store_true",
                         help="Suppress the per-batch progress bar.")
    p_score.set_defaults(func=cmd_score)

    return p


def main(argv: Optional[List[str]] = None):
    args = build_parser().parse_args(argv)
    args.func(args)


if __name__ == "__main__":
    main()
