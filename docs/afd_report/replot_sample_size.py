"""Re-plot the AFD fold-vs-fold sample-size figures for the report.

Changes vs. re_val/plots.py:sample_size:
  (a) drop the "mean over folds" connecting line — on QM9 the N=10,000 mean is
      slightly negative (it has reached the numerical floor), which a log axis
      cannot draw, so the original line plunged vertically before the last point.
      We instead show the individual per-fold points, which is what carries the
      information, plus per-N mean markers.
  (b) fit the finite-N bias floor AFD ~ intercept + c/N only over the full-rank
      points with a positive mean, so the fit curve no longer dives off the axis.

Reads the raw results JSON written by re_val and writes PNGs into
afd_report/figures/.
"""
import json
import os

import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
REVAL = os.path.join(os.path.dirname(HERE), 're_val')
RESULTS = os.path.join(REVAL, 'results')
OUT = os.path.join(HERE, 'figures')

DATASET_COLOR = {'qm9': '#1f77b4', 'mp20': '#d62728'}

JOBS = [
    ('fig1_sample_size__pcq.json', 'fig1_sample_size.png'),
    ('mp20_fig1_sample_size__amp20.json', 'mp20_fig1_sample_size.png'),
]


def replot(res, out_path):
    dataset = res['dataset']
    color = DATASET_COLOR.get(dataset, 'C0')
    N = np.asarray(res['N'], float)
    afd = np.asarray(res['afd'], float)
    s = res['summary']
    sizes = np.asarray(s['N'], float)
    mean = np.asarray(s['mean'], float)
    fdim = res.get('feature_dim', 256)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4.2))

    # (a) per-fold points (no mean line); positive means as ticks; feature-dim rank line
    pos = afd > 0
    ax1.scatter(N[pos], afd[pos], s=16, alpha=0.45, color=color, label='individual folds')
    mpos = mean > 0
    ax1.scatter(sizes[mpos], mean[mpos], s=45, color='k', marker='_',
                linewidths=1.8, zorder=5, label='per-N mean')
    ax1.axvline(fdim, ls=':', color='r', lw=1, label=f'feature dim ({fdim})')
    ax1.set_xscale('log'); ax1.set_yscale('log')
    ax1.set_xlabel('fold size N'); ax1.set_ylabel('AFD (fold vs fold)')
    ax1.set_title('(a) fold-vs-fold baseline (true distance = 0)')
    ax1.legend(fontsize=9)

    # (b) finite-N bias: full-rank regime (N>=500), fit floor ~ intercept + c/N
    full = (sizes >= 500) & (mean > 0)
    ax2.scatter(sizes[full], mean[full], s=45, color='k', zorder=5, label='mean (N≥500)')
    if full.sum() >= 2:
        slope, intercept = np.polyfit(1.0 / sizes[full], mean[full], 1)
        xx = np.linspace(sizes[full].min(), sizes[full].max(), 200)
        ax2.plot(xx, intercept + slope / xx, '-', color='C1',
                 label=f'fit ≈ {intercept:.1e} + {slope:.2g}/N')
    ax2.set_xscale('log'); ax2.set_yscale('log')
    ax2.set_xlabel('fold size N'); ax2.set_ylabel('AFD')
    ax2.set_title('(b) finite-N bias, full-rank regime')
    ax2.legend(fontsize=9)

    fig.suptitle(f'AFD fold-vs-fold baseline & minimum sample size — {dataset}', y=1.02)
    fig.text(0.995, 0.01, f'[{dataset} · {res["scorer_label"]}]', ha='right',
             va='bottom', fontsize=8, color='0.5', style='italic')
    fig.tight_layout()
    fig.savefig(out_path, dpi=200, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved {out_path}')


def main():
    os.makedirs(OUT, exist_ok=True)
    for src, dst in JOBS:
        res = json.load(open(os.path.join(RESULTS, src)))
        replot(res, os.path.join(OUT, dst))


if __name__ == '__main__':
    main()
