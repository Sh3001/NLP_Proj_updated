"""Is the test F0.5 difference between two matchers real? Paired bootstrap over S1 entities.

Both runs must already be final (threshold tuned on val, test scored once); this only compares their saved
predictions, it tunes nothing.

    python3 analysis/paired_bootstrap.py --gt <er-small>/test_labels/test_ground_truth.tsv \
        --a <run A>/preds/<method>/matching_results.tsv --b <run B>/preds/<method>/matching_results.tsv

Prints macro F0.5 of A and B, the difference A - B, how many entities differ, and a 95% bootstrap interval of the
difference (resampling S1 entities with replacement, the same entities for A and B).
"""
import argparse, sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / 'src'))
from evaluate import read_lists, entity_score  # noqa: E402  (the competition metric)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--gt', required=True)
    ap.add_argument('--a', required=True, help='matching_results.tsv of run A')
    ap.add_argument('--b', required=True, help='matching_results.tsv of run B')
    ap.add_argument('--n', type=int, default=10_000, help='bootstrap resamples')
    ap.add_argument('--seed', type=int, default=0)
    a = ap.parse_args()
    gt, pa, pb = read_lists(a.gt), read_lists(a.a), read_lists(a.b)
    ids = sorted(gt)
    fa = np.array([entity_score(gt[s], pa.get(s, set()))[0] for s in ids])
    fb = np.array([entity_score(gt[s], pb.get(s, set()))[0] for s in ids])
    d = fa - fb
    rng = np.random.default_rng(a.seed)
    bs = np.array([d[rng.integers(0, len(d), len(d))].mean() for _ in range(a.n)])
    lo, hi = np.percentile(bs, [2.5, 97.5])
    print(f'{len(ids):,} S1 entities | A {fa.mean():.5f} | B {fb.mean():.5f} | A - B {d.mean():+.5f}')
    print(f'entities that differ: {(d != 0).sum():,} (A better {(d > 0).sum():,}, B better {(d < 0).sum():,})')
    print(f'95% bootstrap interval of A - B: [{lo:+.5f}, {hi:+.5f}] '
          f'({"includes 0: not significant" if lo <= 0 <= hi else "excludes 0"})')


if __name__ == '__main__':
    sys.exit(main())
