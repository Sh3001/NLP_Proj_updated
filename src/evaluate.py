"""
Score predictions on a held-out split with the competition metric.

  macro F0.5 over Source-1 entities, singletons included:
    true = {}       -> 1.0 if predicted {} else 0.0
    true != {}      -> F0.5(P, R) ; 0.0 when nothing is predicted
  plus macro/micro precision & recall, singleton accuracy, per-country breakdown, and
  (with --candidates) blocking recall, candidates per S1 and the blocking ceiling
  (F0.5 of an oracle that keeps exactly the true ids among the candidates).

A Source-1 entity missing from the prediction file counts as an empty prediction (warned).

Usage:
  python3 src/evaluate.py --gt data/full/test_labels/test_ground_truth.tsv \
      --pred out/matching_results.tsv [--candidates out/candidate_pairs.tsv] \
      [--s1 data/full/test/test_source1.tsv] [--json metrics.json]
"""
from __future__ import annotations

import argparse
import collections
import json


def read_lists(path):
    d = {}
    with open(path, encoding="utf-8") as f:
        next(f)
        for line in f:
            s, _, m = line.rstrip("\r\n").partition("\t")
            if s:
                d[s] = {x.strip() for x in m.split(",") if x.strip()}
    return d


def f05(p, r):
    return 0.0 if p + r == 0 else 1.25 * p * r / (0.25 * p + r)


def entity_score(true, pred):
    if not true:
        return (1.0 if not pred else 0.0), None, None
    if not pred:
        return 0.0, None, 0.0
    tp = len(true & pred)
    p, r = tp / len(pred), tp / len(true)
    return f05(p, r), p, r


def score(gt, pred, country=None):
    acc = collections.defaultdict(lambda: collections.defaultdict(float))
    for s, true in gt.items():
        pr = pred.get(s, set())
        f, p, r = entity_score(true, pr)
        for g in ("ALL", country.get(s, "?") if country else None):
            if g is None:
                continue
            a = acc[g]
            a["n"] += 1
            a["f05"] += f
            a["tp"] += len(true & pr)
            a["fp"] += len(pr - true)
            a["fn"] += len(true - pr)
            a["exact"] += true == pr
            if not true:
                a["singletons"] += 1
                a["singleton_ok"] += not pr
            # macro P / R as in the existing pipelines: P over S1 with a prediction, R over S1 with true links
            if pr:
                a["p_sum"] += len(true & pr) / len(pr)
                a["p_n"] += 1
            if true:
                a["r_sum"] += len(true & pr) / len(true)
                a["r_n"] += 1
    out = {}
    for g, a in acc.items():
        mp = a["tp"] / max(a["tp"] + a["fp"], 1)
        mr = a["tp"] / max(a["tp"] + a["fn"], 1)
        out[g] = {
            "n_s1": int(a["n"]), "macro_f05": a["f05"] / a["n"],
            "macro_precision": a["p_sum"] / max(a["p_n"], 1), "macro_recall": a["r_sum"] / max(a["r_n"], 1),
            "micro_precision": mp, "micro_recall": mr, "micro_f05": f05(mp, mr),
            "exact_set_acc": a["exact"] / a["n"],
            "singletons": int(a["singletons"]),
            "singleton_acc": a["singleton_ok"] / max(a["singletons"], 1),
            "TP": int(a["tp"]), "FP": int(a["fp"]), "FN": int(a["fn"]),
        }
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--gt", required=True)
    ap.add_argument("--pred", required=True)
    ap.add_argument("--candidates")
    ap.add_argument("--s1", help="split's source1 tsv, for a per-country breakdown")
    ap.add_argument("--json", help="write metrics here")
    a = ap.parse_args()

    gt = read_lists(a.gt)
    pred = read_lists(a.pred)
    missing = len(set(gt) - set(pred))
    extra = len(set(pred) - set(gt))
    if missing:
        print(f"WARNING: {missing:,} S1 entities have no prediction row (scored as empty)")
    if extra:
        print(f"WARNING: {extra:,} prediction rows are not in the ground truth (ignored)")

    country = None
    if a.s1:
        country = {}
        with open(a.s1, encoding="utf-8") as f:
            next(f)
            for line in f:
                p = line.rstrip("\r\n").split("\t")
                country[p[0]] = p[-1]

    res = {"prediction": score(gt, pred, country)}

    if a.candidates:
        cand = read_lists(a.candidates)
        oracle = {s: gt[s] & cand.get(s, set()) for s in gt}
        n_true = sum(len(v) for v in gt.values())
        found = sum(len(v) for v in oracle.values())
        not_in_cand = sum(len(pred.get(s, set()) - cand.get(s, set())) for s in gt)
        res["blocking"] = {
            "pair_recall": found / max(n_true, 1),
            "cands_per_s1": sum(len(cand.get(s, ())) for s in gt) / len(gt),
            "ceiling_macro_f05": score(gt, oracle)["ALL"]["macro_f05"],
            "matches_not_in_candidates": not_in_cand,
        }

    for g, m in sorted(res["prediction"].items(), key=lambda x: x[0] != "ALL"):
        print(f"[{g}] macro F0.5 {m['macro_f05']:.5f} | P {m['macro_precision']:.4f} R {m['macro_recall']:.4f} "
              f"| micro P {m['micro_precision']:.4f} R {m['micro_recall']:.4f} | exact {m['exact_set_acc']:.4f} "
              f"| singleton acc {m['singleton_acc']:.4f} | n {m['n_s1']:,}")
    if "blocking" in res:
        b = res["blocking"]
        print(f"[blocking] pair recall {b['pair_recall']:.4f} | {b['cands_per_s1']:.2f} cands/S1 "
              f"| ceiling F0.5 {b['ceiling_macro_f05']:.5f} | matches outside candidates {b['matches_not_in_candidates']:,}")
    if a.json:
        with open(a.json, "w") as f:
            json.dump(res, f, indent=1)


if __name__ == "__main__":
    main()
