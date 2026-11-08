"""
Build train / val / test splits from the labelled TRAINING data only
(the official test set has no ground truth).

Unit of splitting = Source-1 entity.  Every S2/S3 record is linked to at most one S1
(checked on the data: 7,638,365 links, 0 shared), so a linked record simply follows its
S1 into the same split -> no match can cross a split boundary.
Unlinked S2/S3 records (distractors, ~26 % of the pool) are spread over the splits with the
same proportions, by a deterministic hash of their id, so every split keeps the same
distractor density as the original data.

S1 entities are stratified by  country x number-of-true-matches bucket (0 / 1-3 / 4+).

Optional regional subset (--states): keeps whole states, so the confusing near-duplicate
businesses of a region stay together (a random sample would make the task artificially easy).
States are written differently by each source (TX / Texas, MH / Maharashtra); the mapping
raw-address-component -> canonical state is learned from the ground-truth links.
Records whose state cannot be determined are kept with the country's subset rate (hash),
so the share of addresses with a missing state stays representative.

Output (same file format as the competition, lines copied byte for byte):
  <out>/train/train_source{1,2,3}.tsv   <out>/train/train_ground_truth.tsv
  <out>/val/val_source{1,2,3}.tsv       <out>/val/val_ground_truth.tsv
  <out>/test/test_source{1,2,3}.tsv     (no labels here, like the real test set)
  <out>/test_labels/test_ground_truth.tsv   (only for the final evaluation)
  <out>/s1_splits.tsv   <out>/manifest.json

Stdlib only.  Usage:
  python3 src/make_splits.py --data-dir ../dataset/train --out data/full
  python3 src/make_splits.py --data-dir ../dataset/train --list-states
  python3 src/make_splits.py --data-dir ../dataset/train --out data/small \
      --states "US:OR,US:NM,India:Telangana,India:Kerala"
"""
from __future__ import annotations

import argparse
import collections
import json
import random
import re
import sys
import hashlib
import time
from pathlib import Path

SPLITS = ("train", "val", "test")
_T0 = time.time()


def log(msg):
    print(f"[{time.time() - _T0:7.1f}s] {msg}", flush=True)


# ----------------------------------------------------------------------------- parsing
_NON_ALPHA = re.compile(r"[^a-z\u0080-￿ ]+")


def comp_norm(c: str) -> str:
    return " ".join(_NON_ALPHA.sub(" ", c.lower()).split())


def parse(line: str):
    """-> (entity_id, address, country) ; tolerant to malformed rows."""
    f = line.rstrip("\r\n").split("\t")
    if len(f) >= 4:
        return f[0], f[2], f[3]
    return f[0], (f[2] if len(f) > 2 else ""), (f[-1] if len(f) > 1 else "")


def tail_comps(addr: str, n: int = 3):
    """Last n non-empty normalised comma components, right to left."""
    out = []
    for c in reversed(addr.split(",")):
        c = comp_norm(c)
        if c:
            out.append(c)
            if len(out) == n:
                break
    return out


def frac_hash(key: str, seed: int) -> float:
    # BLAKE2b, not CRC32: CRC is linear, so CRC hashes of the same id with different seeds are correlated
    return int.from_bytes(hashlib.blake2b(f"{seed}|{key}".encode(), digest_size=8).digest(), "big") / 2**64


def iter_lines(path: Path):
    with open(path, encoding="utf-8", newline="") as fh:
        header = fh.readline()
        yield header
        for line in fh:
            if line.strip():
                yield line if line.endswith("\n") else line + "\n"


# ----------------------------------------------------------------------------- main steps
def read_gt(path: Path):
    """-> s1 -> raw line, s1 -> n_true, pool_id -> s1"""
    gt_line, n_true, owner = {}, {}, {}
    it = iter_lines(path)
    header = next(it)
    for line in it:
        s1, _, m = line.rstrip("\r\n").partition("\t")
        ids = [x for x in m.split(",") if x] if m else []
        gt_line[s1] = line
        n_true[s1] = len(ids)
        for x in ids:
            if x in owner:
                sys.exit(f"pool record {x} linked to two S1 ({owner[x]}, {s1}): "
                         "split by connected components needed")
            owner[x] = s1
    return header, gt_line, n_true, owner


def learn_states(data: Path, s1_info, owner, min_s1=1000, min_count=30, min_share=0.9):
    """Canonical states = frequent last components of S1 addresses (per country).
    Map raw component -> canonical state, learned from linked pool records."""
    cnt = collections.Counter((c, t[0]) for c, t in ((v[0], v[1]) for v in s1_info.values()) if t)
    canon = {k for k, n in cnt.items() if n >= min_s1}           # (country, name)
    mapping = {k: k[1] for k in canon}

    def s1_state(tails, country):
        for t in tails:
            if (country, t) in canon:
                return t
        return None

    s1_state_of = {s: s1_state(t, c) for s, (c, t) in s1_info.items()}

    votes = collections.defaultdict(collections.Counter)
    for src in ("train_source2.tsv", "train_source3.tsv"):
        it = iter_lines(data / src)
        next(it)
        for line in it:
            eid, addr, country = parse(line)
            o = owner.get(eid)
            if o is None:
                continue
            st = s1_state_of.get(o)
            if st is None:
                continue
            for t in tail_comps(addr):
                if (country, t) not in canon:
                    votes[(country, t)][st] += 1
    for k, c in votes.items():
        st, n = c.most_common(1)[0]
        tot = sum(c.values())
        if tot >= min_count and n / tot >= min_share:
            mapping[k] = st
    return canon, mapping, s1_state_of


def state_of(addr, country, mapping):
    for t in tail_comps(addr):
        st = mapping.get((country, t))
        if st is not None:
            return st
    return None


def stratified_split(s1_ids, s1_country, n_true, fracs, seed):
    rng = random.Random(seed)
    strata = collections.defaultdict(list)
    for s in s1_ids:
        k = n_true.get(s, 0)
        b = 0 if k == 0 else (1 if k <= 3 else 2)
        strata[(s1_country[s], b)].append(s)
    split = {}
    for key in sorted(strata):
        ids = strata[key]
        rng.shuffle(ids)
        n_test = round(len(ids) * fracs["test"])
        n_val = round(len(ids) * fracs["val"])
        for i, s in enumerate(ids):
            split[s] = "test" if i < n_test else ("val" if i < n_test + n_val else "train")
    return split


def pool_split(eid, fracs, seed):
    h = frac_hash(eid, seed + 1)
    if h < fracs["test"]:
        return "test"
    if h < fracs["test"] + fracs["val"]:
        return "val"
    return "train"


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", required=True, help="folder with train_source{1,2,3}.tsv + train_ground_truth.tsv")
    ap.add_argument("--out", help="output folder")
    ap.add_argument("--val-frac", type=float, default=0.10)
    ap.add_argument("--test-frac", type=float, default=0.20)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--states", default="", help='regional subset, e.g. "US:OR,India:Kerala" (canonical names, see --list-states)')
    ap.add_argument("--list-states", action="store_true", help="print canonical states with S1 counts and exit")
    a = ap.parse_args()

    data = Path(a.data_dir)
    fracs = {"val": a.val_frac, "test": a.test_frac, "train": 1 - a.val_frac - a.test_frac}
    assert fracs["train"] > 0

    # 1. S1 records
    s1_info, s1_country, s1_order = {}, {}, []
    it = iter_lines(data / "train_source1.tsv")
    s1_header = next(it)
    for line in it:
        eid, addr, country = parse(line)
        s1_info[eid] = (country, tail_comps(addr))
        s1_country[eid] = country
        s1_order.append(eid)
    log(f"S1: {len(s1_order):,} records")

    # 2. ground truth
    gt_header, gt_line, n_true, owner = read_gt(data / "train_ground_truth.tsv")
    missing_gt = [s for s in s1_order if s not in gt_line]
    log(f"GT: {len(gt_line):,} S1 rows, {len(owner):,} links, S1 without a GT row: {len(missing_gt):,}")

    # 3. optional regional subset
    keep_s1 = None
    subset_info = None
    if a.states or a.list_states:
        canon, mapping, s1_state_of = learn_states(data, s1_info, owner)
        log(f"states: {len(canon)} canonical, {len(mapping):,} raw spellings mapped")
        if a.list_states:
            c = collections.Counter((s1_country[s], st) for s, st in s1_state_of.items() if st)
            unk = collections.Counter(s1_country[s] for s, st in s1_state_of.items() if not st)
            for (country, st), n in sorted(c.items(), key=lambda x: (x[0][0], -x[1])):
                print(f"  {country}:{st}\t{n:,}")
            print(f"  unknown state: {dict(unk)}")
            return
        wanted = set()
        for tok in a.states.split(","):
            country, _, st = tok.strip().partition(":")
            key = (country.strip(), comp_norm(st))
            if key not in canon:
                sys.exit(f"unknown state {tok!r}; run --list-states")
            wanted.add(key)
        known = collections.Counter()
        kept_known = collections.Counter()
        for s, st in s1_state_of.items():
            if st:
                known[s1_country[s]] += 1
                kept_known[s1_country[s]] += (s1_country[s], st) in wanted
        rate = {c: kept_known[c] / known[c] for c in known}          # used for unknown-state records
        keep_s1 = set()
        for s in s1_order:
            st = s1_state_of[s]
            c = s1_country[s]
            if st is not None:
                ok = (c, st) in wanted
            else:
                ok = frac_hash(s, a.seed + 7) < rate.get(c, 0.0)
            if ok:
                keep_s1.add(s)
        subset_info = {"states": sorted(f"{c}:{s}" for c, s in wanted),
                       "unknown_state_keep_rate": rate}
        log(f"subset: {len(keep_s1):,} S1 kept; unknown-state keep rate {rate}")

        def keep_pool(eid, addr, country):
            o = owner.get(eid)
            if o is not None:
                return o in keep_s1
            st = state_of(addr, country, mapping)
            if st is not None:
                return (country, st) in wanted
            return frac_hash(eid, a.seed + 7) < rate.get(country, 0.0)
    else:
        def keep_pool(eid, addr, country):
            return True

    s1_used = [s for s in s1_order if keep_s1 is None or s in keep_s1]

    # 4. split S1
    split = stratified_split(s1_used, s1_country, n_true, fracs, a.seed)
    log("S1 split: " + ", ".join(f"{k} {sum(v == k for v in split.values()):,}" for k in SPLITS))

    out = Path(a.out)
    for d in ("train", "val", "test", "test_labels"):
        (out / d).mkdir(parents=True, exist_ok=True)

    def gt_path(sp):
        return out / ("test_labels" if sp == "test" else sp) / f"{sp}_ground_truth.tsv"

    stats = {sp: collections.Counter() for sp in SPLITS}

    # 5. write S1 + GT
    fh = {sp: open(out / sp / f"{sp}_source1.tsv", "w", encoding="utf-8", newline="") for sp in SPLITS}
    gh = {sp: open(gt_path(sp), "w", encoding="utf-8", newline="") for sp in SPLITS}
    for sp in SPLITS:
        fh[sp].write(s1_header)
        gh[sp].write(gt_header)
    it = iter_lines(data / "train_source1.tsv")
    next(it)
    for line in it:
        eid = line.split("\t", 1)[0]
        sp = split.get(eid)
        if sp is None:
            continue
        fh[sp].write(line)
        gh[sp].write(gt_line.get(eid, f"{eid}\t\n"))
        st = stats[sp]
        k = n_true.get(eid, 0)
        st["s1"] += 1
        st[f"s1_{s1_country[eid]}"] += 1
        st["singletons"] += k == 0
        st["links"] += k
    for f in list(fh.values()) + list(gh.values()):
        f.close()

    # 6. write S2 / S3
    for src in ("source2", "source3"):
        fh = {sp: open(out / sp / f"{sp}_{src}.tsv", "w", encoding="utf-8", newline="") for sp in SPLITS}
        it = iter_lines(data / f"train_{src}.tsv")
        header = next(it)
        for f in fh.values():
            f.write(header)
        for line in it:
            eid, addr, country = parse(line)
            if not keep_pool(eid, addr, country):
                continue
            o = owner.get(eid)
            sp = split[o] if o is not None else pool_split(eid, fracs, a.seed)
            fh[sp].write(line)
            st = stats[sp]
            st[f"{src}"] += 1
            st[f"{src}_unlinked"] += o is None
        for f in fh.values():
            f.close()
        log(f"{src} written")

    # 7. independent check of the written files
    check = verify(out)
    manifest = {
        "source": str(data.resolve()), "seed": a.seed, "fractions": fracs,
        "stratified_by": "country x n_true bucket (0 / 1-3 / 4+)",
        "subset": subset_info,
        "stats": {sp: dict(stats[sp]) for sp in SPLITS},
        "verify": check,
    }
    with open(out / "manifest.json", "w") as f:
        json.dump(manifest, f, indent=1)
    with open(out / "s1_splits.tsv", "w", encoding="utf-8") as f:
        f.write("entity_id\tcountry\tn_true\tsplit\n")
        for s in s1_used:
            f.write(f"{s}\t{s1_country[s]}\t{n_true.get(s, 0)}\t{split[s]}\n")

    print("\n split   S1        singletons  links      S2        S3        unlinked pool")
    for sp in SPLITS:
        st = stats[sp]
        unl = st["source2_unlinked"] + st["source3_unlinked"]
        pool = st["source2"] + st["source3"]
        print(f" {sp:6} {st['s1']:>9,} {st['singletons']:>10,} {st['links']:>10,} "
              f"{st['source2']:>9,} {st['source3']:>9,}  {unl:,} ({unl / max(pool, 1):.1%})")
    print(f"\n verify: {'PASS' if check['ok'] else 'FAIL'}  {check}")
    log(f"done -> {out}")
    if not check["ok"]:
        sys.exit(1)


def verify(out: Path):
    """Re-read the written files: no id in two splits, every GT match present in its split's pool."""
    res = {"ok": True}
    seen_s1, seen_pool_all = set(), set()
    for sp in SPLITS:
        pool = set()
        for src in ("source2", "source3"):
            it = iter_lines(out / sp / f"{sp}_{src}.tsv")
            next(it)
            for line in it:
                pool.add(line.split("\t", 1)[0])
        s1 = set()
        it = iter_lines(out / sp / f"{sp}_source1.tsv")
        next(it)
        for line in it:
            s1.add(line.split("\t", 1)[0])
        p = out / ("test_labels" if sp == "test" else sp) / f"{sp}_ground_truth.tsv"
        gt_s1, missing = set(), 0
        it = iter_lines(p)
        next(it)
        for line in it:
            s, _, m = line.rstrip("\r\n").partition("\t")
            gt_s1.add(s)
            missing += sum(1 for x in m.split(",") if x and x not in pool)
        r = {"s1_overlap_prev": len(s1 & seen_s1), "pool_overlap_prev": len(pool & seen_pool_all),
             "gt_rows_eq_s1": gt_s1 == s1, "gt_ids_missing_from_pool": missing}
        res[sp] = r
        res["ok"] &= (r["s1_overlap_prev"] == 0 and r["pool_overlap_prev"] == 0
                      and r["gt_rows_eq_s1"] and missing == 0)
        seen_s1 |= s1
        seen_pool_all |= pool
        del pool
    return res


if __name__ == "__main__":
    main()
