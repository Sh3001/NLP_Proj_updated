"""Issue #8: per-script error analysis of every matcher + IndicBERTv2 vs bge-reranker-base.

Inputs (unzipped anywhere under ROOT, any folder layout):
  * er-small         -> test_labels/test_ground_truth.tsv
  * er-03-output     -> prep/rec.parquet, prep/pairs_test.parquet
  * every model zip  -> results/<method>.json, preds/<method>/matching_results.tsv, scores/<method>_test.npy

    python3 analysis/per_script.py ROOT [--out analysis/out_script] [--errors-model "<method>"]

Groups: the script of the S2/S3 record (Latin / Devanagari / Malayalam) x country. S1 records are the
reference side; a link "belongs" to the script of its S2/S3 record.

Per-script F0.5 = the competition metric (macro F0.5 per S1 entity) restricted to the links of one group:
for every S1 that has >= 1 true or predicted link from a record of that group, F0.5 over those links only.
(Singletons have no script; they enter only through false matches. The overall F0.5 is recomputed with
singletons and must equal the notebook's results/*.json: that is the check that this script is right.)

Writes to --out: per_script_f05.{md,csv,png}, per_script_errors.md, ranking_per_script.md,
indic_vs_bge.md, error_examples.csv, summary.json. Needs numpy, pandas, pyarrow, matplotlib.
"""
import argparse, csv, glob, json, re, zlib
from pathlib import Path

import numpy as np
import pandas as pd

INDIC, BGE = 'CE IndicBERTv2 full fine-tune', 'CE bge-reranker-base full fine-tune'
GROUPS = [('US', 'latin'), ('India', 'latin'), ('India', 'devanagari'), ('India', 'malayalam'), ('India', 'other_indic')]
GLABEL = {g: f"{g[0]} {g[1].replace('_', ' ').capitalize()}" for g in GROUPS}
SHORT = [('CE bge-reranker-base', 'bge'), ('Siamese char-', ''), (' full fine-tune', ' full FT'),
         ('GBM (string + parsed-address features)', 'GBM baseline'), (' (vanilla)', ''), ('CE ', '')]
SECTION_ORDER = ['5 Classical baseline', '6 RNN / LSTM', '7 Transformers', '7b Ditto ablation', '8 Fusion']


def short(m):
    for a, b in SHORT: m = m.replace(a, b)
    return m


def safe_name(s): return re.sub(r'[^\w.-]+', '_', s)


def find(root, pattern): return sorted(Path(p) for p in glob.glob(str(Path(root) / '**' / pattern), recursive=True))


def find_one(root, pattern, what):
    h = find(root, pattern)
    if not h: raise SystemExit(f'not found under {root}: {pattern}  ({what})')
    return h[0]


def read_tsv(p): return pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, encoding='utf-8')


def read_links(p):
    d = read_tsv(p); col = d.columns[1]
    return {s: [x for x in m.split(',') if x] for s, m in zip(d.iloc[:, 0], d[col])}


def f05(tp, npred, ntrue):                       # per-S1 F0.5, singletons (ntrue = 0) right iff nothing predicted
    tp, npred, ntrue = (np.asarray(a, float) for a in (tp, npred, ntrue))
    P = np.divide(tp, npred, out=np.zeros_like(tp), where=npred > 0)
    R = np.divide(tp, ntrue, out=np.zeros_like(tp), where=ntrue > 0)
    d = 0.25 * P + R
    f = np.divide(1.25 * P * R, d, out=np.zeros_like(tp), where=d > 0)
    return np.where(ntrue == 0, (npred == 0).astype(float), f)


def auc(y, s):                                   # ROC AUC from ranks (Mann-Whitney U); ties get the mean rank
    y = np.asarray(y, bool); n1, n0 = y.sum(), (~y).sum()
    if n1 == 0 or n0 == 0: return np.nan
    r = pd.Series(s).rank().values
    return (r[y].sum() - n1 * (n1 + 1) / 2) / (n1 * n0)


def md_table(df, floatfmt=None):
    cols = list(df.columns); out = ['| ' + ' | '.join(map(str, cols)) + ' |', '|' + '---|' * len(cols)]
    for r in df.itertuples(index=False):
        out.append('| ' + ' | '.join('' if (isinstance(v, float) and np.isnan(v)) else
                                     (f'{v:{floatfmt}}' if floatfmt and isinstance(v, float) else str(v)) for v in r) + ' |')
    return '\n'.join(out) + '\n'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root'); ap.add_argument('--out', default='analysis/out_script')
    ap.add_argument('--errors-model', default=None, help='model whose errors are sampled (default: best test F0.5)')
    ap.add_argument('--n-examples', type=int, default=8, help='examples per script x error type')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True); root = a.root

    # ---- data ------------------------------------------------------------------------------------
    rec_p = find_one(root, 'prep/rec.parquet', 'output of 03_candidates (er-03-output)')
    REC = pd.read_parquet(rec_p, columns=['id', 'name', 'addr', 'split', 'src', 'country', 'script'])
    REC = REC[REC.split == 'test'].set_index('id')
    PT = pd.read_parquet(find_one(root, 'prep/pairs_test.parquet', 'er-03-output'), columns=['q_id', 's_id'])
    GT = read_links(find_one(root, 'test_labels/test_ground_truth.tsv', 'the er-small dataset'))
    S1 = REC.index[REC.src == 'S1']; GT = {s: GT.get(s, []) for s in S1}
    OWNER = {x: s for s, l in GT.items() for x in l}
    Q = REC[REC.src != 'S1'].copy()
    Q['owner'] = Q.index.map(OWNER)                                        # NaN = unlinked distractor
    Q['grp'] = list(zip(Q.country, Q.script))
    cand = set(zip(PT.q_id, PT.s_id))
    Q['true_in_cand'] = [isinstance(o, str) and (q, o) in cand for q, o in zip(Q.index, Q.owner)]
    PT['label'] = (PT.q_id.map(OWNER) == PT.s_id).values
    PT['grp'] = list(zip(Q.country.reindex(PT.q_id).values, Q.script.reindex(PT.q_id).values))
    print(f'test split: {len(S1):,} S1, {len(Q):,} S2/S3 records ({Q.owner.notna().sum():,} linked), {len(PT):,} candidate pairs')
    print('S2/S3 records by country x script:\n' + Q.groupby(['country', 'script']).size().to_string())
    print('S1 records by script:', REC[REC.src == 'S1'].script.value_counts().to_dict())

    # ---- models ----------------------------------------------------------------------------------
    RES = {}
    for p in find(root, 'results/*.json'):
        r = json.load(open(p)); m = r.get('method')
        if not m: continue
        r['_root'] = p.parent.parent
        if m not in RES or (str(RES[m].get('quick')).lower() == 'true' and str(r.get('quick')).lower() != 'true'): RES[m] = r
    PRED, SCORE = {}, {}
    for m, r in RES.items():
        pp = r['_root'] / 'preds' / safe_name(m) / 'matching_results.tsv'
        pp = pp if pp.exists() else next(iter(find(root, f'preds/{safe_name(m)}/matching_results.tsv')), None)
        if pp is None: print('no preds for', m); continue
        pr = read_links(pp)
        if set(pr) != set(S1): print(f'skip {m}: its predictions cover another S1 set (QUICK_RUN = {r.get("quick")})'); continue
        PRED[m] = {x: s for s, l in pr.items() for x in l}
        sp = r['_root'] / 'scores' / f'{safe_name(m)}_test.npy'
        sp = sp if sp.exists() else next(iter(find(root, f'scores/{safe_name(m)}_test.npy')), None)
        if sp is not None:
            v = np.load(sp)
            if len(v) == len(PT): SCORE[m] = v
    if not PRED: raise SystemExit('no model outputs with predictions found')
    models = sorted(PRED, key=lambda m: (SECTION_ORDER.index(RES[m].get('section')) if RES[m].get('section') in SECTION_ORDER else 9,
                                         -RES[m]['test_f05']))
    print('models:', [short(m) for m in models])

    # ---- per-model, per-group metrics ------------------------------------------------------------
    s1_pos = pd.Series(np.arange(len(S1)), index=S1); n = len(S1)
    ntrue_all = np.array([len(GT[s]) for s in S1])
    rows, err_rows, check = [], [], []
    Q['_oi'] = s1_pos.reindex(Q.owner.fillna('')).values                     # owner position (NaN if unlinked)
    for m in models:
        pred = Q.index.map(PRED[m]); Q['_p'] = pred
        Q['_pi'] = s1_pos.reindex(pd.Series(pred).fillna('')).values
        ok = Q._p.notna() & (Q._p == Q.owner)
        # overall F0.5, singletons included: must equal the notebook's own number
        has = Q._pi.notna()
        npred = np.bincount(Q._pi[has].astype(int), minlength=n); tp = np.bincount(Q._pi[ok].astype(int), minlength=n)
        f_all = f05(tp, npred, ntrue_all).mean(); check.append((m, f_all, RES[m]['test_f05']))
        row = {'model': short(m), 'section': RES[m].get('section'), 'test F0.5': round(f_all, 5)}
        for g in GROUPS:
            sel = (Q.grp == g).values
            hp, ho = sel & has.values, sel & Q.owner.notna().values
            npg = np.bincount(Q._pi[hp].astype(int), minlength=n)
            ntg = np.bincount(Q._oi[ho].astype(int), minlength=n)
            tpg = np.bincount(Q._pi[sel & ok.values].astype(int), minlength=n)
            inc = (npg > 0) | (ntg > 0)
            row[GLABEL[g]] = round(f05(tpg[inc], npg[inc], ntg[inc]).mean(), 5) if inc.any() else np.nan
            linked = ho.sum(); unl = (sel & Q.owner.isna().values).sum()
            e = {'model': short(m), 'group': GLABEL[g], 'records': int(sel.sum()),
                 'wrong match': int((sel & has.values & Q.owner.notna().values & ~ok.values).sum()),
                 'false match': int((sel & has.values & Q.owner.isna().values).sum()),
                 'missed': int((sel & ~has.values & Q.owner.notna().values).sum()),
                 'of which blocking miss': int((sel & Q.owner.notna().values & ~Q.true_in_cand.values & ~ok.values).sum())}
            e['error rate %'] = round(100 * (e['wrong match'] + e['false match'] + e['missed']) / max(sel.sum(), 1), 2)
            e['linked'], e['unlinked'] = int(linked), int(unl)
            err_rows.append(e)
        rows.append(row)
        # record-level correctness: right S1 picked, or correctly left unmatched (unlinked record, no prediction)
        Q[f'ok::{m}'] = ok.values | (~has.values & Q.owner.isna().values); Q[f'pred::{m}'] = Q._p.values

    bad = [(short(m), x, y) for m, x, y in check if abs(x - y) > 2e-5]
    print('\nconsistency check (recomputed overall test F0.5 vs results/*.json):', 'OK' if not bad else f'MISMATCH {bad}')
    T = pd.DataFrame(rows)
    T.to_csv(out / 'per_script_f05.csv', index=False)
    (out / 'per_script_f05.md').write_text(
        md_table(T) + '\nRecords per group (test S2/S3): ' +
        ', '.join(f"{GLABEL[g]} {int((Q.grp == g).sum()):,}" for g in GROUPS) + '\n')
    print('\nper-script F0.5 (links of S2/S3 records of that script):'); print(T.to_string(index=False))
    E = pd.DataFrame(err_rows); (out / 'per_script_errors.md').write_text(md_table(E))
    print('\nerrors per group:'); print(E.to_string(index=False))

    # ---- threshold-free ranking quality per script (all models with test scores) -----------------
    rk = []
    has_true = PT.groupby('q_id').label.transform('any').values
    for m in [m for m in models if m in SCORE]:
        PT['_s'] = SCORE[m]
        top = PT.loc[PT.groupby('q_id')._s.idxmax()]
        r = {'model': short(m)}
        for g in GROUPS:
            sel = (PT.grp == g).values
            r[f'AUC {GLABEL[g]}'] = round(auc(PT.label.values[sel], SCORE[m][sel]), 5)
            tg = top[(top.grp == g) & top.q_id.isin(PT.q_id[sel & has_true])]
            r[f'top-1 {GLABEL[g]}'] = round(tg.label.mean(), 4) if len(tg) else np.nan
        rk.append(r)
    RK = pd.DataFrame(rk)
    if len(RK):
        (out / 'ranking_per_script.md').write_text(
            'AUC: pair-level ROC AUC of the test logits. top-1: share of linked records whose true S1 is among the candidates '
            'and gets the highest score (no threshold).\n\n' + md_table(RK))
        print('\nranking quality per script (no threshold):'); print(RK.to_string(index=False))

    # ---- IndicBERTv2 vs bge (06b) ------------------------------------------------------------------
    summary = {'models': models, 'check': check}
    if INDIC in PRED and BGE in PRED:
        lines = ['## IndicBERTv2 (06e) vs bge-reranker-base full fine-tune (06b), India records\n']
        rows = []
        for g in [g for g in GROUPS if g[0] == 'India']:
            sel = Q.grp == g; i_ok, b_ok = Q.loc[sel, f'ok::{INDIC}'], Q.loc[sel, f'ok::{BGE}']
            r = {'script': g[1], 'records': int(sel.sum()),
                 'F0.5 Indic': T.loc[T.model == short(INDIC), GLABEL[g]].item(), 'F0.5 bge': T.loc[T.model == short(BGE), GLABEL[g]].item(),
                 'both right': int((i_ok & b_ok).sum()), 'only Indic right': int((i_ok & ~b_ok).sum()),
                 'only bge right': int((~i_ok & b_ok).sum()), 'both wrong': int((~i_ok & ~b_ok).sum())}
            if INDIC in SCORE and BGE in SCORE:
                ps = (PT.grp == g).values
                r['AUC Indic'] = round(auc(PT.label.values[ps], SCORE[INDIC][ps]), 5)
                r['AUC bge'] = round(auc(PT.label.values[ps], SCORE[BGE][ps]), 5)
            # McNemar (exact two-sided binomial on the discordant records)
            b, c = r['only Indic right'], r['only bge right']
            from math import comb
            k, nn = min(b, c), b + c
            r['McNemar p'] = float(f"{min(1.0, 2 * sum(comb(nn, i) for i in range(k + 1)) / 2 ** nn):.3g}") if nn else 1.0
            rows.append(r)
        IV = pd.DataFrame(rows); lines.append(md_table(IV))
        (out / 'indic_vs_bge.md').write_text('\n'.join(lines))
        print('\nIndicBERTv2 vs bge (06b):'); print(IV.to_string(index=False))
        summary['indic_vs_bge'] = rows
    else:
        print(f'\n(IndicBERTv2 vs bge skipped: need both "{INDIC}" and "{BGE}" outputs)')

    # ---- error examples --------------------------------------------------------------------------
    em = a.errors_model or max(models, key=lambda m: RES[m]['test_f05'])
    if em not in PRED: raise SystemExit(f'--errors-model {em!r} not found; have {models}')
    Q['n_models_wrong'] = (~Q[[f'ok::{m}' for m in models]]).sum(axis=1)
    p, o = Q[f'pred::{em}'], Q.owner
    kind = np.select([p.notna() & o.notna() & (p != o), p.notna() & o.isna(), p.isna() & o.notna()],
                     ['wrong match', 'false match', 'missed'], 'correct')
    Q['error'] = kind
    nm = REC.name.to_dict(); ad = REC.addr.to_dict()
    X = Q[Q.error != 'correct'].copy()
    X['rnd'] = [zlib.crc32(i.encode()) for i in X.index]                     # reproducible shuffle within ties
    X = (X.sort_values(['n_models_wrong', 'rnd'], ascending=[False, True])
           .groupby(['country', 'script', 'error'], group_keys=False).head(a.n_examples))
    EX = pd.DataFrame({'country': X.country, 'script': X.script, 'error': X.error, 'n_models_wrong': X.n_models_wrong,
                       'of_models': len(models), 'true_in_candidates': X.true_in_cand,
                       'record_id': X.index, 'src': X.src, 'record_name': X['name'], 'record_addr': X['addr'],
                       'predicted_s1': X[f'pred::{em}'].fillna(''), 'predicted_name': X[f'pred::{em}'].map(nm).fillna(''),
                       'predicted_addr': X[f'pred::{em}'].map(ad).fillna(''),
                       'true_s1': X.owner.fillna(''), 'true_name': X.owner.map(nm).fillna(''), 'true_addr': X.owner.map(ad).fillna('')})
    EX.to_csv(out / 'error_examples.csv', index=False)
    print(f'\nerror examples of "{short(em)}": {len(EX)} rows -> error_examples.csv '
          f'(sorted by how many of the {len(models)} models also get the record wrong)')
    summary['errors_model'] = em
    json.dump(summary, open(out / 'summary.json', 'w'), indent=1, default=str)

    # ---- chart: per-script F0.5 heatmap ------------------------------------------------------------
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    from matplotlib.colors import LinearSegmentedColormap
    cols = [GLABEL[g] for g in GROUPS]; M = T[cols].values.astype(float)
    cmap = LinearSegmentedColormap.from_list('seq', ['#cde2fb', '#86b6ef', '#3987e5', '#256abf', '#104281'])
    fig, ax = plt.subplots(figsize=(8, 0.42 * len(T) + 1.6))
    main = [i for i, g in enumerate(GROUPS) if g[1] != 'other_indic']             # tiny group: do not let it set the scale
    lo, hi = np.nanmin(M[:, main]) - 0.002, np.nanmax(M[:, main]); im = ax.imshow(np.clip(M, lo, hi), cmap=cmap, vmin=lo, vmax=hi, aspect='auto')
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f'{M[i, j]:.4f}', ha='center', va='center', fontsize=8,
                        color='#ffffff' if (M[i, j] - lo) / max(hi - lo, 1e-9) > 0.6 else '#0b0b0b')
    ax.set_xticks(range(len(cols))); ax.set_xticklabels([c.replace(' ', '\n', 1) for c in cols], fontsize=9)
    ax.set_yticks(range(len(T))); ax.set_yticklabels(T.model, fontsize=9)
    ax.tick_params(length=0); [s.set_visible(False) for s in ax.spines.values()]
    ax.set_title('Test F0.5 per script of the S2/S3 record\n(colour scale clipped; Other Indic = 283 records)', loc='left', fontsize=10)
    plt.colorbar(im, ax=ax, fraction=0.03, pad=0.02).outline.set_visible(False)
    plt.tight_layout(); plt.savefig(out / 'per_script_f05.png', dpi=160); plt.close()
    print('wrote', ', '.join(sorted(p.name for p in out.iterdir())))


if __name__ == '__main__':
    main()
