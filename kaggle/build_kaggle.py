"""Builds the Kaggle notebooks of the NLP project (one per model) + a .py copy of each for local smoke tests.

    python3 kaggle/build_kaggle.py          -> kaggle/notebooks/*.ipynb  and  kaggle/scripts/*.py
"""
import json
from pathlib import Path

HERE = Path(__file__).parent
(HERE / "notebooks").mkdir(exist_ok=True)
(HERE / "scripts").mkdir(exist_ok=True)


def md(s):
    return {"cell_type": "markdown", "metadata": {}, "source": s.strip("\n")}


def code(s):
    return {"cell_type": "code", "metadata": {}, "execution_count": None, "outputs": [], "source": s.strip("\n")}


# =============================================================================== shared cells
def setup_cell(pkgs, torch=True):
    return code(r'''
# ==== Setup (Kaggle: Settings -> Internet ON; GPU T4 where the notebook needs one) ====
import os, sys, subprocess, importlib, time, json, math, re, random, gc, unicodedata, collections, csv, hashlib, glob, zlib
from pathlib import Path
for mod, pip in %s:
    try: importlib.import_module(mod)
    except ImportError:
        if subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pip]).returncode != 0: print('WARNING: could not install', pip)
import numpy as np, pandas as pd
from anyascii import anyascii
import warnings; warnings.filterwarnings('ignore', category=FutureWarning); warnings.filterwarnings('ignore', category=UserWarning)
%s
_T0 = time.time()
def log(*a): print(f'[{(time.time() - _T0) / 60:6.1f} min]', *a, flush=True)
''' % (repr(pkgs), r'''import torch, torch.nn as nn, torch.nn.functional as F
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP = DEVICE == 'cuda'
print('device:', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '(CPU)')''' if torch else "DEVICE = 'cpu'"))


def config_cell(prep_mode, extra, quick_extra):
    return code(r'''
# ==== Configuration ====
QUICK_RUN = False      # True = tiny sample (3,000 / 600 / 600 S1) to check the notebook runs; False = real run
PREP_MODE = %r         # 'build' = make records from the raw split, 'build_save' = also save them, 'load' = take them from 03_candidates

CFG = dict(SEED=42, B_FRAC=0.15, K_TFIDF=10, K_DENSE=10, K_FINAL=5, MAX_LEN=96, SUBSAMPLE_S1=None, %s)
if QUICK_RUN:
    CFG.update(SUBSAMPLE_S1={'train': 3000, 'val': 600, 'test': 600}, %s)
if os.environ.get('NLP_ER_SMOKE') == '1':                         # local smoke test only
    CFG.update(json.loads(os.environ.get('NLP_ER_CFG', '{}')))
SEED = CFG['SEED']
random.seed(SEED); np.random.seed(SEED)
try: torch.manual_seed(SEED)
except NameError: pass

INPUT_ROOT = Path(os.environ.get('NLP_ER_INPUT', '/kaggle/input'))
OUT = Path(os.environ.get('NLP_ER_OUT', '/kaggle/working'))
for d in ('prep', 'scores', 'results', 'preds'): (OUT / d).mkdir(parents=True, exist_ok=True)

def find_all(pattern):
    return sorted(Path(p) for p in glob.glob(str(INPUT_ROOT / '**' / pattern), recursive=True))
def find_one(pattern):
    h = find_all(pattern); return h[0] if h else None

_d = find_one('train/train_source1.tsv')
assert _d is not None, 'Add the dataset (folder with train/ val/ test/ test_labels/) as an input of this notebook'
DATA = _d.parent.parent
print('data:', DATA, '| output:', OUT)
''' % (prep_mode, extra, quick_extra))


DATA_CELL = code(r'''
# ==== Records: load the split, normalise, learn the transliteration lexicon ====
def h01(key):          # uniform [0,1) hash (BLAKE2b; CRC32 hashes of the same id are correlated)
    return int.from_bytes(hashlib.blake2b(key.encode(), digest_size=8).digest(), 'big') / 2**64

def read_tsv(p):
    return pd.read_csv(p, sep='\t', dtype=str, keep_default_na=False, quoting=csv.QUOTE_NONE, encoding='utf-8')

def read_gt(p):
    g = read_tsv(p)
    return {s: [x for x in m.split(',') if x] for s, m in zip(g.source1_entity_id, g.matched_entity_ids)}

SPLITS = ['train', 'val', 'test']
ABBR = {'pvt': 'private', 'ltd': 'limited', 'co': 'company', 'corp': 'corporation', 'inc': 'incorporated',
        'intl': 'international', 'mfg': 'manufacturing', 'svcs': 'services', 'svc': 'service', 'bros': 'brothers',
        'assoc': 'associates', 'mgmt': 'management', 'tech': 'technologies', 'rd': 'road', 'st': 'street',
        'str': 'street', 'ave': 'avenue', 'av': 'avenue', 'blvd': 'boulevard', 'dr': 'drive', 'ln': 'lane',
        'ct': 'court', 'pl': 'place', 'hwy': 'highway', 'pkwy': 'parkway', 'sq': 'square', 'ste': 'suite',
        'apt': 'apartment', 'fl': 'floor', 'bldg': 'building', 'nr': 'near', 'opp': 'opposite', 'cir': 'circle',
        'ter': 'terrace', 'mkt': 'market', 'ngr': 'nagar', 'clny': 'colony'}
LEGAL = {'private', 'limited', 'llp', 'llc', 'incorporated', 'corporation', 'company', 'the', 'plc', 'opc', 'and'}
SCRIPTS = ['latin', 'devanagari', 'malayalam', 'other_indic']
_PUNCT = re.compile(r"[^\w\s/-]+")
INDIC_FIX = [('റ്റ', 'ട്ട')]        # Malayalam ṟṟa: anyascii gives 'rr' where English has 't' (limirrd -> limittd)
_MG = re.compile(r'mg\b')            # anusvara before g: 'marketimg' -> 'marketing'

def romanise(s):
    if s.isascii(): return s
    for a, b in INDIC_FIX: s = s.replace(a, b)
    return _MG.sub('ng', anyascii(s))

def norm(s, expand=True, roman=True, lex=None):
    s = unicodedata.normalize('NFKC', s)
    if roman and not s.isascii(): s = romanise(s)
    toks = _PUNCT.sub(' ', s.lower().replace('&', ' and ')).split()
    if lex: toks = [lex.get(t, t) for t in toks]
    if expand: toks = [ABBR.get(t, t) for t in toks]
    return ' '.join(toks)

def script_of(s):
    if s.isascii(): return 'latin'
    kinds = set()
    for ch in s:
        o = ord(ch)
        if 0x0900 <= o <= 0x097F: kinds.add('devanagari')
        elif 0x0D00 <= o <= 0x0D7F: kinds.add('malayalam')
        elif 0x0980 <= o <= 0x0DFF: kinds.add('other_indic')
    for k in ('devanagari', 'malayalam', 'other_indic'):
        if k in kinds: return k
    return 'latin'                               # accented Latin (e.g. French) counts as Latin

def build_rec():
    parts = []
    for sp in SPLITS:
        for src in ('source1', 'source2', 'source3'):
            d = read_tsv(DATA / sp / f'{sp}_{src}.tsv'); d['split'] = sp; d['src'] = 'S' + src[-1]; parts.append(d)
    R = pd.concat(parts, ignore_index=True).rename(columns={'entity_id': 'id', 'business_name': 'name', 'business_address': 'addr'})
    if CFG['SUBSAMPLE_S1']:          # keep n S1 per split + their records + the same share of unlinked records
        keep = set()
        for sp, n in CFG['SUBSAMPLE_S1'].items():
            g = read_gt(DATA / ('test_labels' if sp == 'test' else sp) / f'{sp}_ground_truth.tsv')   # row selection only
            s1 = R.id[(R.split == sp) & (R.src == 'S1')]
            ks = set(s1.sample(min(n, len(s1)), random_state=SEED)); keep |= ks
            owner = {x: s for s, l in g.items() for x in l}; frac = len(ks) / len(s1)
            for x in R.id[(R.split == sp) & (R.src != 'S1')]:
                o = owner.get(x)
                if (o in ks) if o is not None else (h01(f'sub|{x}') < frac): keep.add(x)
        R = R[R.id.isin(keep)].reset_index(drop=True)
    R['script'] = [script_of(a + ' ' + b) for a, b in zip(R.name, R.addr)]
    return R

def learn_lexicon(R, owner, min_n=3, min_share=0.6):
    # romanised non-Latin word -> English word, aligned by position in linked name pairs of train part A
    ix = pd.Series(np.arange(len(R)), index=R.id.values)
    cnt = collections.defaultdict(collections.Counter)
    for i in np.where((R.split.values == 'train') & (R.src.values != 'S1') & (R.script.values != 'latin'))[0]:
        o = owner.get(R.id.values[i])
        if o is None or h01(f'AB{SEED}|{o}') < CFG['B_FRAC']: continue              # part B excluded
        a = norm(R.name.values[i], expand=False).split(); b = norm(R.name.values[ix[o]], expand=False).split()
        if len(a) == len(b):
            for x, y in zip(a, b):
                if x != y and y.isascii(): cnt[x][y] += 1
    lex = {}
    for x, c in cnt.items():
        y, n = c.most_common(1)[0]
        if n >= min_n and n / sum(c.values()) >= min_share: lex[x] = y
    return lex

def norm_rec(s, expand=True): return norm(s, expand=expand, lex=None if s.isascii() else LEX)
def ce_text(name, addr):
    n = name if name.isascii() else f'{name} / {norm_rec(name, expand=False)}'
    a = addr if addr.isascii() else romanise(addr)
    return f'{n} | {a if a.strip() else "-"}'
def add_text_cols(R):          # every derived text column the matchers read
    R['nn'] = R.name.map(norm_rec); R['na'] = R.addr.map(norm_rec)
    R['ncore'] = [' '.join(t for t in x.split() if t not in LEGAL) or x for x in R.nn]
    R['ce_text'] = [ce_text(a, b) for a, b in zip(R.name, R.addr)]
    R['ltext'] = (R.nn + ' | ' + R.na).str.slice(0, 100)
    R['bl_text'] = R.nn + ' ' + R.na
    return R

if PREP_MODE == 'load':
    _p = find_one('prep/rec.parquet')
    assert _p is not None, 'Add the output of notebook 03_candidates as an input of this notebook'
    REC = pd.read_parquet(_p); PREP_DIR = _p.parent
    LEX = json.load(open(PREP_DIR / 'translit_lexicon.json'))
    _meta = json.load(open(PREP_DIR / 'prep_meta.json'))
    if _meta.get('quick') != QUICK_RUN: print(f"NOTE: 03_candidates ran with QUICK_RUN={_meta.get('quick')}; its sample is used")
    CFG.update({k: _meta['cfg'][k] for k in ('K_TFIDF', 'K_DENSE', 'K_FINAL', 'B_FRAC', 'SEED')})
else:
    REC = build_rec()
    _gt = read_gt(DATA / 'train' / 'train_ground_truth.tsv')
    LEX = learn_lexicon(REC, {x: s for s, l in _gt.items() for x in l})
    print(f'transliteration lexicon: {len(LEX):,} words, e.g.', dict(list(LEX.items())[:12]))
    REC = add_text_cols(REC)
    if PREP_MODE == 'build_save':
        REC.to_parquet(OUT / 'prep' / 'rec.parquet')
        json.dump(LEX, open(OUT / 'prep' / 'translit_lexicon.json', 'w'), ensure_ascii=False)
        json.dump({'quick': QUICK_RUN, 'cfg': CFG, 'n': len(REC)}, open(OUT / 'prep' / 'prep_meta.json', 'w'), indent=1)

ID2IX = pd.Series(np.arange(len(REC)), index=REC.id.values)
_s1 = set(REC.id[REC.src == 'S1'])
GT = {sp: {s: l for s, l in read_gt(DATA / sp / f'{sp}_ground_truth.tsv').items() if s in _s1} for sp in ('train', 'val')}
OWNER = {sp: {x: s for s, l in GT[sp].items() for x in l} for sp in ('train', 'val')}
print(REC.groupby(['split', 'src', 'country']).size().unstack([1, 2]).to_string())
''')

METRIC_CELL = code(r'''
# ==== Metric (macro F0.5 per S1, singletons included), decision rule, result files ====
def f05_parts(tp, npred, ntrue):
    P = np.divide(tp, npred, out=np.zeros(len(tp)), where=npred > 0)
    R = np.divide(tp, ntrue, out=np.zeros(len(tp)), where=ntrue > 0)
    d = 0.25 * P + R
    f = np.divide(1.25 * P * R, d, out=np.zeros(len(tp)), where=d > 0)
    return np.where(ntrue == 0, (npred == 0).astype(float), f), P, R

class Evaluator:
    def __init__(self, sp, gt):
        self.s1 = REC.id.values[(REC.split.values == sp) & (REC.src.values == 'S1')]
        self.pos = pd.Series(np.arange(len(self.s1)), index=self.s1)
        self.ntrue = np.array([len(gt.get(s, [])) for s in self.s1])
        self.owner = {x: s for s, l in gt.items() for x in l}
        self.country = REC.country.values[ID2IX[self.s1].values]
    def links(self, q_ids, s_ids):
        si = self.pos[s_ids].values if len(s_ids) else np.zeros(0, int)
        return si, np.array([self.owner.get(q) == s for q, s in zip(q_ids, s_ids)], float)
    def metrics(self, si, ok):
        n = len(self.s1); npred = np.bincount(si, minlength=n); tp = np.bincount(si, weights=ok, minlength=n)
        f, P, R = f05_parts(tp, npred, self.ntrue)
        out = {'f05': f.mean(), 'P': P[npred > 0].mean() if (npred > 0).any() else 0.0,
               'R': R[self.ntrue > 0].mean() if (self.ntrue > 0).any() else 0.0,
               'singleton_acc': (npred[self.ntrue == 0] == 0).mean() if (self.ntrue == 0).any() else float('nan')}
        for c in np.unique(self.country): out[f'f05_{c}'] = f[self.country == c].mean()
        return out
    def tune(self, best, grid=np.round(np.arange(0.05, 0.991, 0.01), 3)):
        si, ok = self.links(best.q_id.values, best.s_id.values); p = best.p.values
        curve = [(t, self.metrics(si[p >= t], ok[p >= t])['f05']) for t in grid]
        t, f = max(curve, key=lambda x: x[1]); return t, f, curve
    def at(self, best, thr):
        sel = best.p.values >= thr
        return self.metrics(*self.links(best.q_id.values[sel], best.s_id.values[sel]))

_EV = {}
def ev(sp):                                      # test labels are read ONLY here, for the final scoring
    if sp not in _EV:
        g = GT[sp] if sp != 'test' else read_gt(DATA / 'test_labels' / 'test_ground_truth.tsv')
        _EV[sp] = Evaluator(sp, {s: l for s, l in g.items() if s in _s1})
    return _EV[sp]

def best_per_record(pairs, p):                   # decision rule: every S2/S3 record -> its best S1 candidate
    d = pd.DataFrame({'q': pairs.qi.values, 'p': p, 'q_id': pairs.q_id.values, 's_id': pairs.s_id.values})
    return d.sort_values(['q', 'p'], ascending=[True, False]).drop_duplicates('q')

def sigmoid(x): return 1 / (1 + np.exp(-np.clip(x, -30, 30)))
def safe_name(s): return re.sub(r'[^\w.-]+', '_', s)

def report(section, method, p_val, p_test, **extra):
    # threshold tuned on val; test scored once; writes results/<method>.json and preds/<method>/matching_results.tsv
    thr, f_val, _ = ev('val').tune(best_per_record(PAIRS['val'], p_val))
    bt = best_per_record(PAIRS['test'], p_test); m = ev('test').at(bt, thr)
    row = {'section': section, 'method': method, 'threshold': thr, 'val_f05': round(f_val, 5),
           'test_f05': round(m['f05'], 5), 'test_P': round(m['P'], 4), 'test_R': round(m['R'], 4),
           'test_singleton_acc': round(m['singleton_acc'], 4), 'quick': QUICK_RUN,
           **{k.replace('f05_', 'test_f05_'): round(v, 5) for k, v in m.items() if k.startswith('f05_')}, **extra}
    json.dump(row, open(OUT / 'results' / f'{safe_name(method)}.json', 'w'), indent=1, default=str)
    sel = bt[bt.p >= thr]; grp = sel.groupby('s_id').q_id.apply(lambda x: ','.join(x))
    s1 = REC.id.values[(REC.split.values == 'test') & (REC.src.values == 'S1')]
    d = OUT / 'preds' / safe_name(method); d.mkdir(exist_ok=True)
    pd.DataFrame({'source1_entity_id': s1, 'matched_entity_ids': grp.reindex(s1).fillna('').values}).to_csv(
        d / 'matching_results.tsv', sep='\t', index=False)
    print(f"{method}: threshold {thr:.2f} | val F0.5 {f_val:.5f} | TEST F0.5 {m['f05']:.5f}  P {m['P']:.4f}  R {m['R']:.4f}  "
          + ' '.join(f"{k[4:]} {v:.4f}" for k, v in m.items() if k.startswith('f05_')))
    return row

def save_scores(method, d):                      # all matchers save LOGITS for B / val / test pairs
    for k, v in d.items(): np.save(OUT / 'scores' / f'{safe_name(method)}_{k}.npy', np.asarray(v, np.float32))

def load_pairs():
    base = OUT / 'prep' if (OUT / 'prep' / 'pairs_test.parquet').exists() else find_one('prep/pairs_test.parquet').parent
    P = {sp: pd.read_parquet(base / f'pairs_{sp}.parquet') for sp in SPLITS}
    P['A'] = P['train'][P['train'].part == 'A'].reset_index(drop=True)
    P['B'] = P['train'][P['train'].part == 'B'].reset_index(drop=True)
    return P
''')

RULES_CELL = code(r'''
# ==== Address tokens + rule-based silver parser ====
TAGS = ['HOUSE', 'STREET', 'LOCALITY', 'LANDMARK', 'CITY', 'STATE', 'POSTCODE', 'SEP']
T2I = {t: i for i, t in enumerate(TAGS)}
STREET_WORDS = {'road', 'rd', 'street', 'st', 'lane', 'ln', 'drive', 'dr', 'avenue', 'ave', 'boulevard', 'blvd', 'way',
                'court', 'ct', 'place', 'pl', 'highway', 'hwy', 'parkway', 'pkwy', 'circle', 'cir', 'terrace', 'marg',
                'path', 'trail', 'pike', 'loop', 'row', 'expressway', 'bypass', 'salai'}
LANDMARK_WORDS = {'near', 'nr', 'opp', 'opposite', 'behind', 'beside', 'next', 'adjacent', 'adj', 'infront', 'facing', 'besides'}
HOUSE_WORDS = {'no', 'no.', 'h.no', 'h.no.', 'hno', 'plot', 'door', 'flat', 'house', 'shop', 'kh', 'survey', 'sy', 'unit',
               'apt', 'suite', 'ste', '#', 'bldg', 'building', 'floor', 'fl', 'khasra', 'ward', 'room', 'block'}
HOUSE_NAME_END = {'house', 'bhavan', 'villa', 'nivas', 'manzil'}      # Kerala-style house names
POST_RE = re.compile(r'^\d{6}$|^\d{5}(-\d{4})?$')

def addr_tokens(a):
    a = unicodedata.normalize('NFKC', a)
    if not a.isascii(): a = romanise(a)
    return re.findall(r',|[^\s,]+', a.lower())[:40]

def _is_post(t): return bool(POST_RE.match(t.strip('-')))
_last, _any = collections.Counter(), collections.Counter()
for a in REC.addr.values[REC.split.values == 'train']:        # state spellings = components that (almost) always come last
    comps = [' '.join(t for t in c.split() if not _is_post(t) and t.strip(':-') not in ('pin', 'pincode', 'zip')).strip(' -')
             for c in ' '.join(addr_tokens(a)).split(',')]
    comps = [c for c in comps if c and not any(ch.isdigit() for ch in c)]
    for c in comps: _any[c] += 1
    if comps: _last[comps[-1]] += 1
STATE_VOCAB = {c for c, n in _last.items() if n >= 30 and n / _any[c] >= 0.9 and len(c.split()) <= 3}
print(len(STATE_VOCAB), 'state spellings, e.g.', sorted(STATE_VOCAB)[:15])

def rule_tags(toks):
    tags = ['LOCALITY'] * len(toks); comps, cur = [], []
    for i, t in enumerate(toks):
        if t == ',': tags[i] = 'SEP'; comps.append(cur); cur = []
        else: cur.append(i)
    comps.append(cur); comps = [c for c in comps if c]
    if not comps: return tags
    tail = len(comps)
    while tail > 0:                                  # trailing postcode / state components
        c = comps[tail - 1]; w = [toks[i] for i in c]; j = len(c)
        while j > 0 and _is_post(w[j - 1]): tags[c[j - 1]] = 'POSTCODE'; j -= 1
        if 0 < j < len(c) and w[j - 1].strip(':-') in ('pin', 'pincode', 'zip'): tags[c[j - 1]] = 'POSTCODE'; j -= 1
        if j == 0: tail -= 1; continue
        if ' '.join(w[:j]).strip(' -') in STATE_VOCAB:
            for i in c[:j]: tags[i] = 'STATE'
            tail -= 1; continue
        break
    if tail >= 2:                                    # city = the component before state/postcode (or the last one)
        for i in comps[tail - 1]: tags[i] = 'CITY'
        body = comps[:tail - 1]
    else:
        body = comps[:tail]
    for ci, c in enumerate(body):
        w = [toks[i] for i in c]
        if ci == 0 and len(w) > 1 and w[-1] in HOUSE_NAME_END:
            for i in c: tags[i] = 'HOUSE'
            continue
        lab = 'LANDMARK' if w[0] in LANDMARK_WORDS else ('STREET' if any(x in STREET_WORDS for x in w) else 'LOCALITY')
        start = 0
        if ci == 0:
            while start < len(c) and (any(ch.isdigit() for ch in w[start]) or w[start] in HOUSE_WORDS):
                tags[c[start]] = 'HOUSE'; start += 1
        for i in c[start:]: tags[i] = lab
    return tags

def fields_from(toks, tags):
    f = collections.defaultdict(list)
    for t, g in zip(toks, tags):
        if g != 'SEP' and t != ',': f[g].append(t)
    return {k: ' '.join(v) for k, v in f.items()}
''')

TFIDF_CELL = code(r'''
# ==== TF-IDF retrieval helpers ====
from sklearn.feature_extraction.text import TfidfVectorizer

def split_idx(sp, s1=True):
    m = (REC.split.values == sp) & ((REC.src.values == 'S1') if s1 else (REC.src.values != 'S1'))
    return np.where(m)[0]

def topk_sparse(Qm, Sm, q_country, s_country, k, chunk=2048):
    # per country label (an open set): cosine top-k S1 rows for every record
    I = np.full((Qm.shape[0], k), -1, np.int64); V = np.zeros((Qm.shape[0], k), np.float32)
    for c in np.unique(q_country):
        qi = np.where(q_country == c)[0]; si = np.where(s_country == c)[0]
        if len(si) == 0: continue
        St = Sm[si].T.tocsr(); kk = min(k, len(si))
        for a in range(0, len(qi), chunk):
            rows = qi[a:a + chunk]; M = (Qm[rows] @ St).toarray()
            part = np.argpartition(-M, kk - 1, axis=1)[:, :kk]
            pv = np.take_along_axis(M, part, 1); o = np.argsort(-pv, 1)
            I[rows, :kk] = si[np.take_along_axis(part, o, 1)]; V[rows, :kk] = np.take_along_axis(pv, o, 1)
    return I, V

def recall_at(I, sp, s_ix, q_ix, ks=(1, 5, 10)):
    # share of linked records whose true S1 is among the top k
    owner = OWNER[sp]; s_pos = {REC.id.values[i]: j for j, i in enumerate(s_ix)}
    true = np.array([s_pos.get(owner.get(REC.id.values[i]), -2) for i in q_ix]); lk = true >= 0
    return {f'R@{k}': float((I[lk, :k] == true[lk, None]).any(1).mean()) for k in ks if k <= I.shape[1]}
''')

HF_CELL = code(r'''
# ==== Transformer helpers ====
from transformers import AutoTokenizer, AutoModel, AutoModelForSequenceClassification, get_linear_schedule_with_warmup

def make_scaler(enabled):
    try: return torch.amp.GradScaler('cuda', enabled=enabled)
    except Exception: return torch.cuda.amp.GradScaler(enabled=enabled)

def autocast(enabled=True):
    if DEVICE == 'cuda': return torch.autocast(device_type='cuda', dtype=torch.float16, enabled=USE_AMP and enabled)
    return torch.autocast(device_type='cpu', enabled=False)

def mean_pool(out, mask):
    m = mask.unsqueeze(-1).to(out.last_hidden_state.dtype)
    return (out.last_hidden_state * m).sum(1) / m.sum(1).clamp(min=1e-6)
''')

PAIRS_CELL = code(r'''
PAIRS = load_pairs()
print({sp: f"{len(PAIRS[sp]):,} pairs" + (f", positive rate {PAIRS[sp].label.mean():.3f}" if 'label' in PAIRS[sp] else '')
       for sp in ('A', 'B', 'val', 'test')})
''')

# =============================================================================== code shared by training notebooks and 08_demo
PARSER_SRC = r'''
def wkey(t): return re.sub(r'\d', '0', t)

class BiLSTMCRF(nn.Module):
    def __init__(self, n_words, n_chars, n_tags, wdim=100, cdim=32, chid=32, hid=128):
        super().__init__()
        self.wemb = nn.Embedding(n_words, wdim, padding_idx=0); self.cemb = nn.Embedding(n_chars, cdim, padding_idx=0)
        self.clstm = nn.LSTM(cdim, chid, batch_first=True, bidirectional=True)
        self.lstm = nn.LSTM(wdim + 2 * chid, hid, num_layers=2, batch_first=True, bidirectional=True, dropout=0.2)
        self.drop = nn.Dropout(0.2); self.out = nn.Linear(2 * hid, n_tags); self.crf = CRF(n_tags, batch_first=True)
    def emissions(self, W, C):
        B, T, L = C.shape
        _, (h, _) = self.clstm(self.cemb(C.view(B * T, L)))
        x = self.drop(torch.cat([self.wemb(W), torch.cat([h[0], h[1]], -1).view(B, T, -1)], -1))
        y, _ = self.lstm(x); return self.out(self.drop(y))
    def loss(self, W, C, Y, M): return -self.crf(self.emissions(W, C), Y, mask=M, reduction='mean')
    def decode(self, W, C, M): return self.crf.decode(self.emissions(W, C), mask=M)

def encode_batch(seqs, maxc=16):
    T = max(len(s) for s in seqs)
    W = np.zeros((len(seqs), T), np.int64); C = np.zeros((len(seqs), T, maxc), np.int64); M = np.zeros((len(seqs), T), bool)
    for b, s in enumerate(seqs):
        W[b, :len(s)] = [W2I.get(wkey(t), 1) for t in s]; M[b, :len(s)] = True
        for t, tok in enumerate(s): C[b, t, :min(len(tok), maxc)] = [C2I.get(ch, 1) for ch in tok[:maxc]]
    return torch.from_numpy(W), torch.from_numpy(C), torch.from_numpy(M)

def strip_commas(toks, tags):
    k = [i for i, t in enumerate(toks) if t != ','];  return [toks[i] for i in k], [tags[i] for i in k]

def parser_decode(model, seqs, bs=512):
    model.eval(); out = [[] for _ in seqs]; order = np.argsort([len(s) for s in seqs])
    with torch.no_grad():
        for a in range(0, len(order), bs):
            idx = [i for i in order[a:a + bs] if len(seqs[i]) > 0]
            if not idx: continue
            W, C, M = encode_batch([seqs[i] for i in idx])
            for i, p in zip(idx, model.decode(W.to(DEVICE), C.to(DEVICE), M.to(DEVICE))): out[i] = [TAGS[t] for t in p]
    return out
'''

EMBED_SRC = r'''
@torch.no_grad()
def embed(model, tok, texts, prefix, bs=512, max_len=64):
    model.eval(); embs = []
    order = np.argsort([len(t) for t in texts]); inv = np.empty_like(order); inv[order] = np.arange(len(order))
    for a in range(0, len(texts), bs):
        enc = tok([prefix + texts[i] for i in order[a:a + bs]], truncation=True, max_length=max_len, padding=True, return_tensors='pt').to(DEVICE)
        with autocast(): e = F.normalize(mean_pool(model(**enc), enc['attention_mask']).float(), dim=-1)
        embs.append(e.half().cpu())
    return torch.cat(embs)[torch.as_tensor(inv)]
'''

GBM_FEAT_SRC = r'''
NUM_RE = re.compile(r'\d+')
def add_gbm_cols(R):
    R['nums'] = [frozenset(NUM_RE.findall(a)) for a in R.na]; R['first_num'] = [(NUM_RE.findall(a) or [''])[0] for a in R.na]
    R['ntok'] = [frozenset(x.split()) for x in R.nn]; R['atok'] = [frozenset(x.split()) for x in R.na]
    R['legal'] = [frozenset(t for t in x.split() if t in LEGAL) for x in R.nn]
    return R

def jac(a, b): return len(a & b) / len(a | b) if (a or b) else -1.0
def eq_or_missing(a, b): return np.where((a != '') & (b != ''), (a == b).astype(np.float32), -1.0)
def ratio_or_missing(a, b, scorer=fuzz.ratio):
    return np.where((a != '') & (b != ''), cpdist(list(a), list(b), scorer=scorer, workers=-1).astype(np.float32), -1.0)

def pair_features(P):
    qi, si = P.qi.values, P.si.values; c = lambda col: REC[col].values
    qn, sn, qa, sa = c('nn')[qi], c('nn')[si], c('na')[qi], c('na')[si]; Fd = {}
    for nm, sc in (('ratio', fuzz.ratio), ('tset', fuzz.token_set_ratio), ('tsort', fuzz.token_sort_ratio), ('partial', fuzz.partial_ratio)):
        Fd[f'n_{nm}'] = cpdist(list(qn), list(sn), scorer=sc, workers=-1); Fd[f'a_{nm}'] = cpdist(list(qa), list(sa), scorer=sc, workers=-1)
    Fd['n_jw'] = cpdist(list(qn), list(sn), scorer=JaroWinkler.normalized_similarity, workers=-1)
    Fd['core_ratio'] = cpdist(list(c('ncore')[qi]), list(c('ncore')[si]), scorer=fuzz.ratio, workers=-1)
    Fd['core_eq'] = (c('ncore')[qi] == c('ncore')[si]).astype(np.float32)
    for col, nm in (('ntok', 'n_jac'), ('atok', 'a_jac'), ('nums', 'num_jac')):
        A, B = c(col)[qi], c(col)[si]; Fd[nm] = np.array([jac(x, y) for x, y in zip(A, B)], np.float32)
    A, B = c('nums')[qi], c('nums')[si]
    Fd['num_onlyq'] = np.array([len(x - y) for x, y in zip(A, B)], np.float32); Fd['num_onlys'] = np.array([len(y - x) for x, y in zip(A, B)], np.float32)
    Fd['first_num_eq'] = eq_or_missing(c('first_num')[qi], c('first_num')[si])
    A, B = c('legal')[qi], c('legal')[si]
    Fd['legal_conflict'] = np.array([float(bool(x) and bool(y) and x != y) for x, y in zip(A, B)], np.float32)
    A, B = c('ntok')[qi], c('ntok')[si]
    Fd['n_extra_q'] = np.array([len(x - y) for x, y in zip(A, B)], np.float32); Fd['n_extra_s'] = np.array([len(y - x) for x, y in zip(A, B)], np.float32)
    Fd['house_eq'] = eq_or_missing(c('p_house')[qi], c('p_house')[si]); Fd['house_ratio'] = ratio_or_missing(c('p_house')[qi], c('p_house')[si])
    Fd['post_eq'] = eq_or_missing(c('p_postcode')[qi], c('p_postcode')[si]); Fd['city_ratio'] = ratio_or_missing(c('p_city')[qi], c('p_city')[si])
    Fd['street_tset'] = ratio_or_missing(c('p_street')[qi], c('p_street')[si], fuzz.token_set_ratio)
    Fd['locality_tset'] = ratio_or_missing(c('p_locality')[qi], c('p_locality')[si], fuzz.token_set_ratio)
    Fd['q_landmark'] = (c('p_landmark')[qi] != '').astype(np.float32)
    for s_ in SCRIPTS: Fd[f'q_{s_}'] = (c('script')[qi] == s_).astype(np.float32)
    Fd['q_src3'] = (c('src')[qi] == 'S3').astype(np.float32); Fd['q_addr_empty'] = (qa == '').astype(np.float32)
    Fd['q_nlen'] = np.array([len(x) for x in qn], np.float32); Fd['s_nlen'] = np.array([len(x) for x in sn], np.float32)
    Fd['q_alen'] = np.array([len(x) for x in qa], np.float32)
    for col in ('tf_cos', 'de_cos', 'tf_rank', 'de_rank', 'rrf_rank', 'n_cand'): Fd[col] = P[col].values.astype(np.float32)
    X = pd.DataFrame(Fd); g = P.qi.values
    for col in ('n_ratio', 'a_tset', 'tf_cos', 'de_cos', 'core_ratio'): X[f'{col}_gap'] = X[col] - X.groupby(g)[col].transform('max').values
    X['n_ratio_rank'] = X.groupby(g)['n_ratio'].rank(ascending=False, method='min').values
    return X.astype(np.float32)
'''

RNN_CLASS_SRC = r'''
class SiameseRNN(nn.Module):
    def __init__(self, n_chars, cell='lstm', emb=64, hid=128):
        super().__init__()
        self.emb = nn.Embedding(n_chars, emb, padding_idx=0)
        self.rnn = {'lstm': nn.LSTM, 'gru': nn.GRU, 'rnn': nn.RNN}[cell](emb, hid, batch_first=True, bidirectional=True)
        self.head = nn.Sequential(nn.Linear(16 * hid, 256), nn.ReLU(), nn.Dropout(0.1), nn.Linear(256, 1))
    def encode(self, x, lens):
        packed = pack_padded_sequence(self.emb(x), lens.cpu(), batch_first=True, enforce_sorted=False)
        out, _ = pad_packed_sequence(self.rnn(packed)[0], batch_first=True, total_length=x.size(1))
        m = (x > 0).unsqueeze(-1)
        return torch.cat([out.masked_fill(~m, -1e4).max(1).values, (out * m).sum(1) / m.sum(1).clamp(min=1)], -1)
    def pair(self, u, v): return self.head(torch.cat([u, v, (u - v).abs(), u * v], -1)).squeeze(-1)
    def forward(self, xa, la, xb, lb): return self.pair(self.encode(xa, la), self.encode(xb, lb))
'''

CE_CLASS_SRC = r'''
class MoEHead(nn.Module):
    # top-k gated mixture of expert MLPs over the [CLS] vector, Switch-style load-balancing loss
    def __init__(self, d, n_exp=4, k=2, hid=256):
        super().__init__(); self.k, self.n = k, n_exp; self.gate = nn.Linear(d, n_exp)
        self.experts = nn.ModuleList([nn.Sequential(nn.Dropout(0.1), nn.Linear(d, hid), nn.GELU(), nn.Linear(hid, 1)) for _ in range(n_exp)])
    def forward(self, h):
        probs = self.gate(h).float().softmax(-1); topv, topi = probs.topk(self.k, -1)
        w = torch.zeros_like(probs).scatter(-1, topi, topv); w = w / w.sum(-1, keepdim=True)
        out = torch.stack([e(h).squeeze(-1).float() for e in self.experts], -1)
        f = torch.zeros_like(probs).scatter(-1, topi[:, :1], 1.0).mean(0)          # share routed (top-1) to each expert
        return (w * out).sum(-1), self.n * (f * probs.mean(0)).sum(), probs

class CEMoE(nn.Module):
    def __init__(self, name):
        super().__init__(); self.enc = AutoModel.from_pretrained(name); self.moe = MoEHead(self.enc.config.hidden_size)
    def forward(self, **enc): return self.moe(self.enc(**enc).last_hidden_state[:, 0])

def build_ce(kind, name):
    if kind == 'moe': return CEMoE(name)
    m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, ignore_mismatched_sizes=True)
    if kind == 'lora':
        from peft import LoraConfig, get_peft_model, TaskType
        m = get_peft_model(m, LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=['query', 'key', 'value']))
        m.print_trainable_parameters()
    return m
'''

GATE_FEAT_SRC = r'''
NUM_RE = re.compile(r'\d+')
def gate_features(P):
    c = lambda col: REC[col].values[P.qi.values]
    G = [(c('script') == s_).astype(np.float32) for s_ in SCRIPTS]
    G += [(c('src') == 'S3').astype(np.float32), (c('na') == '').astype(np.float32),
          np.array([bool(NUM_RE.search(x)) for x in c('na')], np.float32),
          np.array([len(x) for x in c('nn')], np.float32) / 40, np.array([len(x) for x in c('na')], np.float32) / 80,
          P.tf_cos.values.astype(np.float32), P.de_cos.values.astype(np.float32),
          P.rrf_rank.values.astype(np.float32) / CFG['K_FINAL'], P.n_cand.values.astype(np.float32) / CFG['K_FINAL']]
    return np.stack(G, 1)
'''

GATED_MOE_SRC = r'''
class GatedMoE(nn.Module):
    def __init__(self, n_exp, n_feat, input_gate=True):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_exp)); self.b = nn.Parameter(torch.zeros(n_exp)); self.c = nn.Parameter(torch.zeros(1))
        self.input_gate = input_gate; self.w0 = nn.Parameter(torch.zeros(n_exp))
        self.gate = nn.Sequential(nn.Linear(n_feat, 32), nn.ReLU(), nn.Linear(32, n_exp)) if input_gate else None
    def forward(self, z, g):
        w = self.gate(g).softmax(-1) if self.input_gate else self.w0.softmax(-1).expand(len(z), -1)
        return (w * (self.a * z + self.b)).sum(-1) + self.c, w
'''


RUN_NOTE = r"""
**Run on Kaggle:** *Settings → Accelerator: %s, Internet: On.* Inputs: %s.
Then *Save Version → Save & Run All*. Results are in the notebook's **Output** tab, and later
notebooks take that output as an input.
Set `QUICK_RUN = True` first for a short check run.
"""

DATASET = "your dataset with `data/small` (train/ val/ test/ test_labels/)"


ZIP_CELL = r'''# ---- Download: pack this notebook's output into one zip --------------------------------------
# Output panel -> refresh -> hover over out_%(tag)s.zip -> three dots -> Download.
# Keep the session running until the download has finished.
# Later, upload the zip as a Kaggle Dataset and attach it to the notebooks that need it.
import zipfile
ZIP_NAME = 'out_%(tag)s.zip'
KEEP_PREP = %(keep_prep)r      # prep/ is shipped only by the notebook that builds it (03; 02 adds parsed_fields)
_zp = OUT / ZIP_NAME
with zipfile.ZipFile(_zp, 'w', zipfile.ZIP_DEFLATED) as zf:
    for f in sorted(OUT.rglob('*')):
        rel = f.relative_to(OUT)
        if not f.is_file() or f.suffix == '.zip' or rel.parts[0].startswith('.'):
            continue
        if rel.parts[0] == 'prep' and not (KEEP_PREP is True or (KEEP_PREP and rel.name in KEEP_PREP)):
            continue
        zf.write(f, rel.as_posix(), compress_type=zipfile.ZIP_STORED if f.suffix in ('.pt', '.safetensors', '.bin', '.npy') else zipfile.ZIP_DEFLATED)
print(f'{_zp}  {_zp.stat().st_size / 2**20:.1f} MiB')
with zipfile.ZipFile(_zp) as zf:
    for i in zf.infolist(): print(f'  {i.file_size / 2**20:8.2f} MiB  {i.filename}')
try:
    from IPython.display import FileLink, display
    display(FileLink(ZIP_NAME))
except Exception:
    pass'''


def zip_cell(name):
    keep = True if name.startswith("03") else (["parsed_fields.parquet"] if name.startswith("02") else False)
    return code(ZIP_CELL % {"tag": name.split("_")[0], "keep_prep": keep})


def write(name, title, intro, accel, inputs, cells_):
    if not name.startswith("08"):
        cells_ = list(cells_) + [md("## Download the output\n\nThe cell below packs everything this notebook "
                                    "wrote into one zip, ready to download from the Output panel."), zip_cell(name)]
    nb_cells = [dict(c) for c in [md(f"# {title}\n\n{intro.strip()}\n\n" + (RUN_NOTE % (accel, inputs)).strip())] + cells_]
    for c in nb_cells:
        s = c["source"]; lines = s.split("\n")
        c["source"] = [l + "\n" for l in lines[:-1]] + [lines[-1]]
    nb = {"cells": nb_cells,
          "metadata": {"kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
                       "language_info": {"name": "python"}},
          "nbformat": 4, "nbformat_minor": 4}
    (HERE / "notebooks" / f"{name}.ipynb").write_text(json.dumps(nb, indent=1, ensure_ascii=False))
    (HERE / "scripts" / f"{name}.py").write_text(
        "\n\n".join("".join(c["source"]) for c in nb_cells if c["cell_type"] == "code"))
    print("wrote", name)


BASE_PKGS = [("anyascii", "anyascii")]

# =============================================================================== 01 tokenization
write("01_tokenization", "01 · Tokenization: tokenizers, normalisation and blocking recall", r"""
**Course topic: Tokenization.**

**(a)** How whitespace, char 3-gram, WordPiece (mBERT, IndicBERTv2) and SentencePiece (XLM-R,
mDeBERTa) split business names, per script: fertility (tokens per word), `[UNK]` rate, and share of
words split. The romanised versions of the Devanagari and Malayalam names are included.

**(b)** How the tokenizer and the normalisation affect **TF-IDF blocking recall** on the val split.
Analyzers: word / char n-gram / XLM-R subword. Text: raw, romanised, + abbreviation expansion,
+ the learned transliteration lexicon.
""", "None (CPU is enough)", DATASET, [
    setup_cell(BASE_PKGS + [("transformers", "transformers"), ("sentencepiece", "sentencepiece"), ("google.protobuf", "protobuf")], torch=False),
    config_cell("build", "TOK_SAMPLE=2000, ABL_FIT=150_000", "TOK_SAMPLE=300, ABL_FIT=20_000"),
    DATA_CELL, TFIDF_CELL,
    code(r'''
# ==== (a) Tokenizer statistics ====
from transformers import AutoTokenizer
TOK_MODELS = {'mBERT (WordPiece)': 'google-bert/bert-base-multilingual-cased',      # Apache-2.0
              'IndicBERTv2 (WordPiece)': 'ai4bharat/IndicBERTv2-MLM-only',          # MIT
              'XLM-R (SentencePiece)': 'BAAI/bge-reranker-base',                    # MIT (XLM-R tokenizer)
              'mDeBERTa (SentencePiece)': 'microsoft/mdeberta-v3-base'}             # MIT
TOKS = {}
for label, name in TOK_MODELS.items():
    try: TOKS[label] = AutoTokenizer.from_pretrained(name)
    except Exception as e: print(f'skip {label}: {type(e).__name__}: {str(e)[:150]}')

tr_pool = REC[(REC.split == 'train') & (REC.src != 'S1')]
groups = {'US latin': tr_pool[(tr_pool.country == 'US') & (tr_pool.script == 'latin')].name,
          'India latin': tr_pool[(tr_pool.country == 'India') & (tr_pool.script == 'latin')].name}
for sc in ('devanagari', 'malayalam'):
    x = tr_pool[tr_pool.script == sc].name
    groups[sc] = x; groups[sc + ' (romanised)'] = x.map(romanise)
rows = []
for g, ser in groups.items():
    sample = ser.sample(min(CFG['TOK_SAMPLE'], len(ser)), random_state=SEED).tolist()
    if not sample: continue
    n_words = sum(len(s.split()) for s in sample)
    rows.append({'group': g, 'tokenizer': 'whitespace', 'fertility': 1.0, 'unk_rate': 0.0, 'split_words': 0.0, 'vocab': None})
    rows.append({'group': g, 'tokenizer': 'char 3-gram', 'fertility': sum(max(len(s) - 2, 1) for s in sample) / n_words,
                 'unk_rate': 0.0, 'split_words': 1.0, 'vocab': None})
    words = [w for s in sample for w in s.split()][:5000]
    for label, tok in TOKS.items():
        ids = tok(sample, add_special_tokens=False)['input_ids']; n_tok = sum(len(i) for i in ids)
        unk = sum(sum(t == tok.unk_token_id for t in i) for i in ids)
        wid = tok(words, add_special_tokens=False)['input_ids']
        rows.append({'group': g, 'tokenizer': label, 'fertility': n_tok / n_words, 'unk_rate': unk / max(n_tok, 1),
                     'split_words': float(np.mean([len(i) > 1 for i in wid])), 'vocab': len(tok)})
TOK_STATS = pd.DataFrame(rows); TOK_STATS.to_csv(OUT / 'tokenizer_stats.csv', index=False)
for v, title in (('fertility', 'fertility (tokens per word)'), ('unk_rate', 'UNK rate'), ('split_words', 'share of words split into >1 piece')):
    print(f'\n{title}:'); print(TOK_STATS.pivot(index='tokenizer', columns='group', values=v).round(3).to_string())
ex = tr_pool[tr_pool.script == 'malayalam'].name.head(1).tolist() + tr_pool[tr_pool.script == 'devanagari'].name.head(1).tolist() \
     + tr_pool[tr_pool.script == 'latin'].name.head(1).tolist()
for s in ex:
    print('\n', s, '  ->  romanised:', romanise(s), ' | normalised:', norm(s, lex=None if s.isascii() else LEX))
    for label, tok in TOKS.items(): print(f'   {label:26s}', tok.tokenize(s)[:18])
'''),
    code(r'''
# ==== (b) Tokenizer / normalisation effect on TF-IDF blocking (val split) ====
tr_ix = np.concatenate([split_idx('train'), split_idx('train', False)])
fit_ix = np.random.default_rng(SEED).choice(tr_ix, min(CFG['ABL_FIT'], len(tr_ix)), replace=False)
vs, vq = split_idx('val'), split_idx('val', False)
need = np.unique(np.concatenate([fit_ix, vs, vq]))
def texts_for(fn):
    out = np.empty(len(REC), dtype=object); out[:] = ''
    for i in need: out[i] = fn(REC.name.values[i], REC.addr.values[i])
    return out
lexn = lambda s, **k: norm(s, lex=None if s.isascii() else LEX, **k)
T = {'raw': texts_for(lambda a, b: (a + ' ' + b).lower()),
     'romanised': texts_for(lambda a, b: norm(a, expand=False) + ' ' + norm(b, expand=False)),
     'romanised + abbrev.': texts_for(lambda a, b: norm(a) + ' ' + norm(b)),
     'romanised + abbrev. + lexicon': texts_for(lambda a, b: lexn(a) + ' ' + lexn(b)),
     'not romanised (+ abbrev.)': texts_for(lambda a, b: norm(a, roman=False) + ' ' + norm(b, roman=False))}
W_ = dict(analyzer='word', token_pattern=r'(?u)\b\w+\b'); C3 = dict(analyzer='char_wb', ngram_range=(3, 3))
variants = [('word', 'raw', W_), ('word', 'romanised', W_), ('word', 'romanised + abbrev.', W_),
            ('word', 'romanised + abbrev. + lexicon', W_),
            ('char 3-gram', 'not romanised (+ abbrev.)', C3), ('char 3-gram', 'romanised + abbrev.', C3),
            ('char 3-gram', 'romanised + abbrev. + lexicon', C3),
            ('char 2-4-gram', 'romanised + abbrev. + lexicon', dict(analyzer='char_wb', ngram_range=(2, 4)))]
if 'XLM-R (SentencePiece)' in TOKS:
    tk = TOKS['XLM-R (SentencePiece)']
    for src in ('raw', 'romanised + abbrev. + lexicon'):
        pieces = np.empty(len(REC), dtype=object); pieces[:] = ''
        for i, x in zip(need, tk(list(T[src][need]), add_special_tokens=False)['input_ids']): pieces[i] = ' '.join(map(str, x))
        T['XLM-R pieces of ' + src] = pieces
        variants.append(('XLM-R subwords', 'XLM-R pieces of ' + src, dict(analyzer=str.split)))
rows = []
for an, src, kw in variants:
    t = time.time(); txt = T[src]
    vec = TfidfVectorizer(sublinear_tf=True, min_df=2, dtype=np.float32, **kw).fit(txt[fit_ix])
    I, _ = topk_sparse(vec.transform(txt[vq]), vec.transform(txt[vs]), REC.country.values[vq], REC.country.values[vs], 10)
    rec_all = recall_at(I, 'val', vs, vq)
    nl = vq[REC.script.values[vq] != 'latin']; Inl = I[REC.script.values[vq] != 'latin']
    rec_nl = recall_at(Inl, 'val', vs, nl) if len(nl) else {}
    rows.append({'analyzer': an, 'text': src.replace('XLM-R pieces of ', ''), **{k: round(v, 4) for k, v in rec_all.items()},
                 'R@1 non-Latin records': round(rec_nl.get('R@1', float('nan')), 4), 'vocab': len(vec.vocabulary_),
                 'sec': round(time.time() - t, 1)})
    print(rows[-1])
BLOCK_ABL = pd.DataFrame(rows); BLOCK_ABL.to_csv(OUT / 'tokenization_blocking.csv', index=False)
print(BLOCK_ABL.to_string(index=False))
'''),
])

# =============================================================================== 02 parser
write("02_parser_bilstm_crf", "02 · Address parsing with a BiLSTM-CRF", r"""
**Course topics: Parsing, LSTM.**

There are no gold address parses. So **silver labels** come from rules that use the comma structure:
trailing postcode/state, the city before it, house-number tokens first, landmark phrases
(*near / opp / behind*), and street vs locality by street words. A **BiLSTM-CRF** (word embedding
+ character BiLSTM per token, 2-layer BiLSTM, CRF) is trained on them. **Half of the training
addresses have their commas removed**, so the model has to learn the structure from the words.
The rules cannot do that.

Tested on val addresses against the silver labels (with and without commas), with the rules on
the same input as the baseline. Output: `prep/parsed_fields.parquet` (house no., street,
locality, landmark, city, state, postcode for every record). It is used by `04_gbm`.
""", "GPU T4", DATASET, [
    setup_cell(BASE_PKGS + [("torchcrf", "pytorch-crf"), ("sklearn", "scikit-learn")]),
    config_cell("build", "PARSER_TRAIN=120_000, PARSER_EPOCHS=3", "PARSER_TRAIN=5000, PARSER_EPOCHS=1"),
    DATA_CELL, RULES_CELL,
    code(r'''
REC_TOKS = [addr_tokens(a) for a in REC.addr.values]
for i in np.random.default_rng(0).choice(len(REC), 4, replace=False):
    print(list(zip(REC_TOKS[i], rule_tags(REC_TOKS[i]))))
'''),
    code(r'''
# ==== BiLSTM-CRF ====
from torchcrf import CRF
from sklearn.metrics import classification_report, f1_score
''' + PARSER_SRC + r'''
rng = np.random.default_rng(SEED)
_tr = [i for i in np.where(REC.split.values == 'train')[0] if REC_TOKS[i]]
_va = [i for i in np.where(REC.split.values == 'val')[0] if REC_TOKS[i]]
tr = rng.choice(_tr, min(CFG['PARSER_TRAIN'], len(_tr)), replace=False)
va = rng.choice(_va, min(10_000, len(_va)), replace=False)
wc = collections.Counter(wkey(t) for i in tr for t in REC_TOKS[i]); cc = collections.Counter(ch for i in tr for t in REC_TOKS[i] for ch in t)
W2I = {w: j + 2 for j, (w, n) in enumerate(wc.most_common(30_000)) if n >= 3}
C2I = {c: j + 2 for j, (c, n) in enumerate(cc.most_common()) if n >= 5}
train_data = []
for n, i in enumerate(tr):
    toks, tags = REC_TOKS[i], rule_tags(REC_TOKS[i])
    if n % 2: toks, tags = strip_commas(toks, tags)                 # half without commas
    if toks: train_data.append((toks, [T2I[t] for t in tags]))
PARSER = BiLSTMCRF(len(W2I) + 2, len(C2I) + 2, len(TAGS)).to(DEVICE)
opt = torch.optim.Adam(PARSER.parameters(), lr=2e-3); t0 = time.time()
for ep in range(CFG['PARSER_EPOCHS']):
    PARSER.train(); random.shuffle(train_data); tot = 0
    for a in range(0, len(train_data), 256):
        b = train_data[a:a + 256]; W, C, M = encode_batch([x[0] for x in b]); Y = torch.zeros_like(W)
        for k, (_, y) in enumerate(b): Y[k, :len(y)] = torch.tensor(y)
        loss = PARSER.loss(W.to(DEVICE), C.to(DEVICE), Y.to(DEVICE), M.to(DEVICE))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(PARSER.parameters(), 5.0); opt.step(); tot += loss.item() * len(b)
    log(f'epoch {ep + 1}: loss {tot / len(train_data):.4f}')
parser_time = time.time() - t0
'''),
    code(r'''
# ==== Evaluation against silver labels (val) ====
rows = []; labs = [t for t in TAGS if t != 'SEP']
for cond in ('with commas', 'commas removed'):
    seqs, gold = [], []
    for i in va:
        toks, tags = REC_TOKS[i], rule_tags(REC_TOKS[i])
        if cond == 'commas removed': toks, tags = strip_commas(toks, tags)
        if toks: seqs.append(toks); gold.append(tags)
    pred = parser_decode(PARSER, seqs); rule_pred = [rule_tags(s) for s in seqs]
    for name, P_ in (('BiLSTM-CRF', pred), ('rules', rule_pred)):
        g = [t for s in gold for t in s]; p = [t for s in P_ for t in s]
        rows.append({'input': cond, 'parser': name, 'token_acc': np.mean([a == b for a, b in zip(g, p)]),
                     'macro_f1': f1_score(g, p, labels=labs, average='macro', zero_division=0),
                     'exact_address': np.mean([a == b for a, b in zip(gold, P_)])})
    if cond == 'commas removed':
        print('BiLSTM-CRF on comma-free val addresses (vs silver labels):')
        print(classification_report([t for s in gold for t in s], [t for s in pred for t in s], labels=labs, zero_division=0, digits=3))
PARSER_EVAL = pd.DataFrame(rows).round(4); PARSER_EVAL.to_csv(OUT / 'parser_eval.csv', index=False)
print(PARSER_EVAL.to_string(index=False))
n_par = sum(p.numel() for p in PARSER.parameters())
json.dump({'params': n_par, 'train_min': parser_time / 60, 'n_train': len(train_data)}, open(OUT / 'parser_meta.json', 'w'))
print(f'{n_par / 1e6:.2f}M parameters, trained in {parser_time / 60:.1f} min')
i = va[0]; s = [t for t in REC_TOKS[i] if t != ',']
print('\nexample without commas:', list(zip(s, parser_decode(PARSER, [s])[0])))
'''),
    code(r'''
# ==== Parse every address (with commas) -> prep/parsed_fields.parquet ====
tags_all = parser_decode(PARSER, REC_TOKS)
fl = [fields_from(t, g) for t, g in zip(REC_TOKS, tags_all)]
pf = pd.DataFrame({f'p_{k.lower()}': [d.get(k, '') for d in fl] for k in TAGS if k != 'SEP'}); pf.insert(0, 'id', REC.id.values)
pf.to_parquet(OUT / 'prep' / 'parsed_fields.parquet')
print(pd.concat([REC[['addr']], pf.drop(columns='id')], axis=1).sample(8, random_state=1).to_string())
(OUT / 'models').mkdir(exist_ok=True)                                  # weights for 08_demo
torch.save({'state_dict': PARSER.state_dict(), 'W2I': W2I, 'C2I': C2I, 'TAGS': TAGS}, OUT / 'models' / 'parser_bilstm_crf.pt')
'''),
])

# =============================================================================== 03 candidates
write("03_candidates", "03 · Candidate generation: TF-IDF + transformer bi-encoder (zero-shot vs fine-tuned)", r"""
**Course topics: Transformers, Fine-tuning (+ the tokenization choices from 01).**
**Every later notebook needs this notebook's output.**

* Normalised records and the learned transliteration lexicon → `prep/rec.parquet`.
* **TF-IDF** char 3-grams of the romanised, normalised `name + address`, fitted on train text. For each
  S2/S3 record: the top-k S1 with the same country label (an open set).
* **Bi-encoder** `intfloat/multilingual-e5-small` (MIT, 118M): **zero-shot** first, then **fine-tuned
  contrastively** on (record, true S1) pairs of train part A (InfoNCE with in-batch negatives,
  same-S1 duplicates masked).
* **Fusion**: reciprocal-rank fusion of both lists. The top `K_FINAL` per record are the candidate pairs
  that every matcher scores → `prep/pairs_{train,val,test}.parquet`. Also: blocking recall, the
  blocking ceiling, and `candidate_pairs.tsv` for the test split.
""", "GPU T4", DATASET, [
    setup_cell(BASE_PKGS + [("sklearn", "scikit-learn"), ("transformers", "transformers")]),
    config_cell("build_save", "BI_MODEL='intfloat/multilingual-e5-small', BI_PAIRS=100_000, BI_BS=128, BI_LR=3e-5, BI_LEN=64, BI_EPOCHS=1, USE_DENSE=True",
                "BI_PAIRS=3000"),
    DATA_CELL, METRIC_CELL, TFIDF_CELL, HF_CELL,
    code(r'''
# ==== TF-IDF retrieval ====
import pickle
S_IX = {sp: split_idx(sp) for sp in SPLITS}; Q_IX = {sp: split_idx(sp, False) for sp in SPLITS}
t = time.time()
VEC = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), sublinear_tf=True, min_df=2, dtype=np.float32)
VEC.fit(REC.bl_text.values[np.concatenate([S_IX['train'], Q_IX['train']])])
CT = {}
for sp in SPLITS:
    CT[sp] = topk_sparse(VEC.transform(REC.bl_text.values[Q_IX[sp]]), VEC.transform(REC.bl_text.values[S_IX[sp]]),
                         REC.country.values[Q_IX[sp]], REC.country.values[S_IX[sp]], CFG['K_TFIDF'])
    log(f'tfidf {sp}: {len(Q_IX[sp]):,} records x {len(S_IX[sp]):,} S1')
BLOCK_ROWS = [{'retriever': 'TF-IDF char 3-gram', **recall_at(CT['val'][0], 'val', S_IX['val'], Q_IX['val'])}]
print(BLOCK_ROWS[-1], f'{time.time() - t:.0f}s')
'''),
    code(r'''
# ==== Bi-encoder: zero-shot vs contrastive fine-tuning ====
def part_of(ids):          # train S1 -> A/B by hash; linked records follow their S1; unlinked by hash
    o = OWNER['train']; return np.array(['B' if h01(f"AB{SEED}|{o.get(x, x)}") < CFG['B_FRAC'] else 'A' for x in ids])

''' + EMBED_SRC + r'''
@torch.no_grad()
def dense_topk(Eq, Es, q_country, s_country, k, chunk=4096):
    I = np.full((len(Eq), k), -1, np.int64); V = np.zeros((len(Eq), k), np.float32)
    for c in np.unique(q_country):
        qi = np.where(q_country == c)[0]; si = np.where(s_country == c)[0]
        if len(si) == 0: continue
        S = Es[si].to(DEVICE).float(); kk = min(k, len(si))
        for a in range(0, len(qi), chunk):
            rows = qi[a:a + chunk]; v, j = (Eq[rows].to(DEVICE).float() @ S.T).topk(kk, dim=1)
            I[rows, :kk] = si[j.cpu().numpy()]; V[rows, :kk] = v.cpu().numpy()
    return I, V

def texts(ix): return REC.ce_text.values[ix].tolist()
CD = None
if CFG['USE_DENSE']:
    tok_bi = AutoTokenizer.from_pretrained(CFG['BI_MODEL']); bi = AutoModel.from_pretrained(CFG['BI_MODEL']).to(DEVICE); L = CFG['BI_LEN']
    def dense_split(model, sp):
        Es = embed(model, tok_bi, texts(S_IX[sp]), 'passage: ', max_len=L); Eq = embed(model, tok_bi, texts(Q_IX[sp]), 'query: ', max_len=L)
        I, V = dense_topk(Eq, Es, REC.country.values[Q_IX[sp]], REC.country.values[S_IX[sp]], CFG['K_DENSE']); return I, V, Eq, Es
    zs = recall_at(dense_split(bi, 'val')[0], 'val', S_IX['val'], Q_IX['val']); print('zero-shot:', zs)
    BLOCK_ROWS.append({'retriever': 'e5-small zero-shot', **zs})
    ids = REC.id.values[Q_IX['train']]; own = np.array([OWNER['train'].get(x, '') for x in ids])
    pos = np.where((own != '') & (part_of(ids) == 'A'))[0]
    pos = np.random.default_rng(SEED).choice(pos, min(CFG['BI_PAIRS'], len(pos)), replace=False)
    q_txt = REC.ce_text.values[Q_IX['train'][pos]]; s_ids = own[pos]; s_txt = REC.ce_text.values[ID2IX[s_ids].values]
    opt = torch.optim.AdamW(bi.parameters(), lr=CFG['BI_LR'], weight_decay=0.01); bs = CFG['BI_BS']
    steps = CFG['BI_EPOCHS'] * math.ceil(len(pos) / bs)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); scaler = make_scaler(USE_AMP); t0 = time.time(); step = 0
    for ep in range(CFG['BI_EPOCHS']):
        bi.train()
        for b in np.array_split(np.random.default_rng(SEED + ep).permutation(len(pos)), max(1, len(pos) // bs)):
            if len(b) < 2: continue
            eq = tok_bi(['query: ' + x for x in q_txt[b]], truncation=True, max_length=L, padding=True, return_tensors='pt').to(DEVICE)
            es = tok_bi(['passage: ' + x for x in s_txt[b]], truncation=True, max_length=L, padding=True, return_tensors='pt').to(DEVICE)
            with autocast():
                u = F.normalize(mean_pool(bi(**eq), eq['attention_mask']).float(), dim=-1)
                v = F.normalize(mean_pool(bi(**es), es['attention_mask']).float(), dim=-1)
            logits = u @ v.T / 0.05
            sid = s_ids[b]; same = torch.as_tensor(sid[:, None] == sid[None, :]) & ~torch.eye(len(b), dtype=torch.bool)
            logits = logits.masked_fill(same.to(DEVICE), -1e4); lab = torch.arange(len(b), device=DEVICE)
            loss = (F.cross_entropy(logits, lab) + F.cross_entropy(logits.T, lab)) / 2
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.step(opt); scaler.update(); sch.step(); step += 1
            if step % 200 == 0: log(f'bi-encoder step {step}/{steps} loss {loss.item():.4f}')
    bi_time = time.time() - t0
    CD = {sp: dense_split(bi, sp) for sp in SPLITS}
    BLOCK_ROWS.append({'retriever': 'e5-small fine-tuned', **recall_at(CD['val'][0], 'val', S_IX['val'], Q_IX['val']),
                       'train_min': round(bi_time / 60, 1)})
    _d = OUT / 'models' / 'e5_finetuned'; _d.mkdir(parents=True, exist_ok=True)      # weights for 08_demo (fp16)
    bi.half().save_pretrained(_d); tok_bi.save_pretrained(_d)
    del bi; gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()
print(pd.DataFrame(BLOCK_ROWS).round(4).to_string(index=False))
'''),
    code(r'''
# ==== Fused candidate set -> prep/pairs_{split}.parquet ====
def fuse(sp):
    nq = len(Q_IX[sp]); lists = [('tf', CT[sp][0])] + ([('de', CD[sp][0])] if CD else []); dfs = []
    for name, I in lists:
        k = I.shape[1]; d = pd.DataFrame({'q': np.repeat(np.arange(nq), k), 's': I.ravel(), f'{name}_rank': np.tile(np.arange(k), nq)})
        dfs.append(d[d.s >= 0])
    df = dfs[0] if len(dfs) == 1 else dfs[0].merge(dfs[1], on=['q', 's'], how='outer')
    df['rrf'] = sum(1 / (60 + df[f'{n}_rank'].fillna(1e6)) for n, _ in lists)
    df = df.sort_values(['q', 'rrf'], ascending=[True, False]); df['rrf_rank'] = df.groupby('q').cumcount()
    df = df[df.rrf_rank < CFG['K_FINAL']].reset_index(drop=True)
    df['tf_rank'] = df['tf_rank'].fillna(CFG['K_TFIDF'])
    df['de_rank'] = df['de_rank'].fillna(CFG['K_DENSE']) if 'de_rank' in df else float(CFG['K_DENSE'])
    df['n_cand'] = df.groupby('q').s.transform('size')
    qi, si = Q_IX[sp][df.q.values], S_IX[sp][df.s.values]
    df['qi'], df['si'] = qi, si; df['q_id'], df['s_id'] = REC.id.values[qi], REC.id.values[si]
    Qm, Sm = VEC.transform(REC.bl_text.values[qi]), VEC.transform(REC.bl_text.values[si])
    df['tf_cos'] = np.asarray(Qm.multiply(Sm).sum(1)).ravel().astype(np.float32)
    df['de_cos'] = (CD[sp][2][df.q.values].float() * CD[sp][3][df.s.values].float()).sum(1).numpy() if CD else 0.0
    if sp != 'test': df['label'] = np.array([OWNER[sp].get(q) == s for q, s in zip(df.q_id, df.s_id)], np.int8)
    df['part'] = part_of(df.q_id.values) if sp == 'train' else sp
    return df.drop(columns=['q', 's'])

for sp in SPLITS:
    fuse(sp).to_parquet(OUT / 'prep' / f'pairs_{sp}.parquet'); log(f'pairs {sp} written')
PAIRS = load_pairs()
lk = [x for x in REC.id.values[Q_IX['val']] if x in OWNER['val']]; hit = set(zip(PAIRS['val'].q_id, PAIRS['val'].s_id))
BLOCK_ROWS.append({'retriever': f"fused (RRF), top {CFG['K_FINAL']}", f"R@{CFG['K_FINAL']}": float(np.mean([(x, OWNER['val'][x]) in hit for x in lk]))})
def ceiling(sp):
    d = PAIRS[sp]; si, ok = ev(sp).links(d.q_id.values, d.s_id.values); return ev(sp).metrics(si[ok == 1], ok[ok == 1])['f05']
BLOCK = {'val_ceiling_f05': ceiling('val'), 'test_ceiling_f05': ceiling('test'),
         **{f'{sp}_pairs': len(PAIRS[sp]) for sp in SPLITS}, 'pairs_per_record_test': len(PAIRS['test']) / len(Q_IX['test'])}
json.dump({'rows': BLOCK_ROWS, **BLOCK, 'quick': QUICK_RUN}, open(OUT / 'blocking.json', 'w'), indent=1, default=float)
print(pd.DataFrame(BLOCK_ROWS).round(4).to_string(index=False)); print(BLOCK)
g = PAIRS['test'].groupby('s_id').q_id.apply(lambda x: ','.join(x)); s1 = REC.id.values[S_IX['test']]
pd.DataFrame({'source1_entity_id': s1, 'candidate_entity_ids': g.reindex(s1).fillna('').values}).to_csv(
    OUT / 'preds' / 'candidate_pairs.tsv', sep='\t', index=False)
print({sp: f"{len(PAIRS[sp]):,} pairs, positive rate {PAIRS[sp].label.mean():.3f}" for sp in ('A', 'B', 'val')})
'''),
])

# =============================================================================== 04 GBM
write("04_gbm_baseline", "04 · Classical baseline: LightGBM on string-similarity + parsed-address features", r"""
The non-neural reference point. Features per candidate pair: fuzzy name and address scores
(ratio, token-set, partial, Jaro-Winkler), word Jaccard, house-number agreement, legal-suffix conflict,
**the parsed address fields from 02** (house no., postcode, city, street, locality), retrieval
scores, and the competition among a record's candidates. Trained on part A, early-stopped on val.
If 02 is not attached, the rule parser is used instead.
""", "None (CPU)", "the dataset, the output of `03_candidates`, and (recommended) the output of `02_parser_bilstm_crf`", [
    setup_cell(BASE_PKGS + [("rapidfuzz", "rapidfuzz>=3.6"), ("lightgbm", "lightgbm")], torch=False),
    config_cell("load", "", ""),
    DATA_CELL, METRIC_CELL, PAIRS_CELL, RULES_CELL,
    code(r'''
# ==== Parsed address fields (from 02, else rules) ====
_pf = find_one('prep/parsed_fields.parquet')
if _pf is not None:
    pf = pd.read_parquet(_pf).set_index('id').reindex(REC.id.values).fillna(''); PARSE_SRC = 'BiLSTM-CRF (02)'
else:
    print('02 output not attached -> rule parser'); toks = [addr_tokens(a) for a in REC.addr.values]
    fl = [fields_from(t, rule_tags(t)) for t in toks]
    pf = pd.DataFrame({f'p_{k.lower()}': [d.get(k, '') for d in fl] for k in TAGS if k != 'SEP'}, index=REC.id.values); PARSE_SRC = 'rules'
for c in pf.columns: REC[c] = pf[c].values
print('parsed fields from', PARSE_SRC)
'''),
    code(r'''
# ==== Pair features ====
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
import lightgbm as lgb
''' + GBM_FEAT_SRC + r'''
REC = add_gbm_cols(REC)
X = {}
for sp in ('A', 'B', 'val', 'test'):
    t = time.time(); X[sp] = pair_features(PAIRS[sp]); log(f'features {sp}: {X[sp].shape} in {time.time() - t:.0f}s')
'''),
    code(r'''
# ==== LightGBM ====
t = time.time()
gbm = lgb.LGBMClassifier(n_estimators=3000, learning_rate=0.05, num_leaves=63, min_child_samples=50, subsample=0.8,
                         subsample_freq=1, colsample_bytree=0.8, verbose=-1, random_state=SEED)
gbm.fit(X['A'], PAIRS['A'].label, eval_set=[(X['val'], PAIRS['val'].label)], callbacks=[lgb.early_stopping(100, verbose=False)])
gbm_time = time.time() - t
logit = lambda p: np.log(np.clip(p, 1e-6, 1 - 1e-6) / (1 - np.clip(p, 1e-6, 1 - 1e-6)))
S = {sp: logit(gbm.predict_proba(X[sp])[:, 1]) for sp in ('B', 'val', 'test')}
METHOD = 'GBM (string + parsed-address features)'
save_scores(METHOD, S)
imp = pd.Series(gbm.booster_.feature_importance('gain'), index=X['A'].columns).sort_values(ascending=False)
imp.to_csv(OUT / 'gbm_feature_importance.csv'); print(imp.head(20).round(0).to_string())
report('5 Classical baseline', METHOD, sigmoid(S['val']), sigmoid(S['test']), params=f'{gbm.best_iteration_} trees',
       train_min=round(gbm_time / 60, 1), parsed_fields=PARSE_SRC)
(OUT / 'models').mkdir(exist_ok=True)                                  # model for 08_demo
gbm.booster_.save_model(str(OUT / 'models' / 'gbm.txt'), num_iteration=gbm.best_iteration_)
json.dump({'method': METHOD, 'features': list(X['A'].columns), 'parse_src': PARSE_SRC}, open(OUT / 'models' / 'gbm_meta.json', 'w'), indent=1)
'''),
])

# =============================================================================== 05 RNN family
RNN_TEMPLATE = r'''
# ==== Siamese character-level recurrent matcher ====
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
CELL, METHOD = %r, %r
CHAR_MAX = 100
_cc = collections.Counter(ch for t in REC.ltext.values[REC.split.values == 'train'] for ch in t)
CH2I = {c: i + 2 for i, (c, n) in enumerate(_cc.most_common(250)) if n >= 20}
XCH = np.zeros((len(REC), CHAR_MAX), np.uint8)
for r, t in enumerate(REC.ltext.values):
    XCH[r, :min(len(t), CHAR_MAX)] = [CH2I.get(ch, 1) for ch in t[:CHAR_MAX]]
XLEN = np.maximum((XCH > 0).sum(1), 1)

''' + RNN_CLASS_SRC + r'''
def batch_x(ix):
    L = int(XLEN[ix].max()); return torch.as_tensor(XCH[ix, :L].astype(np.int64), device=DEVICE), torch.as_tensor(XLEN[ix])

@torch.no_grad()
def rnn_scores(model, P, bs=2048):
    # each record is encoded once, then the pair head runs on all candidate pairs
    model.eval(); uniq = np.unique(np.concatenate([P.qi.values, P.si.values]))
    E = torch.zeros(len(uniq), model.head[0].in_features // 4, device=DEVICE)
    pos = pd.Series(np.arange(len(uniq)), index=uniq); order = uniq[np.argsort(XLEN[uniq])]
    for a in range(0, len(order), bs):
        ix = order[a:a + bs]; E[pos[ix].values] = model.encode(*batch_x(ix))
    qa = torch.as_tensor(pos[P.qi.values].values, device=DEVICE); sa = torch.as_tensor(pos[P.si.values].values, device=DEVICE)
    return torch.cat([model.pair(E[qa[a:a + 65536]], E[sa[a:a + 65536]]) for a in range(0, len(qa), 65536)]).float().cpu().numpy()

model = SiameseRNN(len(CH2I) + 2, CELL).to(DEVICE)
opt = torch.optim.Adam(model.parameters(), lr=1e-3)
A = PAIRS['A']; y_all = A.label.values.astype(np.float32); rng = np.random.default_rng(SEED); t0 = time.time()
for ep in range(CFG['EPOCHS']):
    model.train(); idx = rng.permutation(len(A))[:CFG['PAIRS_PER_EPOCH']]; tot = 0
    for a in range(0, len(idx), 512):
        b = idx[a:a + 512]
        loss = F.binary_cross_entropy_with_logits(model(*batch_x(A.qi.values[b]), *batch_x(A.si.values[b])), torch.as_tensor(y_all[b], device=DEVICE))
        opt.zero_grad(); loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0); opt.step(); tot += loss.item() * len(b)
    pv = rnn_scores(model, PAIRS['val'])
    vl = F.binary_cross_entropy_with_logits(torch.as_tensor(pv), torch.as_tensor(PAIRS['val'].label.values, dtype=torch.float32)).item()
    log(f'{CELL} epoch {ep + 1}/{CFG["EPOCHS"]}: train loss {tot / len(idx):.4f}  val loss {vl:.4f}')
train_time = time.time() - t0
S = {sp: rnn_scores(model, PAIRS[sp]) for sp in ('B', 'val', 'test')}
save_scores(METHOD, S)
n_par = sum(p.numel() for p in model.parameters())
report('6 RNN / LSTM', METHOD, sigmoid(S['val']), sigmoid(S['test']), params=f'{n_par / 1e6:.2f}M', train_min=round(train_time / 60, 1))
(OUT / 'models').mkdir(exist_ok=True)                                  # weights for 08_demo
torch.save({'method': METHOD, 'cell': CELL, 'ch2i': CH2I, 'char_max': CHAR_MAX, 'state_dict': model.state_dict()},
           OUT / 'models' / f'{safe_name(METHOD)}.rnn.pt')
'''

for nb, cell, method, title, epochs in (("05a_siamese_bilstm", "lstm", "Siamese char-BiLSTM", "Siamese character-level BiLSTM", 3),
                                        ("05b_siamese_bigru", "gru", "Siamese char-BiGRU", "Siamese character-level BiGRU", 3),
                                        ("05c_siamese_birnn", "rnn", "Siamese char-BiRNN (vanilla)", "Siamese character-level vanilla BiRNN", 3)):
    write(nb, f"{nb[:3]} · {title} matcher", rf"""
**Course topics: RNN / LSTM.** Each record (`name | address`, romanised and normalised, ≤ 100
characters) is encoded by a bidirectional {cell.upper()} over characters (max + mean pooling). A
pair is classified from `[u, v, |u − v|, u ⊙ v]` with an MLP (a Siamese network). Trained on the part-A
candidate pairs from 03, so the negatives are hard: they are what blocking retrieved. The three
notebooks 05a/b/c differ **only in the recurrent cell**: LSTM vs GRU vs vanilla tanh-RNN.
""", "GPU T4", "the dataset and the output of `03_candidates`", [
        setup_cell(BASE_PKGS),
        config_cell("load", f"EPOCHS={epochs}, PAIRS_PER_EPOCH=600_000", "EPOCHS=1, PAIRS_PER_EPOCH=50_000"),
        DATA_CELL, METRIC_CELL, PAIRS_CELL,
        code(RNN_TEMPLATE % (cell, method)),
    ])

# =============================================================================== 06 cross-encoders
CE_TEMPLATE = r'''
# ==== Cross-encoder: %s ====
METHOD, KIND, MODEL, LR, TRAIN, AMP = %r, %r, %r, %r, %r, %r
MODEL = os.environ.get('NLP_ER_MODEL_OVERRIDE', MODEL)            # local smoke test only
''' + CE_CLASS_SRC + r'''
def ce_forward(model, enc):
    if KIND == 'moe': return model(**enc)
    return model(**enc).logits[:, 0].float(), 0.0, None

def tok_pairs(P, rows):
    return TOK(list(REC.ce_text.values[P.qi.values[rows]]), list(REC.ce_text.values[P.si.values[rows]]),
               truncation=True, max_length=CFG['MAX_LEN'], padding=True, return_tensors='pt').to(DEVICE)

@torch.no_grad()
def ce_score(model, P):
    model.eval()
    lens = np.array([len(a) + len(b) for a, b in zip(REC.ce_text.values[P.qi.values], REC.ce_text.values[P.si.values])])
    order = np.argsort(lens); out = np.zeros(len(P), np.float32); gates = np.zeros((len(P), 4), np.float32) if KIND == 'moe' else None
    for a in range(0, len(order), CFG['EVAL_BS']):
        rows = order[a:a + CFG['EVAL_BS']]
        with autocast(AMP): y, _, probs = ce_forward(model, tok_pairs(P, rows))
        out[rows] = y.float().cpu().numpy()
        if gates is not None: gates[rows] = probs.float().cpu().numpy()
    return out, gates

TOK = AutoTokenizer.from_pretrained(MODEL); model = build_ce(KIND, MODEL).to(DEVICE)
n_all = sum(p.numel() for p in model.parameters())
n_tr = sum(p.numel() for p in model.parameters() if p.requires_grad) if TRAIN else 0
print(f'{n_all / 1e6:.0f}M parameters, {n_tr / 1e6:.2f}M trained')
t0 = time.time()
if TRAIN:
    A = PAIRS['A']; y = A.label.values.astype(np.float32)
    qs = A.qi.unique(); rng = np.random.default_rng(SEED); rng.shuffle(qs)          # sample whole candidate groups
    cum = A.groupby('qi').size().reindex(qs).cumsum().values
    rows = np.where(A.qi.isin(set(qs[:np.searchsorted(cum, CFG['TRAIN_PAIRS']) + 1])).values)[0]; rng.shuffle(rows)
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=LR, weight_decay=0.01)
    steps = math.ceil(len(rows) / CFG['BS']); sch = get_linear_schedule_with_warmup(opt, int(0.06 * steps), steps)
    scaler = make_scaler(USE_AMP and AMP); model.train()
    print(f'training on {len(rows):,} pairs ({y[rows].mean():.3f} positive), {steps:,} steps')
    for st, a in enumerate(range(0, len(rows), CFG['BS'])):
        b = rows[a:a + CFG['BS']]
        with autocast(AMP): logit, aux, _ = ce_forward(model, tok_pairs(A, b))
        loss = F.binary_cross_entropy_with_logits(logit.float(), torch.as_tensor(y[b], device=DEVICE)) + 0.01 * aux
        opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
        nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step()
        if (st + 1) %% 500 == 0: log(f'step {st + 1}/{steps} loss {loss.item():.4f}')
train_time = time.time() - t0
S, G = {}, {}
for sp in ('B', 'val', 'test'):
    S[sp], G[sp] = ce_score(model, PAIRS[sp]); log(f'scored {sp}')
save_scores(METHOD, S)
report('7 Transformers', METHOD, sigmoid(S['val']), sigmoid(S['test']),
       params=f'{n_all / 1e6:.0f}M ({100 * n_tr / n_all:.1f}%% trained)', train_min=round(train_time / 60, 1))
(OUT / 'models').mkdir(exist_ok=True)                                  # weights for 08_demo: trained parameters only
_w = {n: (p.detach().half() if AMP else p.detach()).cpu() for n, p in model.named_parameters() if p.requires_grad} if TRAIN else {}
torch.save({'method': METHOD, 'kind': KIND, 'model': MODEL, 'max_len': CFG['MAX_LEN'], 'amp': AMP, 'trained': _w},
           OUT / 'models' / f'{safe_name(METHOD)}.ce.pt')
print(f"saved {sum(v.numel() for v in _w.values()) / 1e6:.1f}M trained parameters for the demo")
'''

MOE_GATE_CELL = code(r'''
# ==== Which expert of the MoE head handles which records? (test pairs) ====
g = G['test']; P = PAIRS['test']
df = pd.DataFrame(g, columns=[f'expert {i}' for i in range(g.shape[1])])
df['script'] = REC.script.values[P.qi.values]; df['country'] = REC.country.values[P.qi.values]; df['source'] = REC.src.values[P.qi.values]
df['is_match_top1'] = P.rrf_rank.values == 0
for col in ('script', 'country', 'source'):
    print(f'\nmean gate probability by {col}:'); print(df.groupby(col).mean(numeric_only=True).round(3).to_string())
df.groupby('script').mean(numeric_only=True).to_csv(OUT / 'moe_head_gates_by_script.csv')
print('\nshare of pairs whose top-1 expert is each expert:', np.bincount(g.argmax(1), minlength=g.shape[1]) / len(g))
''')

CE_RUNS = [
    ("06a_ce_bge_zeroshot", "CE bge-reranker-base zero-shot", "seq", "BAAI/bge-reranker-base", 0.0, False, True,
     "the pre-trained multilingual reranker scores the pairs as they are. **No training**: the zero-shot reference for 06b–06d."),
    ("06b_ce_bge_full_finetune", "CE bge-reranker-base full fine-tune", "seq", "BAAI/bge-reranker-base", 2e-5, True, True,
     "**full fine-tuning**: all 278M weights, BCE loss, 1 epoch over `TRAIN_PAIRS` part-A candidate pairs."),
    ("06c_ce_bge_lora", "CE bge-reranker-base LoRA r=16", "lora", "BAAI/bge-reranker-base", 2e-4, True, True,
     "**LoRA** (parameter-efficient fine-tuning): frozen backbone plus rank-16 adapters on query/key/value; about 1 % of the weights are trained."),
    ("06d_ce_bge_moe_head", "CE bge-reranker-base + MoE head (top-2 of 4)", "moe", "BAAI/bge-reranker-base", 2e-5, True, True,
     "**Mixture-of-Experts head**: the [CLS] vector goes to 4 expert MLPs, a gate picks the **top 2** per pair, and a Switch-style load-balancing loss is added. Fully fine-tuned. The gate analysis at the end shows which expert handles which script / country / source."),
    ("06e_ce_indicbert", "CE IndicBERTv2 full fine-tune", "seq", "ai4bharat/IndicBERTv2-MLM-only", 3e-5, True, True,
     "**IndicBERTv2** (MIT, 278M), a BERT pre-trained on Indian languages with its own WordPiece vocabulary (see 01), fully fine-tuned as a cross-encoder with a new head."),
    ("06f_ce_mdeberta", "CE mDeBERTa-v3 full fine-tune (fp32)", "seq", "microsoft/mdeberta-v3-base", 2e-5, True, False,
     "**mDeBERTa-v3-base** (MIT, 279M), fully fine-tuned. It runs in fp32, because DeBERTa-v3 gives NaN losses in fp16 on a T4, so it is about 2–3× slower than 06b. Optional."),
]
for nb, method, kind, model, lr, train, amp, desc in CE_RUNS:
    cells_ = [setup_cell(BASE_PKGS + [("transformers", "transformers")] + ([("peft", "peft")] if kind == "lora" else [])),
              config_cell("load", "TRAIN_PAIRS=160_000, BS=32, EVAL_BS=256", "TRAIN_PAIRS=4000"),
              DATA_CELL, METRIC_CELL, PAIRS_CELL, HF_CELL,
              code(CE_TEMPLATE % (method, method, kind, model, lr, train, amp))]
    if kind == "moe":
        cells_.append(MOE_GATE_CELL)
    write(nb, f"{nb[:3]} · {method}", rf"""
**Course topics: Transformers, Fine-tuning{', Mixture of Experts' if kind == 'moe' else ''}.**
A cross-encoder reads both records in one input (`record [SEP] S1`), so attention works across the
two strings. Model `{model}`: {desc}
Scores the same candidate pairs as every other matcher (from 03). Output: logits for B / val / test
(used by 07) and the test metrics.
""", "GPU T4", "the dataset and the output of `03_candidates`", cells_)

# =============================================================================== 07 fusion + results
write("07_moe_fusion_and_results", "07 · Mixture of Experts over all matchers + final results", r"""
**Course topic: Mixture of Experts.** The matchers from 04–06 are the **experts**. A gating network
picks, per pair, how much to trust each expert: p = σ(Σᵢ wᵢ(x) · (aᵢ zᵢ + bᵢ)), with w(x) = softmax(gate(x)).
The gate sees *record properties*: script, source, empty address, lengths, retrieval scores.
**It never sees the country label**, so an unseen country is routed by its script and data quality.
Trained on **part B** (pairs that no expert was trained on).

Compared with: the mean of calibrated experts, logistic-regression stacking, MoE with a constant gate
(learned weights that don't depend on the input, an ablation), and LightGBM stacking on the same inputs.

Then it builds the **final results table and charts** from every attached notebook.
""", "None (CPU)", "the dataset, the output of `03_candidates`, and the outputs of every model notebook you ran (04, 05a–c, 06a–f; 01 and 02 optional)", [
    setup_cell(BASE_PKGS + [("lightgbm", "lightgbm"), ("sklearn", "scikit-learn"), ("matplotlib", "matplotlib")]),
    config_cell("load", "EXCLUDE_EXPERTS=['Siamese char-BiRNN (vanilla)'], MOE_EPOCHS=30", ""),
    DATA_CELL, METRIC_CELL, PAIRS_CELL,
    code(r'''
# ==== Collect expert scores from the attached notebooks ====
RES = {}
for p in find_all('results/*.json'):
    r = json.load(open(p)); RES[r['method']] = r
EXPERTS, Z = [], {s: [] for s in ('B', 'val', 'test')}
for m, r in RES.items():
    if m in CFG['EXCLUDE_EXPERTS']: continue
    f = {s: find_one(f'scores/{safe_name(m)}_{s}.npy') for s in ('B', 'val', 'test')}
    if all(f.values()):
        arr = {s: np.asarray(np.load(f[s]), dtype=np.float32).reshape(-1) for s in f}
        if all(len(arr[s]) == len(PAIRS[s]) for s in arr) and all(np.all(np.isfinite(arr[s])) for s in arr):
            EXPERTS.append(m); [Z[s].append(arr[s]) for s in arr]
        else: print('skip (pair count differs or non-finite values, other QUICK_RUN setting?):', m)
print('experts:', EXPERTS)
assert len(EXPERTS) >= 2, 'attach the outputs of at least two model notebooks'
Z = {s: np.stack(v, 1) for s, v in Z.items()}
mu, sd = Z['B'].mean(0), Z['B'].std(0) + 1e-6; Z = {s: (v - mu) / sd for s, v in Z.items()}
''' + GATE_FEAT_SRC + r'''
GF = {s: gate_features(PAIRS[s]) for s in ('B', 'val', 'test')}
gm, gs = GF['B'].mean(0), GF['B'].std(0) + 1e-6; GF = {s: (v - gm) / gs for s, v in GF.items()}
yB = PAIRS['B'].label.values
'''),
    code(r'''
# ==== Fusion methods ====
import torch, torch.nn as nn, torch.nn.functional as F
from sklearn.linear_model import LogisticRegression
import lightgbm as lgb
torch.manual_seed(SEED)
''' + GATED_MOE_SRC + r'''
def fit_moe(input_gate, cal):
    # experts' affine calibration starts from their own logistic fits -> training starts at the "mean of calibrated experts"
    torch.manual_seed(SEED); m = GatedMoE(Z['B'].shape[1], GF['B'].shape[1], input_gate)
    with torch.no_grad():
        m.a.copy_(torch.tensor([c.coef_[0, 0] for c in cal], dtype=torch.float32))
        m.b.copy_(torch.tensor([c.intercept_[0] for c in cal], dtype=torch.float32))
    Zt, Gt, yt = (torch.as_tensor(a, dtype=torch.float32) for a in (Z['B'], GF['B'], yB))
    bs = 2048; per_ep = math.ceil(len(Zt) / bs); steps = max(CFG['MOE_EPOCHS'] * per_ep, 2000)
    opt = torch.optim.Adam(m.parameters(), lr=1e-2, weight_decay=1e-5)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps); st = 0
    while st < steps:
        perm = torch.randperm(len(Zt))
        for a in range(0, len(perm), bs):
            b = perm[a:a + bs]; out, _ = m(Zt[b], Gt[b])
            loss = F.binary_cross_entropy_with_logits(out, yt[b]); opt.zero_grad(); loss.backward(); opt.step(); sch.step()
            st += 1
            if st >= steps: break
    print(f"MoE ({'input' if input_gate else 'constant'} gate): {steps} steps, final batch loss {loss.item():.4f}")
    return m

def moe_predict(m, s):
    with torch.no_grad(): out, w = m(torch.as_tensor(Z[s], dtype=torch.float32), torch.as_tensor(GF[s], dtype=torch.float32))
    return sigmoid(out.numpy()), w.numpy()

cal = [LogisticRegression().fit(Z['B'][:, [i]], yB) for i in range(len(EXPERTS))]
mp = {s: np.mean([c.predict_proba(Z[s][:, [i]])[:, 1] for i, c in enumerate(cal)], 0) for s in ('val', 'test')}
report('8 Fusion', f'Mean of {len(EXPERTS)} calibrated experts', mp['val'], mp['test'])
lr_ = LogisticRegression(max_iter=1000).fit(Z['B'], yB)
report('8 Fusion', 'Logistic-regression stacking', lr_.predict_proba(Z['val'])[:, 1], lr_.predict_proba(Z['test'])[:, 1])
print('LR weights:', dict(zip(EXPERTS, lr_.coef_[0].round(3))))
m0 = fit_moe(False, cal)
report('8 Fusion', 'MoE, constant gate (ablation)', moe_predict(m0, 'val')[0], moe_predict(m0, 'test')[0])
t = time.time(); m1 = fit_moe(True, cal)
pv, _ = moe_predict(m1, 'val'); pt, WT = moe_predict(m1, 'test')
ROW_MOE = report('8 Fusion', 'Gated Mixture of Experts', pv, pt, train_min=round((time.time() - t) / 60, 2), experts=len(EXPERTS))
(OUT / 'models').mkdir(exist_ok=True)                                  # fusion model for 08_demo
torch.save({'experts': EXPERTS, 'state_dict': m1.state_dict(), 'mu': mu.tolist(), 'sd': sd.tolist(), 'gm': gm.tolist(),
            'gs': gs.tolist(), 'n_feat': int(GF['B'].shape[1]), 'threshold': ROW_MOE['threshold']}, OUT / 'models' / 'moe_fusion.pt')
st = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, verbose=-1, random_state=SEED)
st.fit(np.hstack([Z['B'], GF['B']]), yB)
report('8 Fusion', 'LightGBM stacking (same inputs)', st.predict_proba(np.hstack([Z['val'], GF['val']]))[:, 1],
       st.predict_proba(np.hstack([Z['test'], GF['test']]))[:, 1])

short = lambda e: e.replace('CE bge-reranker-base', 'CE').replace(' (string + parsed-address features)', '')
W = pd.DataFrame(WT, columns=[short(e) for e in EXPERTS]); P = PAIRS['test']
W['script'] = REC.script.values[P.qi.values]; W['country'] = REC.country.values[P.qi.values]
GATE_BY_SCRIPT = W.groupby('script').mean(numeric_only=True); GATE_BY_SCRIPT.to_csv(OUT / 'moe_gate_by_script.csv')
GATE_BY_COUNTRY = W.groupby('country').mean(numeric_only=True); GATE_BY_COUNTRY.to_csv(OUT / 'moe_gate_by_country.csv')
print('\nmean gate weight per expert, by script of the S2/S3 record:'); print(GATE_BY_SCRIPT.round(3).T.to_string())
print('\nby country (the gate never sees this label):'); print(GATE_BY_COUNTRY.round(3).T.to_string())
'''),
    code(r'''
# ==== Final results table + charts ====
import matplotlib.pyplot as plt
for p in find_all('results/*.json') + sorted((OUT / 'results').glob('*.json')):
    r = json.load(open(p)); RES[r['method']] = r
R = pd.DataFrame(RES.values())
if R.quick.astype(str).str.lower().eq('true').any(): print('NOTE: some rows come from QUICK_RUN notebooks:', R[R.quick.astype(str).str.lower() == 'true'].method.tolist())
order = {s: i for i, s in enumerate(['5 Classical baseline', '6 RNN / LSTM', '7 Transformers', '8 Fusion'])}
R = R.assign(o=R.section.map(order)).sort_values(['o', 'test_f05'], ascending=[True, False]).drop(columns='o')
cols = [c for c in ['section', 'method', 'val_f05', 'test_f05', 'test_P', 'test_R', 'test_f05_US', 'test_f05_India',
                    'test_singleton_acc', 'params', 'train_min', 'threshold'] if c in R]
print(R[cols].to_string(index=False)); R[cols].to_csv(OUT / 'final_results.csv', index=False)
bl_p = find_one('blocking.json'); bl = json.load(open(bl_p)) if bl_p else {}
if bl:
    print('\nblocking:'); print(pd.DataFrame(bl['rows']).round(4).to_string(index=False))
    print(f"ceiling F0.5: val {bl['val_ceiling_f05']:.5f} | test {bl['test_ceiling_f05']:.5f} | {bl['pairs_per_record_test']:.2f} candidates per record")
with open(OUT / 'final_results.md', 'w') as f:
    f.write('| ' + ' | '.join(cols) + ' |\n|' + '---|' * len(cols) + '\n')
    for _, r in R[cols].iterrows(): f.write('| ' + ' | '.join('' if pd.isna(r[c]) else str(r[c]) for c in cols) + ' |\n')
    if bl: f.write(f"\nBlocking ceiling (test macro F0.5): {bl['test_ceiling_f05']:.5f}\n")
    for name in ('blocking.json', 'tokenizer_stats.csv', 'tokenization_blocking.csv', 'parser_eval.csv'):
        p = find_one(name)
        if p is not None and name.endswith('.csv'):
            d = pd.read_csv(p)
            try: txt = d.to_markdown(index=False)
            except ImportError: txt = d.to_string(index=False)
            f.write(f'\n### {name}\n\n{txt}\n'); print(f'\n{name}:'); print(d.to_string(index=False))

fig, ax = plt.subplots(figsize=(10, 0.45 * len(R) + 1.2))
colors = {'5 Classical baseline': '#8c8c8c', '6 RNN / LSTM': '#4C78A8', '7 Transformers': '#F58518', '8 Fusion': '#54A24B'}
ax.barh(R.method, R.test_f05, color=[colors.get(s, '#999') for s in R.section])
if bl: ax.axvline(bl['test_ceiling_f05'], ls='--', c='k', lw=1, label='blocking ceiling'); ax.legend(loc='lower right')
ax.set_xlim(max(0.0, R.test_f05.min() - 0.02), 1.0); ax.invert_yaxis()
for yy, v in enumerate(R.test_f05): ax.text(v + 0.001, yy, f'{v:.4f}', va='center', fontsize=8)
ax.set_xlabel('test macro F0.5'); ax.set_title('Held-out test split (data/small)')
plt.tight_layout(); plt.savefig(OUT / 'results_test_f05.png', dpi=150); plt.show()
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 4.5))
GATE_BY_SCRIPT.T.plot.bar(ax=ax1, title='MoE gate weight per expert, by script')
ax1.set_ylabel('mean weight'); ax1.grid(axis='y', alpha=0.3)
GATE_BY_COUNTRY.T.plot.bar(ax=ax2, title='MoE gate weight per expert, by country (post-hoc)')
ax2.set_ylabel('mean weight'); ax2.grid(axis='y', alpha=0.3)
plt.tight_layout(); plt.savefig(OUT / 'moe_gates.png', dpi=150); plt.show()
'''),
])

# =============================================================================== 08 live demo
DEMO_LOAD = r'''
# ==== Load every model and output that is attached ====
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
import lightgbm as lgb
try: from torchcrf import CRF
except ImportError: CRF = None
''' + PARSER_SRC + EMBED_SRC + GBM_FEAT_SRC + RNN_CLASS_SRC + CE_CLASS_SRC + GATE_FEAT_SRC + GATED_MOE_SRC + r'''
_ovr = os.environ.get('NLP_ER_MODEL_OVERRIDE')                     # local smoke test only
DEMO = dict(
    TOKENIZERS={'mBERT (WordPiece)': 'google-bert/bert-base-multilingual-cased',
                'IndicBERTv2 (WordPiece)': 'ai4bharat/IndicBERTv2-MLM-only',
                'XLM-R (SentencePiece)': 'BAAI/bge-reranker-base',
                'mDeBERTa (SentencePiece)': 'microsoft/mdeberta-v3-base'},
    E5_FALLBACK=_ovr or _meta['cfg'].get('BI_MODEL', 'intfloat/multilingual-e5-small'),   # zero-shot e5 if 03's fine-tuned e5 is missing
    ZERO_SHOT_CE=_ovr or 'BAAI/bge-reranker-base',     # 06a needs no training, so it is loaded from Hugging Face (None = skip)
    LOAD_CE=True,                                      # False = skip all cross-encoders (e.g. a CPU-only session)
    BI_LEN=_meta['cfg'].get('BI_LEN', 64), EXAMPLES_PER_GROUP=3)
STATUS = []
def status(part, how): STATUS.append({'component': part, 'runs as': how}); print(f'{part:48s} {how}')

IS_TEST = REC.split.values == 'test'
S1_TEST = np.where(IS_TEST & (REC.src.values == 'S1'))[0]                  # the candidate pool: held-out test S1
_tl = find_one('test_labels/test_ground_truth.tsv')
TEST_OWNER = {x: s for s, l in read_gt(_tl).items() for x in l}          # only to show right / wrong
COUNTRIES = sorted(set(REC.country.values[S1_TEST]))
P_COLS = [f'p_{t.lower()}' for t in TAGS if t != 'SEP']

# stored candidate pairs (03) and stored test scores (04-06)
PT = pd.read_parquet(find_one('prep/pairs_test.parquet'))
PT_ROWS = dict(PT.groupby('qi').indices)
RESULTS = {}
for _p in find_all('results/*.json'):
    _r = json.load(open(_p)); RESULTS[_r['method']] = _r
STORED = {}
for _m in RESULTS:
    _f = find_one(f'scores/{safe_name(_m)}_test.npy')
    if _f is None: continue
    _a = np.load(_f)
    if len(_a) == len(PT): STORED[_m] = _a
    else: print('skip stored scores of', _m, '(pair count differs: another QUICK_RUN setting?)')

# 02 parser
PARSER = None
_p = find_one('models/parser_bilstm_crf.pt')
if _p is not None and CRF is not None:
    _ck = torch.load(_p, map_location='cpu', weights_only=False); assert _ck['TAGS'] == TAGS
    W2I, C2I = _ck['W2I'], _ck['C2I']
    PARSER = BiLSTMCRF(len(W2I) + 2, len(C2I) + 2, len(TAGS)); PARSER.load_state_dict(_ck['state_dict']); PARSER.to(DEVICE).eval()
status('Address parser: BiLSTM-CRF (02)', 'live' if PARSER is not None else 'weights not attached -> rule parser only')
_pf = find_one('prep/parsed_fields.parquet')
PF = pd.read_parquet(_pf).set_index('id') if _pf is not None else None

# 01 tokenizers
TOKS = {}
for _l, _n in DEMO['TOKENIZERS'].items():
    try: TOKS[_l] = AutoTokenizer.from_pretrained(_ovr or _n)
    except Exception as e: print(f'skip tokenizer {_l}: {type(e).__name__}')
status('Tokenizers (01)', ', '.join(TOKS) or 'none loaded')

# 03 retrieval: TF-IDF is refitted exactly as in 03 (deterministic); e5 from 03's weights
VEC = TfidfVectorizer(analyzer='char_wb', ngram_range=(3, 3), sublinear_tf=True, min_df=2, dtype=np.float32)
VEC.fit(REC.bl_text.values[REC.split.values == 'train'])
S1M = VEC.transform(REC.bl_text.values[S1_TEST]); C3 = VEC.build_analyzer()
status('TF-IDF char 3-gram (03)', f'live ({len(VEC.vocabulary_):,} 3-grams, fitted on train)')
_e = find_one('models/e5_finetuned/config.json'); E5_FT = _e is not None
_n = str(_e.parent) if E5_FT else DEMO['E5_FALLBACK']
TOK_E5 = AutoTokenizer.from_pretrained(_n); E5 = AutoModel.from_pretrained(_n).float().to(DEVICE).eval()
ES = embed(E5, TOK_E5, list(REC.ce_text.values[S1_TEST]), 'passage: ', max_len=DEMO['BI_LEN'])
status('Bi-encoder e5-small (03)', 'live, fine-tuned' if E5_FT else
       'fine-tuned weights not attached -> zero-shot e5 (test records still use the stored 03 candidates)')

# 04-06 matchers
LIVE = {}
GBM = GBM_META = None
_g = find_one('models/gbm.txt')
if _g is not None:
    GBM = lgb.Booster(model_file=str(_g)); GBM_META = json.load(open(_g.parent / 'gbm_meta.json'))
    LIVE[GBM_META['method']] = dict(type='gbm')
GBM_PARSE = 'rules' if (GBM_META and GBM_META['parse_src'] == 'rules') else 'bilstm'
for _p in find_all('models/*.rnn.pt'):
    _ck = torch.load(_p, map_location='cpu', weights_only=False)
    _m = SiameseRNN(len(_ck['ch2i']) + 2, _ck['cell']); _m.load_state_dict(_ck['state_dict'])
    LIVE[_ck['method']] = dict(type='rnn', model=_m.to(DEVICE).eval(), ch2i=_ck['ch2i'], char_max=_ck['char_max'])
for _p in (find_all('models/*.ce.pt') if DEMO['LOAD_CE'] else []):
    _ck = torch.load(_p, map_location='cpu', weights_only=False)
    _m = build_ce(_ck['kind'], _ck['model'])
    if _ck['trained']:
        _own = dict(_m.named_parameters()); _bad = [k for k in _ck['trained'] if k not in _own]
        assert not _bad, f"{_ck['method']}: {len(_bad)} saved weights do not fit the model, e.g. {_bad[:3]}"
        with torch.no_grad():
            for k, v in _ck['trained'].items(): _own[k].copy_(v.to(_own[k].dtype))
    LIVE[_ck['method']] = dict(type='ce', model=_m.to(DEVICE).eval(), kind=_ck['kind'], max_len=_ck['max_len'], amp=_ck['amp'],
                               tok=AutoTokenizer.from_pretrained(_ck['model']))
    del _ck; gc.collect()
ZS = 'CE bge-reranker-base zero-shot'
if DEMO['LOAD_CE'] and DEMO['ZERO_SHOT_CE'] and ZS not in LIVE:
    LIVE[ZS] = dict(type='ce', model=build_ce('seq', DEMO['ZERO_SHOT_CE']).to(DEVICE).eval(), kind='seq', max_len=CFG['MAX_LEN'],
                    amp=True, tok=AutoTokenizer.from_pretrained(DEMO['ZERO_SHOT_CE']))
SEC = {m: r.get('section', '') for m, r in RESULTS.items()}
MATCHERS = sorted({m for m in list(LIVE) + list(STORED) if not SEC.get(m, '').startswith('8')},
                  key=lambda m: (SEC.get(m, '7'), -RESULTS.get(m, {}).get('test_f05', 0)))
THR = {m: RESULTS[m]['threshold'] if m in RESULTS else 0.5 for m in MATCHERS}
for _m in MATCHERS:
    status(_m, 'live' if _m in LIVE else 'weights not attached -> stored test scores (test records only)')

# 07 fusion
FUSION = None
_f = find_one('models/moe_fusion.pt')
if _f is not None:
    _ck = torch.load(_f, map_location='cpu', weights_only=False)
    _m = GatedMoE(len(_ck['experts']), _ck['n_feat']); _m.load_state_dict(_ck['state_dict']); _m.eval()
    FUSION = dict(model=_m, experts=_ck['experts'], thr=_ck['threshold'], **{k: np.array(_ck[k], np.float32) for k in ('mu', 'sd', 'gm', 'gs')})
status('Gated Mixture of Experts (07)', 'live' if FUSION else 'not attached -> the best single matcher decides')
log('demo ready')
'''

DEMO_PIPE = r'''
# ==== The pipeline for one record ====
import html as _html
SHORT = [('CE IndicBERTv2 full fine-tune', 'IndicBERTv2'), ('CE mDeBERTa-v3 full fine-tune (fp32)', 'mDeBERTa'),
         ('GBM (string + parsed-address features)', 'LightGBM'), ('Siamese char-', ''), (' (vanilla)', ''),
         ('CE bge-reranker-base ', 'bge '), ('full fine-tune', 'full FT'), ('+ MoE head (top-2 of 4)', 'MoE head'), ('LoRA r=16', 'LoRA')]
def short(m):
    for a, b in SHORT: m = m.replace(a, b)
    return m
TOPIC = {'5': 'classical', '6': 'RNN / LSTM', '7': 'Transformer'}

class use_rec:                       # run the shared feature code on a small frame: [query, candidates...]
    def __init__(self, R): self.R = R
    def __enter__(self): global REC; self.old = REC; REC = self.R
    def __exit__(self, *a): global REC; REC = self.old

def query_typed(name, addr, country, src):
    R = pd.DataFrame([{'id': 'typed-record', 'name': name.strip(), 'addr': addr.strip(), 'country': country, 'split': 'typed', 'src': src}])
    R['script'] = [script_of(R.name[0] + ' ' + R.addr[0])]
    return add_text_cols(R)

def parse(addr):
    toks = addr_tokens(addr); flat = [t for t in toks if t != ',']
    out = dict(toks=toks, flat=flat, rule=rule_tags(toks), rule_flat=rule_tags(flat), lstm=None, lstm_flat=None)
    if PARSER is not None: out['lstm'], out['lstm_flat'] = parser_decode(PARSER, [toks, flat])
    return out

def fields_for(rid, addr):           # the parsed fields the GBM sees (same source as in 04)
    if GBM_PARSE != 'rules' and PF is not None and rid in PF.index: return {c: PF.at[rid, c] for c in P_COLS}
    toks = addr_tokens(addr)
    tags = parser_decode(PARSER, [toks])[0] if (GBM_PARSE != 'rules' and PARSER is not None and toks) else rule_tags(toks)
    f = fields_from(toks, tags); return {c: f.get(c[2:].upper(), '') for c in P_COLS}

def retrieve(q):
    pool = np.where(REC.country.values[S1_TEST] == q.country.values[0])[0]
    if len(pool) == 0: return None
    tf = np.full(len(S1_TEST), -1.0, np.float32); de = np.full(len(S1_TEST), -1.0, np.float32)
    tf[pool] = (VEC.transform(q.bl_text.values) @ S1M[pool].T).toarray()[0]
    eq = embed(E5, TOK_E5, [q.ce_text.values[0]], 'query: ', max_len=DEMO['BI_LEN'])[0].float()
    de[pool] = (ES[pool].float() @ eq).numpy()
    kt, kd, kf = CFG['K_TFIDF'], CFG['K_DENSE'], CFG['K_FINAL']
    tf_top = pool[np.argsort(-tf[pool], kind='stable')[:kt]]; de_top = pool[np.argsort(-de[pool], kind='stable')[:kd]]
    tr = {p: r for r, p in enumerate(tf_top)}; dr = {p: r for r, p in enumerate(de_top)}
    rrf = {p: 1 / (60 + tr.get(p, 1e6)) + 1 / (60 + dr.get(p, 1e6)) for p in set(tr) | set(dr)}     # reciprocal-rank fusion
    fused = sorted(rrf, key=lambda p: -rrf[p])[:kf]
    P = pd.DataFrame({'si_rec': S1_TEST[fused], 'tf_rank': [float(tr.get(p, kt)) for p in fused], 'de_rank': [float(dr.get(p, kd)) for p in fused],
                      'rrf': [rrf[p] for p in fused], 'rrf_rank': np.arange(len(fused)), 'tf_cos': tf[fused], 'de_cos': de[fused]})
    P['n_cand'] = len(P)
    return dict(tf_top=S1_TEST[tf_top], tf_v=tf[tf_top], de_top=S1_TEST[de_top], de_v=de[de_top], P=P)

def stored_P(i):
    rows = PT_ROWS.get(i)
    if rows is None: return None, None
    rows = rows[np.argsort(PT.rrf_rank.values[rows])]; d = PT.iloc[rows]
    P = pd.DataFrame({'si_rec': d.si.values, **{c: d[c].values for c in ('tf_rank', 'de_rank', 'rrf', 'rrf_rank', 'tf_cos', 'de_cos', 'n_cand')}})
    return P, rows

def build_sub(q, P):
    S = pd.concat([q, REC.iloc[P.si_rec.values]], ignore_index=True)
    F_ = [fields_for(r, a) for r, a in zip(S.id.values, S.addr.values)]
    for c in P_COLS: S[c] = [f[c] for f in F_]
    S = add_gbm_cols(S); Ps = P.copy(); Ps['qi'] = 0; Ps['si'] = np.arange(1, len(P) + 1)
    return S, Ps

def logit_np(p): p = np.clip(p, 1e-6, 1 - 1e-6); return np.log(p / (1 - p))

@torch.no_grad()
def live_logits(m, S, Ps):
    L = LIVE[m]
    if L['type'] == 'gbm':
        with use_rec(S): X = pair_features(Ps)
        return logit_np(GBM.predict(X[GBM_META['features']])), None
    if L['type'] == 'rnn':
        def ids(texts):
            X = np.zeros((len(texts), L['char_max']), np.int64)
            for r, t in enumerate(texts): X[r, :min(len(t), L['char_max'])] = [L['ch2i'].get(ch, 1) for ch in t[:L['char_max']]]
            n = np.maximum((X > 0).sum(1), 1); return torch.as_tensor(X[:, :n.max()], device=DEVICE), torch.as_tensor(n)
        u = L['model'].encode(*ids(S.ltext.values[:1])); v = L['model'].encode(*ids(S.ltext.values[1:]))
        return L['model'].pair(u.expand(len(v), -1), v).float().cpu().numpy(), None
    enc = L['tok']([S.ce_text.values[0]] * (len(S) - 1), list(S.ce_text.values[1:]), truncation=True, max_length=L['max_len'],
                   padding=True, return_tensors='pt').to(DEVICE)
    with autocast(L['amp']):
        if L['kind'] == 'moe':
            y, _, probs = L['model'](**enc); return y.float().cpu().numpy(), probs.float().cpu().numpy()
        return L['model'](**enc).logits[:, 0].float().cpu().numpy(), None

def score_all(S, Ps, rows):
    out = {}
    for m in MATCHERS:
        st = STORED[m][rows] if (rows is not None and m in STORED) else None
        if m in LIVE:
            z, extra = live_logits(m, S, Ps); out[m] = dict(z=z, src='live', extra=extra, stored=st)
        elif st is not None:
            out[m] = dict(z=st, src='stored', extra=None, stored=None)
    return out

def fuse(S, Ps, sc):
    if FUSION is None: return None
    miss = [e for e in FUSION['experts'] if e not in sc]
    if miss: return dict(missing=miss)
    Z = (np.stack([sc[e]['z'] for e in FUSION['experts']], 1) - FUSION['mu']) / FUSION['sd']
    with use_rec(S): G = gate_features(Ps)
    with torch.no_grad():
        o, w = FUSION['model'](torch.as_tensor(Z, dtype=torch.float32), torch.as_tensor((G - FUSION['gm']) / FUSION['gs'], dtype=torch.float32))
    return dict(p=sigmoid(o.numpy()), w=w.numpy(), G=G)

def decide(p, thr):                  # the decision rule of every method: best candidate, if p >= threshold
    j = int(np.argmax(p)); return j if p[j] >= thr else None

def run(i=None, typed=None):
    t0 = time.time()
    q = REC.iloc[[i]].reset_index(drop=True) if i is not None else query_typed(*typed)
    ret = retrieve(q); sp, rows = stored_P(i) if i is not None else (None, None)
    P = sp if sp is not None else (ret['P'] if ret else None)
    R = dict(q=q, i=i, ret=ret, P=P, rows=rows, parse=parse(q.addr.values[0]),
             truth=TEST_OWNER.get(q.id.values[0]) if i is not None else None, known=i is not None)
    if P is not None and len(P):
        R['S'], R['Ps'] = build_sub(q, P); R['sc'] = score_all(R['S'], R['Ps'], rows); R['fu'] = fuse(R['S'], R['Ps'], R['sc'])
    R['sec'] = time.time() - t0
    return R
'''

DEMO_RENDER = r'''
# ==== HTML for the five steps ====
CSS = """
.er{font-family:system-ui,-apple-system,'Segoe UI',Roboto,sans-serif;font-size:14px;line-height:1.45;color:#1f2328;background:#ffffff;
    padding:12px 14px;border-radius:8px;border:1px solid #d0d7de}
.er h3{margin:14px 0 6px;font-size:16px;color:#1f2328} .er h4{margin:10px 0 4px;font-size:14px;color:#1f2328}
.er table{border-collapse:collapse;margin:6px 0 12px;font-size:13px}
.er th,.er td{border:1px solid #d0d7de;padding:4px 8px;text-align:left;vertical-align:top;color:#1f2328}
.er th{background:#f6f8fa}
.er .tok{display:inline-block;margin:2px;padding:1px 6px;border-radius:4px;background:#eef1f4;border:1px solid #d0d7de;
         font-family:ui-monospace,Menlo,Consolas,monospace;font-size:12.5px;color:#1f2328}
.er .unk{background:#ffd8d8;border-color:#e5534b} .er .cont{background:#e6f0ff;border-color:#9ec5fe}
.er .chg{background:#fff1b8;border-color:#d4a72c} .er .oov{color:#8c959f;text-decoration:line-through}
.er .muted{color:#57606a;font-size:12px} .er .ok{color:#1a7f37;font-weight:600} .er .bad{color:#cf222e;font-weight:600}
.er .bar{display:inline-block;height:11px;background:#54A24B;vertical-align:middle;border-radius:2px}
.er .banner{font-size:15px;padding:8px 10px;border-radius:6px;background:#f6f8fa;border:1px solid #d0d7de}
"""
TAG_COL = {'HOUSE': '#ffd8a8', 'STREET': '#c3fae8', 'LOCALITY': '#d0ebff', 'LANDMARK': '#eebefa', 'CITY': '#b2f2bb',
           'STATE': '#ffec99', 'POSTCODE': '#ffc9c9', 'SEP': '#e9ecef'}
def esc(x): return _html.escape(str(x))
def chip(t, cls='', style='', title=''): return f'<span class="tok {cls}" style="{style}" title="{esc(title)}">{esc(t)}</span>'
def box(body): return f'<div class="er">{body}</div>'
def tbl(head, rows):
    return ('<table><tr>' + ''.join(f'<th>{h}</th>' for h in head) + '</tr>'
            + ''.join('<tr>' + ''.join(f'<td>{c}</td>' for c in r) + '</tr>' for r in rows) + '</table>')
def bar(v, w=120): return f'<span class="bar" style="width:{max(1, int(w * float(v)))}px"></span> {float(v):.2f}'
def pcell(p, picked=False):
    a = 0.12 + 0.6 * float(p)
    return f'<span style="background:rgba(84,162,75,{a:.2f});padding:1px 6px;border-radius:3px">{float(p):.3f}{" ★" if picked else ""}</span>'
def s1_label(ix): return f'<b>{esc(REC.name.values[ix])}</b><br><span class="muted">{esc(REC.addr.values[ix])} · {esc(REC.id.values[ix])}</span>'
def truth_mark(R, ix): return ' <span class="ok">✓ true S1</span>' if R['known'] and REC.id.values[ix] == R['truth'] else ''

def norm_steps(s):
    st = [('raw', s.split(), None)]
    t = unicodedata.normalize('NFKC', s)
    if t != s: st.append(('Unicode NFKC', t.split(), None))
    if not t.isascii(): t = romanise(t); st.append(('romanised (anyascii + Malayalam / anusvara fixes)', t.split(), None))
    toks = _PUNCT.sub(' ', t.lower().replace('&', ' and ')).split(); st.append(('lower-case, punctuation removed', toks, None))
    if not s.isascii():
        new = [LEX.get(x, x) for x in toks]; st.append(('learned transliteration lexicon', new, [a != b for a, b in zip(toks, new)])); toks = new
    new = [ABBR.get(x, x) for x in toks]; st.append(('abbreviations expanded', new, [a != b for a, b in zip(toks, new)]))
    return st

def tok_chips(tok, text):
    pieces = tok.tokenize(text)[:60]; sp = any(p.startswith('▁') for p in pieces); out = []
    for p in pieces:
        cls = 'unk' if p == tok.unk_token else ('cont' if ((not p.startswith('▁')) if sp else p.startswith('##')) else '')
        out.append(chip(p, cls))
    return ''.join(out), len(pieces)

def header(R):
    q = R['q']; r = q.iloc[0]
    return (f'<div class="banner"><b>{esc(r["name"])}</b><br>{esc(r["addr"]) or "<i>(no address)</i>"}<br>'
            f'<span class="muted">source {esc(r["src"])} · country {esc(r["country"])} · script {esc(r["script"])} · id {esc(r["id"])}</span></div>')

def panel_tokens(R):
    q = R['q'].iloc[0]; h = header(R)
    for field, s in (('Name', q['name']), ('Address', q['addr'])):
        h += f'<h3>Normalisation of the {field.lower()}</h3>'
        rows = [[esc(st), ''.join(chip(t, 'chg' if (chg and chg[k]) else '') for k, t in enumerate(toks)) or '<i>empty</i>']
                for st, toks, chg in norm_steps(s)]
        h += tbl(['step', 'tokens (yellow = changed by this step)'], rows)
    h += '<h3>How the tokenizers split the name</h3><p class="muted">blue = continuation piece (word was split), red = unknown token</p>'
    rows = []
    for label, tok in TOKS.items():
        for kind, text in (('raw', q['name']), ('normalised', q['nn'])):
            c, n = tok_chips(tok, text); rows.append([esc(label), kind, c, f'{n} / {max(1, len(text.split()))} = {n / max(1, len(text.split())):.2f}'])
    grams = C3(q['nn'])[:48]
    rows.append(['char 3-grams (TF-IDF)', 'normalised', ''.join(chip(g.replace(' ', '␣'), '' if g in VEC.vocabulary_ else 'oov') for g in grams),
                 f'{len(C3(q["nn"]))} grams; grey = not in the TF-IDF vocabulary'])
    h += tbl(['tokenizer', 'input', 'pieces', 'tokens / words'], rows)
    h += '<h3>What the models read</h3>' + tbl(['model', 'input text'], [
        ['TF-IDF', esc(q['bl_text'])], ['e5 bi-encoder', esc('query: ' + q['ce_text'])],
        ['cross-encoders', esc(q['ce_text']) + ' <span class="muted">[SEP] candidate S1</span>'], ['Siamese RNNs (characters)', esc(q['ltext'])]])
    return box(h)

def tag_chips(toks, tags):
    if tags is None: return '<i>parser weights not attached</i>'
    return ''.join(chip(t, style=f'background:{TAG_COL.get(g, "#eee")}', title=g) for t, g in zip(toks, tags)) or '<i>empty</i>'

def panel_parse(R):
    p = R['parse']; h = header(R)
    h += '<h3>Address parsing: rules vs BiLSTM-CRF</h3><p>' + ''.join(chip(t, style=f'background:{c}') for t, c in TAG_COL.items() if t != 'SEP') + '</p>'
    h += tbl(['input', 'parser', 'tagged tokens (hover for the tag)'], [
        ['with commas', 'rules', tag_chips(p['toks'], p['rule'])], ['with commas', 'BiLSTM-CRF (02)', tag_chips(p['toks'], p['lstm'])],
        ['commas removed', 'rules', tag_chips(p['flat'], p['rule_flat'])], ['commas removed', 'BiLSTM-CRF (02)', tag_chips(p['flat'], p['lstm_flat'])]])
    h += ('<p class="muted">The rules depend on commas; without them they fall back to one component. The BiLSTM-CRF was trained with '
          'half of the addresses stripped of commas, so it reads the structure from the words.</p>')
    if 'S' in R:
        S = R['S']; j = pick_final(R)[0]; j = 0 if j is None else j
        rows = [[c[2:].upper(), esc(S[c].values[0]) or '–', esc(S[c].values[1 + j]) or '–',
                 ('<span class="ok">same</span>' if S[c].values[0] == S[c].values[1 + j] else '<span class="bad">differs</span>')
                 if S[c].values[0] and S[c].values[1 + j] else '']
                for c in P_COLS]
        h += (f'<h3>Parsed fields: this record vs the chosen candidate</h3>'
              + tbl(['field', 'this record', esc(S.name.values[1 + j]), 'agree?'], rows)
              + '<p class="muted">These fields become LightGBM features (house-number and postcode agreement, city / street similarity).</p>')
    return box(h)

def panel_retrieve(R):
    h = header(R); ret = R['ret']
    if ret is None: return box(h + '<p>No S1 record with this country label in the test split.</p>')
    known = R['known']
    for title, ix, v in ((f"TF-IDF char 3-gram: top {CFG['K_TFIDF']}", ret['tf_top'], ret['tf_v']),
                         (f"e5-small bi-encoder ({'fine-tuned' if E5_FT else 'zero-shot'}): top {CFG['K_DENSE']}", ret['de_top'], ret['de_v'])):
        h += f'<h3>{title}</h3>' + tbl(['rank', 'S1 record', 'cosine'], [[k + 1, s1_label(s) + truth_mark(R, s), f'{c:.3f}'] for k, (s, c) in enumerate(zip(ix, v))])
    P = ret['P']
    h += f"<h3>Reciprocal-rank fusion → the {CFG['K_FINAL']} candidates every matcher scores</h3>"
    h += tbl(['#', 'S1 record', 'TF-IDF rank', 'e5 rank', 'RRF score'],
             [[k + 1, s1_label(s) + truth_mark(R, s), '–' if a >= CFG['K_TFIDF'] else int(a) + 1, '–' if b >= CFG['K_DENSE'] else int(b) + 1, f'{c:.4f}']
              for k, (s, a, b, c) in enumerate(zip(P.si_rec, P.tf_rank, P.de_rank, P.rrf))])
    if known:
        st = R['P'] if R['rows'] is not None else None
        same = st is not None and set(st.si_rec) == set(P.si_rec)
        h += f'<p class="muted">Stored candidates from 03 for this record: {"identical ✓" if same else "differ (the stored set from 03 is scored below)"}.</p>'
        if R['truth'] is None: h += '<p><b>This record has no true S1</b>: the right answer is "no match".</p>'
        elif R['truth'] not in set(REC.id.values[R['P'].si_rec.values]): h += '<p class="bad">Blocking miss: the true S1 is not among the candidates.</p>'
    return box(h)

def pick_final(R):
    # final decision: gated MoE if available, else the best single matcher (highest test F0.5)
    fu = R.get('fu')
    if fu and 'p' in fu: return decide(fu['p'], FUSION['thr']), fu['p'], 'Gated Mixture of Experts (07)', FUSION['thr']
    if R.get('sc'):
        m = max(R['sc'], key=lambda m: RESULTS.get(m, {}).get('test_f05', 0)); p = sigmoid(R['sc'][m]['z'])
        return decide(p, THR[m]), p, m, THR[m]
    return None, None, None, None

def verdict(R, j):
    if not R['known']: return ''
    pred = REC.id.values[R['P'].si_rec.values[j]] if j is not None else None
    if pred == R['truth']: return '<span class="ok">✓ correct</span>'
    return '<span class="bad">✗ wrong</span>' + (' (should be: no match)' if R['truth'] is None else '')

def panel_scores(R):
    h = header(R)
    if 'sc' not in R: return box(h + '<p>No candidates.</p>')
    P, sc = R['P'], R['sc']; ms = [m for m in MATCHERS if m in sc]
    probs = {m: sigmoid(sc[m]['z']) for m in ms}; picks = {m: decide(probs[m], THR[m]) for m in ms}
    head = ['#', 'S1 candidate'] + [f'{esc(short(m))}<br><span class="muted">{TOPIC.get(SEC.get(m, "7")[:1], "")}</span>' for m in ms]
    rows = [[k + 1, s1_label(s) + truth_mark(R, s)] + [pcell(probs[m][k], picks[m] == k) for m in ms] for k, s in enumerate(P.si_rec)]
    rows.append(['', '<b>threshold</b> (tuned on val)'] + [f'{THR[m]:.2f}' for m in ms])
    rows.append(['', '<b>decision</b>'] + [('#' + str(picks[m] + 1) if picks[m] is not None else 'no match') + ' ' + verdict(R, picks[m]) for m in ms])
    rows.append(['', 'runs'] + [sc[m]['src'] + (f'<br><span class="muted">|Δ| vs stored {np.abs(sc[m]["z"] - sc[m]["stored"]).max():.3f}</span>'
                                               if sc[m].get('stored') is not None else '') for m in ms])
    rows.append(['', 'test F0.5 (whole split)'] + [f"{RESULTS[m]['test_f05']:.4f}" if m in RESULTS else '–' for m in ms])
    h += '<h3>Match probability of every candidate, per model</h3>' + tbl(head, rows)
    h += '<p class="muted">★ = the model\'s pick: its best candidate, if p ≥ its threshold. "stored" = scores from the notebook run (model weights not attached).</p>'
    moe = [m for m in ms if sc[m].get('extra') is not None]
    for m in moe:
        g = sc[m]['extra']
        h += (f'<h3>Inside {esc(short(m))}: gate of the MoE head (4 experts, top 2 used)</h3>'
              + tbl(['#', 'S1 candidate'] + [f'expert {e}' for e in range(g.shape[1])],
                    [[k + 1, esc(REC.name.values[s])] + [bar(x, 80) for x in g[k]] for k, s in enumerate(P.si_rec)]))
    return box(h)

GATE_NAMES = [f'script = {s}' for s in SCRIPTS] + ['source S3', 'address empty', 'address has a number', 'name length / 40',
                                                    'address length / 80', 'TF-IDF cosine', 'e5 cosine', 'fused rank / K', 'candidates / K']
def panel_fusion(R):
    h = header(R)
    if 'sc' not in R: return box(h + '<p>No candidates.</p>')
    fu = R.get('fu'); P = R['P']
    if FUSION is None: h += '<p>The fusion model of 07 is not attached. The best single matcher decides instead.</p>'
    elif 'missing' in fu: h += '<p>Fusion needs every expert it was trained with. Missing here: ' + esc(', '.join(fu['missing'])) + '</p>'
    else:
        ex = FUSION['experts']; j = decide(fu['p'], FUSION['thr']); k0 = int(np.argmax(fu['p']))
        h += ('<h3>Gated Mixture of Experts over the matchers</h3><p class="muted">p = σ(Σᵢ wᵢ(x)·(aᵢ zᵢ + bᵢ) + c). '
              'The gate w(x) sees properties of the record and of the retrieval, never the country label.</p>')
        h += tbl(['#', 'S1 candidate', 'fused p'] + [esc(short(e)) for e in ex],
                 [[k + 1, s1_label(s) + truth_mark(R, s), pcell(fu['p'][k], j == k)] + [bar(w, 60) for w in fu['w'][k]] for k, s in enumerate(P.si_rec)])
        h += '<p class="muted">Bars = gate weight of each expert for that pair (each row sums to 1).</p>'
        h += '<h3>What the gate saw (best candidate)</h3>' + tbl(['gate input', 'value'], [[esc(n), f'{v:.3f}'] for n, v in zip(GATE_NAMES, fu['G'][k0])])
    h += final_html(R)
    return box(h)

def final_html(R):
    j, p, by, thr = pick_final(R)
    if p is None: return '<h3>Decision</h3><p>No candidates, so: no match.</p>'
    k0 = int(np.argmax(p)); s = R['P'].si_rec.values[k0]
    if j is None: txt = f'<b>No match</b>: the best candidate ({esc(REC.name.values[s])}) has p = {p[k0]:.3f} < threshold {thr:.2f}, so the record is treated as a new entity.'
    else: txt = f'<b>Match</b> → {s1_label(s)}<br>p = {p[j]:.3f} ≥ threshold {thr:.2f}'
    tr = ''
    if R['known']:
        tr = ('<br>Ground truth: ' + ('no true S1 (distractor)' if R['truth'] is None else
              esc(REC.name.values[ID2IX[R['truth']]]) + f' <span class="muted">{esc(R["truth"])}</span>') + ' → ' + verdict(R, j))
    return f'<h3>Decision</h3><div class="banner">{txt}<br><span class="muted">decided by {esc(by)}</span>{tr}</div>'

def render(R):
    top = box(header(R) + final_html(R) + f'<p class="muted">pipeline time {R["sec"]:.2f} s on {DEVICE}</p>')
    return [top, panel_tokens(R), panel_parse(R), panel_retrieve(R), panel_scores(R), panel_fusion(R)]

def safe_render(**kw):
    try: return render(run(**kw))
    except Exception as e:
        import traceback; return [box(f'<pre>{esc(traceback.format_exc())}</pre>')] + [''] * 5
'''

DEMO_EXAMPLES = r'''
# ==== Example records from the held-out test split ====
def pick_examples(n=DEMO['EXAMPLES_PER_GROUP']):
    d = PT[['qi', 'q_id', 's_id', 'si', 'tf_rank', 'rrf_rank']].copy()
    d['true'] = [TEST_OWNER.get(a) == b for a, b in zip(d.q_id, d.s_id)]
    q = pd.DataFrame({'qi': list(PT_ROWS)}); q['q_id'] = REC.id.values[q.qi]
    for c in ('script', 'country', 'addr', 'name', 'src'): q[c] = REC[c].values[q.qi]
    q['linked'] = q.q_id.isin(set(TEST_OWNER))
    t = d[d.true].drop_duplicates('qi').set_index('qi'); q['found'] = q.qi.isin(t.index)
    q['t_tf'] = q.qi.map(t.tf_rank); q['t_rrf'] = q.qi.map(t.rrf_rank)
    d['core'] = REC.ncore.values[d.si.values]; d['tcore'] = d.qi.map(t.si.map(lambda s: REC.ncore.values[s]))
    sib = set(d.qi[(~d.true) & (d.core == d.tcore)])
    q['ok'] = True
    if STORED:                                          # prefer records the best stored matcher gets right (except "hard")
        best = max(STORED, key=lambda m: RESULTS[m]['test_f05']); d['p'] = sigmoid(STORED[best])
        top = d.sort_values(['qi', 'p'], ascending=[True, False]).drop_duplicates('qi').set_index('qi')
        pred = pd.Series(np.where(top.p >= THR.get(best, 0.5), top.s_id, ''), index=top.index)
        q['ok'] = q.qi.map(pred).values == q.q_id.map(TEST_OWNER).fillna('').values
    nw = q.addr.str.split().str.len()
    G = {'Easy · US, Latin script': q.found & (q.country == 'US') & (q.script == 'latin') & (q.t_tf == 0) & (q.t_rrf == 0),
         'Devanagari script': q.found & (q.script == 'devanagari'),
         'Malayalam script': q.found & (q.script == 'malayalam'),
         'TF-IDF ranks the true S1 lower': q.found & (q.t_tf > 0),
         'Sibling branch among the candidates': q.found & q.qi.isin(sib),
         'Address without commas': q.found & ~q.addr.str.contains(',') & (nw >= 5),
         'No true match (distractor)': ~q.linked}
    if STORED: G[f'Hard · {short(best)} gets it wrong'] = ~q.ok
    EX = {}
    for g, m in G.items():
        pool = q[m & q.ok] if (STORED and not g.startswith('Hard') and (m & q.ok).sum() >= n) else q[m]
        for _, r in pool.sample(min(n, len(pool)), random_state=SEED).iterrows():
            EX[f'{g} · {r["name"][:45]} ({r["src"]}, {r["country"]})'] = int(r.qi)
    return EX
EXAMPLES = pick_examples()
print(len(EXAMPLES), 'examples'); print('\n'.join(list(EXAMPLES)[:30]))
'''

DEMO_INLINE = r'''
# ==== Fallback without Gradio: render one record right here in the notebook ====
try: from IPython.display import HTML, display
except ImportError: HTML, display = str, print                 # outside Jupyter
def show(label=None, i=None, typed=None):
    if label is not None: i = EXAMPLES[label]
    parts = safe_render(i=i, typed=typed)
    titles = ['', '1 · Normalise + tokenize', '2 · Parse the address', '3 · Retrieve candidates', '4 · Score (every matcher)', '5 · Fuse (MoE) + decide']
    display(HTML('<style>' + CSS + '</style>' + ''.join((f'<h2>{t}</h2>' if t else '') + p for t, p in zip(titles, parts))))

show(list(EXAMPLES)[0])
# show(typed=('Sri Lakshmi Traders', 'Shop No 4, MG Road, Kochi, Kerala 682035', 'India', 'S2'))
'''

DEMO_CURATED_MD = r'''
## Four curated demo records (Issue #15)

The four records below are selected deterministically (seed 42) from the held-out
test split to illustrate different aspects of the task:

| # | Category | Purpose |
|---|---|---|
| 1 | Easy · US, Latin script | True S1 is TF-IDF rank 1; all models agree. |
| 2 | Devanagari or Malayalam script | Tests romanisation, transliteration lexicon, IndicBERTv2. |
| 3 | Sibling branch / distractor | Strong hard-negative with the same core name appears among candidates. |
| 4 | Failure case | The best single matcher is wrong; shows what the MoE gating does. |
'''

DEMO_CURATED_CODE = r'''
# ==== Four curated demo records (Issue #15, Task 3) ====
# Deterministic selection from EXAMPLES; groups come from pick_examples().
# Falls back gracefully if a category has no examples in this dataset attachment.

def pick_demo4():
    'Return ordered dict {label: qi} for the four presentation records.'
    hard_keys = [k for k in EXAMPLES if k.startswith('Hard')]
    slots = [
        ('Easy US Latin',      ['Easy · US, Latin script']),
        ('Indic script',       ['Devanagari script', 'Malayalam script']),
        ('Sibling/distractor', ['Sibling branch among the candidates',
                                'No true match (distractor)']),
        ('Failure case',       hard_keys if hard_keys else ['TF-IDF ranks the true S1 lower']),
    ]
    demo4 = {}; used = set()
    for slot_name, groups in slots:
        matched = False
        for g in groups:
            cands = [(k, v) for k, v in EXAMPLES.items() if k.startswith(g) and v not in used]
            if cands:
                label, qi = cands[0]
                demo4[f'{slot_name}: {label}'] = qi
                used.add(qi); matched = True; break
        if not matched:
            remaining = [(k, v) for k, v in EXAMPLES.items() if v not in used]
            if remaining:
                label, qi = remaining[0]
                demo4[f'{slot_name} (fallback): {label}'] = qi; used.add(qi)
    return demo4

DEMO4 = pick_demo4()
print('=== Four demo records ===')
for idx, (label, qi) in enumerate(DEMO4.items(), 1):
    r = REC.iloc[qi]
    tid = TEST_OWNER.get(r.id)
    tname = REC.name.values[ID2IX[tid]] if tid and tid in ID2IX.index else '(no true S1)'
    print(f'[{idx}] {label}')
    print(f'    name   : {r["name"]}')
    print(f'    addr   : {r["addr"]}')
    print(f'    country: {r["country"]} | script: {r["script"]} | src: {r["src"]} | qi: {qi}')
    print(f'    truth  : {tname} ({tid})')
    print()

# Render all four records inline in the notebook
for label, qi in DEMO4.items():
    show(i=qi)
'''

DEMO_FALLBACK_MD = r'''
## Fallback HTML (Issue #15, Task 5)

The cell below saves a **self-contained HTML file** (`demo_fallback.html`) with
all four demo records rendered using the same five-panel layout as the Gradio app.
This file opens in any browser **without** a live Gradio server.
'''

DEMO_FALLBACK_CODE = r'''
# ==== Save fallback HTML for the four demo records (Issue #15, Task 5) ====
import html as _html2
_TITLES = ['', '1 · Normalise + tokenize', '2 · Parse the address',
           '3 · Retrieve candidates', '4 · Score (every matcher)', '5 · Fuse (MoE) + decide']

_status_rows = ''.join(
    f'<tr><td>{_html2.escape(r["component"])}</td><td>{_html2.escape(r["runs as"])}</td></tr>'
    for r in STATUS
)
_status_html = ('<table style="border-collapse:collapse;font-size:13px">'
               '<tr><th style="border:1px solid #d0d7de;padding:4px 8px;background:#f6f8fa">component</th>'
               '<th style="border:1px solid #d0d7de;padding:4px 8px;background:#f6f8fa">runs as</th></tr>'
               + _status_rows + '</table>')

_nav = ''.join(
    f'<a href="#rec{i+1}" style="padding:7px 14px;background:#54A24B;color:#fff;'
    f'border-radius:5px;text-decoration:none;font-weight:600">'
    f'Record {i+1}: {_html2.escape(list(DEMO4)[i].split(":")[0])}</a> '
    for i in range(len(DEMO4))
)

_parts_all = []
for rec_num, (label, qi) in enumerate(DEMO4.items(), 1):
    r = REC.iloc[qi]
    tid = TEST_OWNER.get(r.id)
    tname = (REC.name.values[ID2IX[tid]] if tid and tid in ID2IX.index else '(no true S1)')
    meta = (f'<p style="margin:6px 0"><b>Name:</b> {_html2.escape(r["name"])} &nbsp;|&nbsp; '
            f'<b>Addr:</b> {_html2.escape(r["addr"]) or "(empty)"} &nbsp;|&nbsp; '
            f'<b>Country:</b> {_html2.escape(r["country"])} &nbsp;|&nbsp; '
            f'<b>Script:</b> {_html2.escape(r["script"])} &nbsp;|&nbsp; '
            f'<b>GT:</b> {_html2.escape(tname)}</p>')
    panels = safe_render(i=qi)
    sec_html = f'<section id="rec{rec_num}" style="margin-bottom:60px;border-top:2px solid #d0d7de;padding-top:20px">'
    sec_html += f'<h2 style="color:#1f2328;border-left:4px solid #54A24B;padding-left:10px">Record {rec_num} – {_html2.escape(label)}</h2>'
    sec_html += meta
    for title, panel in zip(_TITLES, panels):
        if title: sec_html += f'<h3 style="color:#1f2328">{_html2.escape(title)}</h3>'
        sec_html += panel
    sec_html += '</section>'
    _parts_all.append(sec_html)

_html_doc = (
    '<!DOCTYPE html>\n<html lang="en"><head><meta charset="UTF-8">\n'
    '<title>Entity Resolution Demo - Four Records (Issue #15 fallback)</title>\n'
    '<style>\n'
    'body{font-family:system-ui,-apple-system,sans-serif;max-width:1200px;margin:0 auto;padding:20px;background:#f8fafc;color:#1f2328}\n'
    'h1{color:#1f2328;border-bottom:2px solid #d0d7de;padding-bottom:8px}\n'
    + CSS + '\n'
    '</style></head><body>\n'
    '<h1>Business Entity Resolution - Demo: Four Records</h1>\n'
    '<p>Fallback HTML for <a href="https://github.com/Sh3001/NLP_Proj_updated/issues/15">Issue #15</a>. '
    f'Generated from the held-out test split with <code>QUICK_RUN={QUICK_RUN}</code>. '
    'Opens in any browser without a live Gradio server.</p>\n'
    '<h2 style="color:#1f2328">Model status at load time</h2>\n'
    f'<div style="background:#f6f8fa;border:1px solid #d0d7de;border-radius:6px;padding:12px;margin-bottom:20px">'
    f'<p><b>DEVICE:</b> {DEVICE}</p>{_status_html}</div>\n'
    f'<div style="display:flex;gap:10px;flex-wrap:wrap;margin:16px 0">{_nav}</div>\n'
    + ''.join(_parts_all) + '\n</body></html>'
)

_html_path = OUT / 'demo_fallback.html'
_html_path.write_text(_html_doc, encoding='utf-8')
print(f'Fallback HTML saved -> {_html_path}  ({_html_path.stat().st_size / 1024:.1f} kB)')
try:
    from IPython.display import FileLink, display as _ipy_display
    _ipy_display(FileLink('demo_fallback.html'))
except Exception:
    pass
'''

DEMO_GRADIO = r'''
# ==== Gradio app: click the public *.gradio.live link printed below ====
try:
    import gradio as gr
except ImportError:
    gr = None; print('gradio not installed: use show(...) from the cell above')
if gr is not None:
    TITLES = ['1 · Normalise + tokenize', '2 · Parse the address', '3 · Retrieve candidates', '4 · Score (every matcher)', '5 · Fuse (MoE) + decide']
    with gr.Blocks(title='Entity resolution: follow one record', css=CSS) as demo:
        gr.Markdown('## Follow one record through the pipeline\nPick an S2/S3 record from the **held-out test split** '
                    '(or type one). Every step runs on it: tokenization → parsing (BiLSTM-CRF) → retrieval (TF-IDF + e5) → '
                    'matchers (LightGBM, RNNs, cross-encoders) → Mixture-of-Experts fusion → decision.')
        with gr.Row():
            dd = gr.Dropdown(choices=list(EXAMPLES), value=list(EXAMPLES)[0], label='Example from the test split', scale=4)
            b_ex = gr.Button('Run', variant='primary', scale=1)
        with gr.Accordion('…or type a record / enter a test-split id', open=False):
            with gr.Row():
                t_name = gr.Textbox(label='Business name', scale=2); t_addr = gr.Textbox(label='Address', scale=3)
            with gr.Row():
                t_cty = gr.Dropdown(COUNTRIES, value=COUNTRIES[0], label='Country'); t_src = gr.Radio(['S2', 'S3'], value='S2', label='Source')
                b_t = gr.Button('Run typed record')
            with gr.Row():
                t_id = gr.Textbox(label='S2/S3 record id from the test split'); b_id = gr.Button('Run id')
        summary = gr.HTML()
        with gr.Tabs():
            tabs = []
            for t in TITLES:
                with gr.Tab(t): tabs.append(gr.HTML())
        outs = [summary] + tabs
        def by_id(x):
            x = x.strip()
            if x not in ID2IX.index or REC.split.values[ID2IX[x]] != 'test' or REC.src.values[ID2IX[x]] == 'S1':
                return [box(f'<p>{esc(x)} is not an S2/S3 record of the test split.</p>')] + [''] * 5
            return safe_render(i=int(ID2IX[x]))
        b_ex.click(lambda l: safe_render(i=EXAMPLES[l]), dd, outs)
        dd.change(lambda l: safe_render(i=EXAMPLES[l]), dd, outs)
        b_t.click(lambda n, a, c, s: safe_render(typed=(n, a, c, s)), [t_name, t_addr, t_cty, t_src], outs)
        b_id.click(by_id, t_id, outs)
        demo.load(lambda: safe_render(i=EXAMPLES[list(EXAMPLES)[0]]), None, outs)
    if os.environ.get('NLP_ER_SMOKE') == '1': print('smoke test: app built, not launched')
    else: demo.launch(share=True, debug=False)
'''

write("08_demo", "08 · Live demo: follow one record through the pipeline", r"""
**All course topics on one record.** Pick a business record from the **held-out test split** (or type
one) and watch every stage run on it:

1. **Normalise + tokenize**: Unicode NFKC → romanisation → learned transliteration lexicon → abbreviation
   expansion; the same name split by WordPiece, SentencePiece and character 3-grams (unknown and split pieces highlighted).
2. **Parse the address**: rule parser vs the **BiLSTM-CRF** from 02, with and without commas.
3. **Retrieve**: TF-IDF and the fine-tuned **e5 transformer** bi-encoder, fused by reciprocal rank → 5 candidates.
4. **Score**: every matcher (LightGBM, Siamese **LSTM / GRU / RNN**, **cross-encoders**: zero-shot, full
   fine-tune, LoRA, **MoE head**, IndicBERTv2, mDeBERTa) gives each candidate a probability.
5. **Fuse + decide**: the gated **Mixture of Experts** from 07 weights the matchers per pair; the decision
   rule (best candidate if p ≥ threshold) gives the answer, checked against the ground truth.

**Inputs:** the dataset, and the outputs (notebook versions or uploaded zips) of 02, 03, 04, 05a–c, 06a–f and 07,
**run with the version of the notebooks that saves model weights** (`models/` folder in their output).
Whatever is missing degrades gracefully: a model without weights shows its stored test scores (test
records only), a missing 07 lets the best single matcher decide. The status table after loading shows
what runs live.

**Run interactively** (not *Save & Run All*): *Run All*, then open the public `https://….gradio.live`
link printed by the last cell. It stays up while the session runs. The cell before it shows one
record inline, as a fallback without Gradio.
""", "GPU T4", "the dataset and the outputs of 02, 03, 04, 05a–c, 06a–f, 07", [
    setup_cell(BASE_PKGS + [("sklearn", "scikit-learn"), ("transformers", "transformers"), ("rapidfuzz", "rapidfuzz>=3.6"),
                            ("lightgbm", "lightgbm"), ("torchcrf", "pytorch-crf"), ("peft", "peft"), ("sentencepiece", "sentencepiece"),
                            ("google.protobuf", "protobuf"), ("gradio", "gradio")]),
    config_cell("load", "", ""),
    DATA_CELL, METRIC_CELL, RULES_CELL, TFIDF_CELL, HF_CELL,
    code(DEMO_LOAD), code(DEMO_PIPE), code(DEMO_RENDER), code(DEMO_EXAMPLES), code(DEMO_INLINE),
    md(DEMO_CURATED_MD), code(DEMO_CURATED_CODE), md(DEMO_FALLBACK_MD), code(DEMO_FALLBACK_CODE),
    code(DEMO_GRADIO),
])
