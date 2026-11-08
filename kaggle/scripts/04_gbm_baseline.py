# ==== Setup (Kaggle: Settings -> Internet ON; GPU T4 where the notebook needs one) ====
import os, sys, subprocess, importlib, time, json, math, re, random, gc, unicodedata, collections, csv, hashlib, glob, zlib
from pathlib import Path
for mod, pip in [('anyascii', 'anyascii'), ('rapidfuzz', 'rapidfuzz>=3.6'), ('lightgbm', 'lightgbm')]:
    try: importlib.import_module(mod)
    except ImportError:
        if subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pip]).returncode != 0: print('WARNING: could not install', pip)
import numpy as np, pandas as pd
from anyascii import anyascii
import warnings; warnings.filterwarnings('ignore', category=FutureWarning); warnings.filterwarnings('ignore', category=UserWarning)
DEVICE = 'cpu'
_T0 = time.time()
def log(*a): print(f'[{(time.time() - _T0) / 60:6.1f} min]', *a, flush=True)

# ==== Configuration ====
QUICK_RUN = False      # True = tiny sample (3,000 / 600 / 600 S1) to check the notebook runs; False = real run
PREP_MODE = 'load'         # 'build' = make records from the raw split, 'build_save' = also save them, 'load' = take them from 03_candidates

CFG = dict(SEED=42, B_FRAC=0.15, K_TFIDF=10, K_DENSE=10, K_FINAL=5, MAX_LEN=96, SUBSAMPLE_S1=None, )
if QUICK_RUN:
    CFG.update(SUBSAMPLE_S1={'train': 3000, 'val': 600, 'test': 600}, )
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

PAIRS = load_pairs()
print({sp: f"{len(PAIRS[sp]):,} pairs" + (f", positive rate {PAIRS[sp].label.mean():.3f}" if 'label' in PAIRS[sp] else '')
       for sp in ('A', 'B', 'val', 'test')})

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

# ==== Pair features ====
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
import lightgbm as lgb

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

REC = add_gbm_cols(REC)
X = {}
for sp in ('A', 'B', 'val', 'test'):
    t = time.time(); X[sp] = pair_features(PAIRS[sp]); log(f'features {sp}: {X[sp].shape} in {time.time() - t:.0f}s')

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

# ---- Download: pack this notebook's output into one zip --------------------------------------
# Output panel -> refresh -> hover over out_04.zip -> three dots -> Download.
# Keep the session running until the download has finished.
# Later, upload the zip as a Kaggle Dataset and attach it to the notebooks that need it.
import zipfile
ZIP_NAME = 'out_04.zip'
KEEP_PREP = False      # prep/ is shipped only by the notebook that builds it (03; 02 adds parsed_fields)
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
    pass