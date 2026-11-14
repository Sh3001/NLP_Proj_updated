# ==== Setup (Kaggle: Settings -> Internet ON; GPU T4 where the notebook needs one) ====
import os, sys, subprocess, importlib, time, json, math, re, random, gc, unicodedata, collections, csv, hashlib, glob, zlib
from pathlib import Path
for mod, pip in [('anyascii', 'anyascii'), ('transformers', 'transformers'), ('sentencepiece', 'sentencepiece'), ('google.protobuf', 'protobuf'), ('matplotlib', 'matplotlib')]:
    try: importlib.import_module(mod)
    except ImportError:
        if subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', pip]).returncode != 0: print('WARNING: could not install', pip)
import numpy as np, pandas as pd
from anyascii import anyascii
import warnings; warnings.filterwarnings('ignore', category=FutureWarning); warnings.filterwarnings('ignore', category=UserWarning)
import torch, torch.nn as nn, torch.nn.functional as F
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
USE_AMP = DEVICE == 'cuda'
print('device:', DEVICE, torch.cuda.get_device_name(0) if DEVICE == 'cuda' else '(CPU)')
_T0 = time.time()
def log(*a): print(f'[{(time.time() - _T0) / 60:6.1f} min]', *a, flush=True)

# ==== Configuration ====
QUICK_RUN = False      # True = tiny sample (3,000 / 600 / 600 S1) to check the notebook runs; False = real run
PREP_MODE = 'load'         # 'build' = make records from the raw split, 'build_save' = also save them, 'load' = take them from 03_candidates

CFG = dict(SEED=42, B_FRAC=0.15, K_TFIDF=10, K_DENSE=10, K_FINAL=5, MAX_LEN=96, SUBSAMPLE_S1=None, RUN_SET='ablation', DITTO_LEN=96, DITTO_BS=32, DITTO_EVAL_BS=256, DITTO_N_ABL=40_000, DITTO_N_OP=20_000, DITTO_N_LOW=[2_000, 10_000], DITTO_N_FINAL=160_000, DITTO_MIN_STEPS=600, DITTO_SU_WORDS=16, DITTO_ALPHA=0.8, DITTO_FINAL='auto')
if QUICK_RUN:
    CFG.update(SUBSAMPLE_S1={'train': 3000, 'val': 600, 'test': 600}, DITTO_N_ABL=3000, DITTO_N_OP=2000, DITTO_N_LOW=[500, 1000], DITTO_N_FINAL=4000, DITTO_MIN_STEPS=20)
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
UNIT_WORDS = {'fl', 'floor', 'suite', 'ste', 'unit', 'apt', 'room', 'flat', 'door', 'shop', 'bldg', 'building', '#'}   # unit/floor parts
HOUSE_NAME_END = {'house', 'bhavan', 'villa', 'nivas', 'manzil'}      # Kerala-style house names
POST_RE = re.compile(r'^\d{6}$|^\d{5}(-\d{4})?$')

NULL_TOKS = {'null', '<null>', 'n/a', 'none', 'nan', '<na>'}      # missing-value markers in the raw addresses

def addr_tokens(a):
    a = unicodedata.normalize('NFKC', a)
    if not a.isascii(): a = romanise(a)
    return [t for t in re.findall(r',|[^\s,]+', a.lower()) if t not in NULL_TOKS][:40]

def _is_post(t): return bool(POST_RE.match(t.strip('-')))
_last, _any = collections.Counter(), collections.Counter()
for a in REC.addr.values[REC.split.values == 'train']:        # state spellings = components that (almost) always come last
    comps = [' '.join(t for t in c.split() if not _is_post(t) and t.strip(':-') not in ('pin', 'pincode', 'zip')).strip(' -')
             for c in ' '.join(addr_tokens(a)).split(',')]
    comps = [c for c in comps if c and not any(ch.isdigit() for ch in c)]
    for c in comps: _any[c] += 1
    if comps: _last[comps[-1]] += 1
# share >= 0.6, not 0.9: ~10-14 % of addresses have shuffled components ('61022 Holly Court, OR, Bend'), so real states
# are last in only 86-90 % of their occurrences. On train, every component with share >= 0.5 is a state or a state
# abbreviation; city names stay below 0.45 ('washington' 0.44, 'delhi' 0.32 are both).
STATE_VOCAB = {c for c, n in _last.items() if n >= 30 and n / _any[c] >= 0.6 and len(c.split()) <= 3}
print(len(STATE_VOCAB), 'state spellings:', sorted(STATE_VOCAB, key=lambda c: -_last[c]))

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
    # city = the last component before the trailing state/postcode that has no digits and is not a state
    # ('BRIGHAM CITY, UT, 612 675' -> brigham city, not '612 675')
    k = next((k for k in range(tail - 1, -1, -1) if not any(ch.isdigit() for i in comps[k] for ch in toks[i])
              and ' '.join(toks[i] for i in comps[k]).strip(' -') not in STATE_VOCAB), None)
    if tail >= 2 and k is not None:
        for i in comps[k]: tags[i] = 'CITY'
        body = comps[:k] + comps[k + 1:tail]
    else:
        body = comps[:tail]
    for ci, c in enumerate(body):
        w = [toks[i] for i in c]
        if ' '.join(w).strip(' -') in STATE_VOCAB:   # a state that is not at the end (shuffled components)
            for i in c: tags[i] = 'STATE'
            continue
        if ci == 0 and len(w) > 1 and w[-1] in HOUSE_NAME_END:
            for i in c: tags[i] = 'HOUSE'
            continue
        lab = 'LANDMARK' if w[0] in LANDMARK_WORDS else ('STREET' if any(x in STREET_WORDS for x in w) else 'LOCALITY')
        start = 0
        # leading house-number / unit tokens: in the first component, and in any component that starts with a unit word
        # ('fl 1', 'suite no.889 valamkottil towers' -> HOUSE only for 'suite no.889')
        if ci == 0 or (w[0].rstrip('.') in UNIT_WORDS and any(ch.isdigit() for x in w for ch in x)):
            while start < len(c) and (any(ch.isdigit() for ch in w[start]) or w[start] in HOUSE_WORDS):
                tags[c[start]] = 'HOUSE'; start += 1
        for i in c[start:]: tags[i] = lab
    return tags

def fields_from(toks, tags):
    f = collections.defaultdict(list)
    for t, g in zip(toks, tags):
        if g != 'SEP' and t != ',': f[g].append(t)
    return {k: ' '.join(v) for k, v in f.items()}

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

# ==== Ditto (Li et al., VLDB 2021): serialization, DK, SU, DA / MixDA, model ====
DITTO_COLS = ['house', 'street', 'locality', 'landmark', 'city', 'state', 'postcode']
P_COLS = ['p_' + c for c in DITTO_COLS]                       # parsed address fields (02, or the rules)
MARKERS = ['[COL]', '[VAL]', '[NUM]']
AUG_OPS = ['span_del', 'span_shuffle', 'attr_del', 'attr_shuffle', 'entry_swap']
_HOUSE_NOISE = set(HOUSE_WORDS) | {'h.no', 'no', '#'}

def norm_house(s):                                            # span normalisation: 'h.no. ##110' -> ['110']
    out = []
    for t in s.split():
        t = t.strip('#.,:-')
        if t and t not in _HOUSE_NOISE: out.append(t)
    return out

def _expand(s):                                               # span normalisation: abbreviations, '##110' -> '110'
    out = []
    for t in s.split():
        t = t.lstrip('#') or t; out.append(ABBR.get(t, t))
    return out

def attrs_of(ix, dk, summ=None):
    # one record -> [(column, [words])]. plain: name + address as in the other cross-encoders.
    # dk (domain knowledge): span normalisation (romanisation, learned lexicon, abbreviations, house numbers)
    #                        + span typing (the parser's HOUSE / STREET / LOCALITY / LANDMARK / CITY / STATE / POSTCODE spans)
    if not dk:
        name, addr = REC.name.values[ix], REC.addr.values[ix]
        n = name if name.isascii() else f'{name} / {norm_rec(name, expand=False)}'
        a = addr if addr.isascii() else romanise(addr)
        attrs = [('name', n.split()), ('address', a.split())]
    else:
        attrs = [('name', REC.nn.values[ix].split())]
        for c in DITTO_COLS:
            v = REC['p_' + c].values[ix]
            ws = norm_house(v) if c == 'house' else _expand(v)
            if ws: attrs.append((c, ws))
        if len(attrs) == 1 and REC.na.values[ix]: attrs.append(('address', REC.na.values[ix].split()))
    return summ(attrs) if summ is not None else attrs

def serialize(attrs, mark_num=False):
    # [COL] name [VAL] balaji care private limited [COL] house [VAL] [NUM] 110 ...
    out = []
    for c, ws in attrs:
        out += ['[COL]', c, '[VAL]']
        for w in ws:
            if mark_num and any(ch.isdigit() for ch in w): out.append('[NUM]')      # span typing: numbers
            out.append(w)
    return ' '.join(out)

class Summarizer:
    # SU: keep the words with the highest TF-IDF instead of cutting the end. Every attribute keeps its best word.
    def __init__(self, dk, budget, idf=None, default=None):
        self.dk, self.budget, self.idf, self.default = dk, budget, idf or {}, default or 1.0
    def fit(self, ix):
        df = collections.Counter(); n = 0
        for i in ix:
            df.update({w for _, ws in attrs_of(int(i), self.dk) for w in ws}); n += 1
        self.idf = {w: math.log((n + 1) / (c + 1)) + 1 for w, c in df.items() if c >= 2}
        self.default = math.log(n + 1) + 1                    # unseen / rare words are the most informative
        return self
    def select(self, attrs):                                   # set of kept (attr, word) positions, None = keep all
        flat = [(ai, wi, w) for ai, (_, ws) in enumerate(attrs) for wi, w in enumerate(ws)]
        if len(flat) <= self.budget: return None
        score = lambda w: self.idf.get(w, self.default)
        keep = set()
        for ai, (_, ws) in enumerate(attrs):
            if ws: keep.add((ai, max(range(len(ws)), key=lambda k: score(ws[k]))))
        rest = sorted((t for t in flat if (t[0], t[1]) not in keep), key=lambda t: (-score(t[2]), t[0], t[1]))
        keep |= {(t[0], t[1]) for t in rest[:max(0, self.budget - len(keep))]}
        return keep
    def __call__(self, attrs):
        keep = self.select(attrs)
        if keep is None: return attrs
        return [(c, [w for wi, w in enumerate(ws) if (ai, wi) in keep]) for ai, (c, ws) in enumerate(attrs)]

def augment(A, B, op, rng):
    # DA operators of Ditto on a pair of attribute lists
    if op == 'all': op = AUG_OPS[int(rng.integers(len(AUG_OPS)))]
    A = [(c, list(w)) for c, w in A]; B = [(c, list(w)) for c, w in B]
    if op == 'entry_swap': return B, A
    if op == 'attr_shuffle':
        return [A[i] for i in rng.permutation(len(A))], [B[i] for i in rng.permutation(len(B))]
    X = A if rng.random() < 0.5 else B
    if op == 'attr_del':
        if len(X) > 1: del X[int(rng.integers(len(X)))]
        return A, B
    ok = [k for k, (_, ws) in enumerate(X) if len(ws) >= 2]
    if not ok: return A, B
    ws = X[ok[int(rng.integers(len(ok)))]][1]
    if op == 'span_del':
        L = int(rng.integers(1, min(3, len(ws) - 1) + 1)); s = int(rng.integers(0, len(ws) - L + 1)); del ws[s:s + L]
    else:                                                      # span_shuffle
        L = int(rng.integers(2, min(4, len(ws)) + 1)); s = int(rng.integers(0, len(ws) - L + 1))
        seg = ws[s:s + L]; rng.shuffle(seg); ws[s:s + L] = seg
    return A, B

def ditto_tokenizer(name):
    tok = AutoTokenizer.from_pretrained(name)
    tok.add_special_tokens({'additional_special_tokens': MARKERS})
    return tok

class DittoModel(nn.Module):
    # pre-trained LM + linear layer on the [CLS] vector. MixDA: h = lam * h(x) + (1 - lam) * h(augmented x)
    def __init__(self, name, tok, pretrained=True):
        super().__init__()
        from transformers import AutoConfig
        self.enc = (AutoModel.from_pretrained(name) if pretrained else AutoModel.from_config(AutoConfig.from_pretrained(name))).float()   # fp32 master weights, AMP runs the forward in fp16
        ids = tok.convert_tokens_to_ids(MARKERS)
        if max(ids) >= self.enc.get_input_embeddings().weight.shape[0]: self.enc.resize_token_embeddings(len(tok))
        if pretrained:                                         # the new markers start at the mean embedding
            with torch.no_grad():
                w = self.enc.get_input_embeddings().weight; keep = torch.ones(w.shape[0], dtype=torch.bool); keep[ids] = False
                w[ids] = w[keep].mean(0)
        self.drop = nn.Dropout(0.1); self.fc = nn.Linear(self.enc.config.hidden_size, 1)
    def forward(self, enc, n=None, lam=None):
        h = self.enc(**enc).last_hidden_state[:, 0]
        if lam is not None: h = lam[:, None] * h[:n] + (1 - lam[:, None]) * h[n:]     # first n rows = originals, rest = augmented
        return self.fc(self.drop(h)).squeeze(-1).float()

# ==== Parsed address fields for the domain-knowledge attributes (from 02, else the rule parser) ====
_pf = find_one('prep/parsed_fields.parquet')
if _pf is not None:
    pf = pd.read_parquet(_pf).set_index('id').reindex(REC.id.values).fillna(''); PARSE_SRC = 'BiLSTM-CRF (02)'
else:
    print('02 output not attached -> rule parser'); toks = [addr_tokens(a) for a in REC.addr.values]
    fl = [fields_from(t, rule_tags(t)) for t in toks]
    pf = pd.DataFrame({f'p_{k.lower()}': [d.get(k, '') for d in fl] for k in TAGS if k != 'SEP'}, index=REC.id.values); PARSE_SRC = 'rules'
for c in P_COLS: REC[c] = pf[c].values
print('parsed fields from', PARSE_SRC)
for i in np.random.default_rng(1).choice(len(REC), 3, replace=False):
    print('\nplain :', serialize(attrs_of(i, False))); print('DK    :', serialize(attrs_of(i, True), True))

# ==== Ditto: one training + evaluation run per configuration ====
BBN = {'xlmr': 'xlm-roberta-base', 'bge': 'BAAI/bge-reranker-base'}      # both MIT
LRS = {'xlmr': 3e-5, 'bge': 2e-5}
BS, L_MAX, ALPHA = CFG['DITTO_BS'], CFG['DITTO_LEN'], CFG['DITTO_ALPHA']
SEC_ABL = '7b Ditto ablation'

def C(dk=False, su=False, da=None, bb='xlmr', n=None, final=False):
    return dict(dk=dk, su=su, da=da, bb=bb, n=n, final=final)

def run_name(c):
    fl = [x for x in ('DK' if c['dk'] else '', 'SU' if c['su'] else '', f"DA-{c['da']}" if c['da'] else '') if x] or ['plain']
    n = c['n']; nl = f'{n // 1000}k' if n >= 1000 and n % 1000 == 0 else str(n)
    return f"Ditto{' final' if c['final'] else ''} {c['bb']} {'+'.join(fl)} {nl}"

SUMM = {}
def get_summ(dk):                                    # IDF from 100k train records (never val / test)
    if dk not in SUMM:
        tr = np.where(REC.split.values == 'train')[0]
        ix = np.random.default_rng(SEED).choice(tr, min(100_000, len(tr)), replace=False)
        SUMM[dk] = Summarizer(dk, CFG['DITTO_SU_WORDS']).fit(ix)
    return SUMM[dk]

SER = {'key': None, 'arr': None}
def ser_for(dk, su, idx):                            # serialized text of every record in idx (cached for the current dk / su)
    if SER['key'] != (dk, su): SER.update(key=(dk, su), arr=np.full(len(REC), None, dtype=object))
    arr = SER['arr']; summ = get_summ(dk) if su else None
    for i in np.unique(idx):
        if arr[i] is None: arr[i] = serialize(attrs_of(int(i), dk, summ), dk)
    return arr

@torch.no_grad()
def ditto_score(model, tok, P, ser, bs):
    model.eval(); ta, tb = ser[P.qi.values], ser[P.si.values]
    order = np.argsort([len(a) + len(b) for a, b in zip(ta, tb)]); out = np.zeros(len(P), np.float32)
    for a in range(0, len(order), bs):
        rows = order[a:a + bs]
        enc = tok(list(ta[rows]), list(tb[rows]), truncation=True, max_length=L_MAX, padding=True, return_tensors='pt').to(DEVICE)
        with autocast(): out[rows] = model(enc).float().cpu().numpy()
    return out

def pair_labels(sp):                                 # the test labels are read here, for the final scoring only
    if sp == 'test':
        own = ev('test').owner; return np.array([own.get(q) == s for q, s in zip(PAIRS['test'].q_id, PAIRS['test'].s_id)], np.int8)
    return PAIRS[sp].label.values

def _f1(y, pred):
    tp = int((pred & (y == 1)).sum()); P = tp / max(int(pred.sum()), 1); R = tp / max(int(y.sum()), 1)
    return (2 * P * R / (P + R) if P + R else 0.0), P, R

def pair_f1(p_val, p_test):                          # pair-level F1 on the candidate pairs (Ditto's metric), threshold tuned on val
    yv, yt = pair_labels('val'), pair_labels('test')
    t = max(np.arange(0.05, 0.96, 0.01), key=lambda t: _f1(yv, p_val >= t)[0]); f, P, R = _f1(yt, p_test >= t)
    return {'test_pair_f1': round(f, 5), 'test_pair_P': round(P, 4), 'test_pair_R': round(R, 4), 'pair_thr': round(float(t), 2)}

def script_f05(bt, thr):                             # record-level F0.5 per script of the S2/S3 record (test)
    own = ev('test').owner; sel = bt[bt.p >= thr]
    qids = REC.id.values[(REC.split.values == 'test') & (REC.src.values != 'S1')]; qs = REC.script.values[ID2IX[qids].values]
    out = {}
    for s in SCRIPTS:
        nt = sum(1 for q, sc in zip(qids, qs) if sc == s and q in own)
        m = REC.script.values[ID2IX[sel.q_id.values].values] == s; npred = int(m.sum())
        tp = sum(1 for q, r in zip(sel.q_id.values[m], sel.s_id.values[m]) if own.get(q) == r)
        P = tp / max(npred, 1); R = tp / max(nt, 1); out[f'script_f05_{s}'] = round(1.25 * P * R / (0.25 * P + R), 5) if P + R else 0.0
        out[f'script_n_{s}'] = nt
    return out

def run_ditto(c):
    t_all = time.time(); dk, su, da, bkey, n, final = c['dk'], c['su'], c['da'], c['bb'], c['n'], c['final']
    method = run_name(c); name = os.environ.get('NLP_ER_MODEL_OVERRIDE', BBN[bkey]); summ = get_summ(dk) if su else None
    print(f'\n######## {method}  ({name})')
    tok = ditto_tokenizer(name); model = DittoModel(name, tok).to(DEVICE)
    A = PAIRS['A']; y = A.label.values.astype(np.float32)                    # training pairs: whole candidate groups of part A
    qs = A.qi.unique(); rng = np.random.default_rng(SEED); rng.shuffle(qs)
    cum = A.groupby('qi').size().reindex(qs).cumsum().values
    rows = np.where(A.qi.isin(set(qs[:np.searchsorted(cum, n) + 1])).values)[0]; rng.shuffle(rows)
    epochs = max(1, math.ceil(CFG['DITTO_MIN_STEPS'] * BS / len(rows)))      # small training sets are repeated (as in the paper)
    steps = epochs * math.ceil(len(rows) / BS)
    ser = ser_for(dk, su, np.concatenate([A.qi.values[rows], A.si.values[rows]]))
    opt = torch.optim.AdamW(model.parameters(), lr=LRS[bkey], weight_decay=0.01)
    sch = get_linear_schedule_with_warmup(opt, int(0.1 * steps), steps); scaler = make_scaler(USE_AMP)
    arng = np.random.default_rng(SEED + 1); st = 0; t0 = time.time(); model.train()
    if DEVICE == 'cuda': torch.cuda.reset_peak_memory_stats()
    print(f'{len(rows):,} pairs ({y[rows].mean():.3f} positive), {epochs} epoch(s), {steps:,} steps, DK={dk} SU={su} DA={da}')
    for e in range(epochs):
        order = rows if e == 0 else rows[np.random.default_rng(SEED + e).permutation(len(rows))]
        for a in range(0, len(order), BS):
            b = order[a:a + BS]; qi, si = A.qi.values[b], A.si.values[b]; nb = len(b)
            ta, tb = list(ser[qi]), list(ser[si]); lam = None
            if da:                                                           # MixDA: encode the augmented copy in the same batch
                for q_, s_ in zip(qi, si):
                    X, Y = augment(attrs_of(int(q_), dk, summ), attrs_of(int(s_), dk, summ), da, arng)
                    ta.append(serialize(X, dk)); tb.append(serialize(Y, dk))
                lam = arng.beta(ALPHA, ALPHA, nb); lam = torch.as_tensor(np.maximum(lam, 1 - lam), dtype=torch.float32, device=DEVICE)
            enc = tok(ta, tb, truncation=True, max_length=L_MAX, padding=True, return_tensors='pt').to(DEVICE)
            with autocast(): logit = model(enc, nb, lam) if da else model(enc)
            loss = F.binary_cross_entropy_with_logits(logit, torch.as_tensor(y[b], device=DEVICE))
            assert math.isfinite(loss.item()), 'loss is NaN/inf: training diverged, stopping instead of scoring a broken model'
            opt.zero_grad(); scaler.scale(loss).backward(); scaler.unscale_(opt)
            nn.utils.clip_grad_norm_(model.parameters(), 1.0); scaler.step(opt); scaler.update(); sch.step(); st += 1
            if st % 500 == 0: log(f'step {st}/{steps} loss {loss.item():.4f}')
    train_time = time.time() - t0
    gpu_gb = round(torch.cuda.max_memory_allocated() / 2**30, 2) if DEVICE == 'cuda' else None
    S = {}
    for sp in (('B', 'val', 'test') if final else ('val', 'test')):
        S[sp] = ditto_score(model, tok, PAIRS[sp], ser_for(dk, su, np.concatenate([PAIRS[sp].qi.values, PAIRS[sp].si.values])), CFG['DITTO_EVAL_BS']); log(f'scored {sp}')
    pv, pt = sigmoid(S['val']), sigmoid(S['test'])
    thr, _, _ = ev('val').tune(best_per_record(PAIRS['val'], pv))
    extra = {**pair_f1(pv, pt), **script_f05(best_per_record(PAIRS['test'], pt), thr)}
    n_par = sum(p.numel() for p in model.parameters())
    row = report('7 Transformers' if final else SEC_ABL, method, pv, pt, params=f'{n_par / 1e6:.0f}M', train_min=round(train_time / 60, 1), gpu_mem_gb=gpu_gb,
                 ditto_cfg=c, ditto_pairs=len(rows), ditto_epochs=epochs, parsed_fields=PARSE_SRC, **extra)
    if final:                                                                # only the final run feeds 07 and the demo
        save_scores(method, S); (OUT / 'models').mkdir(exist_ok=True)
        torch.save({'method': method, 'cfg': c, 'backbone': name, 'max_len': L_MAX, 'su_words': CFG['DITTO_SU_WORDS'],
                    'idf': SUMM[dk].idf if su else {}, 'default_idf': SUMM[dk].default if su else 1.0,
                    'state_dict': {k: (v.half() if v.is_floating_point() else v) for k, v in model.state_dict().items()}},
                   OUT / 'models' / f'{safe_name(method)}.ditto.pt')
        print('saved the final model for the demo')
    print(f'run time {(time.time() - t_all) / 60:.1f} min'); del model, opt; return row

nA, nO, nL, nF = CFG['DITTO_N_ABL'], CFG['DITTO_N_OP'], CFG['DITTO_N_LOW'], CFG['DITTO_N_FINAL']
SETS = {'ablation': [C(n=nA), C(dk=True, n=nA), C(su=True, n=nA), C(da='all', n=nA), C(True, True, 'all', 'xlmr', nA), C(True, True, 'all', 'bge', nA)],
        'operators': [C(da=op, n=nO) for op in AUG_OPS + ['all']],
        'label_eff': [C(n=k) for k in nL] + [C(True, True, 'all', 'xlmr', k) for k in nL]}

def pick_final():                                    # best ablation run by VAL F0.5 (an ablation run from this or an attached notebook)
    base = C(True, True, 'all', 'xlmr', nF, True); mode = CFG['DITTO_FINAL']
    if mode == 'full': return base
    if mode == 'full-bge': return {**base, 'bb': 'bge'}
    cands = {}
    for p in find_all('results/*.json') + sorted((OUT / 'results').glob('*.json')):
        r = json.load(open(p)); c = r.get('ditto_cfg')
        if c and not c.get('final') and c.get('n') == nA: cands[r['method']] = r
    if not cands: print('no ablation results found -> full Ditto'); return base
    best = max(cands.values(), key=lambda r: r['val_f05']); c = best['ditto_cfg']
    print(f"final configuration = best ablation run by val F0.5: {best['method']} (val {best['val_f05']})")
    return C(c['dk'], c['su'], c['da'], c['bb'], nF, True)

# ==== Run the selected set (RUN_SET in the configuration cell) ====
# 'ablation' (R1-R6) | 'operators' (each augmentation operator) | 'label_eff' (2k / 10k pairs) | 'final' | 'all'
_sets = ['ablation', 'operators', 'label_eff'] if CFG['RUN_SET'] == 'all' else ([] if CFG['RUN_SET'] == 'final' else [CFG['RUN_SET']])
DONE = []
def go(c):
    try: DONE.append(run_ditto(c))
    except Exception:
        import traceback; traceback.print_exc(); print('RUN FAILED:', run_name(c))     # one failure does not stop the session
    gc.collect()
    if DEVICE == 'cuda': torch.cuda.empty_cache()
for s_ in _sets:
    for c_ in SETS[s_]: go(c_)
if CFG['RUN_SET'] in ('all', 'final'): go(pick_final())
print(f'\n{len(DONE)} runs finished')

# ==== Ablation table, label-efficiency plot, per-script scores ====
import matplotlib.pyplot as plt
RES = {}
for p in find_all('results/*.json') + sorted((OUT / 'results').glob('*.json')):
    r = json.load(open(p)); RES[r['method']] = r
rows = []
for m, r in RES.items():
    c = r.get('ditto_cfg')
    if c: rows.append({'method': m, 'DK': c['dk'], 'SU': c['su'], 'DA': c['da'] or '-', 'backbone': c['bb'], 'pairs': r.get('ditto_pairs', c['n']),
                       'final': c['final'], **{k: r.get(k) for k in ('val_f05', 'test_f05', 'test_P', 'test_R', 'test_pair_f1', 'train_min')},
                       **{k.replace('script_f05_', 'f05_'): r.get(k) for k in r if k.startswith('script_f05_')}})
D = pd.DataFrame(rows)
if len(D):
    D = D.sort_values(['final', 'backbone', 'pairs', 'DK', 'SU', 'DA']).reset_index(drop=True)
    ref = RES.get('CE bge-reranker-base full fine-tune')                               # R0: plain text + bge, from 06b (if attached)
    if ref: print('R0 (06b, plain text, bge, full fine-tune): test F0.5', ref['test_f05'], '| val', ref['val_f05'])
    D.to_csv(OUT / 'ditto_ablation.csv', index=False); print(D.drop(columns=['final']).round(4).to_string(index=False))
    full = D[(D.DK) & (D.SU) & (D.DA == 'all') & (D.backbone == 'xlmr') & (~D.final)]; plain = D[(~D.DK) & (~D.SU) & (D.DA == '-') & (D.backbone == 'xlmr') & (~D.final)]
    if len(full) > 1 or len(plain) > 1:
        fig, ax = plt.subplots(figsize=(6, 3.6))
        for d_, lab in ((plain, 'plain serialization (R1)'), (full, 'DK + SU + DA (R5)')):
            d_ = d_.sort_values('pairs'); ax.plot(d_.pairs, d_.test_f05, 'o-', label=lab)
        ax.set_xscale('log'); ax.set_xlabel('training pairs'); ax.set_ylabel('test macro F0.5'); ax.legend(); ax.set_title('Label efficiency')
        plt.tight_layout(); plt.savefig(OUT / 'ditto_label_efficiency.png', dpi=150); plt.show()
    ops = D[(~D.DK) & (~D.SU) & (D.DA != '-') & (D.backbone == 'xlmr') & (~D.final)]
    if len(ops): print('\naugmentation operators (plain serialization):'); print(ops[['DA', 'pairs', 'val_f05', 'test_f05', 'test_pair_f1']].round(4).to_string(index=False))

# ==== Token lengths with and without summarisation (val pairs), per script of the S2/S3 record ====
_tk = ditto_tokenizer(os.environ.get('NLP_ER_MODEL_OVERRIDE', BBN['xlmr']))
_P = PAIRS['val'].sample(min(4000, len(PAIRS['val'])), random_state=SEED).reset_index(drop=True); _rows = []
for dk_, su_ in ((False, False), (False, True), (True, False), (True, True)):
    _s = ser_for(dk_, su_, np.concatenate([_P.qi.values, _P.si.values]))
    ln = np.array([len(x) for x in _tk(list(_s[_P.qi.values]), list(_s[_P.si.values]), truncation=False)['input_ids']])
    sc_ = REC.script.values[_P.qi.values]
    for g in ['all'] + SCRIPTS:
        m = np.ones(len(ln), bool) if g == 'all' else sc_ == g
        if m.any(): _rows.append({'DK': dk_, 'SU': su_, 'script': g, 'pairs': int(m.sum()), 'mean_tokens': round(float(ln[m].mean()), 1),
                                  f'share_truncated_at_{L_MAX}': round(float((ln[m] > L_MAX).mean()), 4)})
LEN = pd.DataFrame(_rows); LEN.to_csv(OUT / 'ditto_lengths.csv', index=False); print(LEN.to_string(index=False))

# ---- Download: pack this notebook's output into one zip --------------------------------------
# Output panel -> refresh -> hover over out_06g.zip -> three dots -> Download.
# Keep the session running until the download has finished.
# Later, upload the zip as a Kaggle Dataset and attach it to the notebooks that need it.
import zipfile
ZIP_NAME = 'out_06g.zip'
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