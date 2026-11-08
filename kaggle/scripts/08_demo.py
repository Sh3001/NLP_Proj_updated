# ==== Setup (Kaggle: Settings -> Internet ON; GPU T4 where the notebook needs one) ====
import os, sys, subprocess, importlib, time, json, math, re, random, gc, unicodedata, collections, csv, hashlib, glob, zlib
from pathlib import Path
for mod, pip in [('anyascii', 'anyascii'), ('sklearn', 'scikit-learn'), ('transformers', 'transformers'), ('rapidfuzz', 'rapidfuzz>=3.6'), ('lightgbm', 'lightgbm'), ('torchcrf', 'pytorch-crf'), ('peft', 'peft'), ('sentencepiece', 'sentencepiece'), ('google.protobuf', 'protobuf'), ('gradio', 'gradio')]:
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

# ==== Load every model and output that is attached ====
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
from rapidfuzz import fuzz
from rapidfuzz.distance import JaroWinkler
from rapidfuzz.process import cpdist
import lightgbm as lgb
try: from torchcrf import CRF
except ImportError: CRF = None

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

@torch.no_grad()
def embed(model, tok, texts, prefix, bs=512, max_len=64):
    model.eval(); embs = []
    order = np.argsort([len(t) for t in texts]); inv = np.empty_like(order); inv[order] = np.arange(len(order))
    for a in range(0, len(texts), bs):
        enc = tok([prefix + texts[i] for i in order[a:a + bs]], truncation=True, max_length=max_len, padding=True, return_tensors='pt').to(DEVICE)
        with autocast(): e = F.normalize(mean_pool(model(**enc), enc['attention_mask']).float(), dim=-1)
        embs.append(e.half().cpu())
    return torch.cat(embs)[torch.as_tensor(inv)]

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

class GatedMoE(nn.Module):
    def __init__(self, n_exp, n_feat, input_gate=True):
        super().__init__()
        self.a = nn.Parameter(torch.ones(n_exp)); self.b = nn.Parameter(torch.zeros(n_exp)); self.c = nn.Parameter(torch.zeros(1))
        self.input_gate = input_gate; self.w0 = nn.Parameter(torch.zeros(n_exp))
        self.gate = nn.Sequential(nn.Linear(n_feat, 32), nn.ReLU(), nn.Linear(32, n_exp)) if input_gate else None
    def forward(self, z, g):
        w = self.gate(g).softmax(-1) if self.input_gate else self.w0.softmax(-1).expand(len(z), -1)
        return (w * (self.a * z + self.b)).sum(-1) + self.c, w

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