# ==== Setup (Kaggle: Settings -> Internet ON; GPU T4 where the notebook needs one) ====
import os, sys, subprocess, importlib, time, json, math, re, random, gc, unicodedata, collections, csv, hashlib, glob, zlib
from pathlib import Path
for mod, pip in [('anyascii', 'anyascii'), ('torchcrf', 'pytorch-crf'), ('sklearn', 'scikit-learn')]:
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
PREP_MODE = 'build'         # 'build' = make records from the raw split, 'build_save' = also save them, 'load' = take them from 03_candidates

CFG = dict(SEED=42, B_FRAC=0.15, K_TFIDF=10, K_DENSE=10, K_FINAL=5, MAX_LEN=96, SUBSAMPLE_S1=None, PARSER_TRAIN=120_000, PARSER_EPOCHS=3)
if QUICK_RUN:
    CFG.update(SUBSAMPLE_S1={'train': 3000, 'val': 600, 'test': 600}, PARSER_TRAIN=5000, PARSER_EPOCHS=1)
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

REC_TOKS = [addr_tokens(a) for a in REC.addr.values]
for i in np.random.default_rng(0).choice(len(REC), 4, replace=False):
    print(list(zip(REC_TOKS[i], rule_tags(REC_TOKS[i]))))

# ==== BiLSTM-CRF ====
from torchcrf import CRF
from sklearn.metrics import classification_report, f1_score

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

# ==== Evaluation against silver labels (val) ====
rows = []
for cond in ('with commas', 'commas removed'):
    seqs, gold = [], []
    for i in va:
        toks, tags = REC_TOKS[i], rule_tags(REC_TOKS[i])
        if cond == 'commas removed': toks, tags = strip_commas(toks, tags)
        if toks: seqs.append(toks); gold.append(tags)
    _seen = {t for s in gold for t in s}                 # macro F1 over the tags that occur (no POSTCODE in this data: it would count as 0)
    labs = [t for t in TAGS if t != 'SEP' and t in _seen]
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

# ==== Parse every address (with commas) -> prep/parsed_fields.parquet ====
tags_all = parser_decode(PARSER, REC_TOKS)
fl = [fields_from(t, g) for t, g in zip(REC_TOKS, tags_all)]
pf = pd.DataFrame({f'p_{k.lower()}': [d.get(k, '') for d in fl] for k in TAGS if k != 'SEP'}); pf.insert(0, 'id', REC.id.values)
pf.to_parquet(OUT / 'prep' / 'parsed_fields.parquet')
print(pd.concat([REC[['addr']], pf.drop(columns='id')], axis=1).sample(8, random_state=1).to_string())
(OUT / 'models').mkdir(exist_ok=True)                                  # weights for 08_demo
torch.save({'state_dict': PARSER.state_dict(), 'W2I': W2I, 'C2I': C2I, 'TAGS': TAGS}, OUT / 'models' / 'parser_bilstm_crf.pt')

# ---- Download: pack this notebook's output into one zip --------------------------------------
# Output panel -> refresh -> hover over out_02.zip -> three dots -> Download.
# Keep the session running until the download has finished.
# Later, upload the zip as a Kaggle Dataset and attach it to the notebooks that need it.
import zipfile
ZIP_NAME = 'out_02.zip'
KEEP_PREP = ['parsed_fields.parquet']      # prep/ is shipped only by the notebook that builds it (03; 02 adds parsed_fields)
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