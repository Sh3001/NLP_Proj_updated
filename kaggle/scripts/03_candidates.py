# ==== Setup (Kaggle: Settings -> Internet ON; GPU T4 where the notebook needs one) ====
import os, sys, subprocess, importlib, time, json, math, re, random, gc, unicodedata, collections, csv, hashlib, glob, zlib
from pathlib import Path
for mod, pip in [('anyascii', 'anyascii'), ('sklearn', 'scikit-learn'), ('transformers', 'transformers')]:
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
PREP_MODE = 'build_save'         # 'build' = make records from the raw split, 'build_save' = also save them, 'load' = take them from 03_candidates

CFG = dict(SEED=42, B_FRAC=0.15, K_TFIDF=10, K_DENSE=10, K_FINAL=5, MAX_LEN=96, SUBSAMPLE_S1=None, BI_MODEL='intfloat/multilingual-e5-small', BI_PAIRS=100_000, BI_BS=128, BI_LR=3e-5, BI_LEN=64, BI_EPOCHS=1, USE_DENSE=True)
if QUICK_RUN:
    CFG.update(SUBSAMPLE_S1={'train': 3000, 'val': 600, 'test': 600}, BI_PAIRS=3000)
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

# ==== Bi-encoder: zero-shot vs contrastive fine-tuning ====
def part_of(ids):          # train S1 -> A/B by hash; linked records follow their S1; unlinked by hash
    o = OWNER['train']; return np.array(['B' if h01(f"AB{SEED}|{o.get(x, x)}") < CFG['B_FRAC'] else 'A' for x in ids])


@torch.no_grad()
def embed(model, tok, texts, prefix, bs=512, max_len=64):
    model.eval(); embs = []
    order = np.argsort([len(t) for t in texts]); inv = np.empty_like(order); inv[order] = np.arange(len(order))
    for a in range(0, len(texts), bs):
        enc = tok([prefix + texts[i] for i in order[a:a + bs]], truncation=True, max_length=max_len, padding=True, return_tensors='pt').to(DEVICE)
        with autocast(): e = F.normalize(mean_pool(model(**enc), enc['attention_mask']).float(), dim=-1)
        embs.append(e.half().cpu())
    return torch.cat(embs)[torch.as_tensor(inv)]

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

# ---- Download: pack this notebook's output into one zip --------------------------------------
# Output panel -> refresh -> hover over out_03.zip -> three dots -> Download.
# Keep the session running until the download has finished.
# Later, upload the zip as a Kaggle Dataset and attach it to the notebooks that need it.
import zipfile
ZIP_NAME = 'out_03.zip'
KEEP_PREP = True      # prep/ is shipped only by the notebook that builds it (03; 02 adds parsed_fields)
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