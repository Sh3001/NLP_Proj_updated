# ==== Setup (Kaggle: Settings -> Internet ON; GPU T4 where the notebook needs one) ====
import os, sys, subprocess, importlib, time, json, math, re, random, gc, unicodedata, collections, csv, hashlib, glob, zlib
from pathlib import Path
for mod, pip in [('anyascii', 'anyascii'), ('transformers', 'transformers')]:
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

CFG = dict(SEED=42, B_FRAC=0.15, K_TFIDF=10, K_DENSE=10, K_FINAL=5, MAX_LEN=96, SUBSAMPLE_S1=None, TRAIN_PAIRS=160_000, BS=32, EVAL_BS=256)
if QUICK_RUN:
    CFG.update(SUBSAMPLE_S1={'train': 3000, 'val': 600, 'test': 600}, TRAIN_PAIRS=4000)
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

# ==== Cross-encoder: CE bge-reranker-base + MoE head (top-2 of 4) ====
METHOD, KIND, MODEL, LR, TRAIN, AMP = 'CE bge-reranker-base + MoE head (top-2 of 4)', 'moe', 'BAAI/bge-reranker-base', 2e-05, True, True
MODEL = os.environ.get('NLP_ER_MODEL_OVERRIDE', MODEL)            # local smoke test only

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
        super().__init__(); self.enc = AutoModel.from_pretrained(name).float(); self.moe = MoEHead(self.enc.config.hidden_size)
    def forward(self, **enc): return self.moe(self.enc(**enc).last_hidden_state[:, 0])

def build_ce(kind, name):
    # .float(): transformers v5 loads weights in the checkpoint's dtype; mDeBERTa's checkpoint is fp16, and pure-fp16
    # training gives NaN losses. The master weights must be fp32 (autocast still runs the forward pass in fp16 where AMP is on).
    if kind == 'moe': return CEMoE(name)
    m = AutoModelForSequenceClassification.from_pretrained(name, num_labels=1, ignore_mismatched_sizes=True).float()
    if kind == 'lora':
        from peft import LoraConfig, get_peft_model, TaskType
        m = get_peft_model(m, LoraConfig(task_type=TaskType.SEQ_CLS, r=16, lora_alpha=32, lora_dropout=0.1, target_modules=['query', 'key', 'value']))
        m.print_trainable_parameters()
    return m

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
if DEVICE == 'cuda': torch.cuda.reset_peak_memory_stats()
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
        if (st + 1) % 500 == 0:
            log(f'step {st + 1}/{steps} loss {loss.item():.4f}')
            assert math.isfinite(loss.item()), 'loss is NaN/inf: training diverged, stopping instead of scoring a broken model'
train_time = time.time() - t0
GPU_GB = round(torch.cuda.max_memory_allocated() / 2**30, 2) if DEVICE == 'cuda' else None     # peak during training
print('peak GPU memory during training:', GPU_GB, 'GB')
S, G = {}, {}
for sp in ('B', 'val', 'test'):
    S[sp], G[sp] = ce_score(model, PAIRS[sp]); log(f'scored {sp}')
save_scores(METHOD, S)
report('7 Transformers', METHOD, sigmoid(S['val']), sigmoid(S['test']),
       params=f'{n_all / 1e6:.0f}M ({100 * n_tr / n_all:.1f}% trained)', train_min=round(train_time / 60, 1), gpu_mem_gb=GPU_GB)
(OUT / 'models').mkdir(exist_ok=True)                                  # weights for 08_demo: trained parameters only
_w = {n: (p.detach().half() if AMP else p.detach()).cpu() for n, p in model.named_parameters() if p.requires_grad} if TRAIN else {}
torch.save({'method': METHOD, 'kind': KIND, 'model': MODEL, 'max_len': CFG['MAX_LEN'], 'amp': AMP, 'trained': _w},
           OUT / 'models' / f'{safe_name(METHOD)}.ce.pt')
print(f"saved {sum(v.numel() for v in _w.values()) / 1e6:.1f}M trained parameters for the demo")

# ==== Which expert of the MoE head handles which records? (test pairs) ====
g = G['test']; P = PAIRS['test']
df = pd.DataFrame(g, columns=[f'expert {i}' for i in range(g.shape[1])])
df['script'] = REC.script.values[P.qi.values]; df['country'] = REC.country.values[P.qi.values]; df['source'] = REC.src.values[P.qi.values]
df['is_match_top1'] = P.rrf_rank.values == 0
for col in ('script', 'country', 'source'):
    print(f'\nmean gate probability by {col}:'); print(df.groupby(col).mean(numeric_only=True).round(3).to_string())
df.groupby('script').mean(numeric_only=True).to_csv(OUT / 'moe_head_gates_by_script.csv')
print('\nshare of pairs whose top-1 expert is each expert:', np.bincount(g.argmax(1), minlength=g.shape[1]) / len(g))

# ---- Download: pack this notebook's output into one zip --------------------------------------
# Output panel -> refresh -> hover over out_06d.zip -> three dots -> Download.
# Keep the session running until the download has finished.
# Later, upload the zip as a Kaggle Dataset and attach it to the notebooks that need it.
import zipfile
ZIP_NAME = 'out_06d.zip'
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