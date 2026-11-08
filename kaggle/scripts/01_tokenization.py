# ==== Setup (Kaggle: Settings -> Internet ON; GPU T4 where the notebook needs one) ====
import os, sys, subprocess, importlib, time, json, math, re, random, gc, unicodedata, collections, csv, hashlib, glob, zlib
from pathlib import Path
for mod, pip in [('anyascii', 'anyascii'), ('transformers', 'transformers'), ('sentencepiece', 'sentencepiece'), ('google.protobuf', 'protobuf')]:
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
PREP_MODE = 'build'         # 'build' = make records from the raw split, 'build_save' = also save them, 'load' = take them from 03_candidates

CFG = dict(SEED=42, B_FRAC=0.15, K_TFIDF=10, K_DENSE=10, K_FINAL=5, MAX_LEN=96, SUBSAMPLE_S1=None, TOK_SAMPLE=2000, ABL_FIT=150_000)
if QUICK_RUN:
    CFG.update(SUBSAMPLE_S1={'train': 3000, 'val': 600, 'test': 600}, TOK_SAMPLE=300, ABL_FIT=20_000)
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

# ---- Download: pack this notebook's output into one zip --------------------------------------
# Output panel -> refresh -> hover over out_01.zip -> three dots -> Download.
# Keep the session running until the download has finished.
# Later, upload the zip as a Kaggle Dataset and attach it to the notebooks that need it.
import zipfile
ZIP_NAME = 'out_01.zip'
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