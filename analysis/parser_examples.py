"""Issue #4: example addresses for the BiLSTM-CRF parser (02) vs the rule parser.

Needs the er-small splits and the unzipped out_02 (for models/parser_bilstm_crf.pt):

    python3 analysis/parser_examples.py --data <folder with er-small> --out02 <unzipped out_02> [--out analysis/out_parser]

The rule parser, tokenizer and model code are taken from kaggle/scripts/02_parser_bilstm_crf.py (the cells up to
training), so they are exactly what the notebook ran. Writes parser_examples.md with three groups (val split):

  - how many raw val addresses have no comma at all (in er-small: only empty addresses)
  B. commas removed: the BiLSTM-CRF matches the silver parse exactly and the rules do not
  C. commas removed: both differ from the silver parse
Needs torch, pytorch-crf, pandas, scikit-learn, anyascii.
"""
import argparse, os, shutil, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--data', required=True, help='folder that contains train/train_source1.tsv (er-small)')
    ap.add_argument('--out02', required=True, help='unzipped out_02 (has models/parser_bilstm_crf.pt)')
    ap.add_argument('--out', default=str(ROOT / 'analysis' / 'out_parser'))
    ap.add_argument('--n', type=int, default=25, help='candidates listed per group')
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    # run the notebook's own setup / data / rules cells (no training) with the notebook's local-run switches
    os.environ.update(NLP_ER_SMOKE='1', NLP_ER_INPUT=str(Path(a.data).resolve()), NLP_ER_OUT=str(out / '_nb_scratch'))
    src = (ROOT / 'kaggle' / 'scripts' / '02_parser_bilstm_crf.py').read_text()
    head = src[:src.index('# ==== BiLSTM-CRF')]
    model_src = src[src.index('def wkey'):src.index('rng = np.random.default_rng(SEED)')]
    g = {'__name__': 'nb02'}
    exec(compile(head, '02_parser_bilstm_crf.py', 'exec'), g)
    exec(compile('from torchcrf import CRF\n' + model_src, '02_parser_bilstm_crf.py', 'exec'), g)
    torch, np, REC, REC_TOKS = g['torch'], g['np'], g['REC'], g['REC_TOKS']
    rule_tags, strip_commas, fields_from = g['rule_tags'], g['strip_commas'], g['fields_from']

    ck = torch.load(Path(a.out02) / 'models' / 'parser_bilstm_crf.pt', map_location='cpu', weights_only=False)
    g['W2I'], g['C2I'] = ck['W2I'], ck['C2I']
    model = g['BiLSTMCRF'](len(ck['W2I']) + 2, len(ck['C2I']) + 2, len(ck['TAGS']))
    model.load_state_dict(ck['state_dict']); model.to(g['DEVICE'])
    decode = lambda seqs: g['parser_decode'](model, seqs)

    def fields(toks, tags):
        f = fields_from(toks, tags)
        return '; '.join(f'{k}: {f[k]}' for k in ('HOUSE', 'STREET', 'LOCALITY', 'LANDMARK', 'CITY', 'STATE') if f.get(k))

    va = np.where(REC.split.values == 'val')[0]
    rng = np.random.default_rng(0)
    lines = ['# Parser examples (val split)\n']

    no_comma = [i for i in va if ',' not in REC.addr.values[i]]
    n_nonempty = sum(1 for i in no_comma if REC.addr.values[i].strip())
    lines.append(f'{len(no_comma):,} of {len(va):,} val records have no comma in the address; {n_nonempty:,} of them are non-empty. '
                 'So the comma-free condition below is a robustness test (commas removed on purpose), not a case found in this data.\n')

    # B / C. commas removed, compared with the silver parse of the comma version
    S = [i for i in rng.permutation(va) if len(REC_TOKS[i]) >= 6][:20_000]
    gold, seqs = [], []
    for i in S:
        t, y = strip_commas(REC_TOKS[i], rule_tags(REC_TOKS[i])); seqs.append(t); gold.append(y)
    pb = decode(seqs)
    B = [k for k in range(len(S)) if pb[k] == gold[k] and rule_tags(seqs[k]) != gold[k]
         and {'CITY', 'STATE'} <= set(gold[k]) and len(set(gold[k])) >= 4][:a.n]
    C = [k for k in range(len(S)) if pb[k] != gold[k] and rule_tags(seqs[k]) != gold[k]
         and sum(x != y for x, y in zip(pb[k], gold[k])) >= 2][:a.n]
    n_b = sum(pb[k] == gold[k] for k in range(len(S)))
    for title, ks in ((f'B. Commas removed: BiLSTM-CRF exact, rules wrong (BiLSTM-CRF exact on {n_b:,} of {len(S):,})', B),
                      ('C. Commas removed: both wrong (vs the silver parse of the comma version)', C)):
        lines.append(f'\n## {title}\n')
        for k in ks:
            i, t = S[k], seqs[k]
            lines.append(f'- `{REC.addr.values[i]}` ({REC.country.values[i]})\n  - silver (with commas): {fields(t, gold[k])}'
                         f'\n  - BiLSTM-CRF (no commas): {fields(t, pb[k])}\n  - rules (no commas): {fields(t, rule_tags(t))}')
    shutil.rmtree(out / '_nb_scratch', ignore_errors=True)
    (out / 'parser_examples.md').write_text('\n'.join(lines) + '\n')
    print('wrote', out / 'parser_examples.md')


if __name__ == '__main__':
    sys.exit(main())
