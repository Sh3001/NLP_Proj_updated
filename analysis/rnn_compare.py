"""Issue #7: compare the Siamese char-BiLSTM / BiGRU / vanilla BiRNN runs (05a / 05b / 05c).

Unzip out_05a.zip, out_05b.zip, out_05c.zip (any folder layout), then:

    python3 analysis/rnn_compare.py <folder with the unzipped outputs> [--out analysis/out_rnn]

Writes rnn_table.md (test F0.5, P, R, per-country F0.5, parameters, training minutes),
rnn_epochs.md (per-epoch losses, val F0.5, gradient norms), rnn_val_loss.png and rnn_grad_by_distance.png.
Needs only numpy, pandas and matplotlib.
"""
import argparse, glob, json
from pathlib import Path

import pandas as pd

ORDER = ['Siamese char-BiLSTM', 'Siamese char-BiGRU', 'Siamese char-BiRNN (vanilla)']
SHORT = {'Siamese char-BiLSTM': 'BiLSTM', 'Siamese char-BiGRU': 'BiGRU', 'Siamese char-BiRNN (vanilla)': 'BiRNN (vanilla)'}
# categorical slots 1-3 of the validated default palette (light mode) + distinct line styles (not colour alone)
STYLE = {'BiLSTM': ('#2a78d6', '-', 'o'), 'BiGRU': ('#eb6834', '--', 's'), 'BiRNN (vanilla)': ('#1baf7a', ':', '^')}
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def find(root, pattern):
    return sorted(Path(p) for p in glob.glob(str(Path(root) / '**' / pattern), recursive=True))


def md_table(df):
    cols = list(df.columns)
    lines = ['| ' + ' | '.join(cols) + ' |', '|' + '---|' * len(cols)]
    lines += ['| ' + ' | '.join('' if pd.isna(v) else str(v) for v in r) + ' |' for r in df.itertuples(index=False)]
    return '\n'.join(lines) + '\n'


def style_axes(ax):
    for s in ('top', 'right'): ax.spines[s].set_visible(False)
    for s in ('left', 'bottom'): ax.spines[s].set_color(INK2)
    ax.tick_params(colors=INK2, labelsize=9); ax.grid(True, color=GRID, lw=0.8); ax.set_axisbelow(True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('root', help='folder that contains the unzipped out_05a/b/c outputs')
    ap.add_argument('--out', default='analysis/out_rnn')
    a = ap.parse_args(); out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    res = {}
    for p in find(a.root, 'results/*.json'):
        r = json.load(open(p))
        if r.get('method') in ORDER: res[r['method']] = r
    hist = {}
    for p in find(a.root, 'diagnostics/*.json'):
        h = json.load(open(p))
        if h.get('method') in ORDER: hist[h['method']] = h
    methods = [m for m in ORDER if m in res]
    if not methods: raise SystemExit(f'no 05a/b/c results/*.json found under {a.root}')
    missing = [SHORT[m] for m in ORDER if m not in res]
    if missing: print('missing runs:', missing)
    if any(str(res[m].get('quick')).lower() == 'true' for m in methods): print('WARNING: some runs are QUICK_RUN = True')

    # ---- table ---------------------------------------------------------------------------------
    rows = []
    for m in methods:
        r = res[m]; h = hist.get(m, {})
        ep = h.get('epochs', [])
        rows.append({'model': SHORT[m], 'val F0.5': r['val_f05'], 'test F0.5': r['test_f05'], 'test P': r['test_P'],
                     'test R': r['test_R'], 'F0.5 US': r.get('test_f05_US'), 'F0.5 India': r.get('test_f05_India'),
                     'singleton acc': r.get('test_singleton_acc'), 'threshold': r['threshold'], 'params': r.get('params'),
                     'train min': r.get('train_min'), 'final val loss': round(ep[-1]['val_loss'], 4) if ep else None})
    T = pd.DataFrame(rows)
    print(T.to_string(index=False)); (out / 'rnn_table.md').write_text(md_table(T))

    if not hist:
        print('no diagnostics/*.json: these runs were made with the old notebooks (val loss only in the log)'); return
    E = pd.DataFrame([{'model': SHORT[m], **e} for m in methods if m in hist for e in hist[m]['epochs']])
    E = E.round({'train_loss': 4, 'val_loss': 4, 'val_f05': 5, 'grad_norm_mean': 3, 'grad_norm_max': 3, 'clipped_share': 3, 'minutes': 1})
    print(E.to_string(index=False)); (out / 'rnn_epochs.md').write_text(md_table(E))

    # gradient reaching k characters back, relative to k = 0 (after training)
    G = []
    for m in methods:
        if m not in hist: continue
        g = hist[m]['grad_by_distance']; last = g[sorted(k for k in g if k.startswith('epoch'))[-1]]
        G.append({'model': SHORT[m], **{f'k={k}': f'{last[k] / last[0]:.1e}' for k in (10, 25, 50, 75, 99) if k < len(last)},
                  **{f'init k={k}': f"{g['init'][k] / g['init'][0]:.1e}" for k in (50, 99) if k < len(g['init'])}})
    G = pd.DataFrame(G); print(G.to_string(index=False)); (out / 'rnn_grad_by_distance.md').write_text(md_table(G))

    # ---- plots ---------------------------------------------------------------------------------
    import matplotlib; matplotlib.use('Agg'); import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'text.color': INK, 'axes.labelcolor': INK2})

    fig, axs = plt.subplots(1, 2, figsize=(10, 3.8))
    for key, ax, title in (('val_loss', axs[0], 'Validation loss (BCE)'), ('train_loss', axs[1], 'Training loss (BCE)')):
        for name, d in E.groupby('model', sort=False):
            c, ls, mk = STYLE[name]
            ax.plot(d.epoch, d[key], color=c, ls=ls, marker=mk, ms=8, lw=2, label=name)
            if key == 'val_loss':                         # direct labels on one panel only (the legend names all)
                ax.annotate(name, (d.epoch.iloc[-1], d[key].iloc[-1]), xytext=(6, 0), textcoords='offset points',
                            va='center', fontsize=9, color=INK)
        ax.set_title(title, loc='left', fontsize=11); ax.set_xlabel('epoch'); ax.set_xticks(sorted(E.epoch.unique()))
        ax.set_xlim(E.epoch.min() - 0.2, E.epoch.max() + 0.9); style_axes(ax)
    axs[0].legend(frameon=False, fontsize=9)
    plt.tight_layout(); plt.savefig(out / 'rnn_val_loss.png', dpi=160); plt.close()

    fig, ax = plt.subplots(figsize=(7, 4))
    for m in methods:
        if m not in hist: continue
        name = SHORT[m]; c, ls, _ = STYLE[name]; g = hist[m]['grad_by_distance']
        last = g[sorted(k for k in g if k.startswith('epoch'))[-1]]
        ax.semilogy(range(len(last)), last, color=c, ls=ls, lw=2, label=name)
        ax.annotate(name, (len(last) - 1, last[-1]), xytext=(6, 0), textcoords='offset points', va='center', fontsize=9, color=INK)
    ax.set_title('How far back the last state still "sees": ‖∂h_T / ∂x_(T−k)‖ after training', loc='left', fontsize=10)
    ax.set_xlabel('k = characters before the last one'); ax.set_ylabel('gradient norm (log scale)')
    ax.legend(frameon=False, fontsize=9); style_axes(ax); ax.set_xlim(0, 118)
    plt.tight_layout(); plt.savefig(out / 'rnn_grad_by_distance.png', dpi=160); plt.close()
    print('wrote', *sorted(p.name for p in out.iterdir()))


if __name__ == '__main__':
    main()
