"""Feature importance of the LightGBM baseline (04), by feature and by feature family.

    python3 analysis/gbm_importance_plot.py [analysis/out_gbm/gbm_feature_importance.csv]

Input: `gbm_feature_importance.csv` from the `er-04-output` Kaggle dataset (total gain per feature, run with the
BiLSTM-CRF fields from 02). Writes `analysis/out_gbm/gbm_feature_gain.png` and `gbm_feature_families.md`.
"""
import csv, sys
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

OUT = Path(__file__).resolve().parent / 'out_gbm'

RETRIEVAL = {'tf_cos', 'de_cos', 'tf_rank', 'de_rank', 'rrf_rank', 'n_cand', 'tf_cos_gap', 'de_cos_gap'}
PARSED = {'house_eq', 'house_ratio', 'post_eq', 'city_ratio', 'street_tset', 'locality_tset', 'q_landmark'}
META = {'q_src3', 'q_addr_empty', 'q_nlen', 's_nlen', 'q_alen', 'q_latin', 'q_devanagari', 'q_malayalam', 'q_other_indic'}
NUMBERS = {'num_jac', 'num_onlyq', 'num_onlys', 'first_num_eq'}

# (family, sub-family) -> label; colours: the validated categorical slots 1-4, in fixed order
FAMILIES = [('retrieval', "retrieval scores and ranks from 03 (TF-IDF, fine-tuned e5, RRF)", '#2a78d6'),
            ('string', 'string, edit-distance and number features', '#eb6834'),
            ('meta', 'record metadata (lengths, script, source)', '#1baf7a'),
            ('parsed', 'parsed-address fields (02)', '#eda100')]
COLOR = {k: c for k, _, c in FAMILIES}


def family(f):
    if f in RETRIEVAL: return 'retrieval'
    if f in PARSED: return 'parsed'
    if f in META: return 'meta'
    return 'string'


def sub_family(f):   # finer split of the string family, for the table only
    if f in NUMBERS: return 'numbers in the address'
    if f.startswith('a_'): return 'address string similarity'
    return 'name string similarity'


def main():
    src = Path(sys.argv[1]) if len(sys.argv) > 1 else OUT / 'gbm_feature_importance.csv'
    imp = {r[0]: float(r[1]) for r in list(csv.reader(open(src)))[1:]}
    tot = sum(imp.values())
    share = {f: 100 * v / tot for f, v in sorted(imp.items(), key=lambda x: -x[1])}

    fam_share = {k: sum(s for f, s in share.items() if family(f) == k) for k, _, _ in FAMILIES}
    fam_n = {k: sum(family(f) == k for f in share) for k, _, _ in FAMILIES}
    sub = {}
    for f, s in share.items():
        if family(f) == 'string': sub[sub_family(f)] = sub.get(sub_family(f), 0) + s

    lines = ['| feature family | features | share of gain |', '|---|---|---|']
    for k, label, _ in FAMILIES:
        lines.append(f'| {label} | {fam_n[k]} | {fam_share[k]:.1f}% |')
        if k == 'string':
            for name in ('numbers in the address', 'name string similarity', 'address string similarity'):
                lines.append(f'| &nbsp;&nbsp;of which {name} | {sum(sub_family(f) == name for f in share if family(f) == "string")} | {sub[name]:.1f}% |')
    (OUT / 'gbm_feature_families.md').write_text('\n'.join(lines) + '\n')
    print('\n'.join(lines))

    # ---- figure: top 15 features (left), families (right); one x-scale per panel, both "% of total gain"
    plt.rcParams.update({'font.size': 9, 'font.family': 'DejaVu Sans', 'axes.edgecolor': '#c8c7c2',
                         'axes.labelcolor': '#52514e', 'xtick.color': '#52514e', 'ytick.color': '#0b0b0b'})
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(9.2, 4.4), gridspec_kw={'width_ratios': [1.55, 1], 'wspace': 0.62})
    fig.patch.set_facecolor('white')
    top = list(share.items())[:15][::-1]
    ax.barh([f for f, _ in top], [s for _, s in top], height=0.7, color=[COLOR[family(f)] for f, _ in top],
            edgecolor='white', linewidth=2)
    for i, (f, s) in enumerate(top):
        ax.text(s + 0.8, i, f'{s:.1f}%', va='center', fontsize=8, color='#0b0b0b')
    ax.set_xlim(0, 60); ax.set_xlabel('share of total gain (%)')
    ax.set_title('Top 15 of 48 features', loc='left', fontsize=10, color='#0b0b0b')
    for a in (ax, ax2):
        a.spines[['top', 'right']].set_visible(False); a.tick_params(length=0)
        a.grid(axis='x', color='#ebeae6', linewidth=0.8); a.set_axisbelow(True)
    for t in ax.get_yticklabels(): t.set_fontfamily('DejaVu Sans Mono')

    fams = [(k, label) for k, label, _ in FAMILIES][::-1]
    short = {'retrieval': 'retrieval (03)', 'string': 'string / number', 'meta': 'metadata', 'parsed': 'parsed fields (02)'}
    ax2.barh([short[k] for k, _ in fams], [fam_share[k] for k, _ in fams], height=0.5,
             color=[COLOR[k] for k, _ in fams], edgecolor='white', linewidth=2)
    for i, (k, _) in enumerate(fams):
        ax2.text(fam_share[k] + 1.5, i, f'{fam_share[k]:.1f}%  ({fam_n[k]} feat.)', va='center', fontsize=8, color='#0b0b0b')
    ax2.set_xlim(0, 150); ax2.set_xticks([0, 25, 50, 75, 100]); ax2.set_xlabel('share of total gain (%)')
    ax2.set_title('By feature family', loc='left', fontsize=10, color='#0b0b0b')

    handles = [plt.Rectangle((0, 0), 1, 1, color=c) for _, _, c in FAMILIES]
    fig.legend(handles, [label for _, label, _ in FAMILIES], loc='lower center', ncol=2, frameon=False, fontsize=8,
               bbox_to_anchor=(0.5, -0.02))
    fig.subplots_adjust(left=0.13, right=0.98, top=0.91, bottom=0.25)
    fig.savefig(OUT / 'gbm_feature_gain.png', dpi=200)
    print('wrote', OUT / 'gbm_feature_gain.png')


if __name__ == '__main__':
    sys.exit(main())
