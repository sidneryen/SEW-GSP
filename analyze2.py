"""Revision analysis: 9 optimisers x 31 datasets x 3 clusterers, plus direct-clustering baselines.
Writes results/rev_*.csv and prints a summary used in the manuscript."""
import json
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
import sys
sys.path.insert(0, '.')
from sewsp.evalkit import MAIN_DATASETS, HD_DATASETS

M = ['Raw', 'HIBOG', 'HIAC*', 'HIACSP', 'HIACSP-WF', 'LS-GSP', 'MCFS-GSP', 'SEW-GSP']
df = pd.read_csv('results/main_all.csv')
df = df[df.method.isin(M)]
ALLD = MAIN_DATASETS + HD_DATASETS
GROUPS = {'all': ALLD, 'orig16': MAIN_DATASETS, 'hd15': HD_DATASETS}


def p1(d):
    return d.loc[d.groupby(['dataset', 'method', 'alg'])['ARI'].idxmax()]


def p2(d):
    return d[(d.K == 10) & (d.d == 5)]


def holm(ps):
    ps = np.asarray(ps); o = np.argsort(ps); m = len(ps); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * ps[i]); adj[i] = min(1, run)
    return adj


summary = {}
for pname, f in [('P1', p1), ('P2', p2)]:
    sel = f(df)
    for metric in ['ARI', 'NMI', 'ACC']:
        tab = sel.pivot_table(index='dataset', columns=['alg', 'method'], values=metric)
        tab = tab.reindex(ALLD) * 100
        tab.round(1).to_csv(f'results/rev_{pname}_{metric}_full.csv')
    for gname, ds in GROUPS.items():
        s = sel[sel.dataset.isin(ds)]
        piv = s.pivot_table(index=['dataset', 'alg'], columns='method', values='ARI')[M]
        ranks = piv.apply(lambda r: rankdata(-r), axis=1, result_type='expand'); ranks.columns = M
        fr = friedmanchisquare(*[piv[m] for m in M]).pvalue
        comp = {}
        raw_p = []
        for m in M[:-1]:
            diff = piv['SEW-GSP'] - piv[m]
            w = (diff > 1e-9).sum(); t = (diff.abs() <= 1e-9).sum(); l = len(diff) - w - t
            p = wilcoxon(piv['SEW-GSP'], piv[m]).pvalue if (diff.abs() > 1e-9).any() else 1.0
            raw_p.append(p); comp[m] = dict(W=int(w), T=int(t), L=int(l), p=float(p))
        for m, ph in zip(M[:-1], holm(raw_p)):
            comp[m]['p_holm'] = float(ph)
        means = {}
        for metric in ['ARI', 'NMI', 'ACC']:
            mm = s.pivot_table(index=['dataset', 'alg'], columns='method', values=metric)[M]
            means[metric] = (mm.mean() * 100).round(2).to_dict()
            means[metric + '_by_alg'] = (mm.groupby(level='alg').mean() * 100).round(2).to_dict(orient='index')
        summary[f'{pname}_{gname}'] = dict(n_blocks=len(piv), friedman_p=float(fr),
                                           mean_rank=ranks.mean().round(3).to_dict(), means=means, vs=comp)
        print(f'== {pname} {gname} (blocks={len(piv)}) Friedman p={fr:.2e}')
        print('   ARI', means['ARI'])
        print('   rank', ranks.mean().round(2).to_dict())
        print('   ', {m: (c['W'], c['T'], c['L'], round(c['p_holm'], 4)) for m, c in comp.items()})

# ---------------------------------------------------------------- direct clustering comparison
import os
if not os.path.exists('results/direct_all.csv'):
    json.dump(summary, open('results/rev_summary.json', 'w'), indent=1); sys.exit()
dr = pd.read_csv('results/direct_all.csv')
rows = {}
for pname in ['P1', 'P2']:
    if pname == 'P1':
        dsel = dr.loc[dr.groupby(['dataset', 'method'])['ARI'].idxmax()]
    else:
        dsel = dr[dr['default']]
    piv = dsel.pivot_table(index='dataset', columns='method', values=['ARI', 'NMI', 'ACC'])
    sew = (p1 if pname == 'P1' else p2)(df)
    sew = sew[sew.method == 'SEW-GSP'].pivot_table(index='dataset', columns='alg', values=['ARI', 'NMI', 'ACC'])
    out = {}
    for metric in ['ARI', 'NMI', 'ACC']:
        t = pd.concat([piv[metric], sew[metric].add_prefix('SEW-GSP+')], axis=1).reindex(ALLD) * 100
        t.round(1).to_csv(f'results/rev_direct_{pname}_{metric}.csv')
        out[metric] = {g: t.loc[ds].mean().round(2).to_dict() for g, ds in GROUPS.items()}
    tA = pd.concat([piv['ARI'], sew['ARI'].add_prefix('SEW-GSP+')], axis=1).reindex(ALLD)
    best_sew = tA[[c for c in tA.columns if c.startswith('SEW')]].max(1)
    out['wins_bestSEW_vs'] = {c: [int((best_sew > tA[c] + 1e-9).sum()), int((best_sew < tA[c] - 1e-9).sum()),
                                  float(wilcoxon(best_sew, tA[c]).pvalue)] for c in ['EWKM', 'SparseKM', 'Spectral']}
    out['wins_SEWKM_vs'] = {c: [int((tA['SEW-GSP+KM'] > tA[c] + 1e-9).sum()), int((tA['SEW-GSP+KM'] < tA[c] - 1e-9).sum()),
                                float(wilcoxon(tA['SEW-GSP+KM'], tA[c]).pvalue)] for c in ['EWKM', 'SparseKM', 'Spectral']}
    rows[pname] = out
    print('== direct', pname, json.dumps(out['ARI']['all']), out['wins_bestSEW_vs'], out['wins_SEWKM_vs'])
summary['direct'] = rows
json.dump(summary, open('results/rev_summary.json', 'w'), indent=1)
