"""Collect all numbers used by paper/build.js into results/paper_numbers.json."""
import json, sys, os
import numpy as np, pandas as pd
sys.path.insert(0, '.')
from sewsp.evalkit import load, MAIN_DATASETS
M = ['Raw', 'HIBOG', 'HIACSP', 'HIACSP-WF', 'SEW-GSP']
out = {'datasets': []}
for d in MAIN_DATASETS:
    X, y = load(d); out['datasets'].append([d, int(X.shape[0]), int(X.shape[1]), int(len(np.unique(y)))])
for p in ['P1', 'P2']:
    for m in ['ARI', 'NMI', 'ACC']:
        t = pd.read_csv(f'results/table_{p}_{m}.csv', index_col=[0, 1])
        out[f'{p}_{m}'] = {a: {me: {c: round(float(t.loc[(a, me), c]), 1) for c in t.columns} for me in M} for a in ['KM', 'AGG', 'DPC']}
a = pd.read_csv('results/ablation.csv')
out['ablation'] = (a.groupby('variant')[['ARI', 'NMI', 'ACC']].mean() * 100).round(2).reset_index().values.tolist()
ab = (a.pivot_table(index='variant', columns='dataset', values='ARI') * 100).round(1)
out['ablation_ds'] = {v: ab.loc[v][['iris+N20', 'wine+N20', 'glass', 'statlog', 'iris', 'wdbc']].tolist() for v in ab.index}
out['scal'] = pd.read_csv('results/scalability.csv').round(4).fillna(-1).values.tolist()
n = pd.read_csv('results/noise.csv')
out['noise'] = (n.groupby(['data', 'm', 'method'])['ARI'].mean() * 100).round(1).unstack().reset_index().values.tolist()
json.dump(out, open('results/paper_numbers.json', 'w'), indent=0)
print('ok')

# ---------------------------------------------------------------- revision (31 datasets, 9 methods)
from sewsp.evalkit import HD_DATASETS
M9 = ['Raw', 'HIBOG', 'HIAC*', 'HIACSP', 'HIACSP-WF', 'LS-GSP', 'MCFS-GSP', 'SEW-GSP']
dfm = pd.read_csv('results/main_all.csv')
ALLD = MAIN_DATASETS + HD_DATASETS
rev = json.load(open('results/rev_summary.json'))
for pname in ['P1', 'P2']:
    sel = dfm.loc[dfm.groupby(['dataset', 'method', 'alg'])['ARI'].idxmax()] if pname == 'P1' else dfm[(dfm.K == 10) & (dfm.d == 5)]
    t = (sel.groupby(['dataset', 'method'])['ARI'].mean().unstack()[M9].reindex(ALLD) * 100).round(1)
    rev[f'perds_{pname}'] = {ds: t.loc[ds].tolist() for ds in ALLD}
rp = pd.read_csv('results/relevance_profile.csv').set_index('dataset')
rev['relprof'] = {ds: round(float(rp.loc[ds, 'frac_relevant']) * 100, 1) for ds in ALLD}
rev['datasets_all'] = []
for d in ALLD:
    X, y = load(d); rev['datasets_all'].append([d, int(X.shape[0]), int(X.shape[1]), int(len(np.unique(y)))])
json.dump(rev, open('results/rev_numbers.json', 'w'), indent=0)
print('rev ok')

# ---------------------------------------------------------------- large-scale (run on a desktop machine)
lg = {}
for nm in ['mnist', 'fashion']:
    f = f'results/large_{nm}_70000.csv'
    if os.path.exists(f):
        lg[nm] = pd.read_csv(f).fillna(-1).to_dict(orient='records')
rev['large'] = lg
json.dump(rev, open('results/rev_numbers.json', 'w'), indent=0)
print('large ok', list(lg))
