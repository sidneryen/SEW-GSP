"""Summarise results: protocol P1 (best over K x d, selected by ARI) and P2 (K=10, d=5)."""
import numpy as np, pandas as pd
from scipy.stats import friedmanchisquare, wilcoxon, rankdata
M = ['Raw', 'HIBOG', 'HIACSP', 'HIACSP-WF', 'SEW-GSP']
df = pd.read_csv('results/main_all.csv')
dsets = list(dict.fromkeys(df.dataset))

def p1(df):
    idx = df.groupby(['dataset', 'method', 'alg'])['ARI'].idxmax()
    return df.loc[idx]

def p2(df):
    return df[((df.K == 10) & (df.d == 5))]

out = {}
for pname, f in [('P1', p1), ('P2', p2)]:
    sel = f(df)
    for metric in ['ARI', 'NMI', 'ACC']:
        tab = sel.pivot_table(index=['alg', 'method'], columns='dataset', values=metric)[dsets] * 100
        tab['Avg'] = tab.mean(axis=1)
        out[(pname, metric)] = tab
        tab.round(1).to_csv(f'results/table_{pname}_{metric}.csv')
    # stats on ARI: blocks = dataset x clusterer
    piv = sel.pivot_table(index=['dataset', 'alg'], columns='method', values='ARI')[M]
    fr = friedmanchisquare(*[piv[m] for m in M])
    ranks = piv.apply(lambda r: rankdata(-r), axis=1, result_type='expand'); ranks.columns = M
    print(f'== {pname}: mean ARI per method', (piv.mean() * 100).round(2).to_dict())
    print('   mean rank', ranks.mean().round(2).to_dict(), 'Friedman p=%.2e' % fr.pvalue)
    for m in M[:-1]:
        w = wilcoxon(piv['SEW-GSP'], piv[m])
        wins = (piv['SEW-GSP'] > piv[m] + 1e-9).sum(); ties = (abs(piv['SEW-GSP'] - piv[m]) <= 1e-9).sum()
        print(f'   vs {m}: W/T/L = {wins}/{ties}/{len(piv)-wins-ties}, Wilcoxon p={w.pvalue:.2e}')
    ranks.mean().to_csv(f'results/ranks_{pname}.csv')
for k, t in out.items():
    if k[1] == 'ARI':
        print(k); print(t.round(1).to_string())
