"""Fraction of features whose CNLR relevance exceeds the 95th percentile of a permutation null
(how much 'irrelevant-feature' structure a dataset contains)."""
import sys, numpy as np, pandas as pd
sys.path.insert(0, '.')
from sewsp.core import minmax, relevance_cross
from sewsp.evalkit import load, MAIN_DATASETS, HD_DATASETS
rows = []
for n in MAIN_DATASETS + HD_DATASETS:
    X, y = load(n); X0 = minmax(X)
    if X0.shape[0] > 3000:
        X0 = X0[np.random.default_rng(0).choice(X0.shape[0], 3000, replace=False)]
    D = X0.shape[1]
    keep = np.where(X0.std(0) > 0)[0]; X0 = X0[:, keep]
    r = relevance_cross(X0, np.ones(X0.shape[1]), 10)
    rng = np.random.default_rng(1)
    Xp = np.column_stack([rng.permutation(X0[:, m]) for m in range(X0.shape[1])])
    rn = relevance_cross(np.hstack([X0, Xp]), np.ones(2 * X0.shape[1]), 10)[X0.shape[1]:]
    q = np.quantile(rn, 0.95)
    rows.append(dict(dataset=n, D=D, nonconst=len(keep), frac_relevant=float((r > q).mean()),
                     median_r=float(np.median(r)), null_q95=float(q)))
    print(rows[-1], flush=True)
pd.DataFrame(rows).to_csv('results/relevance_profile.csv', index=False)
