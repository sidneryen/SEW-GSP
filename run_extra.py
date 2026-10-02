"""Experiments 2-5: noise robustness, weight recovery, ablation, scalability.
Usage: python run_extra.py [noise|weights|ablation|scal|all]
"""
import os, sys, time
import numpy as np, pandas as pd
from sklearn.metrics import roc_auc_score
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sewsp.core import (METHODS, minmax, knn_graph, geodesic_knn_full, geodesic_knn_truncated,
                        sewgsp, hiacsp, variance_weights, laplacian_scores, sew_weights, EPS)
from sewsp.evalkit import load, cluster, scores, synthetic_gauss, add_noise_features, MAIN_DATASETS

ALGS = ['KM', 'AGG', 'DPC']
K, D = 10, 5
os.makedirs('results', exist_ok=True)


def eval_all(Z, y):
    k = len(np.unique(y))
    return {a: scores(y, cluster(a, Z, k)) for a in ALGS}


# --------------------------------------------------------------------------- E2
def exp_noise():
    rows = []
    for seed in range(5):
        for m in [0, 5, 10, 20, 40, 80]:
            X, y = synthetic_gauss(n=500, k=5, d_inf=5, d_noise=m, sep=2.5, seed=seed)
            for meth, f in METHODS.items():
                Z = f(X, K, D)[-1]
                for a, s in eval_all(Z, y).items():
                    rows.append(dict(data='Synthetic', m=m, seed=seed, method=meth, alg=a, **s))
        print('noise syn seed', seed, flush=True)
    for name in ['iris', 'wine', 'wdbc']:
        X0, y = load(name)
        for seed in range(3):
            for m in [0, 10, 20, 40]:
                X = add_noise_features(X0, m, seed) if m else X0
                for meth, f in METHODS.items():
                    Z = f(X, K, D)[-1]
                    for a, s in eval_all(Z, y).items():
                        rows.append(dict(data=name, m=m, seed=seed, method=meth, alg=a, **s))
        print('noise', name, flush=True)
    pd.DataFrame(rows).to_csv('results/noise.csv', index=False)


# --------------------------------------------------------------------------- E3
def exp_weights():
    rows = []
    for seed, m, nt in [(s_, m_, t_) for s_ in range(10) for m_ in [5, 10, 20, 40, 80]
                        for t_ in ['uniform', 'gaussian']]:
        if True:
            # 5 high-variance informative dims + 2 low-variance informative dims + m noise dims
            X, y = synthetic_gauss(n=500, k=5, d_inf=5, d_noise=m, sep=2.5, seed=seed, low_var_inf=2, noise=nt)
            X0 = minmax(X)
            truth = np.r_[np.ones(7), np.zeros(m)]
            ws = {'Variance (HIACSP-WF)': variance_weights(X0),
                  'Laplacian score': np.clip(1 - laplacian_scores(X0, X0, K), 0, None),
                  'CNLR (proposed)': sew_weights(X0, K, 'shrink')}
            for nm, w in ws.items():
                rows.append(dict(seed=seed, m=m, noise=nt, weighting=nm, AUC=roc_auc_score(truth, w),
                                 mass_inf=w[:7].sum() / w.sum(),
                                 auc_lowvar=roc_auc_score(np.r_[np.ones(2), np.zeros(m)],
                                                          np.r_[w[5:7], w[7:]])))
    pd.DataFrame(rows).to_csv('results/weights.csv', index=False)
    # weight profile on wine+N20 for a figure
    X, y = load('wine+N20'); X0 = minmax(X)
    prof = pd.DataFrame({'feature': np.arange(X0.shape[1]),
                         'Variance (HIACSP-WF)': variance_weights(X0),
                         'CNLR (proposed)': sew_weights(X0, K, 'shrink')})
    prof.to_csv('results/weights_wine_n20.csv', index=False)


# --------------------------------------------------------------------------- E4
VARIANTS = {
    'A0 HIACSP (equal w, hard)':     lambda X: hiacsp(X, K, D, truncated=True),
    'A1 +variance w (=HIACSP-WF)':   lambda X: sewgsp(X, K, D, soft=False, weighting='variance'),
    'A2 +Laplacian-score w':          lambda X: sewgsp(X, K, D, soft=False, weighting='ls'),
    'A3a +CNLR w, no shrinkage (hard)': lambda X: sewgsp(X, K, D, soft=False, weighting='cnlr', wmode='linear'),
    'A3 +CNLR w (hard boundary)':     lambda X: sewgsp(X, K, D, soft=False, weighting='cnlr'),
    'A4 +CNLR w +soft boundary (full)': lambda X: sewgsp(X, K, D, soft=True, weighting='cnlr'),
    'A5 full +weight co-evolution':   lambda X: sewgsp(X, K, D, soft=True, weighting='cnlr', coevolve=True),
    'W-only CNLR (no optimisation)':  lambda X: [minmax(X) * np.sqrt(sew_weights(minmax(X), K, 'shrink'))],
}


def exp_ablation():
    rows = []
    for ds in MAIN_DATASETS:
        X, y = load(ds)
        for v, f in VARIANTS.items():
            Z = f(X)[-1]
            for a, s in eval_all(Z, y).items():
                rows.append(dict(dataset=ds, variant=v, alg=a, **s))
        print('ablation', ds, flush=True)
    pd.DataFrame(rows).to_csv('results/ablation.csv', index=False)


# --------------------------------------------------------------------------- E5
def exp_scal():
    rows = []
    rng = np.random.default_rng(0)
    for n in [1000, 2000, 4000, 8000, 16000, 32000, 64000]:
        C = rng.uniform(0, 30, size=(20, 2))
        X = C[rng.integers(0, 20, n)] + rng.normal(0, 1, (n, 2))
        Z = minmax(X)
        G, _, _ = knn_graph(Z, K)
        geodesic_knn_truncated(G, K)            # JIT warm-up
        t = time.time(); a, da = geodesic_knn_truncated(G, K); tt = time.time() - t
        tf, same = np.nan, np.nan
        if n <= 8000:
            t = time.time(); b, db = geodesic_knn_full(G, K); tf = time.time() - t
            same = float(np.allclose(np.sort(da, 1), np.sort(db, 1)))
        t = time.time(); sewgsp(X, K, D); tw = time.time() - t
        rows.append(dict(n=n, t_full=tf, t_trunc=tt, exact=same, t_sewgsp=tw,
                         mem_full_MB=n * n * 8 / 2 ** 20, mem_trunc_MB=n * K * 16 / 2 ** 20))
        print(rows[-1], flush=True)
    pd.DataFrame(rows).to_csv('results/scalability.csv', index=False)


if __name__ == '__main__':
    what = sys.argv[1] if len(sys.argv) > 1 else 'all'
    for nm, fn in [('weights', exp_weights), ('scal', exp_scal), ('ablation', exp_ablation),
                   ('noise', exp_noise)]:
        if what in (nm, 'all'):
            fn()
