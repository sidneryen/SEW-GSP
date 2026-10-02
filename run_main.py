"""Experiment 1: main comparison of dataset-optimisation methods with K-means, AGG and DPC.

For every (dataset, method, K) the optimiser is run once for d_max iterations and every
snapshot d in D_EVAL is clustered, so both protocols can be derived from one CSV:
  P1 (grid / oracle, as in the HIBOG-HIACSP-WF literature): best over K x d
  P2 (label-free defaults): K = 10, d = 5
Usage: python run_main.py [n_jobs]
"""
import os, sys, time, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sewsp.core import ALL_METHODS as METHODS
from sewsp.evalkit import load, cluster, scores, MAIN_DATASETS, HD_DATASETS

OUT = 'results/main'
K_GRID = [5, 10, 15, 20, 25, 30]
D_EVAL = [1, 2, 3, 4, 5, 6, 8, 10]
ALGS = ['KM', 'AGG', 'DPC']


def job(args):
    ds, method = args
    f = f'{OUT}/{ds}__{method}.csv'
    if os.path.exists(f):
        return f
    X, y = load(ds)
    k = len(np.unique(y))
    rows = []
    big = len(y) > 9000                     # USPS: reduced grid (same for every method)
    kg, dg = ([10, 20, 30], [2, 5, 10]) if big else (K_GRID, D_EVAL)
    for K in ([10] if method == 'Raw' else kg):
        t0 = time.time()
        snaps = METHODS[method](X, K, max(dg))
        t_opt = time.time() - t0
        for d in ([5] if method == 'Raw' else dg):
            Z = snaps[d - 1]
            for a in ALGS:
                s = scores(y, cluster(a, Z, k))
                rows.append(dict(dataset=ds, method=method, K=K, d=d, alg=a, time=t_opt, **s))
    pd.DataFrame(rows).to_csv(f, index=False)
    print('done', ds, method, flush=True)
    return f


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    if len(sys.argv) > 1 and sys.argv[1] == '--list':      # print pending jobs (for xargs -P)
        for d_, m_ in itertools.product(MAIN_DATASETS + HD_DATASETS, METHODS.keys()):
            if not os.path.exists(f'{OUT}/{d_}__{m_}.csv'):
                print(d_, m_)
        sys.exit()
    if len(sys.argv) > 1 and sys.argv[1] == '--job':       # run one job: --job DATASET METHOD
        job((sys.argv[2], sys.argv[3])); sys.exit()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    jobs = list(itertools.product(MAIN_DATASETS + HD_DATASETS, METHODS.keys()))
    # big datasets first for better load balance
    jobs.sort(key=lambda j: -len(load(j[0])[1]))
    with Pool(n) as p:
        list(p.imap_unordered(job, jobs))
    df = pd.concat([pd.read_csv(f'{OUT}/{a}__{b}.csv') for a, b in jobs])
    df.to_csv('results/main_all.csv', index=False)
