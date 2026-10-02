"""Experiment 1b (revision): SEW-GSP + simple clusterer versus clustering algorithms that
handle feature relevance / non-convex structure themselves:
  EWKM (entropy-weighted k-means), Sparse k-means, Spectral clustering.
Each is run on the min-max normalised data with a parameter grid (P1 = best by ARI) and a
fixed default (P2).  Usage: python run_direct.py [n_jobs]
"""
import os, sys, itertools
import numpy as np, pandas as pd
from multiprocessing import Pool
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sewsp.core import minmax
from sewsp.evalkit import load, scores, ewkm, sparse_kmeans, spectral, MAIN_DATASETS, HD_DATASETS

OUT = 'results/direct'


def grids(D):
    sq = np.sqrt(D)
    return {
        'EWKM':      ([0.1, 0.5, 1, 2, 5, 10, 20], 1),
        'SparseKM':  (sorted({1.5, 2.0, max(1.5, sq / 4), max(1.5, sq / 2), max(1.5, 0.75 * sq)}), max(1.5, sq / 2)),
        'Spectral':  ([5, 10, 15, 20, 25, 30], 10),
    }


def run(alg, X, k, p):
    if alg == 'EWKM':
        return ewkm(X, k, gamma=p)
    if alg == 'SparseKM':
        return sparse_kmeans(X, k, s=p)
    return spectral(X, k, n_neighbors=p)


def job(args):
    ds, alg = args
    f = f'{OUT}/{ds}__{alg}.csv'
    if os.path.exists(f):
        return
    X, y = load(ds); X = minmax(X); k = len(np.unique(y))
    grid, default = grids(X.shape[1])[alg]
    rows = []
    for p in grid:
        try:
            s = scores(y, run(alg, X, k, p))
        except Exception as e:                      # e.g. disconnected graph in spectral
            print('fail', ds, alg, p, e, flush=True); continue
        rows.append(dict(dataset=ds, method=alg, param=float(p), default=bool(np.isclose(p, default)), **s))
    pd.DataFrame(rows).to_csv(f, index=False)
    print('done', ds, alg, flush=True)


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2
    jobs = list(itertools.product(MAIN_DATASETS + HD_DATASETS, ['EWKM', 'SparseKM', 'Spectral']))
    jobs.sort(key=lambda j: -len(load(j[0])[1]))
    with Pool(n) as p:
        list(p.imap_unordered(job, jobs))
    pd.concat([pd.read_csv(f'{OUT}/{a}__{b}.csv') for a, b in jobs
               if os.path.exists(f'{OUT}/{a}__{b}.csv')]).to_csv('results/direct_all.csv', index=False)
