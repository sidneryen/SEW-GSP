"""Large-scale experiment (to be run on a machine with many cores / >= 32 GB RAM).

Dataset: MNIST (70,000 x 784) or Fashion-MNIST, downloaded from the clustering benchmark
repository of Gagolewski (github.com/gagolews/clustering-data-v1).
Methods: Raw, HIACSP (all-pairs Dijkstra -> infeasible, reported as OOM), HIACSP (truncated
Dijkstra), HIACSP-WF (truncated), SEW-GSP (CNLR weights estimated on a 5000-object subsample).
Clusterers: K-means (10 restarts). AGG / DPC need an N x N matrix (39 GB for N = 70,000) and
are run only if --quadratic is given.

Usage:
  python run_large.py --data mnist  [--n 70000] [--K 10] [--d 5] [--quadratic]
  python run_large.py --data fashion
"""
import os, sys, time, argparse, gzip, urllib.request
import numpy as np, pandas as pd
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from sewsp.core import minmax, hiacsp, sewgsp
from sewsp.evalkit import cluster, scores

URL = 'https://raw.githubusercontent.com/gagolews/clustering-data-v1/master/mnist/{}.{}.gz'


def fetch(name):
    os.makedirs('data/large', exist_ok=True)
    npz = f'data/large/{name}.npz'          # offline copy (if GitHub is unreachable)
    if os.path.exists(npz):
        z = np.load(npz)
        return z['X'].astype(float), z['y'].astype(int)
    out = {}
    for part in ['data', 'labels0']:
        f = f'data/large/{name}.{part}.gz'
        if not os.path.exists(f):
            print('downloading', f, flush=True)
            urllib.request.urlretrieve(URL.format(name, part), f)
        out[part] = f
    X = np.loadtxt(out['data'], ndmin=2)
    y = np.loadtxt(out['labels0'], dtype=int)
    return X, y


def peak_mb():
    """Peak resident memory in MB (Linux/macOS via resource, Windows via psutil if installed)."""
    try:
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1024 / (1024 if sys.platform == 'darwin' else 1)
    except ImportError:
        try:
            import psutil
            return psutil.Process().memory_info().peak_wset / 2 ** 20
        except Exception:
            return float('nan')


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--data', default='mnist', choices=['mnist', 'fashion'])
    ap.add_argument('--n', type=int, default=0, help='random subsample size (0 = all)')
    ap.add_argument('--K', type=int, default=10)
    ap.add_argument('--d', type=int, default=5)
    ap.add_argument('--quadratic', action='store_true', help='also run AGG and DPC (O(N^2) memory)')
    a = ap.parse_args()
    name = {'mnist': 'digits', 'fashion': 'fashion'}[a.data]
    X, y = fetch(name)
    if a.n and a.n < len(y):
        idx = np.random.default_rng(0).choice(len(y), a.n, replace=False); X, y = X[idx], y[idx]
    k = len(np.unique(y)); N = len(y)
    print(f'{a.data}: N={N}, D={X.shape[1]}, k={k}', flush=True)
    algs = ['KM'] + (['AGG', 'DPC'] if a.quadratic else [])
    methods = {
        'Raw':                lambda: [minmax(X)],
        'HIACSP (all-pairs)': (lambda: hiacsp(X, a.K, a.d, truncated=False)) if N * N * 8 < 4 * 2 ** 30 else None,
        'HIACSP':             lambda: hiacsp(X, a.K, a.d, truncated=True),
        'HIACSP-WF':          lambda: hiacsp(X, a.K, a.d, weights='variance', truncated=True),
        'SEW-GSP':            lambda: sewgsp(X, a.K, a.d, weight_sample=5000),
    }
    rows = []
    for m, f in methods.items():
        if f is None:
            need = N * N * 8 / 2 ** 30
            print(f'{m}: skipped, distance matrix alone needs {need:.1f} GB', flush=True)
            rows.append(dict(method=m, alg='-', time=np.nan, note=f'needs {need:.1f} GB'))
            continue
        t = time.time(); Z = f()[-1]; t = time.time() - t
        for al in algs:
            tc = time.time(); s = scores(y, cluster(al, Z, k)); tc = time.time() - tc
            rows.append(dict(method=m, alg=al, time=t, t_cluster=tc, peak_MB=peak_mb(), **s))
            print(rows[-1], flush=True)
    os.makedirs('results', exist_ok=True)
    pd.DataFrame(rows).to_csv(f'results/large_{a.data}_{N}.csv', index=False)
