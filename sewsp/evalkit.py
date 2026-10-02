"""Datasets, clustering algorithms (K-means, Agglomerative/Ward, DPC) and metrics."""
import os
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import pdist, squareform
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import adjusted_rand_score, normalized_mutual_info_score
from sklearn import datasets as skd

DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'data')


# ----------------------------------------------------------------------------
# datasets
# ----------------------------------------------------------------------------
def _load_bench(name):
    X = np.loadtxt(os.path.join(DATA_DIR, f'{name}.data.gz'), ndmin=2)
    y = np.loadtxt(os.path.join(DATA_DIR, f'{name}.labels0.gz'), dtype=int)
    return X, y


def add_noise_features(X, m, seed=0):
    rng = np.random.default_rng(seed)
    lo, hi = X.min(0), X.max(0)
    N = rng.uniform(0, 1, size=(X.shape[0], m)) * (hi - lo).mean() + lo.mean()
    return np.hstack([X, N])


def synthetic_gauss(n=600, k=5, d_inf=5, d_noise=20, sep=3.0, seed=0, low_var_inf=0, noise='uniform'):
    """k Gaussian clusters living in d_inf informative dims, plus uniform noise dims.
    low_var_inf extra informative dims that only separate one small cluster
    (discriminative but low-variance after min-max scaling)."""
    rng = np.random.default_rng(seed)
    C = rng.normal(0, sep, size=(k, d_inf))
    y = rng.integers(0, k, n)
    Xi = C[y] + rng.normal(0, 1, size=(n, d_inf))
    parts = [Xi]
    if low_var_inf:
        L = rng.normal(0, 1, size=(n, low_var_inf))
        L[y == 0] += 6.0
        parts.append(L)
    lo, hi = Xi.min(), Xi.max()
    if noise == 'uniform':
        parts.append(rng.uniform(lo, hi, size=(n, d_noise)))
    else:   # Gaussian irrelevant features (unimodal, lower variance after min-max scaling)
        parts.append(rng.normal(0, Xi.std(), size=(n, d_noise)))
    return np.hstack(parts), y


HD_DATASETS = ['leukemia', 'colon', 'lung', 'lymphoma', 'GLIOMA', 'Carcinom', 'nci9',   # gene expression
               'COIL20', 'ORL', 'Yale', 'warpPIE10P', 'warpAR10P', 'USPS',               # images
               'Isolet', 'PCMAC']                                                        # speech, text


def _load_hd(name):
    import scipy.io as sio
    m = sio.loadmat(os.path.join(DATA_DIR, 'hd', f'{name}.mat'))
    X = m['X']
    X = X.toarray() if hasattr(X, 'toarray') else np.asarray(X)
    return X.astype(float), np.asarray(m['Y']).ravel().astype(int)


def load(name):
    if name in HD_DATASETS:
        return _load_hd(name)
    if name == 'iris':
        d = skd.load_iris(); return d.data, d.target
    if name == 'digits':
        d = skd.load_digits(); return d.data, d.target
    if name.endswith('+N20'):
        X, y = load(name[:-4]); return add_noise_features(X, 20), y
    return _load_bench(name)


MAIN_DATASETS = ['a2', 'd31', 's1', 'unbalance',                       # synthetic 2-D
                 'iris', 'wine', 'ecoli', 'glass', 'yeast', 'wdbc',
                 'ionosphere', 'sonar', 'statlog', 'digits',            # real
                 'iris+N20', 'wine+N20']                                # real + irrelevant features


# ----------------------------------------------------------------------------
# clustering
# ----------------------------------------------------------------------------
def dpc(X, k, pct=2.0, chunk=2000, exact_max=12000):
    """Density peaks clustering (Rodriguez & Laio, 2014), Gaussian kernel, top-k gamma centres.
    For N > exact_max, distances are processed in row chunks (O(N * chunk) memory) and the
    cut-off quantile is estimated from 2e6 random pairs; otherwise the computation is exact."""
    from sklearn.metrics import pairwise_distances
    n = len(X)
    rng = np.random.default_rng(0)
    if n <= exact_max:
        tri = pdist(X)
        dc = np.percentile(tri, pct) + 1e-12
        D = squareform(tri); del tri
        rho = np.concatenate([np.exp(-(D[a:a + chunk] / dc) ** 2).sum(1) for a in range(0, n, chunk)]) - 1
        rows = lambda a, b: D[a:b]
    else:
        i = rng.integers(0, n, 2_000_000); j = rng.integers(0, n, 2_000_000); m = i != j
        dc = np.percentile(np.linalg.norm(X[i[m]] - X[j[m]], axis=1), pct) + 1e-12
        rho = np.empty(n)
        for a in range(0, n, chunk):
            Dc = pairwise_distances(X[a:a + chunk], X)
            rho[a:a + chunk] = np.exp(-(Dc / dc) ** 2).sum(1) - 1
        rows = lambda a, b: pairwise_distances(X[a:b], X)
    rho = rho + 1e-9 * np.random.default_rng(0).random(n)    # break ties
    order = np.argsort(-rho)
    rank = np.empty(n, int); rank[order] = np.arange(n)
    delta = np.zeros(n); nneigh = np.zeros(n, int)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        Dc = rows(a, b)
        for r_, i_ in enumerate(range(a, b)):
            higher = rank < rank[i_]
            if not higher.any():
                delta[i_] = Dc[r_].max(); continue
            dd = np.where(higher, Dc[r_], np.inf)
            j_ = int(np.argmin(dd)); delta[i_] = dd[j_]; nneigh[i_] = j_
    gamma = rho * delta
    centres = np.argsort(-gamma)[:k]
    lab = -np.ones(n, int)
    lab[centres] = np.arange(k)
    if lab[order[0]] < 0:                 # densest point must be a centre
        lab[centres[-1]] = -1; lab[order[0]] = k - 1
    for i in order:
        if lab[i] < 0:
            lab[i] = lab[nneigh[i]]
    return lab


def ewkm(X, k, gamma=1.0, n_init=10, iters=50, seed=0):
    """Entropy-weighted k-means (Jing, Ng & Huang, TKDE 2007): cluster-specific feature weights
    w_lj ~ exp(-D_lj / gamma)."""
    rng = np.random.default_rng(seed)
    n, D = X.shape
    best, best_obj = None, np.inf
    for _ in range(n_init):
        C = X[rng.choice(n, k, replace=False)].copy()
        W = np.full((k, D), 1.0 / D)
        lab = None
        for _ in range(iters):
            dist = np.stack([((X - C[l]) ** 2 * W[l]).sum(1) for l in range(k)], 1)
            new = dist.argmin(1)
            if lab is not None and np.array_equal(new, lab):
                break
            lab = new
            for l in range(k):
                m = lab == l
                if m.any():
                    C[l] = X[m].mean(0)
                    Dl = ((X[m] - C[l]) ** 2).sum(0)
                    a = -(Dl - Dl.min()) / gamma
                    W[l] = np.exp(a) / np.exp(a).sum()
        obj = sum((((X[lab == l] - C[l]) ** 2) * W[l]).sum() + gamma * (W[l] * np.log(W[l] + 1e-300)).sum()
                  for l in range(k))
        if obj < best_obj:
            best, best_obj = lab.copy(), obj
    return best


def sparse_kmeans(X, k, s=None, iters=6, seed=0):
    """Sparse k-means (Witten & Tibshirani, JASA 2010); s = L1 bound on the feature weights."""
    n, D = X.shape
    s = s if s is not None else np.sqrt(D) / 2
    s = float(np.clip(s, 1.0001, np.sqrt(D)))
    w = np.full(D, 1 / np.sqrt(D))
    tss = ((X - X.mean(0)) ** 2).sum(0)
    lab = None
    for _ in range(iters):
        Xw = X * np.sqrt(w)
        lab = KMeans(k, n_init=10, random_state=seed).fit_predict(Xw)
        wcss = sum(((X[lab == l] - X[lab == l].mean(0)) ** 2).sum(0) for l in np.unique(lab))
        a = np.maximum(tss - wcss, 0)
        if a.max() <= 0:
            break
        lo, hi = 0.0, a.max()
        for _ in range(60):                       # binary search for the soft threshold
            mid = (lo + hi) / 2
            v = np.maximum(a - mid, 0); v = v / (np.linalg.norm(v) + 1e-12)
            if v.sum() > s: lo = mid
            else: hi = mid
        v = np.maximum(a - hi, 0); w_new = v / (np.linalg.norm(v) + 1e-12)
        if np.abs(w_new - w).sum() / (np.abs(w).sum() + 1e-12) < 1e-4:
            w = w_new; break
        w = w_new
    return lab


def spectral(X, k, n_neighbors=10, seed=0):
    from sklearn.cluster import SpectralClustering
    return SpectralClustering(k, affinity='nearest_neighbors', n_neighbors=n_neighbors,
                              assign_labels='cluster_qr', random_state=seed).fit_predict(X)


def cluster(alg, X, k, seed=0):
    if alg == 'KM':
        return KMeans(k, n_init=10, random_state=seed).fit_predict(X)
    if alg == 'AGG':
        return AgglomerativeClustering(k, linkage='ward').fit_predict(X)
    if alg == 'DPC':
        return dpc(X, k)
    raise ValueError(alg)


# ----------------------------------------------------------------------------
# metrics
# ----------------------------------------------------------------------------
def acc(y, p):
    yu, pu = np.unique(y), np.unique(p)
    M = np.zeros((len(pu), len(yu)))
    for i, a in enumerate(pu):
        for j, b in enumerate(yu):
            M[i, j] = np.sum((p == a) & (y == b))
    r, c = linear_sum_assignment(-M)
    return M[r, c].sum() / len(y)


def scores(y, p):
    return {'ARI': adjusted_rand_score(y, p),
            'NMI': normalized_mutual_info_score(y, p),
            'ACC': acc(y, p)}
