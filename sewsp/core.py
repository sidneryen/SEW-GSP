"""
Core algorithms for clustering-oriented dataset optimization.

Implements
  * HIBOG-style gravitation on Euclidean kNN (all objects move)
  * HIACSP  : shortest-path (geodesic) kNN + boundary-object optimisation
  * HIACSP-WF: HIACSP + variance-based global feature weights (Liao & Gu, 2026)
  * SEW-GSP (proposed): Structure-aware Entropy-regularised feature Weighting
               + exact truncated-Dijkstra Geodesic kNN
               + soft boundary-aware displacement with weight co-evolution

All optimisers return a list of snapshots (one per iteration) so that the
iteration count d can be evaluated without re-running the optimisation.
"""
import time
import heapq
import numpy as np
from numba import njit
from scipy import sparse
from scipy.sparse.csgraph import dijkstra
from sklearn.neighbors import NearestNeighbors

EPS = 1e-12


# ----------------------------------------------------------------------------
# basic utilities
# ----------------------------------------------------------------------------
def minmax(X):
    X = np.asarray(X, dtype=float)
    lo, hi = X.min(0), X.max(0)
    rng = np.where(hi - lo > 0, hi - lo, 1.0)
    return (X - lo) / rng


def knn_graph(Z, k):
    """Symmetric kNN graph (CSR) with Euclidean edge lengths, plus raw kNN."""
    n = Z.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(Z)
    dist, idx = nn.kneighbors(Z)
    dist, idx = dist[:, 1:], idx[:, 1:]
    rows = np.repeat(np.arange(n), idx.shape[1])
    G = sparse.csr_matrix((dist.ravel() + EPS, (rows, idx.ravel())), shape=(n, n))
    G = G.maximum(G.T).tocsr()
    return G, dist, idx


# ----------------------------------------------------------------------------
# geodesic (shortest-path) kNN
# ----------------------------------------------------------------------------
def geodesic_knn_full(G, K):
    """Reference implementation used by HIACSP: all-pairs Dijkstra, O(N^2) memory."""
    D = dijkstra(G, directed=False)
    np.fill_diagonal(D, np.inf)
    idx = np.argpartition(D, K, axis=1)[:, :K]
    # order by distance
    rows = np.arange(D.shape[0])[:, None]
    order = np.argsort(D[rows, idx], axis=1)
    idx = idx[rows, order]
    dist = D[rows, idx]
    return idx, dist


@njit(cache=True)
def _truncated_dijkstra(indptr, indices, data, n, K):
    out_idx = -np.ones((n, K), dtype=np.int64)
    out_d = np.full((n, K), np.inf)
    dist = np.full(n, np.inf)
    done = np.zeros(n, dtype=np.bool_)
    touched = np.empty(n, dtype=np.int64)
    for s in range(n):
        nt = 0
        dist[s] = 0.0
        touched[nt] = s
        nt += 1
        h = [(0.0, s)]
        got = -1  # number of settled nodes excluding the source
        while len(h) > 0 and got < K:
            d, u = heapq.heappop(h)
            if done[u]:
                continue
            done[u] = True
            if got >= 0:
                out_idx[s, got] = u
                out_d[s, got] = d
            got += 1
            for p in range(indptr[u], indptr[u + 1]):
                v = indices[p]
                nd = d + data[p]
                if nd < dist[v]:
                    if dist[v] == np.inf:
                        touched[nt] = v
                        nt += 1
                    dist[v] = nd
                    heapq.heappush(h, (nd, v))
        for t in range(nt):
            dist[touched[t]] = np.inf
            done[touched[t]] = False
    return out_idx, out_d


def geodesic_knn_truncated(G, K):
    """Exact geodesic kNN by early-terminated Dijkstra (stops after K settled nodes).
    Time O(N * K * k_g * log(K k_g)), memory O(N K)."""
    G = G.tocsr()
    idx, dist = _truncated_dijkstra(G.indptr.astype(np.int64), G.indices.astype(np.int64),
                                    G.data.astype(np.float64), G.shape[0], K)
    return idx, dist


def _fill_missing(idx, Z, K):
    """Objects in tiny components may have < K geodesic neighbours: fill with Euclidean kNN."""
    miss = (idx < 0).any(1)
    if miss.any():
        nn = NearestNeighbors(n_neighbors=K + 1).fit(Z)
        _, e_idx = nn.kneighbors(Z[miss])
        idx = idx.copy()
        for r, i in enumerate(np.where(miss)[0]):
            have = [j for j in idx[i] if j >= 0]
            for j in e_idx[r, 1:]:
                if len(have) >= K:
                    break
                if j not in have:
                    have.append(j)
            idx[i] = have[:K]
    return idx


# ----------------------------------------------------------------------------
# feature weighting
# ----------------------------------------------------------------------------
def variance_weights(X):
    """HIACSP-WF weights (Eq. 4 of Liao & Gu), rescaled to mean 1."""
    v = X.var(0, ddof=1)
    w = v / (v.sum() + EPS)
    return w * X.shape[1]


def heat_graph(Z, k):
    """Symmetric heat-kernel kNN affinity graph S and its degree vector."""
    n = Z.shape[0]
    nn = NearestNeighbors(n_neighbors=min(k + 1, n)).fit(Z)
    dist, idx = nn.kneighbors(Z)
    dist, idx = dist[:, 1:], idx[:, 1:]
    sig2 = np.mean(dist ** 2) + EPS
    rows = np.repeat(np.arange(n), idx.shape[1])
    S = sparse.csr_matrix((np.exp(-dist.ravel() ** 2 / sig2), (rows, idx.ravel())), shape=(n, n))
    S = S.maximum(S.T).tocsr()
    return S, np.asarray(S.sum(1)).ravel()


def ls_on_graph(X, S, deg):
    """Laplacian score  f~^T L f~ / f~^T D f~  of every column of X (He et al., 2005)."""
    mu = (deg @ X) / deg.sum()
    F = X - mu
    num_D = (deg[:, None] * F ** 2).sum(0)
    num_L = num_D - (F * (S @ F)).sum(0)
    ls = np.where(num_D > EPS, num_L / (num_D + EPS), 1.0)
    return np.clip(ls, 0, None)


def laplacian_scores(X, Z, k):
    S, deg = heat_graph(Z, k)
    return ls_on_graph(X, S, deg)


def relevance(X, Z, k, n_perm=3, seed=0):
    """Null-calibrated structural relevance r_m = max(0, 1 - LS_m / LS_m^null), in [0,1].
    LS^null is the Laplacian score of the same feature after random permutation over
    samples (it keeps the marginal distribution but destroys any graph structure)."""
    rng = np.random.default_rng(seed)
    ls = laplacian_scores(X, Z, k)
    null = np.zeros_like(ls)
    for _ in range(n_perm):
        Xp = np.column_stack([rng.permutation(X[:, m]) for m in range(X.shape[1])])
        null += laplacian_scores(Xp, Z, k)
    null /= n_perm
    return np.clip(1.0 - ls / (null + EPS), 0.0, 1.0), ls


def relevance_cross(X, w, k, n_groups=16, n_perm=3, seed=0, Xg=None):
    """Cross-feature (leave-group-out) null-calibrated relevance.
    The Laplacian score of feature m is evaluated on a graph built WITHOUT feature m
    (from the remaining, currently weighted, features). A feature is relevant only if it
    agrees with the cluster structure carried by the *other* features; an independent
    noise feature therefore scores LS ~ LS^null and gets r ~ 0, removing the
    self-reference bias of the ordinary Laplacian score."""
    D = X.shape[1]
    rng = np.random.default_rng(seed)
    if D == 1:
        return relevance(X, X, k, n_perm, seed)[0]
    groups = np.array_split(rng.permutation(D), min(D, n_groups))
    Xp = [np.column_stack([rng.permutation(X[:, m]) for m in range(D)]) for _ in range(n_perm)]
    r = np.zeros(D)
    sw = np.sqrt(w)
    Xg = X if Xg is None else Xg      # coordinates used to build the graph
    for g in groups:
        keep = np.setdiff1d(np.arange(D), g)
        Z = Xg[:, keep] * sw[keep]
        S, deg = heat_graph(Z, k)
        ls = ls_on_graph(X[:, g], S, deg)
        null = np.mean([ls_on_graph(P[:, g], S, deg) for P in Xp], axis=0)
        r[g] = np.clip(1.0 - ls / (null + EPS), 0.0, 1.0)
    return r


def relevance_to_weights(r, mode='shrink'):
    """Map relevances r in [0,1]^D to feature weights with mean 1.
    'linear' : w = D r / sum(r)  -- the closed-form maximiser of
               sum_m w_m r_m - (lam/2)||w||^2  s.t. sum_m w_m = D, w >= 0
    'shrink' : w = (1 - rho) * 1 + rho * w_linear,  rho = max_m r_m
               (evidence-adaptive shrinkage towards equal weights; proposed default)
    float g  : w = D softmax(r / g)  (max-entropy alternative, used only for comparison)"""
    D = len(r)
    lin = (r + EPS) / (r.sum() + EPS * D) * D
    if mode == 'linear':
        return lin
    if mode == 'shrink':
        rho = float(r.max())
        return (1.0 - rho) + rho * lin
    a = (r - r.max()) / (float(mode) + EPS)
    w = np.exp(a)
    return w / w.sum() * D


def sew_weights(X, k, mode='shrink', n_refine=3, w0=None, Xg=None):
    """CNLR feature weights with alternating graph <-> weight refinement.
    X  : features whose relevance is measured (min-max normalised)
    Xg : coordinates used to build the kNN graphs (defaults to X)"""
    D = X.shape[1]
    w = np.ones(D) if w0 is None else w0.copy()
    for _ in range(n_refine):
        r = relevance_cross(X, w, k, Xg=Xg)
        w_new = relevance_to_weights(r, mode)
        converged = np.abs(w_new - w).max() < 1e-3
        w = w_new
        if converged:
            break
    return w


# ----------------------------------------------------------------------------
# boundary objects and forces
# ----------------------------------------------------------------------------
def unit_sum(Z, nbr, chunk=None):
    """Sum of unit vectors from each object to its neighbours (chunked to bound memory)."""
    n = Z.shape[0]
    chunk = chunk or max(1, int(3e7 // (nbr.shape[1] * Z.shape[1])))
    E = np.empty_like(Z)
    nrm = np.empty(nbr.shape)
    for a in range(0, n, chunk):
        b = min(n, a + chunk)
        diff = Z[nbr[a:b]] - Z[a:b, None, :]                 # c x K x D
        nr = np.linalg.norm(diff, axis=2) + EPS
        E[a:b] = (diff / nr[..., None]).sum(1)
        nrm[a:b] = nr
    return E, nrm


def boundary_scores(Z, nbr, K):
    """BS(x) = ||sum of unit vectors to kNN|| / rho(x)  (Eq. 7-8 of HIACSP-WF)."""
    E, nrm = unit_sum(Z, nbr)
    nn = NearestNeighbors(n_neighbors=K + 1).fit(Z)
    dk, _ = nn.kneighbors(Z)
    dc = dk[:, -1].mean()
    rho = np.array([len(a) for a in nn.radius_neighbors(Z, radius=dc, return_distance=False)], float)
    bs = np.linalg.norm(E, axis=1) / np.maximum(rho, 1.0)
    return bs, E, nrm


def displacement(E, nrm, T):
    """F_i = (sum e / ||sum e||) * max_j ||x_j - x_i||;  S = F * T  (Eq. 9-10)."""
    dirn = E / (np.linalg.norm(E, axis=1, keepdims=True) + EPS)
    return T * dirn * nrm.max(1, keepdims=True)


# ----------------------------------------------------------------------------
# optimisers
# ----------------------------------------------------------------------------
def hibog(X, K=10, d=5, T=0.5):
    """HIBOG-style: every object attracted by its Euclidean kNN (no boundary selection)."""
    Z = minmax(X)
    snaps = []
    for _ in range(d):
        _, _, idx = knn_graph(Z, K)
        E, nrm = unit_sum(Z, idx)
        Z = Z + displacement(E, nrm, T) * 0.5     # all objects; damped
        snaps.append(Z.copy())
    return snaps


def hiacsp(X, K=10, d=5, T=0.5, weights=None, truncated=False):
    """HIACSP (weights=None) and HIACSP-WF (weights='variance')."""
    X0 = minmax(X)
    w = np.ones(X0.shape[1]) if weights is None else variance_weights(X0)
    Z = X0 * np.sqrt(w)
    snaps = []
    for _ in range(d):
        G, _, _ = knn_graph(Z, K)
        if truncated:
            nbr, _ = geodesic_knn_truncated(G, K)
        else:
            nbr, _ = geodesic_knn_full(G, K)
        nbr = _fill_missing(nbr, Z, K)
        bs, E, nrm = boundary_scores(Z, nbr, K)
        B = bs >= bs.mean()
        S = displacement(E, nrm, T)
        Z = Z + S * B[:, None]
        snaps.append(Z.copy())
    return snaps


# ----------------------------------------------------------------------------
# alternative unsupervised feature scorers (baselines for the weighting stage)
# ----------------------------------------------------------------------------
def _to_unit(score_good):
    s = np.asarray(score_good, float)
    s = s - s.min()
    return s / (s.max() + EPS)


def spec_relevance(X, k):
    """SPEC, ranking function phi_2 (Zhao & Liu, ICML 2007); lower phi_2 = better."""
    S, deg = heat_graph(X, k)
    dm = 1.0 / np.sqrt(deg + EPS)
    Ln = sparse.identity(len(deg)) - sparse.diags(dm) @ S @ sparse.diags(dm)
    xi1 = np.sqrt(deg) / np.linalg.norm(np.sqrt(deg))
    F = np.sqrt(deg)[:, None] * X
    F = F / (np.linalg.norm(F, axis=0, keepdims=True) + EPS)
    num = (F * (Ln @ F)).sum(0)
    den = 1.0 - (xi1 @ F) ** 2
    phi2 = np.where(den > 1e-8, num / (den + EPS), 1.0)
    # phi_2 is a normalised smoothness ratio (~[0, 1]; 0 = perfectly smooth), mapped to
    # relevance exactly like the Laplacian score: r = 1 - phi_2
    return np.clip(1.0 - phi2, 0.0, 1.0)


def mcfs_relevance(X, k, n_eig=None):
    """MCFS (Cai, Zhang & He, KDD 2010): spectral embedding + L1-regularised regression;
    score_m = max_e |a_{e,m}|."""
    from sklearn.linear_model import Lars
    if X.shape[0] > 3000:          # dense eigen-decomposition on a fixed random subsample
        X = X[np.random.default_rng(0).choice(X.shape[0], 3000, replace=False)]
    S, deg = heat_graph(X, k)
    n = X.shape[0]
    n_eig = n_eig or 5
    dm = 1.0 / np.sqrt(deg + EPS)
    A = (sparse.diags(dm) @ S @ sparse.diags(dm)).toarray()
    vals, vecs = np.linalg.eigh(A)
    Y = (dm[:, None] * vecs)[:, np.argsort(-vals)][:, 1:n_eig + 1]
    Xc = X - X.mean(0)
    nnz = int(min(X.shape[1], max(10, X.shape[1] // 5), n - 1))
    sc = np.zeros(X.shape[1])
    for e in range(Y.shape[1]):
        a = Lars(n_nonzero_coefs=nnz, fit_intercept=True).fit(Xc, Y[:, e]).coef_
        sc = np.maximum(sc, np.abs(a))
    return sc / (sc.max() + EPS)          # unselected features keep relevance 0 (MCFS selection)


def relevance_weights(X, k, method):
    """Weights (mean 1, linear in relevance) from an alternative scorer."""
    D = X.shape[1]
    if method == 'ls':
        r = np.clip(1.0 - laplacian_scores(X, X, k), 0, None)
    elif method == 'spec':
        r = spec_relevance(X, k)
    elif method == 'mcfs':
        r = mcfs_relevance(X, k)
    else:
        raise ValueError(method)
    return (r + EPS) / (r.sum() + EPS * D) * D


# ----------------------------------------------------------------------------
# HIAC-style optimiser (valid-neighbour gravitation)
# ----------------------------------------------------------------------------
def _knee(y):
    """Index of the point of a sorted curve farthest from the chord (knee / turning point)."""
    n = len(y)
    x = np.linspace(0, 1, n)
    yy = (y - y.min()) / (y.max() - y.min() + EPS)
    return int(np.argmax(np.abs(yy - x)))


def hiac(X, K=10, d=5, T=0.5):
    """HIAC-style re-implementation (Li et al., Inf. Sci. 2023): the K-NN relations of the
    original data are split into valid / invalid by a data-driven threshold located at the
    turning point (knee) of the sorted K-NN edge-length curve; each object is then attracted
    only by its initial valid neighbours (all objects move, damped step as in HIBOG)."""
    Z = minmax(X)
    _, dist, idx = knn_graph(Z, K)
    e = np.sort(dist.ravel())
    tau = e[_knee(e)]
    valid = dist <= tau
    snaps = []
    for _ in range(d):
        E = np.zeros_like(Z); step = np.zeros((Z.shape[0], 1))
        ch = max(1, int(3e7 // (idx.shape[1] * Z.shape[1])))
        for a in range(0, Z.shape[0], ch):
            b = min(Z.shape[0], a + ch)
            diff = Z[idx[a:b]] - Z[a:b, None, :]
            nrm = np.linalg.norm(diff, axis=2) + EPS
            v = valid[a:b]
            E[a:b] = (diff / nrm[..., None] * v[..., None]).sum(1)
            step[a:b, 0] = np.where(v, nrm, 0).max(1)
        has = valid.any(1)
        dirn = E / (np.linalg.norm(E, axis=1, keepdims=True) + EPS)
        Z = Z + 0.5 * T * dirn * step * has[:, None]
        snaps.append(Z.copy())
    return snaps


def sewgsp(X, K=10, d=5, T=0.5, soft=True, weighting='cnlr', coevolve=False,
           n_refine=3, return_weights=False, wmode='shrink', weight_sample=None, seed=0):
    """Proposed SEW-GSP.
    weighting: 'cnlr'     cross-feature null-calibrated Laplacian relevance, w ~ r (proposed)
               'ls'       ordinary (self-referential) Laplacian score, w ~ 1 - LS
               'variance' variance weights of HIACSP-WF
               'none'     equal weights
    soft     : soft boundary membership (sigmoid of standardised BS) instead of BS >= mean
    coevolve : re-estimate CNLR weights each iteration on the graph of the optimised data
    """
    X0 = minmax(X)
    Dm = X0.shape[1]
    if weighting == 'cnlr':
        Xs = X0
        if weight_sample and X0.shape[0] > weight_sample:   # large N: estimate weights on a subsample
            Xs = X0[np.random.default_rng(seed).choice(X0.shape[0], weight_sample, replace=False)]
        w = sew_weights(Xs, K, wmode, n_refine=n_refine)
    elif weighting == 'ls':
        r = np.clip(1.0 - laplacian_scores(X0, X0, K), 0, None)
        w = (r + EPS) / (r.sum() + EPS * Dm) * Dm
    elif weighting == 'variance':
        w = variance_weights(X0)
    elif weighting in ('spec', 'mcfs'):
        w = relevance_weights(X0, K, weighting)
    else:
        w = np.ones(Dm)
    Xc = X0.copy()
    snaps, wl = [], []
    for it in range(d):
        if it > 0 and coevolve and weighting == 'cnlr':
            w = sew_weights(X0, K, wmode, n_refine=1, w0=w, Xg=Xc)
        sw = np.sqrt(w) + EPS
        Z = Xc * sw
        G, _, _ = knn_graph(Z, K)
        nbr, _ = geodesic_knn_truncated(G, K)
        nbr = _fill_missing(nbr, Z, K)
        bs, E, nrm = boundary_scores(Z, nbr, K)
        if soft:
            zs = (bs - bs.mean()) / (bs.std() + EPS)
            beta = 1.0 / (1.0 + np.exp(-4.0 * zs))
        else:
            beta = (bs >= bs.mean()).astype(float)
        Z = Z + displacement(E, nrm, T) * beta[:, None]
        Xc = Z / sw
        snaps.append(Z.copy())          # emitted in the learned metric space
        wl.append(w.copy())
    return (snaps, wl) if return_weights else snaps


METHODS = {
    'Raw':       lambda X, K, d: [minmax(X)] * d,
    'HIBOG':     lambda X, K, d: hibog(X, K, d),
    'HIACSP':    lambda X, K, d: hiacsp(X, K, d, truncated=True),
    'HIACSP-WF': lambda X, K, d: hiacsp(X, K, d, weights='variance', truncated=True),
    'SEW-GSP':   lambda X, K, d: sewgsp(X, K, d),
}
# extended baselines (revision): HIAC-style optimiser and the SEW-GSP pipeline with the
# CNLR stage replaced by classical unsupervised feature scorers
METHODS_EXT = {
    'HIAC*':     lambda X, K, d: hiac(X, K, d),
    'LS-GSP':    lambda X, K, d: sewgsp(X, K, d, weighting='ls'),
    'SPEC-GSP':  lambda X, K, d: sewgsp(X, K, d, weighting='spec'),
    'MCFS-GSP':  lambda X, K, d: sewgsp(X, K, d, weighting='mcfs'),
}
ALL_METHODS = {**METHODS, **METHODS_EXT}
