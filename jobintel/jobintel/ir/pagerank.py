"""PageRank implemented from scratch (power iteration) + the job link graph it runs on.

    PR(v) = (1 - d)/n + d * ( sum_{u -> v} PR(u) * w(u,v) / W(u)  +  (mass of dangling nodes)/n )          d = damping = 0.85

Dangling nodes (no out-links) spread their rank uniformly, so the scores always sum to 1.  The graph is
    Company -> Career page -> Job -> Related job        (+ Job -> Company back-link)
Edges are OBSERVED hyperlinks when the crawler saw them and CONSTRUCTED otherwise (company<->job from the data, related jobs from
shared rare skills); constructed edges are labelled in job_links.is_constructed.  PageRank is an extra authority signal - it must
never replace textual relevance, and whether it helps is measured in Phase 4, not assumed."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PageRankResult:
    scores: np.ndarray
    iterations: int
    converged: bool
    delta: float


def pagerank(n: int, src: Sequence[int], dst: Sequence[int], weights: Optional[Sequence[float]] = None, *, damping: float = 0.85,
             tol: float = 1e-10, max_iter: int = 200) -> PageRankResult:
    if n == 0:
        return PageRankResult(np.zeros(0), 0, True, 0.0)
    src, dst = np.asarray(src, dtype=np.int64), np.asarray(dst, dtype=np.int64)
    w = np.ones(len(src)) if weights is None else np.asarray(weights, dtype=np.float64)
    out_w = np.bincount(src, weights=w, minlength=n)
    dangling = out_w == 0
    r = np.full(n, 1.0 / n)
    delta = np.inf
    for it in range(1, max_iter + 1):
        contrib = r[src] * w / out_w[src]
        nxt = np.bincount(dst, weights=contrib, minlength=n)
        nxt = damping * (nxt + r[dangling].sum() / n) + (1.0 - damping) / n
        delta = float(np.abs(nxt - r).sum())
        r = nxt
        if delta < tol:
            return PageRankResult(r / r.sum(), it, True, delta)
    return PageRankResult(r / r.sum(), max_iter, False, delta)


class LinkGraph:
    """Nodes are ('job', id) / ('company', id) / ('career', company_id)."""

    def __init__(self) -> None:
        self.node_of: Dict[Tuple[str, int], int] = {}
        self.src: List[int] = []
        self.dst: List[int] = []
        self.w: List[float] = []

    def node(self, kind: str, ident: int) -> int:
        key = (kind, int(ident))
        if key not in self.node_of:
            self.node_of[key] = len(self.node_of)
        return self.node_of[key]

    def edge(self, a: Tuple[str, int], b: Tuple[str, int], weight: float = 1.0) -> None:
        self.src.append(self.node(*a))
        self.dst.append(self.node(*b))
        self.w.append(weight)

    @property
    def n_nodes(self) -> int:
        return len(self.node_of)

    def run(self, **kw) -> Dict[Tuple[str, int], float]:
        res = pagerank(self.n_nodes, self.src, self.dst, self.w, **kw)
        self.last = res
        return {key: float(res.scores[i]) for key, i in self.node_of.items()}


def related_job_edges(job_rows: np.ndarray, skill_cols: np.ndarray, n_jobs: int, n_skills: int, cluster: np.ndarray, *, k: int = 3,
                      min_df: int = 2, max_df: int = 200, min_shared_idf: float = 8.0, chunk: int = 3000) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """'Related job' links: for every job, the k most similar jobs of the same cluster, similarity = sum of IDF of the RARE skills they share
    (skills listed by more than max_df jobs say little about relatedness).  Returns (src, dst, weight) job indices."""
    from scipy import sparse
    X = sparse.csr_matrix((np.ones(len(job_rows)), (job_rows, skill_cols)), shape=(n_jobs, n_skills))
    X.data[:] = 1.0
    df = np.asarray(X.sum(axis=0)).ravel()
    keep = (df >= min_df) & (df <= max_df)
    idf = np.where(keep, np.log(n_jobs / np.maximum(df, 1)), 0.0)
    W = X.multiply(idf[None, :]).tocsr()
    B = X.multiply(keep[None, :].astype(float)).tocsr()
    S, D, WT = [], [], []
    for a in range(0, n_jobs, chunk):
        P = (W[a:a + chunk] @ B.T).tocsr()                       # sum of idf over shared kept skills
        for i in range(P.shape[0]):
            lo, hi = P.indptr[i], P.indptr[i + 1]
            cols, vals = P.indices[lo:hi], P.data[lo:hi]
            ok = (cols != a + i) & (cluster[cols] == cluster[a + i]) & (vals >= min_shared_idf)
            cols, vals = cols[ok], vals[ok]
            if not len(cols):
                continue
            if len(cols) > k:
                top = np.argpartition(-vals, k - 1)[:k]
                cols, vals = cols[top], vals[top]
            S.append(np.full(len(cols), a + i))
            D.append(cols)
            WT.append(vals)
    if not S:
        return np.zeros(0, int), np.zeros(0, int), np.zeros(0)
    return np.concatenate(S), np.concatenate(D), np.concatenate(WT)
