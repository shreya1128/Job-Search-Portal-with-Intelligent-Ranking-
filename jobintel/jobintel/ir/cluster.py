"""Cluster-based retrieval: the collection is organised into logical clusters (Software/IT, Data/AI, Finance, Healthcare, ...).
A query is first ROUTED to the clusters most likely to hold its answers, then only those clusters are searched.

Routing = a naive-Bayes style classifier over cluster language models:
    P(c | q)  proportional to  prod_w ( (cf_c(w) + mu P(w|C)) / (n_c + mu) )^qtf(w)        uniform prior over clusters
and the smallest set of clusters whose posterior mass reaches `mass` (at most `max_clusters`) is searched.
The cluster of a job is a title-keyword heuristic (common.cluster_for_title), not ground truth, so routing can miss relevant jobs that sit in
another cluster - the experiment in docs/phase3_report.md measures how often."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Sequence, Tuple

import numpy as np


@dataclass
class RouteResult:
    clusters: List[str]
    posteriors: Dict[str, float]
    doc_fraction: float

    def to_dict(self) -> Dict:
        return {"clusters": self.clusters, "posteriors": {k: round(v, 4) for k, v in self.posteriors.items()}, "doc_fraction": round(self.doc_fraction, 4)}


class ClusterRouter:
    def __init__(self, labels: List[str], cf: np.ndarray, n_tokens: np.ndarray, docs: Dict[str, np.ndarray], coll_prob: np.ndarray, n_docs: int):
        self.labels, self.cf, self.n_tokens, self.docs, self.coll_prob, self.n_docs = labels, cf, n_tokens, docs, coll_prob, n_docs

    @classmethod
    def from_index(cls, ix) -> "ClusterRouter":
        arr = np.asarray(ix.cluster_arr)
        labels = sorted({str(x) for x in arr if x})
        codes = np.full(len(arr), -1, dtype=np.int32)
        for i, lab in enumerate(labels):
            codes[arr == lab] = i
        tok_cluster = np.repeat(codes, ix.doc_len)
        v = len(ix.indptr) - 1
        cf = np.zeros((len(labels), v), dtype=np.int64)
        for i in range(len(labels)):
            cf[i] = np.bincount(ix.tokens[tok_cluster == i], minlength=v)
        docs = {lab: np.flatnonzero(codes == i).astype(np.int32) for i, lab in enumerate(labels)}
        return cls(labels, cf, cf.sum(axis=1), docs, ix.cf / ix.cf.sum(), len(arr))

    def route(self, qterms: Sequence[Tuple[int, int]], max_clusters: int = 3, mass: float = 0.9, mu: float = 500.0) -> RouteResult:
        logp = np.zeros(len(self.labels))
        for t, qtf in qterms:
            pc = self.coll_prob[t]
            if pc <= 0:
                continue
            logp += qtf * np.log((self.cf[:, t] + mu * pc) / (self.n_tokens + mu))
        post = np.exp(logp - logp.max())
        post /= post.sum()
        order = np.argsort(-post, kind="stable")
        chosen, acc = [], 0.0
        for i in order[:max_clusters]:
            chosen.append(self.labels[i])
            acc += post[i]
            if acc >= mass:
                break
        frac = sum(len(self.docs[c]) for c in chosen) / self.n_docs
        return RouteResult(chosen, {self.labels[i]: float(post[i]) for i in order}, frac)

    def docs_in(self, clusters: Sequence[str]) -> np.ndarray:
        parts = [self.docs[c] for c in clusters if c in self.docs]
        return np.sort(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int32)
