"""Ranking-evaluation metrics, computed from a ranked list of job ids and the relevance judgments of that query.

labels: {job_id: grade}   grade 0 = not relevant, 1 = partially relevant, 2 = clearly relevant.   Documents without a judgment count as 0.
Binary metrics use  relevant = grade >= thr  (thr = 1 by default; thr = 2 gives the strict version).

  P@k      relevant in the top k / k                                 (k even when fewer than k results were returned)
  R@k      relevant in the top k / all relevant judged documents    -> recall WITHIN THE JUDGED POOL (the true number of relevant jobs in the
           collection is unknown), so it is only meaningful for comparing systems that were pooled
  RR       1 / rank of the first relevant result (0 if none);  MRR = mean over queries
  nDCG@k   DCG with gain 2^grade - 1 and discount log2(rank + 1), divided by the DCG of the ideal ordering of the judged documents
Metrics that need at least one relevant judged document (R@k, RR, nDCG) return None for a query that has none: such queries are left out
of those averages (and reported)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence

import numpy as np


def _rel(labels: Dict[int, int], d: int, thr: int) -> bool:
    return labels.get(d, 0) >= thr


def n_relevant(labels: Dict[int, int], thr: int = 1) -> int:
    return sum(1 for g in labels.values() if g >= thr)


def precision_at_k(ranked: Sequence[int], labels: Dict[int, int], k: int, thr: int = 1) -> float:
    return sum(_rel(labels, d, thr) for d in ranked[:k]) / k


def recall_at_k(ranked: Sequence[int], labels: Dict[int, int], k: int, thr: int = 1) -> Optional[float]:
    total = n_relevant(labels, thr)
    return None if total == 0 else sum(_rel(labels, d, thr) for d in ranked[:k]) / total


def reciprocal_rank(ranked: Sequence[int], labels: Dict[int, int], thr: int = 1) -> Optional[float]:
    if n_relevant(labels, thr) == 0:
        return None
    for i, d in enumerate(ranked, 1):
        if _rel(labels, d, thr):
            return 1.0 / i
    return 0.0


def dcg(grades: Sequence[int]) -> float:
    return sum((2 ** g - 1) / math.log2(i + 2) for i, g in enumerate(grades))


def ndcg_at_k(ranked: Sequence[int], labels: Dict[int, int], k: int) -> Optional[float]:
    ideal = dcg(sorted(labels.values(), reverse=True)[:k])
    return None if ideal == 0 else dcg([labels.get(d, 0) for d in ranked[:k]]) / ideal


def judged_at_k(ranked: Sequence[int], labels: Dict[int, int], k: int) -> float:
    """share of the RETURNED top-k results that carries a judgment (a query with fewer than k results is not penalised for the missing ones).
    Low values mean the metrics are biased AGAINST this system (unjudged = not relevant)."""
    top = list(ranked[:k])
    return sum(d in labels for d in top) / len(top) if top else 1.0


def sign_flip_p(d: np.ndarray, n_perm: int = 20000, seed: int = 0) -> float:
    """Paired randomization test: if the two systems were interchangeable, the sign of every per-query difference is arbitrary.  Two-sided
    p-value = share of random sign assignments whose |mean difference| is at least the observed one (exact enumeration for <= 14 non-zero
    differences).  Exact under the null, unlike the percentile bootstrap, which is too optimistic with ~20 queries."""
    d = np.asarray(d, dtype=float)
    d = d[np.abs(d) > 1e-12]
    if len(d) == 0:
        return 1.0
    obs = abs(d.mean())
    if len(d) <= 14:
        signs = ((np.arange(2 ** len(d))[:, None] >> np.arange(len(d))) & 1) * 2 - 1
        return float((np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).mean())
    rng = np.random.default_rng(seed)
    signs = rng.choice([-1.0, 1.0], size=(n_perm, len(d)))
    return float((1 + (np.abs((signs * d).mean(axis=1)) >= obs - 1e-12).sum()) / (n_perm + 1))


def holm(pvals: Sequence[float]) -> List[float]:
    """Holm-Bonferroni adjusted p-values (family-wise error control over all comparisons shown)."""
    order = sorted(range(len(pvals)), key=lambda i: pvals[i])
    adj, running = [0.0] * len(pvals), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (len(pvals) - rank) * pvals[i]))
        adj[i] = running
    return adj


def paired_bootstrap(a: Sequence[float], b: Sequence[float], n_boot: int = 10000, seed: int = 0) -> Dict:
    """Paired comparison of two systems over the same queries: mean difference (a - b), a 95% percentile-bootstrap interval (DESCRIBES the
    size of the effect), a sign-flip randomization p-value (DECIDES whether a difference is supported) and wins / ties / losses."""
    d = np.asarray(a, dtype=float) - np.asarray(b, dtype=float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(mean_diff=float(d.mean()), ci_low=float(lo), ci_high=float(hi), p_value=sign_flip_p(d, seed=seed), wins=int((d > 1e-12).sum()), ties=int((np.abs(d) <= 1e-12).sum()),
                losses=int((d < -1e-12).sum()), n=len(d))


def bootstrap_mean_ci(values: Sequence[float], n_boot: int = 10000, seed: int = 0) -> Dict:
    v = np.asarray(values, dtype=float)
    rng = np.random.default_rng(seed)
    means = v[rng.integers(0, len(v), size=(n_boot, len(v)))].mean(axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    return dict(mean=float(v.mean()), ci_low=float(lo), ci_high=float(hi), n=len(v))
