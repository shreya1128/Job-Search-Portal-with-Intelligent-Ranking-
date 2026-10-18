"""Retrieval / ranking functions, implemented from scratch with NumPy (no TF-IDF / BM25 anywhere).

 * keyword_baseline   coverage of distinct query terms (+ total term frequency as tie-break)
 * unigram_lm         multinomial unigram language model, query likelihood, Dirichlet or Jelinek-Mercer smoothing
 * bigram_lm          bigram language model P(w_i | w_{i-1}, d), smoothed with the unigram model (Dirichlet)
 * dfr                Divergence From Randomness: PL2 (Poisson / Laplace / normalisation 2) and InL2
All scorers take the SAME candidate set (documents containing at least one query term) so models are comparable.

Contract that makes parallel / partitioned search exact: the score of a document depends only on (a) the document itself and
(b) COLLECTION statistics read from `index.cf / index.df / index.n_docs / index.total_tokens / index.avg_doc_len`.  A partition
object exposes GLOBAL values for (b) and LOCAL postings / doc_len / tokens, so a partition scores exactly like the whole index.
A document's score never depends on which other documents happen to be candidates (see tests)."""
from __future__ import annotations

import math
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

LOG2E = math.log2(math.e)
QTerms = List[Tuple[int, int]]          # [(term_id, query term frequency)]


def candidates(index, qterms: QTerms) -> np.ndarray:
    """Documents containing at least one query term (sorted document indices)."""
    parts = [index.postings(t)[0] for t, _ in qterms]
    return np.unique(np.concatenate(parts)) if parts else np.zeros(0, dtype=np.int32)


def top_positions(raw: np.ndarray, n: int) -> np.ndarray:
    """Positions of the n best scores: score descending, ties by position (= by document order) - FULLY deterministic.
    (np.argpartition alone picks an arbitrary subset of the documents tied at the cut-off, which made two correct searches - e.g. a
    partitioned and a single-index one - disagree about which tied documents make the top n.)  A partitioned search that merges
    per-partition top-n lists therefore reproduces the single-index ranking exactly."""
    n = min(n, len(raw))
    if n <= 0:
        return np.zeros(0, dtype=np.int64)
    neg = -np.asarray(raw, dtype=np.float64)
    if n >= len(neg):
        idx = np.arange(len(neg))
    else:
        thr = np.partition(neg, n - 1)[n - 1]                       # the n-th best score (negated)
        strictly = np.flatnonzero(neg < thr)                        # clearly inside the top n
        ties = np.flatnonzero(neg == thr)                           # tied at the boundary, already in position order
        idx = np.concatenate([strictly, ties[: n - len(strictly)]])
    return idx[np.lexsort((idx, neg[idx]))]


def _position_map(index, cands: np.ndarray) -> np.ndarray:
    pos = np.full(len(index.doc_len), -1, dtype=np.int64)
    pos[cands] = np.arange(len(cands))
    return pos


def _hits(pos: np.ndarray, docs: np.ndarray, tf: np.ndarray):
    """Restrict a posting list to the candidate set (documents removed by a filter have position -1 and must not be scored)."""
    p = pos[docs]
    keep = p >= 0
    return p[keep], tf[keep], docs[keep]


def collection_prob(index, tid: int) -> float:
    """P(w | C): collection language model (maximum likelihood)."""
    return float(index.cf[tid]) / index.total_tokens


# =========================================================================== baseline
def keyword_baseline(index, qterms: QTerms, cands: np.ndarray) -> np.ndarray:
    pos = _position_map(index, cands)
    matched = np.zeros(len(cands))
    tf_sum = np.zeros(len(cands))
    for t, qtf in qterms:
        docs, tf = index.postings(t)
        p, tf, _ = _hits(pos, docs, tf)
        matched[p] += 1
        tf_sum[p] += tf
    return matched * 1e6 + tf_sum                                          # lexicographic: coverage first, then total tf


# =========================================================================== unigram language model
def unigram_lm(index, qterms: QTerms, cands: np.ndarray, *, smoothing: str = "dirichlet", mu: float = 200.0, lam: float = 0.3) -> np.ndarray:
    """log P(q | d) = sum_w qtf(w) * log P(w | d)          (query likelihood, multinomial model)

      Dirichlet:        P(w|d) = (tf + mu * P(w|C)) / (|d| + mu)
      Jelinek-Mercer:   P(w|d) = (1 - lam) * tf/|d| + lam * P(w|C)
    A word that is absent from a document therefore keeps a NON-ZERO probability (it falls back on the collection model).
    Query words absent from the whole collection (cf = 0) carry no evidence and are skipped."""
    pos = _position_map(index, cands)
    dl = index.doc_len[cands].astype(np.float64)
    score = np.zeros(len(cands))
    for t, qtf in qterms:
        pc = collection_prob(index, t)
        if pc <= 0:
            continue
        docs, tf = index.postings(t)
        p, tf, _ = _hits(pos, docs, tf)
        tf = tf.astype(np.float64)
        if smoothing == "dirichlet":
            score += qtf * (np.log(mu * pc) - np.log(dl + mu))                # tf = 0 for every candidate ...
            score[p] += qtf * (np.log(tf + mu * pc) - np.log(mu * pc))        # ... corrected where the term occurs
        elif smoothing == "jm":
            base = np.log(lam * pc)
            score += qtf * base
            score[p] += qtf * (np.log((1 - lam) * tf / dl[p] + lam * pc) - base)
        else:
            raise ValueError("smoothing must be 'dirichlet' or 'jm'")
    return score


def doc_term_probability(index, di: int, term: str, *, mu: float = 200.0) -> float:
    """P(term | document) under Dirichlet smoothing - handy for tests and for the 'multinomial distribution' demo."""
    tid = index.term_id(term)
    if tid < 0:
        return 0.0
    docs, tf = index.postings(tid)
    k = np.searchsorted(docs, di)
    f = int(tf[k]) if k < len(docs) and docs[k] == di else 0
    return (f + mu * collection_prob(index, tid)) / (index.doc_len[di] + mu)


# =========================================================================== bigram language model
def _bigram_counts(index, a: int, b: int, docs: np.ndarray) -> np.ndarray:
    out = np.zeros(len(docs))
    for i, d in enumerate(docs):
        arr = index.doc_tokens(int(d))
        out[i] = np.count_nonzero((arr[:-1] == a) & (arr[1:] == b))
    return out


def bigram_collection_counts(index, qseq: Sequence[int]) -> List[float]:
    """cf(w_{i-1} w_i) for every consecutive query pair, over ALL documents of this index (a partition returns its share;
    the coordinator sums the shares to obtain the global value)."""
    q = [t for t in qseq if index.cf[t] > 0]
    out = []
    for a, b in zip(q[:-1], q[1:]):
        both = np.intersect1d(index.postings(a)[0], index.postings(b)[0], assume_unique=True)
        out.append(float(_bigram_counts(index, a, b, both).sum()) if len(both) else 0.0)
    return out


def bigram_lm(index, qseq: Sequence[int], cands: np.ndarray, *, mu: float = 200.0, mu_b: float = 100.0, alpha: float = 0.5,
              cf_bigram: Optional[Sequence[float]] = None) -> np.ndarray:
    """P(q|d) = P(w1|d) * prod_{i>=2} P(w_i | w_{i-1}, d)

       P(w_i | w_{i-1}, d) = ( c(w_{i-1} w_i ; d) + mu_b * P_C(w_i | w_{i-1}) ) / ( c(w_{i-1} ; d) + mu_b )
       P_C(w_i | w_{i-1})  = alpha * cf(w_{i-1} w_i)/cf(w_{i-1}) + (1 - alpha) * P(w_i | C)       (back-off to the unigram model)
    The first word uses the Dirichlet unigram model.  Query terms unseen in the collection are dropped beforehand.
    cf_bigram: GLOBAL bigram collection counts (one per consecutive pair); computed from this index when omitted."""
    qseq = [t for t in qseq if index.cf[t] > 0]
    if not qseq:
        return np.zeros(len(cands))
    pos = _position_map(index, cands)
    score = unigram_lm(index, [(qseq[0], 1)], cands, mu=mu)
    for i, (a, b) in enumerate(zip(qseq[:-1], qseq[1:])):
        da, tfa = index.postings(a)
        both = np.intersect1d(da, index.postings(b)[0], assume_unique=True)
        inc = pos[both] >= 0
        cbi = np.zeros(len(cands))
        if cf_bigram is None:
            cnt_all = _bigram_counts(index, a, b, both) if len(both) else np.zeros(0)
            cf_bi, cnt = float(cnt_all.sum()), cnt_all[inc]
        else:
            cf_bi = float(cf_bigram[i])
            cnt = _bigram_counts(index, a, b, both[inc]) if inc.any() else np.zeros(0)
        cbi[pos[both[inc]]] = cnt
        cprev = np.zeros(len(cands))
        p, tfp, _ = _hits(pos, da, tfa)
        cprev[p] = tfp
        pc_bi = alpha * (cf_bi / index.cf[a]) + (1 - alpha) * collection_prob(index, b)
        score += np.log((cbi + mu_b * pc_bi) / (cprev + mu_b))
    return score


# =========================================================================== Divergence From Randomness
def dfr(index, qterms: QTerms, cands: np.ndarray, *, model: str = "PL2", c: float = 1.0) -> np.ndarray:
    """DFR measures how far a term's frequency in a document diverges from what a random (null) model predicts for the collection.

      normalisation 2:  tfn = tf * log2(1 + c * avg_len / |d|)
      PL2   randomness = Poisson with lambda = cf/N, after-effect = Laplace:
              w = ( tfn*log2(tfn/lambda) + (lambda - tfn)*log2(e) + 0.5*log2(2*pi*tfn) ) / (tfn + 1)      (clipped at 0: a term that is
              not MORE frequent than chance carries no evidence)
      InL2  randomness = inverse document frequency:   w = tfn/(tfn+1) * log2((N+1)/(df+0.5))
    Document score = sum over query terms of qtf * w; terms absent from a document contribute 0."""
    n, avg = index.n_docs, index.avg_doc_len
    pos = _position_map(index, cands)
    score = np.zeros(len(cands))
    for t, qtf in qterms:
        if index.cf[t] <= 0:
            continue
        docs, tf = index.postings(t)
        p, tf, docs = _hits(pos, docs, tf)
        tfn = tf.astype(np.float64) * np.log2(1.0 + c * avg / index.doc_len[docs])
        if model.upper() == "PL2":
            lam = index.cf[t] / n
            w = (tfn * np.log2(tfn / lam) + (lam - tfn) * LOG2E + 0.5 * np.log2(2 * math.pi * tfn)) / (tfn + 1.0)
            w = np.maximum(w, 0.0)
        elif model.upper() == "INL2":
            w = tfn / (tfn + 1.0) * math.log2((n + 1.0) / (index.df[t] + 0.5))
        else:
            raise ValueError("model must be 'PL2' or 'InL2'")
        score[p] += qtf * w
    return score
