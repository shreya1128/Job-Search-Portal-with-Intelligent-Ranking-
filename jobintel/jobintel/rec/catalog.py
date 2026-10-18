"""ItemCatalog: the jobs of an InvertedIndex as an IDF-weighted skill space (+ metadata accessors).  Everything here is derived from the index; no database is needed.

  X[j, s] = 1 if job j lists skill s.   idf[s] = ln((N + 1) / (df_s + 1)) + 1  (rare skills count more).   Xn = row-normalised (X * idf).
  content_scores(profile)  cosine between the profile's idf-weighted skill vector and every job            (symmetric: also penalises jobs with many unrelated skills)
  coverage(profile)        share of the JOB'S idf mass that the profile covers (what a recruiter calls 'how many of the requirements do you have')
Jobs without any skill have no skill evidence: coverage returns the ranker's neutral UNKNOWN_SCORE for them, content 0."""
from __future__ import annotations

import re
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import scipy.sparse as sp

from ..ir.ranker import UNKNOWN_SCORE

_STOP = {"and", "the", "for", "with", "jobs", "job", "senior", "junior", "lead", "sr", "jr"}


class ItemCatalog:
    def __init__(self, ix):
        self.ix = ix
        N = ix.n_docs
        sk_lists = [list(set(s or [])) for s in ix.meta["skills"]]
        self.skills: List[str] = sorted({s for ss in sk_lists for s in ss})
        self.sidx: Dict[str, int] = {s: i for i, s in enumerate(self.skills)}
        rows = np.fromiter((i for i, ss in enumerate(sk_lists) for _ in ss), dtype=np.int64)
        cols = np.fromiter((self.sidx[s] for ss in sk_lists for s in ss), dtype=np.int64)
        self.X = sp.csr_matrix((np.ones(len(rows), dtype=np.float32), (rows, cols)), shape=(N, len(self.skills)))
        df = np.asarray(self.X.sum(axis=0)).ravel()
        self.df = df
        self.idf = (np.log((N + 1.0) / (df + 1.0)) + 1.0).astype(np.float32)
        self.Xw = self.X.multiply(self.idf).tocsr()
        self.mass = np.asarray(self.Xw.sum(axis=1)).ravel()
        norms = np.sqrt(np.asarray(self.Xw.multiply(self.Xw).sum(axis=1)).ravel())
        self.has_skills = norms > 0
        inv = np.where(norms > 0, 1.0 / np.maximum(norms, 1e-12), 0.0).astype(np.float32)
        self.Xn = sp.diags(inv) @ self.Xw
        self.N = N

    # ------------------------------------------------------------------------------------------------ vectors
    def vector(self, weights: Dict[str, float]) -> Tuple[np.ndarray, List[str]]:
        v = np.zeros(len(self.skills), dtype=np.float32)
        unknown = []
        for s, w in weights.items():
            i = self.sidx.get(s)
            if i is None:
                unknown.append(s)
            else:
                v[i] = max(v[i], float(w))
        return v, unknown

    def content_scores(self, weights: Dict[str, float]) -> Optional[np.ndarray]:
        v, _ = self.vector(weights)
        q = v * self.idf
        n = float(np.linalg.norm(q))
        if n == 0:
            return None                                   # none of the profile's skills occurs in any job: no content evidence at all
        return np.asarray(self.Xn @ (q / n)).ravel()

    def coverage(self, weights: Dict[str, float]) -> Optional[np.ndarray]:
        v, _ = self.vector(weights)
        if not v.any():
            return None
        got = np.asarray(self.Xw @ v).ravel()
        out = np.where(self.mass > 0, got / np.maximum(self.mass, 1e-12), UNKNOWN_SCORE)
        return np.clip(out, 0.0, 1.0)

    # ------------------------------------------------------------------------------------------------ jobs
    def job_skills(self, di: int) -> List[str]:
        return sorted(set(self.ix.meta["skills"][di] or []))

    def similar_jobs(self, di: int, k: int = 50) -> np.ndarray:
        if not self.has_skills[di]:
            return np.zeros(0, dtype=np.int64)
        sims = np.asarray(self.Xn @ self.Xn[di].T.toarray().ravel()).ravel()
        sims[di] = -1.0
        top = np.argpartition(-sims, min(k, self.N - 1) - 1)[:k]
        top = top[sims[top] > 0]
        return top[np.argsort(-sims[top])]

    def jobs_by_title(self, role_text: str, limit: int = 300) -> np.ndarray:
        toks = [t for t in re.findall(r"[a-z]+", (role_text or "").lower()) if len(t) > 2 and t not in _STOP]
        if not toks:
            return np.zeros(0, dtype=np.int64)
        mask = np.array([all(t in title for t in toks) for title in self.ix.title_lc])
        return np.where(mask)[0][:limit]

    def market_frequency(self, docs: Sequence[int]) -> Dict[str, float]:
        docs = np.asarray(docs, dtype=np.int64)
        docs = docs[self.has_skills[docs]] if len(docs) else docs
        if not len(docs):
            return {}
        cnt = np.asarray(self.X[docs].sum(axis=0)).ravel()
        nz = np.nonzero(cnt)[0]
        return {self.skills[i]: float(cnt[i] / len(docs)) for i in nz}

    def meta(self, di: int) -> Dict:
        m = self.ix.meta
        g = lambda f: m[f][di]
        return {"job_id": int(self.ix.job_ids[di]), "title": g("title"), "company": g("company"), "city": g("city"), "cities": list(g("cities") or []), "country": g("country"),
                "source": g("source"), "category": g("category"), "employment": g("employment"), "exp_min": float(g("exp_min")), "exp_max": float(g("exp_max")),
                "sal_min": float(g("sal_min")), "sal_max": float(g("sal_max")), "sal_cur": g("sal_cur"), "text_quality": g("text_quality")}
