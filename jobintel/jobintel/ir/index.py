"""Retrieval index = the 'search engine' data structure, kept SEPARATE from the application database.

    Collection  ->  Documents  ->  Terms  ->  Posting information

  vocab            term -> term id
  postings (CSR)   for every term: the documents containing it and the term frequency in each   (indptr / post_doc / post_tf)
  df, cf           document frequency (#docs containing the term) and collection frequency (total occurrences)
  doc_len, tokens  document lengths and the token sequence of every document (needed by the bigram language model)
  meta             per-document fields used for filtering and ranking signals (city, experience, skills, ...)
It is pickled to disk (scripts/build_index.py) so the search application loads it in a few seconds."""
from __future__ import annotations

import pickle
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

META_FIELDS = ["source", "title", "company", "city", "cities", "country", "exp_min", "exp_max", "sal_min", "sal_max", "sal_cur",
               "sal_period", "employment", "cluster", "category", "text_quality", "skills", "posting_date", "seniority", "external_id"]
NUMERIC = {"exp_min", "exp_max", "sal_min", "sal_max"}
FORMAT_VERSION = 2


@dataclass
class IndexDoc:
    job_id: int
    tokens: List[str]
    meta: Dict = field(default_factory=dict)


class InvertedIndex:
    # ------------------------------------------------------------------ construction
    def __init__(self) -> None:
        self.vocab: Dict[str, int] = {}
        self.terms: List[str] = []
        self.indptr = np.zeros(1, dtype=np.int64)
        self.post_doc = np.zeros(0, dtype=np.int32)
        self.post_tf = np.zeros(0, dtype=np.int32)
        self.df = np.zeros(0, dtype=np.int32)
        self.cf = np.zeros(0, dtype=np.int64)
        self.doc_len = np.zeros(0, dtype=np.int32)
        self.doc_offsets = np.zeros(1, dtype=np.int64)
        self.tokens = np.zeros(0, dtype=np.int32)
        self.job_ids = np.zeros(0, dtype=np.int64)
        self.meta: Dict[str, object] = {}
        self.pagerank: Optional[np.ndarray] = None
        self.version = FORMAT_VERSION
        self._aux_built = False

    @classmethod
    def build(cls, docs: Iterable[IndexDoc]) -> "InvertedIndex":
        ix = cls()
        per_doc: List[np.ndarray] = []
        meta_cols: Dict[str, list] = {f: [] for f in META_FIELDS}
        ids: List[int] = []
        for d in docs:
            arr = np.fromiter((ix.vocab.setdefault(t, len(ix.vocab)) for t in d.tokens), dtype=np.int32, count=len(d.tokens))
            per_doc.append(arr)
            ids.append(d.job_id)
            for f in META_FIELDS:
                meta_cols[f].append(d.meta.get(f))
        n, v = len(per_doc), len(ix.vocab)
        ix.terms = [None] * v                                           # type: ignore
        for t, i in ix.vocab.items():
            ix.terms[i] = t
        ix.job_ids = np.array(ids, dtype=np.int64)
        ix.doc_len = np.array([len(a) for a in per_doc], dtype=np.int32)
        ix.doc_offsets = np.concatenate([[0], np.cumsum(ix.doc_len)]).astype(np.int64)
        ix.tokens = np.concatenate(per_doc) if per_doc else np.zeros(0, dtype=np.int32)
        t_ids, d_ids, tfs = [], [], []
        for di, arr in enumerate(per_doc):
            u, c = np.unique(arr, return_counts=True)
            t_ids.append(u)
            d_ids.append(np.full(len(u), di, dtype=np.int32))
            tfs.append(c.astype(np.int32))
        t_ids, d_ids, tfs = (np.concatenate(x) if x else np.zeros(0, dtype=np.int32) for x in (t_ids, d_ids, tfs))
        order = np.argsort(t_ids, kind="stable")                         # stable: postings stay sorted by document inside a term
        ix.post_doc, ix.post_tf = d_ids[order], tfs[order]
        counts = np.bincount(t_ids, minlength=v)
        ix.indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        ix.df = counts.astype(np.int32)
        ix.cf = np.bincount(ix.tokens, minlength=v).astype(np.int64) if len(ix.tokens) else np.zeros(v, dtype=np.int64)
        for f in META_FIELDS:
            col = meta_cols[f]
            ix.meta[f] = np.array([np.nan if x is None else x for x in col], dtype=np.float64) if f in NUMERIC else col
        ix.build_aux()
        return ix

    def build_aux(self) -> None:
        """Lookup tables derived from the metadata: city -> docs, skill -> docs, job_id -> doc."""
        self._pos = {int(j): i for i, j in enumerate(self.job_ids)}
        self._ext = {(self.meta["source"][i], str(self.meta["external_id"][i])): i for i in range(self.n_docs) if self.meta["external_id"][i] is not None}
        city_docs: Dict[str, list] = defaultdict(list)
        skill_docs: Dict[str, list] = defaultdict(list)
        for i in range(self.n_docs):
            cs = self.meta["cities"][i] or ([self.meta["city"][i]] if self.meta["city"][i] else [])
            for c in set(cs):
                city_docs[c.lower()].append(i)
            for s in set(self.meta["skills"][i] or []):
                skill_docs[s].append(i)
        low = lambda col: np.array([(x or "").lower() for x in col], dtype=object)
        self.country_lc, self.employment_lc = low(self.meta["country"]), low(self.meta["employment"])
        self.title_lc, self.category_lc = low(self.meta["title"]), low(self.meta["category"])
        self.cluster_arr, self.source_arr, self.quality_arr = (np.array([x or "" for x in self.meta[f]], dtype=object) for f in ("cluster", "source", "text_quality"))
        self.city_docs = {k: np.array(v, dtype=np.int32) for k, v in city_docs.items()}
        self.skill_docs = {k: np.array(v, dtype=np.int32) for k, v in skill_docs.items()}
        self._aux_built = True

    # ------------------------------------------------------------------ statistics & lookups
    @property
    def n_docs(self) -> int:
        return len(self.job_ids)

    @property
    def n_terms(self) -> int:
        return len(self.terms)

    @property
    def total_tokens(self) -> int:
        return int(self.doc_len.sum())

    @property
    def avg_doc_len(self) -> float:
        return float(self.doc_len.mean()) if self.n_docs else 0.0

    def term_id(self, term: str) -> int:
        return self.vocab.get(term, -1)

    def postings(self, tid: int) -> Tuple[np.ndarray, np.ndarray]:
        a, b = self.indptr[tid], self.indptr[tid + 1]
        return self.post_doc[a:b], self.post_tf[a:b]

    def doc_index(self, job_id: int) -> Optional[int]:
        return self._pos.get(int(job_id))

    def doc_by_external(self, source: str, external_id: str) -> Optional[int]:
        """document index from the STABLE identifier (source, external_id) - the same across databases built from the same raw files."""
        return self._ext.get((source, str(external_id)))

    def doc_tokens(self, di: int) -> np.ndarray:
        return self.tokens[self.doc_offsets[di]:self.doc_offsets[di + 1]]

    def term_info(self, term: str) -> Dict:
        tid = self.term_id(term)
        return {"term": term, "in_vocab": tid >= 0, "df": int(self.df[tid]) if tid >= 0 else 0, "cf": int(self.cf[tid]) if tid >= 0 else 0}

    def doc_distribution(self, di: int, top: int = 10) -> List[Tuple[str, float]]:
        """The document's multinomial distribution over words (maximum-likelihood): P(w|d) = tf / |d|."""
        toks = self.doc_tokens(di)
        if not len(toks):
            return []
        u, c = np.unique(toks, return_counts=True)
        order = np.argsort(-c, kind="stable")[:top]
        return [(self.terms[u[i]], float(c[i]) / len(toks)) for i in order]

    def stats(self) -> Dict:
        return dict(documents=self.n_docs, terms=self.n_terms, tokens=self.total_tokens, avg_doc_len=round(self.avg_doc_len, 1),
                    postings=int(len(self.post_doc)), has_pagerank=self.pagerank is not None)

    # ------------------------------------------------------------------ persistence
    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "wb") as fh:
            pickle.dump(self, fh, protocol=pickle.HIGHEST_PROTOCOL)

    @staticmethod
    def load(path: Path) -> "InvertedIndex":
        with open(path, "rb") as fh:
            ix = pickle.load(fh)
        if getattr(ix, "version", None) != FORMAT_VERSION:
            raise ValueError("index file has an incompatible format; rebuild it with scripts/build_index.py")
        return ix
