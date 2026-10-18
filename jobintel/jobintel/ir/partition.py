"""Partitioned index: the collection is split into P partitions (the 'search nodes' of the parallel / distributed architecture).

    Partition 1 -> documents 0 .. N/P-1        Partition 2 -> N/P .. 2N/P-1       ...      (strategy 'range', as in the project brief)
    strategy 'roundrobin' instead gives document i to partition i mod P (more even work when the collection is ordered by source)

A partition holds LOCAL postings / document lengths / token sequences but exposes GLOBAL collection statistics (cf, df, N, total
tokens, average length): every partition therefore scores a document exactly as the monolithic index would, and the merged ranking is
identical to a single-process search (verified by tests and by scripts/verify_phase3.py)."""
from __future__ import annotations

import pickle
import time
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace
from typing import List, Optional, Sequence

import numpy as np


@dataclass
class GlobalStats:
    n_docs: int
    total_tokens: int
    avg_doc_len: float
    df: np.ndarray
    cf: np.ndarray

    @classmethod
    def from_index(cls, ix) -> "GlobalStats":
        return cls(int(len(ix.doc_len)), int(ix.doc_len.sum()), float(ix.doc_len.mean()), np.asarray(ix.df), np.asarray(ix.cf))


class PartitionIndex:
    def __init__(self, part_id: int, global_idx: np.ndarray, indptr, post_doc, post_tf, doc_len, doc_offsets, tokens, g: GlobalStats):
        self.part_id, self.global_idx = part_id, global_idx
        self.indptr, self.post_doc, self.post_tf = indptr, post_doc, post_tf
        self.doc_len, self.doc_offsets, self.tokens, self.g = doc_len, doc_offsets, tokens, g
        self._lookup: Optional[np.ndarray] = None

    # --- GLOBAL collection statistics (what the scoring functions read) -----------------------------------------------------
    cf = property(lambda self: self.g.cf)
    df = property(lambda self: self.g.df)
    n_docs = property(lambda self: self.g.n_docs)
    total_tokens = property(lambda self: self.g.total_tokens)
    avg_doc_len = property(lambda self: self.g.avg_doc_len)

    # --- LOCAL data ---------------------------------------------------------------------------------------------------------
    @property
    def n_local(self) -> int:
        return len(self.doc_len)

    def postings(self, tid: int):
        a, b = self.indptr[tid], self.indptr[tid + 1]
        return self.post_doc[a:b], self.post_tf[a:b]

    def doc_tokens(self, di: int) -> np.ndarray:
        return self.tokens[self.doc_offsets[di]:self.doc_offsets[di + 1]]

    @property
    def local_lookup(self) -> np.ndarray:
        """global document index -> local index (-1 when the document lives on another partition)."""
        if self._lookup is None:
            self._lookup = np.full(self.g.n_docs, -1, dtype=np.int32)
            self._lookup[self.global_idx] = np.arange(self.n_local, dtype=np.int32)
        return self._lookup

    def memory_mb(self) -> float:
        return sum(a.nbytes for a in (self.indptr, self.post_doc, self.post_tf, self.doc_len, self.doc_offsets, self.tokens, self.global_idx)) / 1e6


def build_partitions(ix, n_parts: int, strategy: str = "range") -> List[PartitionIndex]:
    """Split an index (anything exposing indptr/post_doc/post_tf/doc_len/doc_offsets/tokens/cf/df) into n_parts partitions."""
    n = len(ix.doc_len)
    n_parts = max(1, min(n_parts, n))
    g = GlobalStats.from_index(ix)
    if strategy == "range":
        edges = np.linspace(0, n, n_parts + 1).astype(np.int64)
        sels = [np.arange(edges[i], edges[i + 1], dtype=np.int64) for i in range(n_parts)]
    elif strategy == "roundrobin":
        sels = [np.arange(i, n, n_parts, dtype=np.int64) for i in range(n_parts)]
    else:
        raise ValueError("strategy must be 'range' or 'roundrobin'")
    v = len(ix.indptr) - 1
    term_of_post = np.repeat(np.arange(v, dtype=np.int32), np.diff(ix.indptr))
    parts = []
    for pid, sel in enumerate(sels):
        in_part = np.zeros(n, dtype=bool)
        in_part[sel] = True
        local = np.full(n, -1, dtype=np.int32)
        local[sel] = np.arange(len(sel), dtype=np.int32)
        m = in_part[ix.post_doc]
        counts = np.bincount(term_of_post[m], minlength=v)
        indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        doc_len = ix.doc_len[sel]
        offs = np.concatenate([[0], np.cumsum(doc_len)]).astype(np.int64)
        if strategy == "range" and len(sel):
            tokens = ix.tokens[ix.doc_offsets[sel[0]]:ix.doc_offsets[sel[-1] + 1]]
        else:
            tokens = np.concatenate([ix.tokens[ix.doc_offsets[i]:ix.doc_offsets[i + 1]] for i in sel]) if len(sel) else np.zeros(0, dtype=np.int32)
        parts.append(PartitionIndex(pid, sel, indptr, local[ix.post_doc[m]], ix.post_tf[m], doc_len, offs, tokens, g))
    return parts


def save_partitions(parts: Sequence[PartitionIndex], directory: Path) -> List[Path]:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = []
    for p in parts:
        f = directory / f"partition_{p.part_id:02d}.pkl"
        with open(f, "wb") as fh:
            pickle.dump(p, fh, protocol=pickle.HIGHEST_PROTOCOL)
        paths.append(f)
    return paths


def load_partition(path) -> PartitionIndex:
    with open(path, "rb") as fh:
        return pickle.load(fh)


def replicate(ix, m: int):
    """A m-times larger collection made by repeating every document m times (new document ids).  ONLY for scaling benchmarks: the
    scoring cost per document is realistic, the content is obviously duplicated.  Pure NumPy, no per-document loops."""
    n, v = len(ix.doc_len), len(ix.indptr) - 1
    df = np.diff(ix.indptr)
    term_of_post = np.repeat(np.arange(v, dtype=np.int64), df)
    rank_in_term = np.arange(len(ix.post_doc), dtype=np.int64) - ix.indptr[term_of_post]
    post_doc = np.empty(len(ix.post_doc) * m, dtype=np.int32)
    post_tf = np.tile(ix.post_tf, m)
    for k in range(m):
        dest = m * ix.indptr[term_of_post] + k * df[term_of_post] + rank_in_term
        post_doc[dest] = ix.post_doc + k * n
    post_tf_full = np.empty_like(post_tf)
    for k in range(m):
        dest = m * ix.indptr[term_of_post] + k * df[term_of_post] + rank_in_term
        post_tf_full[dest] = ix.post_tf
    doc_len = np.tile(ix.doc_len, m)
    return SimpleNamespace(indptr=ix.indptr * m, post_doc=post_doc, post_tf=post_tf_full, doc_len=doc_len,
                           doc_offsets=np.concatenate([[0], np.cumsum(doc_len)]).astype(np.int64), tokens=np.tile(ix.tokens, m),
                           cf=ix.cf * m, df=(ix.df * m).astype(np.int32))
