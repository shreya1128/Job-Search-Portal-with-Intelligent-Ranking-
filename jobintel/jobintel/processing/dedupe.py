"""Duplicate detection: exact (SHA-1 of normalised text) and near-duplicate (64-bit SimHash + banded LSH)."""
from __future__ import annotations

import hashlib
import re
from collections import Counter, defaultdict
from functools import lru_cache
from typing import Dict, Iterable, List, Optional

import numpy as np

_WS = re.compile(r"[^a-z0-9]+")


def normalize_for_hash(*parts: Optional[str]) -> str:
    return _WS.sub(" ", " ".join(p for p in parts if p).lower()).strip()


def content_hash(*parts: Optional[str]) -> str:
    return hashlib.sha1(normalize_for_hash(*parts).encode("utf-8")).hexdigest()


@lru_cache(maxsize=300_000)
def _tok_hash(tok: str) -> int:
    return int.from_bytes(hashlib.md5(tok.encode("utf-8")).digest()[:8], "big")


def simhash(tokens: Iterable[str], bits: int = 64) -> Optional[int]:
    """Charikar SimHash: similar token multisets -> hashes with small Hamming distance."""
    tf = Counter(tokens)
    if not tf:
        return None
    h = np.array([_tok_hash(t) for t in tf], dtype=np.uint64)
    w = np.array(list(tf.values()), dtype=np.float64)
    shifts = np.arange(bits, dtype=np.uint64)
    bitmat = ((h[:, None] >> shifts[None, :]) & np.uint64(1)).astype(np.float64)
    score = ((bitmat * 2.0 - 1.0) * w[:, None]).sum(axis=0)
    out = 0
    for i in range(bits):
        if score[i] > 0:
            out |= 1 << i
    return out


def hamming(a: int, b: int) -> int:
    return bin(a ^ b).count("1")        # int.bit_count() would need Python 3.10+


class NearDuplicateIndex:
    """4 bands x 16 bits: any two hashes within Hamming distance <= 3 must agree on at least one band (pigeonhole)."""

    def __init__(self, max_distance: int = 3):
        assert max_distance <= 3, "4-band LSH is only exact for distance <= 3"
        self.max_distance = max_distance
        self._bands: List[Dict[int, List[int]]] = [defaultdict(list) for _ in range(4)]
        self._hash: Dict[int, int] = {}

    def candidates(self, h: int) -> List[int]:
        """Every registered document within the Hamming threshold."""
        seen, out = set(), []
        for b in range(4):
            for cand in self._bands[b].get((h >> (16 * b)) & 0xFFFF, ()):
                if cand not in seen and hamming(h, self._hash[cand]) <= self.max_distance:
                    seen.add(cand)
                    out.append(cand)
        return out

    def find(self, h: int) -> Optional[int]:
        c = self.candidates(h)
        return c[0] if c else None

    def add(self, doc_id: int, h: int) -> None:
        self._hash[doc_id] = h
        for b in range(4):
            self._bands[b][(h >> (16 * b)) & 0xFFFF].append(doc_id)

    def add_or_match(self, doc_id: int, h: int) -> Optional[int]:
        """Return id of an earlier near-duplicate, or register this doc and return None."""
        hit = self.find(h)
        if hit is None:
            self.add(doc_id, h)
        return hit
