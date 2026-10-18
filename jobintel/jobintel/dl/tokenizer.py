"""Deep-learning text preprocessing (the SAME object is used for training and for the deployed application, so they cannot drift apart):

    raw text -> clean/tokenise (Phase-1 tech-aware tokeniser) -> lower-case words, numbers -> <num>, punctuation dropped
             -> vocabulary (training data only; <pad>=0, <unk>=1, <num>=2) -> integer encoding -> truncate / pad to max_len (post-padding)
Persisted as JSON (vocabulary + configuration)."""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Sequence

import numpy as np

from jobintel.processing.text import tokenize

PAD, UNK, NUM = 0, 1, 2
VERSION = 1


def words(text: str) -> List[str]:
    out = []
    for t in tokenize(text or ""):
        if t.kind == "num":
            out.append("<num>")
        elif t.kind in ("word", "abbr"):
            out.append(t.norm.replace(".", "") if t.kind == "abbr" else t.norm)
    return out


class DLTokenizer:
    def __init__(self, vocab: Dict[str, int], max_len: int = 256, config: Dict | None = None):
        self.vocab, self.max_len, self.config = vocab, max_len, config or {}

    @classmethod
    def fit(cls, texts: Iterable[str], vocab_size: int = 20000, min_count: int = 3, max_len: int = 256) -> "DLTokenizer":
        cnt: Counter = Counter()
        for t in texts:
            cnt.update(w for w in words(t) if w != "<num>")
        vocab = {"<pad>": PAD, "<unk>": UNK, "<num>": NUM}
        for w, c in cnt.most_common(vocab_size - 3):
            if c >= min_count:
                vocab[w] = len(vocab)
        return cls(vocab, max_len, {"vocab_size": vocab_size, "min_count": min_count, "fit_tokens": int(sum(cnt.values()))})

    def encode_one(self, text: str) -> List[int]:
        return [NUM if w == "<num>" else self.vocab.get(w, UNK) for w in words(text)][: self.max_len]

    def encode(self, texts: Sequence[str]) -> np.ndarray:
        out = np.zeros((len(texts), self.max_len), dtype=np.int32)
        for i, t in enumerate(texts):
            ids = self.encode_one(t)
            out[i, : len(ids)] = ids
        return out

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"version": VERSION, "max_len": self.max_len, "config": self.config, "vocab": self.vocab}), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "DLTokenizer":
        d = json.loads(Path(path).read_text(encoding="utf-8"))
        assert d["version"] == VERSION, "tokenizer file has an incompatible version"
        return cls(d["vocab"], d["max_len"], d["config"])
