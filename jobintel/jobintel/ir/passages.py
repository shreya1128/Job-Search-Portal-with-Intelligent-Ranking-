"""Passage retrieval: split a long job description into passages, score each against the query, return the best one with the
matching words highlighted (the 'WHY THIS JOB MATCHED' feature).

  description --segment--> passages (respecting section headings, bullets and sentences; 30-90 words)
              --score----> passage language model:  mean over query terms of  log P(w|passage) - log P(w|C)      (Dirichlet, small mu)
              --highlight> HTML with <mark> around every word whose index form equals a query term"""
from __future__ import annotations

import html
import math
import re
from collections import Counter
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Set, Tuple

from jobintel.processing.text import index_tokens, token_stem, tokenize


@dataclass
class Passage:
    no: int
    text: str
    start: int
    end: int
    section: Optional[str]
    n_words: int


def _is_heading(line: str) -> bool:
    words = line.split()
    if not words or len(words) > 6:
        return False
    return line.endswith(":") or (line.isupper() and len(line) > 3 and not line.startswith("•"))


def _units(text: str, max_words: int) -> List[Tuple[int, int, Optional[str], int]]:
    """Atomic pieces (start, end, section, n_words): one per line, long lines split at sentence ends, very long sentences by word windows."""
    out: List[Tuple[int, int, Optional[str], int]] = []
    section: Optional[str] = None
    for m in re.finditer(r"[^\n]+", text):
        raw = m.group(0)
        line = raw.strip()
        if not line:
            continue
        base = m.start() + (len(raw) - len(raw.lstrip()))
        if _is_heading(line):
            section = line.rstrip(":").strip()
            continue
        n = len(line.split())
        if n <= max_words:
            out.append((base, base + len(line), section, n))
            continue
        for sm in re.finditer(r"[^.!?]+(?:[.!?]+|$)\s*", line):
            sent = sm.group(0).strip()
            if not sent:
                continue
            s0 = base + sm.start() + (len(sm.group(0)) - len(sm.group(0).lstrip()))
            words = list(re.finditer(r"\S+", sent))
            for i in range(0, len(words), max_words):
                chunk = words[i:i + max_words]
                out.append((s0 + chunk[0].start(), s0 + chunk[-1].end(), section, len(chunk)))
    return out


def segment(text: str, target_words: int = 55, max_words: int = 90, min_words: int = 8) -> List[Passage]:
    passages: List[Passage] = []
    cur: List[Tuple[int, int, Optional[str], int]] = []

    def flush():
        if cur:
            s, e = cur[0][0], cur[-1][1]
            passages.append(Passage(len(passages), text[s:e], s, e, cur[0][2], sum(u[3] for u in cur)))
            cur.clear()

    for u in _units(text or "", max_words):
        if cur and (u[2] != cur[0][2] or sum(x[3] for x in cur) + u[3] > max_words or sum(x[3] for x in cur) >= target_words):
            flush()
        cur.append(u)
    flush()
    # a fragment of a few words is poor evidence: fold it into the previous passage of the same section
    merged: List[Passage] = []
    for p in passages:
        if merged and p.n_words < min_words and p.section == merged[-1].section:
            q = merged[-1]
            merged[-1] = Passage(q.no, text[q.start:p.end], q.start, p.end, q.section, q.n_words + p.n_words)
        else:
            merged.append(Passage(len(merged), p.text, p.start, p.end, p.section, p.n_words))
    return merged


def score_passages(passages: Sequence[Passage], qterms: Sequence[Tuple[str, int, float]], mu: float = 50.0) -> List[float]:
    """qterms = [(index-form term, query tf, P(term|collection))].  Higher is better; passages with no indexed words get -inf."""
    scores: List[float] = []
    total_q = sum(q for _, q, _ in qterms) or 1
    for p in passages:
        toks = index_tokens(p.text)
        if not toks:
            scores.append(float("-inf"))
            continue
        tf = Counter(toks)
        s = 0.0
        for term, qtf, pc in qterms:
            pc = max(pc, 1e-12)
            s += qtf * (math.log((tf.get(term, 0) + mu * pc) / (len(toks) + mu)) - math.log(pc))
        scores.append(s / total_q)
    return scores


def best_passage(passages: Sequence[Passage], scores: Sequence[float]) -> Tuple[Optional[Passage], float]:
    if not passages:
        return None, float("-inf")
    i = max(range(len(passages)), key=lambda k: scores[k])
    return passages[i], scores[i]


def highlight(text: str, stems: Set[str]) -> str:
    """HTML-escaped text with <mark> around words whose index form is a query term (adjacent marks are merged)."""
    out, last = [], 0
    for t in tokenize(text):
        if token_stem(t) in stems:
            out.append(html.escape(text[last:t.start]))
            out.append("<mark>" + html.escape(text[t.start:t.end]) + "</mark>")
            last = t.end
    out.append(html.escape(text[last:]))
    return re.sub(r"</mark>(\s+)<mark>", r"\1", "".join(out))


def matched_terms(text: str, stems: Set[str]) -> Set[str]:
    return {s for s in (token_stem(t) for t in tokenize(text)) if s in stems}
