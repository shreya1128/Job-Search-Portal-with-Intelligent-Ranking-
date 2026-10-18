"""Finite-state machinery used for query validation and structured-pattern recognition (IR syllabus: finite automata).

Two automata families are implemented from scratch:

1. TokenNFA   - a non-deterministic finite automaton over a TOKEN stream.  Each edge carries a predicate
                (a token class such as NUM, YEAR-WORD, CURRENCY).  It is simulated with the classical
                "set of current states" algorithm, i.e. the NFA is run as an implicit DFA.  Used for
                experience / degree / salary / certification / job-role patterns.
2. AhoCorasick - a deterministic automaton built from a dictionary of multi-word phrases (trie + failure
                links).  One left-to-right pass finds EVERY skill / city / employment-type phrase in a text,
                independent of dictionary size.

What this is NOT: finite automata recognise surface patterns; they do not "understand" meaning.
Semantic matching is handled elsewhere (language models, deep learning, knowledge base).
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple

from jobintel.processing.text import Tok

Pred = Callable[[Tok], bool]


@dataclass(frozen=True)
class FsaMatch:
    start: int          # token index (inclusive)
    end: int            # token index (exclusive)
    tag: str            # accepting-state tag, e.g. 'exp_explicit'
    automaton: str
    payload: Any = None

    def text(self, toks: Sequence[Tok], source: str) -> str:
        return source[toks[self.start].start: toks[self.end - 1].end]


class TokenNFA:
    def __init__(self, name: str):
        self.name = name
        self.start = 0
        self.edges: Dict[int, List[Tuple[str, Pred, int]]] = {0: []}
        self.accepting: Dict[int, Tuple[str, int]] = {}
        self._n = 1

    def new_state(self) -> int:
        s = self._n
        self._n += 1
        self.edges[s] = []
        return s

    def edge(self, src: int, label: str, pred: Pred, dst: int) -> None:
        self.edges[src].append((label, pred, dst))

    def accept(self, state: int, tag: str, priority: int = 0) -> None:
        self.accepting[state] = (tag, priority)

    # ---- simulation -------------------------------------------------------
    def match_at(self, toks: Sequence[Tok], i: int) -> Optional[Tuple[int, str]]:
        """Longest match starting exactly at token i -> (end_index, tag) or None."""
        states = {self.start}
        best = None
        j = i
        while j < len(toks) and states:
            tok = toks[j]
            nxt = {dst for s in states for (_, pred, dst) in self.edges[s] if pred(tok)}
            j += 1
            acc = [self.accepting[s] for s in nxt if s in self.accepting]
            if acc:
                best = (j, max(acc, key=lambda a: a[1])[0])
            states = nxt
        return best

    def scan(self, toks: Sequence[Tok]) -> List[FsaMatch]:
        """Leftmost-longest, non-overlapping matches over the whole token stream."""
        out, i = [], 0
        while i < len(toks):
            m = self.match_at(toks, i)
            if m:
                out.append(FsaMatch(i, m[0], m[1], self.name))
                i = m[0]
            else:
                i += 1
        return out

    # ---- documentation helpers (state diagram for the report) ---------------
    @property
    def n_states(self) -> int:
        return self._n

    @property
    def n_transitions(self) -> int:
        return sum(len(v) for v in self.edges.values())

    def to_dot(self) -> str:
        lines = [f'digraph "{self.name}" {{', "  rankdir=LR;", '  node [shape=circle, fontsize=10];']
        for s, (tag, _) in self.accepting.items():
            lines.append(f'  q{s} [shape=doublecircle, xlabel="{tag}"];')
        lines.append('  start [shape=point]; start -> q0;')
        for s, es in self.edges.items():
            merged: Dict[int, List[str]] = {}
            for label, _, dst in es:
                merged.setdefault(dst, []).append(label)
            for dst, labels in merged.items():
                lines.append(f'  q{s} -> q{dst} [label="{" | ".join(labels)}", fontsize=9];')
        lines.append("}")
        return "\n".join(lines)


# =========================================================================== Aho-Corasick
def light(w: str) -> str:
    """Very light plural folding applied identically to dictionary phrases and text (pandas~panda on both sides)."""
    if len(w) > 3 and w.endswith("s") and not w.endswith(("ss", "us", "is")):
        return w[:-1]
    return w


class AhoCorasick:
    """Multi-pattern matching automaton over token sequences (a trie turned into a DFA with failure links)."""

    def __init__(self) -> None:
        self.goto: List[Dict[str, int]] = [{}]
        self.fail: List[int] = [0]
        self.out: List[List[Tuple[int, Any]]] = [[]]
        self._built = False
        self.n_patterns = 0

    def add(self, key_tokens: Sequence[str], payload: Any) -> None:
        if not key_tokens:
            return
        s = 0
        for t in key_tokens:
            nxt = self.goto[s].get(t)
            if nxt is None:
                self.goto.append({})
                self.fail.append(0)
                self.out.append([])
                nxt = len(self.goto) - 1
                self.goto[s][t] = nxt
            s = nxt
        if not any(p == payload and l == len(key_tokens) for l, p in self.out[s]):
            self.out[s].append((len(key_tokens), payload))
            self.n_patterns += 1
        self._built = False

    def build(self) -> "AhoCorasick":
        q = deque()
        for t, s in self.goto[0].items():
            self.fail[s] = 0
            q.append(s)
        while q:
            r = q.popleft()
            for t, s in self.goto[r].items():
                q.append(s)
                f = self.fail[r]
                while f and t not in self.goto[f]:
                    f = self.fail[f]
                self.fail[s] = self.goto[f].get(t, 0)
                if self.fail[s] == s:
                    self.fail[s] = 0
                self.out[s] = self.out[s] + self.out[self.fail[s]]
        self._built = True
        return self

    @property
    def n_states(self) -> int:
        return len(self.goto)

    def scan(self, toks: Sequence[Tok], skip_syms: frozenset = frozenset({"-", "/"})) -> List[Tuple[int, int, Any]]:
        """All matches as (start_tok, end_tok_exclusive, payload).  Punctuation other than '-' and '/' is a barrier."""
        if not self._built:
            self.build()
        state, kept, res = 0, [], []
        for i, t in enumerate(toks):
            if t.kind == "sym":
                if t.norm in skip_syms:
                    continue
                state, kept = 0, []
                continue
            key = light(t.norm)
            kept.append(i)
            while state and key not in self.goto[state]:
                state = self.fail[state]
            state = self.goto[state].get(key, 0)
            for plen, payload in self.out[state]:
                s = len(kept) - plen
                if s >= 0:
                    res.append((kept[s], i + 1, payload))
        return res


def select_longest(matches: List[Tuple[int, int, Any]]) -> List[Tuple[int, int, Any]]:
    """Maximal-munch: prefer longer matches, then earlier ones; discard overlaps."""
    chosen: List[Tuple[int, int, Any]] = []
    taken: List[Tuple[int, int]] = []
    for s, e, p in sorted(matches, key=lambda m: (-(m[1] - m[0]), m[0])):
        if all(e <= ts or s >= te for ts, te in taken):
            chosen.append((s, e, p))
            taken.append((s, e))
    return sorted(chosen)
