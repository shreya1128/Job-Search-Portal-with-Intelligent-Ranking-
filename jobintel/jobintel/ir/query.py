"""Query processing:  raw text -> validate (finite automaton) -> clean -> tokenise -> recognise entities -> query representation.

    "Entry level machine learning jobs in Pune"
      validation DFA      accepts
      entities            level = entry (0-2 yrs) | skill = machine learning | location = Pune
      retrieval terms     machin, learn            (entity words that act as filters/signals - Pune, entry level - are not matched as text)
"""
from __future__ import annotations

from collections import Counter, OrderedDict
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from jobintel.extraction.extractor import Extractor
from jobintel.ir.index import InvertedIndex
from jobintel.processing.stemmer import stem
from jobintel.processing.text import token_stem, tokenize

MAX_QUERY_CHARS = 200
_GENERIC = {"job", "jobs", "opening", "openings", "vacancy", "vacancies", "position", "positions", "hiring", "hire", "career", "careers",
            "recruitment", "looking", "need", "needed", "wanted", "find", "search", "near", "me", "work", "opportunity", "opportunities"}
GENERIC_STEMS = {stem(w) for w in _GENERIC}
_MASKED_LABELS = {"location", "level", "employment", "experience", "salary"}


class QueryDFA:
    """Deterministic finite automaton that validates the raw query string character by character.

    character classes   L letter/digit   S space   P allowed punctuation   X anything else (control chars, <, >, ;, backslash ...)
    states              START --L--> HAS_ALNUM (accepting) ;  START --P--> ONLY_PUNCT ;  ONLY_PUNCT --L--> HAS_ALNUM ;  any --X--> DEAD (sink)
    """
    START, HAS_ALNUM, ONLY_PUNCT, DEAD = range(4)
    PUNCT = set("+#./,&'-()%$:@_₹")
    TABLE = {START: {"L": HAS_ALNUM, "S": START, "P": ONLY_PUNCT, "X": DEAD},
             HAS_ALNUM: {"L": HAS_ALNUM, "S": HAS_ALNUM, "P": HAS_ALNUM, "X": DEAD},
             ONLY_PUNCT: {"L": HAS_ALNUM, "S": ONLY_PUNCT, "P": ONLY_PUNCT, "X": DEAD},
             DEAD: {"L": DEAD, "S": DEAD, "P": DEAD, "X": DEAD}}

    @classmethod
    def char_class(cls, ch: str) -> str:
        if ch.isalnum():
            return "L"
        if ch == " ":
            return "S"
        return "P" if ch in cls.PUNCT else "X"

    @classmethod
    def validate(cls, q: str) -> Tuple[bool, Optional[str]]:
        if q is None or not isinstance(q, str):
            return False, "query must be text"
        if len(q) > MAX_QUERY_CHARS:
            return False, f"query too long (max {MAX_QUERY_CHARS} characters)"
        state = cls.START
        for ch in q:
            state = cls.TABLE[state][cls.char_class(ch)]
            if state == cls.DEAD:
                return False, f"unsupported character {ch!r}"
        return (True, None) if state == cls.HAS_ALNUM else (False, "query is empty" if state == cls.START else "query has no letters or digits")


@dataclass
class ParsedQuery:
    raw: str
    valid: bool = True
    error: Optional[str] = None
    terms: List[str] = field(default_factory=list)                 # ordered index-form retrieval terms
    surface: Dict[str, str] = field(default_factory=dict)          # index form -> the word the user typed
    skills: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    experience: Optional[Dict] = None                              # {min, max, source: stated|level}
    level: Optional[Dict] = None
    employment_type: Optional[str] = None
    salary: Optional[Dict] = None

    def qterms(self, index: InvertedIndex) -> Tuple[List[Tuple[int, int]], List[str]]:
        """-> ([(term id, query tf)], terms unknown to the index)"""
        cnt = Counter(self.terms)
        known, unknown = [], []
        for t, n in cnt.items():
            tid = index.term_id(t)
            (known if tid >= 0 else unknown).append((tid, n) if tid >= 0 else t)
        return known, unknown

    def sequence(self, index: InvertedIndex) -> List[int]:
        return [index.term_id(t) for t in self.terms if index.term_id(t) >= 0]

    def to_dict(self) -> Dict:
        return {k: getattr(self, k) for k in ("raw", "valid", "error", "terms", "skills", "locations", "experience", "level", "employment_type", "salary")}


class QueryProcessor:
    def __init__(self, extractor: Extractor):
        self.extractor = extractor

    def parse(self, raw: str) -> ParsedQuery:
        ok, why = QueryDFA.validate(raw)
        if not ok:
            return ParsedQuery(raw=str(raw)[:MAX_QUERY_CHARS], valid=False, error=why)
        ex = self.extractor.extract(raw)
        chars = list(raw)
        for sp in ex["spans"]:
            if sp["label"] in _MASKED_LABELS:
                for i in range(sp["start"], sp["end"]):
                    chars[i] = " "
        terms, surface = [], OrderedDict()
        for t in tokenize("".join(chars)):
            s = token_stem(t)
            if s is None or s in GENERIC_STEMS:
                continue
            terms.append(s)
            surface.setdefault(s, t.raw)
        experience = None
        if ex["experience"]:
            experience = {"min": ex["experience"]["min"], "max": ex["experience"]["max"], "source": "stated"}
        elif ex["level"]:
            experience = {"min": ex["level"]["min"], "max": ex["level"]["max"], "source": "level"}
        return ParsedQuery(raw=raw, terms=terms, surface=dict(surface), skills=[s["name"] for s in ex["skills"]], locations=ex["locations"],
                           experience=experience, level=ex["level"], employment_type=ex["employment_type"], salary=ex["salary"])
