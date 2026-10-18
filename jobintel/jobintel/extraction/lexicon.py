"""Skill lexicon: canonical names, aliases, types, ambiguity flag.  Also tag normalisation shared by all loaders."""
from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

from jobintel.extraction.seed_skills import GENERIC_SINGLE_WORDS, SEED_TEXT, TAG_STOPLIST

_PAREN = re.compile(r"\(([^)]*)\)")
_SPACES = re.compile(r"\s+")


def normalize_tag(tag: str) -> str:
    """Lowercase / unicode-fold / trim a raw skill tag; keeps + # . / inside (c++, c#, node.js, ci/cd)."""
    s = unicodedata.normalize("NFKD", str(tag))
    s = "".join(ch for ch in s if not unicodedata.combining(ch)).lower().replace("&", " and ")
    s = re.sub(r"[\"“”‘’`]", "", s)
    s = _SPACES.sub(" ", s).strip(" \t\r\n.,;:|-_")
    return s


@dataclass
class LexEntry:
    canonical: str
    type: str = "technical"
    aliases: List[str] = field(default_factory=list)
    sources: List[str] = field(default_factory=list)
    df: int = 0
    ambiguous: bool = False


class SkillLexicon:
    def __init__(self, entries: Dict[str, LexEntry]):
        self.entries = entries
        self._alias: Dict[str, str] = {}
        for e in entries.values():
            self._alias.setdefault(e.canonical, e.canonical)
            for a in e.aliases:
                self._alias.setdefault(normalize_tag(a), e.canonical)

    # ---- construction -------------------------------------------------------
    @classmethod
    def from_seed(cls) -> "SkillLexicon":
        entries: Dict[str, LexEntry] = {}
        for line in SEED_TEXT.strip().splitlines():
            canonical, typ, aliases = (line.split("|") + ["", ""])[:3]
            canonical = normalize_tag(canonical)
            als = [normalize_tag(a) for a in aliases.split(";") if a.strip()]
            entries[canonical] = LexEntry(canonical, typ.strip(), als, ["seed"], 0, len(canonical) == 1)  # single letters (R, C) are too ambiguous in free text
        return cls(entries)

    @classmethod
    def load(cls, path: Path) -> "SkillLexicon":
        raw = json.loads(Path(path).read_text(encoding="utf-8"))
        return cls({k: LexEntry(**v) for k, v in raw["skills"].items()})

    def save(self, path: Path) -> None:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({"skills": {k: asdict(v) for k, v in self.entries.items()}}, ensure_ascii=False), encoding="utf-8")

    def add(self, canonical: str, typ: str, source: str, df: int = 0, aliases: Optional[List[str]] = None) -> LexEntry:
        canonical = normalize_tag(canonical)
        e = self.entries.get(canonical)
        if e is None:
            single_generic = (len(canonical.split()) == 1 and canonical in GENERIC_SINGLE_WORDS) or canonical in GENERIC_SINGLE_WORDS
            e = LexEntry(canonical, typ, [], [source], df, ambiguous=single_generic)
            self.entries[canonical] = e
            self._alias.setdefault(canonical, canonical)
        else:
            if source not in e.sources:
                e.sources.append(source)
            e.df += df
        for a in aliases or []:
            a = normalize_tag(a)
            if a and a not in e.aliases and a != canonical:
                e.aliases.append(a)
                self._alias.setdefault(a, canonical)
        return e

    # ---- use ----------------------------------------------------------------
    def normalize(self, tag: str) -> Optional[str]:
        """Map a raw dataset tag to its canonical skill name; None if the tag is not a skill at all."""
        full = normalize_tag(tag)
        if full in TAG_STOPLIST or len(full) > 60:
            return None
        if full in self._alias:
            return self._alias[full]
        inner = [normalize_tag(m) for m in _PAREN.findall(full)]
        outer = normalize_tag(_PAREN.sub(" ", full))
        for cand in [outer] + inner:
            if cand in self._alias:
                return self._alias[cand]
        out = outer or full
        return None if out in TAG_STOPLIST else out

    def get(self, canonical: str) -> Optional[LexEntry]:
        return self.entries.get(canonical)

    def phrases(self, include_ambiguous: bool = False) -> Iterator[Tuple[str, str]]:
        for e in self.entries.values():
            if e.ambiguous and not include_ambiguous:
                continue
            yield e.canonical, e.canonical
            for a in e.aliases:
                yield a, e.canonical

    def __len__(self) -> int:
        return len(self.entries)


_cached: Optional[SkillLexicon] = None


def load_lexicon(path: Optional[Path] = None) -> SkillLexicon:
    """Built lexicon if scripts/build_lexicon.py has been run, else the curated seed."""
    global _cached
    from jobintel.config import LEXICON_PATH
    p = Path(path) if path else LEXICON_PATH
    if path is None and _cached is not None:
        return _cached
    lex = SkillLexicon.load(p) if p.exists() else SkillLexicon.from_seed()
    if path is None:
        _cached = lex
    return lex
