"""Document cleaning + tokenisation.

Pipeline for a raw job description (keeps BOTH versions, as the spec requires):
    raw  --fix_mojibake--> --strip_html--> --NFKC/whitespace--> --repair glued sentences-->  description_clean   (for display/passages)
    description_clean --tokenize--> lowercase --tech-aware normalisation--> stop-words --Porter stem--> index tokens
"""
from __future__ import annotations

import html as _html
import re
import unicodedata
import warnings
from typing import Iterable, List, NamedTuple, Optional

from jobintel.processing.stopwords import STOPWORDS
from jobintel.processing.stemmer import stem

# --------------------------------------------------------------------------- mojibake / html
_MOJI_HINT = re.compile(r"[ÃÂâ][\u0080-\u00bf€™œžŸ“”‘’‚„†‡ˆ‰ŠŒŽ–—˜š›]|Â")
_MOJI_SEG = re.compile(r"(?:[ÃÂâ][\u0080-\u00bf€™œžŸ“”‘’‚„†‡ˆ‰ŠŒŽ–—˜š›¡-¿]+)+")


def fix_mojibake(s: str) -> str:
    """Repair UTF-8 text that was decoded as cp1252/latin-1 (e.g. 'CafÃ©', 'AccomplishedÂ ')."""
    if not s or not _MOJI_HINT.search(s):
        return s
    s = s.replace("Â\u00a0", " ").replace("Â ", " ")

    def repl(m: re.Match) -> str:
        seg = m.group(0)
        for enc in ("cp1252", "latin-1"):
            try:
                return seg.encode(enc).decode("utf-8")
            except UnicodeError:
                continue
        return ""
    s = _MOJI_SEG.sub(repl, s)
    return s.replace("Â", "") if "Â" in s and not re.search(r"Â[A-Za-z]", s) else s


_BLOCK_TAGS = ["p", "div", "br", "li", "ul", "ol", "tr", "h1", "h2", "h3", "h4", "h5", "h6", "section", "article", "table", "hr", "dt", "dd"]


def strip_html(raw: str) -> str:
    """Remove tags/entities but keep paragraph and list structure as newlines/bullets (needed for passage splitting)."""
    if not raw:
        return ""
    if "<" not in raw:
        return _html.unescape(raw) if "&" in raw else raw
    from bs4 import BeautifulSoup, MarkupResemblesLocatorWarning
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", MarkupResemblesLocatorWarning)
        soup = BeautifulSoup(raw, "html.parser")
    for t in soup(["script", "style", "noscript", "template"]):
        t.decompose()
    for br in soup.find_all("br"):
        br.replace_with("\n")
    for li in soup.find_all("li"):
        li.insert_before("\n• ")
    for t in soup.find_all([x for x in _BLOCK_TAGS if x not in ("br", "li")]):
        t.insert_before("\n")
        t.insert_after("\n")
    return soup.get_text()


# Text scraped from HTML often loses the separator between a heading/sentence and the next one:
# 'Research InstituteOur passion ...'.  We only split before well-known sentence/heading starters so that
# CamelCase technology names (JavaScript, LinkedIn, TensorFlow) are never broken.
_STARTERS = ("The|We|Our|You|Your|This|These|Join|Work|Responsibilities|Requirements|Qualifications|Benefits|About|Must|"
             "Strong|Experience|Ability|Knowledge|Proven|Minimum|Preferred|Required|Duties|Skills|Education|Salary|"
             "Location|Position|Role|Company|Apply|Please|Candidates|Successful|Ideal|Looking|Excellent|Responsible|"
             "Description|Overview|Summary|Who|What|Why|How|Key|Job")
_GLUE = re.compile(rf"(?<=[a-z])(?=(?:{_STARTERS})(?![a-z]))")
_GLUE_PERIOD = re.compile(r"(?<=[a-z]{3}[.!?])(?!Js\b|Net\b|Io\b)(?=[A-Z][a-z])")
_GLUE_COLON = re.compile(r"(?<=[A-Za-z]):(?=[A-Z])")
_BULLET_START = re.compile(r"^\s*(?:[•◦▪●■□▫·‣⁃➢➤★✔✓►▶◆○\*]|-(?=\s)|–(?=\s)|—(?=\s))\s*")


def normalize_line(line: str) -> str:
    line = re.sub(r"[ \t\r\f\v]+", " ", line).strip()
    if _BULLET_START.match(line):
        line = "• " + _BULLET_START.sub("", line)
    return line


def clean_description(raw: str | None) -> str:
    """Display-quality text: no HTML, no mojibake, normalised unicode, bullets as '• ', blank-line collapsed."""
    if not raw:
        return ""
    s = fix_mojibake(str(raw))
    s = strip_html(s)
    s = unicodedata.normalize("NFKC", s)
    s = s.replace("\u200b", "").replace("\ufeff", "").replace("\xa0", " ")
    s = _GLUE_COLON.sub(": ", _GLUE_PERIOD.sub(" ", _GLUE.sub(" ", s)))
    lines = [normalize_line(l) for l in s.splitlines()]
    out: List[str] = []
    for l in lines:
        if l == "" and (not out or out[-1] == ""):
            continue
        out.append(l)
    return "\n".join(out).strip()


# --------------------------------------------------------------------------- tokenizer
class Tok(NamedTuple):
    raw: str
    norm: str
    start: int
    end: int
    kind: str  # word | abbr | num | sym


_TOKEN_RE = re.compile(
    r"""
    (?P<abbr>(?<![A-Za-z0-9])[A-Za-z]{1,2}\.[A-Za-z]{1,4}(?![A-Za-z])\.?)          # B.Tech  M.Sc  Ph.D  B.E.  e.g.
  | (?P<word>(?:(?<![A-Za-z0-9])\.(?=[A-Za-z]))?[A-Za-z][A-Za-z0-9]*
        (?:[.'’](?=[a-z0-9])[A-Za-z0-9]+)*(?:\+\+|\+(?![0-9])|\#)?)                  # python  node.js  .NET  C++  C#  o'brien
  | (?P<num>\d+(?:,\d{2,3})*(?:\.\d+)?)                                              # 3  1,800  5,00,000  2.5
  | (?P<sym>[^\sA-Za-z0-9])                                                          # any single symbol
    """, re.X)


def tokenize(text: str) -> List[Tok]:
    """Tech-aware tokeniser used by the finite-state components.  norm = lowercase, possessive stripped."""
    toks: List[Tok] = []
    for m in _TOKEN_RE.finditer(text):
        kind = m.lastgroup
        raw = m.group(0)
        norm = raw.lower().replace("’", "'")
        if kind in ("word", "abbr"):
            if norm.endswith("'s"):
                norm = norm[:-2]
            norm = norm.rstrip(".") if kind == "abbr" else norm
        toks.append(Tok(raw, norm, m.start(), m.end(), kind))
    return toks


_TECH_MAP = {"c++": "cpp", "c#": "csharp", ".net": "dotnet", "f#": "fsharp", "node.js": "nodejs", "ci/cd": "cicd"}


def token_stem(t: Tok, *, do_stem: bool = True, remove_stopwords: bool = True) -> Optional[str]:
    """Index form of ONE token (None = not indexed).  Shared by indexing, query processing and highlighting so all three agree."""
    if t.kind not in ("word", "abbr"):
        return None
    w = _TECH_MAP.get(t.norm, t.norm)
    w = w.replace(".", "").replace("'", "").rstrip("+#") if w not in _TECH_MAP.values() else w
    if not w or (len(w) < 2 and w not in ("r", "c")):
        return None
    if remove_stopwords and w in STOPWORDS:
        return None
    return stem(w) if do_stem else w


def index_tokens(text: str, *, do_stem: bool = True, remove_stopwords: bool = True) -> List[str]:
    """Tokens for indexing/retrieval: lowercase, tech-aware, stop-words removed, numbers dropped, Porter-stemmed."""
    out: List[str] = []
    for t in tokenize(text):
        w = token_stem(t, do_stem=do_stem, remove_stopwords=remove_stopwords)
        if w is not None:
            out.append(w)
    return out


_SENT_SPLIT = re.compile(r"(?<=[.!?])\s+(?=[A-Z•])")


def split_sentences(text: str) -> List[str]:
    parts: List[str] = []
    for line in text.split("\n"):
        line = line.strip()
        if line:
            parts.extend(p.strip() for p in _SENT_SPLIT.split(line) if p.strip())
    return parts
