"""Information extraction: runs every automaton over a text and returns structured fields.

    text --tokenize--> experience NFA | degree NFA | salary NFA | certification NFA | role NFA
                       skill AC | employment AC | seniority AC | location AC

Each finite automaton only RECOGNISES a surface pattern; a tiny semantic action then converts the matched
tokens into values (e.g. "3-5 years" -> min 3, max 5).  Ambiguous matches (a bare "2 years") are accepted only
when an experience-related word is nearby.
"""
from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Optional

from jobintel.extraction.lexicon import SkillLexicon, load_lexicon
from jobintel.processing import patterns as P
from jobintel.processing.fsa import AhoCorasick, select_longest
from jobintel.processing.text import Tok, tokenize

# a few safe default cities so the location automaton works before the data-driven gazetteer is built
DEFAULT_CITIES = ["Pune", "Mumbai", "Bengaluru", "Hyderabad", "Chennai", "Delhi NCR", "Gurugram", "Noida", "Kolkata", "Ahmedabad",
                  "Kochi", "Jaipur", "Indore", "Chandigarh", "Thiruvananthapuram", "Coimbatore", "Nagpur", "Singapore", "London",
                  "New York", "San Francisco", "Los Angeles", "Chicago", "Houston", "Seattle", "Boston", "Austin", "Dallas",
                  "Atlanta", "Toronto", "Sydney", "Remote"]
CITY_ALIASES = {"bangalore": "Bengaluru", "gurgaon": "Gurugram", "bombay": "Mumbai", "madras": "Chennai", "calcutta": "Kolkata",
                "new delhi": "Delhi NCR", "delhi": "Delhi NCR", "ncr": "Delhi NCR", "cochin": "Kochi", "trivandrum": "Thiruvananthapuram",
                "mysore": "Mysuru", "work from home": "Remote", "wfh": "Remote", "nyc": "New York", "sf": "San Francisco"}


# Skills that are also everyday English words: accepted in free text only when written Capitalised and not at the
# start of a sentence ("Experience with Spark" yes, "we excel at spark ..." no).
COMMON_WORD_SKILLS = {"go", "swift", "rust", "spark", "pig", "hive", "flask", "lambda", "node", "spring", "excel", "oracle",
                      "julia", "ruby", "react", "scala", "vue", "angular", "kotlin", "airflow", "swift", "triage",
                      "unity", "zoom", "canvas", "chef", "scratch", "puppet", "slack", "eclipse"}


class Extractor:
    def __init__(self, lexicon: Optional[SkillLexicon] = None, cities: Optional[Iterable[str]] = None):
        self.lex = lexicon or load_lexicon()
        self.exp_nfa = P.build_experience_nfa()
        self.deg_nfa = P.build_degree_nfa()
        self.sal_nfa = P.build_salary_nfa()
        self.cert_nfa = P.build_certification_nfa()
        self.role_nfa = P.build_role_nfa()
        self.skill_ac = P.build_phrase_ac({ph: canon for ph, canon in self.lex.phrases()})
        self.emp_ac = P.build_phrase_ac(P.EMPLOYMENT_PHRASES)
        self.level_ac = P.build_phrase_ac(P.LEVEL_PHRASES)
        if cities is None:
            from jobintel.config import GAZETTEER_PATH
            cities = json.loads(Path(GAZETTEER_PATH).read_text(encoding="utf-8")) if Path(GAZETTEER_PATH).exists() else DEFAULT_CITIES
        city_map = {c.lower(): c for c in cities}
        city_map.update({a: c for a, c in CITY_ALIASES.items()})
        self.loc_ac = P.build_phrase_ac(city_map)

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _span_text(text: str, toks: List[Tok], s: int, e: int) -> str:
        return text[toks[s].start: toks[e - 1].end]

    @staticmethod
    def _proper_use(text: str, toks: List[Tok], s: int, raw: str) -> bool:
        if not raw[:1].isupper():
            return False
        if s == 0:
            return False
        gap = text[toks[s - 1].end: toks[s].start]
        return not ("\n" in gap or toks[s - 1].norm in {".", "!", "?", ":", "•"})

    def _exp_context_ok(self, toks: List[Tok], s: int, e: int) -> bool:
        lo, hi = max(0, s - 6), min(len(toks), e + 6)
        return any(t.norm in P.EXP_CONTEXT for t in toks[lo:s] + toks[e:hi])

    # ------------------------------------------------------------------ main API
    def extract(self, text: str, *, skills_only: bool = False) -> Dict:
        toks = tokenize(text or "")
        res: Dict = {"experience": None, "education": [], "salary": None, "certifications": [], "employment_type": None,
                     "level": None, "locations": [], "roles": [], "skills": [], "spans": []}
        # ---- skills (Aho-Corasick, maximal munch) --------------------------
        sk = select_longest(self.skill_ac.scan(toks))
        cnt: Counter = Counter()
        first_span: Dict[str, str] = {}
        for s, e, canon in sk:
            raw = self._span_text(text, toks, s, e)
            if canon in COMMON_WORD_SKILLS and not self._proper_use(text, toks, s, raw):
                continue
            cnt[canon] += 1
            first_span.setdefault(canon, raw)
        res["skills"] = [{"name": c, "type": (self.lex.get(c).type if self.lex.get(c) else "technical"), "mentions": n,
                          "text": first_span[c]} for c, n in cnt.most_common()]
        if skills_only:
            return res
        # ---- experience ----------------------------------------------------
        exps = []
        for m in self.exp_nfa.scan(toks):
            if m.tag == "exp_bare" and not self._exp_context_ok(toks, m.start, m.end):
                continue
            v = P.parse_experience(toks[m.start:m.end])
            if not P.plausible_experience(v):
                continue
            v["text"] = self._span_text(text, toks, m.start, m.end)
            v["explicit"] = m.tag == "exp_explicit"
            exps.append(v)
            res["spans"].append({"label": "experience", "start": toks[m.start].start, "end": toks[m.end - 1].end})
        if exps:
            explicit = [v for v in exps if v["explicit"]]
            res["experience"] = (explicit or exps)[0]          # first explicit mention wins, else first bare one
            res["experience_all"] = exps
        # ---- education -----------------------------------------------------
        for m in self.deg_nfa.scan(toks):
            v = P.parse_degree(toks[m.start:m.end])
            v["text"] = self._span_text(text, toks, m.start, m.end)
            res["education"].append(v)
        # ---- salary --------------------------------------------------------
        for m in self.sal_nfa.scan(toks):
            v = P.parse_salary(toks[m.start:m.end])
            if v:
                v["text"] = self._span_text(text, toks, m.start, m.end)
                res["salary"] = v
                res["spans"].append({"label": "salary", "start": toks[m.start].start, "end": toks[m.end - 1].end})
                break
        # ---- certifications / roles ---------------------------------------
        res["certifications"] = [self._span_text(text, toks, m.start, m.end) for m in self.cert_nfa.scan(toks)]
        res["roles"] = [self._span_text(text, toks, m.start, m.end) for m in self.role_nfa.scan(toks)]
        # ---- phrase dictionaries ------------------------------------------
        emp = select_longest(self.emp_ac.scan(toks))
        if emp:
            res["employment_type"] = Counter(p for _, _, p in emp).most_common(1)[0][0]
            res["spans"] += [{"label": "employment", "start": toks[s].start, "end": toks[e - 1].end} for s, e, _ in emp]
        lv = select_longest(self.level_ac.scan(toks))
        if lv:
            name, lo, hi = lv[0][2]
            res["level"] = {"name": name, "min": lo, "max": hi}
            res["spans"] += [{"label": "level", "start": toks[s].start, "end": toks[e - 1].end} for s, e, _ in lv]
        seen = []
        for s, e, city in select_longest(self.loc_ac.scan(toks)):
            res["spans"].append({"label": "location", "start": toks[s].start, "end": toks[e - 1].end})
            if city not in seen:
                seen.append(city)
        res["locations"] = seen
        return res

    def best_education(self, edu: List[Dict]) -> Optional[str]:
        """Lowest level that is actually demanded ('Bachelor or Master' -> bachelor), with its field if present."""
        cand = [d for d in edu if d["level"] != "high_school" or len(edu) == 1]
        if not cand:
            return None
        d = min(cand, key=lambda x: P.LEVEL_RANK[x["level"]])
        return d["level"] + (f" ({d['field']})" if d.get("field") else "")
