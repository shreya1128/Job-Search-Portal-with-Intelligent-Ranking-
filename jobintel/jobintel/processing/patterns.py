"""Concrete finite automata for the job domain + the small 'semantic actions' that turn a match into values.

Automata (state diagrams can be exported with TokenNFA.to_dot() - see scripts/export_fsa_diagrams.py):
  experience     "3-5 years of experience", "at least 2+ yrs", "five years' experience"
  degree         "Bachelor's degree in Computer Science", "B.Tech", "MCA", "high school diploma"
  salary         "$1,800 to $2,200 monthly", "Rs. 5,00,000 per annum", "8-12 LPA", "50k-70k"
  certification  "AWS certification", "certified scrum master", "valid driver's license"
  role           "Senior Machine Learning Engineer", "Data Analyst"
plus phrase dictionaries (Aho-Corasick) for employment type, seniority level, skills and locations.
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from jobintel.processing.fsa import AhoCorasick, FsaMatch, TokenNFA, light
from jobintel.processing.stopwords import STOPWORDS
from jobintel.processing.text import Tok, tokenize

NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
             "ten": 10, "eleven": 11, "twelve": 12, "fifteen": 15, "twenty": 20}
YEAR_WORDS = {"year", "years", "yr", "yrs", "yoe"}
EXP_WORDS = {"experience", "exp", "expertise"}


# --------------------------------------------------------------------------- predicates
def in_set(s: Iterable[str]):
    s = frozenset(s)
    return lambda t: t.norm in s


def sym(*chars: str):
    s = frozenset(chars)
    return lambda t: t.kind == "sym" and t.norm in s


is_num = lambda t: t.kind == "num"
is_numlike = lambda t: t.kind == "num" or t.norm in NUM_WORDS
is_alpha_word = lambda t: t.kind in ("word", "abbr") and t.norm.replace(".", "").replace("'", "").isalpha()


def to_number(norm: str) -> Optional[float]:
    if norm in NUM_WORDS:
        return float(NUM_WORDS[norm])
    try:
        return float(norm.replace(",", ""))
    except ValueError:
        return None


# =========================================================================== EXPERIENCE
def build_experience_nfa() -> TokenNFA:
    n = TokenNFA("experience")
    q0 = n.start
    qQ, qAt, qUp, qN1, qPlus, qDash, qN2, qY, qOf, qDot, qApos, qExp = (n.new_state() for _ in range(12))
    QUAL = {"minimum", "min", "maximum", "max", "over", "atleast", "upto", "around", "about", "approximately", "approx"}
    n.edge(q0, "QUALIFIER", in_set(QUAL), qQ)
    n.edge(q0, "'at'", in_set({"at"}), qAt)
    n.edge(qAt, "'least'", in_set({"least"}), qQ)
    n.edge(q0, "'up'", in_set({"up"}), qUp)
    n.edge(qUp, "'to'", in_set({"to"}), qQ)
    n.edge(q0, "NUM", is_numlike, qN1)
    n.edge(qQ, "NUM", is_numlike, qN1)
    n.edge(qN1, "'+'", sym("+"), qPlus)
    n.edge(qN1, "'-'|'to'", lambda t: t.norm in {"-", "–", "—", "to"}, qDash)
    n.edge(qN1, "YEAR", in_set(YEAR_WORDS), qY)
    n.edge(qPlus, "YEAR", in_set(YEAR_WORDS), qY)
    n.edge(qDash, "NUM", is_numlike, qN2)
    n.edge(qN2, "'+'", sym("+"), qPlus)
    n.edge(qN2, "YEAR", in_set(YEAR_WORDS), qY)
    n.edge(qY, "'of'", in_set({"of"}), qOf)
    n.edge(qY, "EXP", in_set(EXP_WORDS), qExp)
    n.edge(qOf, "EXP", in_set(EXP_WORDS), qExp)
    n.edge(qY, "'.'", sym("."), qDot)
    n.edge(qDot, "EXP", in_set(EXP_WORDS), qExp)
    n.edge(qY, "apostrophe", sym("'", "’"), qApos)
    n.edge(qApos, "EXP", in_set(EXP_WORDS), qExp)
    n.accept(qY, "exp_bare", 1)
    n.accept(qExp, "exp_explicit", 2)
    return n


EXP_CONTEXT = {"experience", "exp", "expertise", "background", "minimum", "required", "requirement", "working",
               "work", "relevant", "proven", "hands", "industry", "professional"}


MAX_PLAUSIBLE_YEARS = 50      # '20121-3 years' (a year glued to a range by HTML stripping) must not become 20121


def plausible_experience(v: Dict) -> bool:
    return 0 <= v["min"] <= MAX_PLAUSIBLE_YEARS and (v["max"] is None or v["min"] <= v["max"] <= MAX_PLAUSIBLE_YEARS + 10)


def parse_experience(span: Sequence[Tok]) -> Dict:
    nums = [to_number(t.norm) for t in span if is_numlike(t)]
    nums = [x for x in nums if x is not None]
    first = span[0].norm
    plus = any(t.kind == "sym" and t.norm == "+" for t in span)
    if len(nums) >= 2:
        lo, hi = min(nums[0], nums[1]), max(nums[0], nums[1])      # '5-3 years' is a typo for 3-5
    elif first in {"maximum", "max", "upto", "up"}:
        lo, hi = 0.0, nums[0]
    elif plus or first in {"minimum", "min", "at", "over", "atleast"}:
        lo, hi = nums[0], None
    else:
        lo = hi = nums[0]
    return {"min": lo, "max": hi}


# =========================================================================== DEGREE
_BACHELOR = {"bachelor", "bachelors", "undergraduate", "btech", "b.tech", "b.e", "bsc", "b.sc", "bca", "bcom", "b.com",
             "bba", "bs", "b.s", "ba", "b.a", "be/btech"}
_MASTER = {"master", "masters", "postgraduate", "mtech", "m.tech", "m.e", "msc", "m.sc", "mca", "mba", "pgdm", "m.s",
           "m.a", "mcom", "m.com"}
_PHD = {"phd", "ph.d", "doctorate", "doctoral", "dphil"}
DEGREE_LEVEL: Dict[str, str] = {}
for _w in _BACHELOR: DEGREE_LEVEL[_w] = "bachelor"
for _w in _MASTER: DEGREE_LEVEL[_w] = "master"
for _w in _PHD: DEGREE_LEVEL[_w] = "phd"
DEGREE_LEVEL.update({"diploma": "diploma", "ged": "high_school"})
LEVEL_RANK = {"high_school": 1, "diploma": 2, "associate": 3, "bachelor": 4, "master": 5, "phd": 6}
_FIELD_BOUNDARY = {"and", "or", "with", "from", "required", "preferred", "plus", "a", "an", "the", "is", "are", "to", "for",
                   "as", "at", "by", "on", "of", "in", "but", "if", "who", "that", "which", "must", "should", "will",
                   "can", "may", "equivalent", "related", "similar", "relevant", "experience", "years", "year", "degree",
                   "degrees", "preferably", "ideally", "strongly"}


def build_degree_nfa() -> TokenNFA:
    n = TokenNFA("degree")
    q0 = n.start
    qD, qDeg, qAs, qH, qHS, qHS2, qIn, f1, f2, f3 = (n.new_state() for _ in range(10))
    n.edge(q0, "DEGREE-LEVEL", lambda t: t.norm in DEGREE_LEVEL, qD)
    n.edge(q0, "'associate'", in_set({"associate"}), qAs)
    n.edge(qAs, "'degree'", in_set({"degree", "degrees"}), qDeg)
    n.edge(q0, "'high'", in_set({"high"}), qH)
    n.edge(qH, "'school'", in_set({"school"}), qHS)
    n.edge(qHS, "'diploma'|'degree'", in_set({"diploma", "degree", "education", "graduate"}), qHS2)
    n.edge(qD, "'degree'", in_set({"degree", "degrees"}), qDeg)
    n.edge(qD, "'in'|'of'", in_set({"in", "of"}), qIn)
    n.edge(qDeg, "'in'|'of'", in_set({"in", "of"}), qIn)
    field = lambda t: is_alpha_word(t) and t.norm not in _FIELD_BOUNDARY and len(t.norm) > 1
    n.edge(qIn, "FIELD", field, f1)
    n.edge(f1, "FIELD", field, f2)
    n.edge(f1, "'in'", in_set({"in"}), qIn)
    n.edge(f2, "FIELD", field, f3)
    for s, p in ((qD, 1), (qDeg, 2), (qHS, 1), (qHS2, 2), (f1, 3), (f2, 3), (f3, 3)):
        n.accept(s, "degree", p)
    return n


def parse_degree(span: Sequence[Tok]) -> Dict:
    norms = [t.norm for t in span]
    if norms[0] == "associate":
        level = "associate"
    elif norms[0] == "high":
        level = "high_school"
    else:
        level = DEGREE_LEVEL[norms[0]]
    fld = None
    for k in range(len(norms) - 2, 0, -1):          # field = tokens after the LAST 'in'/'of'
        if norms[k] in ("in", "of"):
            fld = " ".join(norms[k + 1:])
            break
    return {"level": level, "field": fld}


# =========================================================================== SALARY
CUR_SYM = {"$": "USD", "₹": "INR", "£": "GBP", "€": "EUR"}
CUR_WORD = {"rs": "INR", "inr": "INR", "usd": "USD", "sgd": "SGD", "eur": "EUR", "gbp": "GBP", "aud": "AUD", "cad": "CAD"}
MULT = {"k": 1e3, "lpa": 1e5, "lakh": 1e5, "lakhs": 1e5, "lac": 1e5, "lacs": 1e5, "crore": 1e7, "crores": 1e7, "million": 1e6}
PERIOD = {"monthly": "month", "month": "month", "mo": "month", "yearly": "year", "year": "year", "annum": "year",
          "annual": "year", "annually": "year", "pa": "year", "hourly": "hour", "hour": "hour", "hr": "hour",
          "weekly": "week", "week": "week", "daily": "day", "day": "day"}


def build_salary_nfa() -> TokenNFA:
    n = TokenNFA("salary")
    q0 = n.start
    sC, sC2, sN1, sK, sR, sRC, sN2, sK2, sPer, sP = (n.new_state() for _ in range(10))
    bN1, bK, bR, bN2, bK2, bPer, bP = (n.new_state() for _ in range(7))
    cur = lambda t: (t.kind == "sym" and t.norm in CUR_SYM) or (t.kind == "word" and t.norm in CUR_WORD)
    dash = lambda t: t.norm in {"-", "–", "—", "to"}
    # --- currency-prefixed track:  CUR [.] NUM [MULT] [ ('-'|'to') [CUR] NUM [MULT] ] [PERIOD | per PERIOD]
    n.edge(q0, "CUR", cur, sC)
    n.edge(sC, "'.'", sym("."), sC2)
    n.edge(sC, "NUM", is_num, sN1)
    n.edge(sC2, "NUM", is_num, sN1)
    n.edge(sN1, "MULT", in_set(MULT), sK)
    n.edge(sN1, "'-'|'to'", dash, sR)
    n.edge(sK, "'-'|'to'", dash, sR)
    n.edge(sR, "CUR", cur, sRC)
    n.edge(sRC, "NUM", is_num, sN2)
    n.edge(sR, "NUM", is_num, sN2)
    n.edge(sN2, "MULT", in_set(MULT), sK2)
    for s in (sN1, sK, sN2, sK2):
        n.edge(s, "PERIOD", in_set(PERIOD), sP)
        n.edge(s, "'per'|'/'|'a'", lambda t: t.norm in {"per", "/", "a"}, sPer)
    n.edge(sPer, "PERIOD", in_set(PERIOD), sP)
    for s, p in ((sN1, 1), (sK, 2), (sN2, 2), (sK2, 3), (sP, 4)):
        n.accept(s, "salary", p)
    # --- bare-number track (needs a unit so '3 years' is never a salary):  NUM (LPA|k|lakh) | NUM per annum
    n.edge(q0, "NUM", is_num, bN1)
    n.edge(bN1, "MULT", in_set(MULT), bK)
    n.edge(bN1, "'-'|'to'", dash, bR)
    n.edge(bR, "NUM", is_num, bN2)
    n.edge(bN2, "MULT", in_set(MULT), bK2)
    n.edge(bK, "'-'|'to'", dash, bR)                      # "50k-70k": unit on the first number too
    for s_ in (bN1, bK, bN2, bK2):
        n.edge(s_, "'per'", in_set({"per"}), bPer)
    n.edge(bPer, "PERIOD", in_set({"annum", "year", "month"}), bP)
    for s, p in ((bK, 1), (bK2, 2), (bP, 3)):
        n.accept(s, "salary", p)
    return n


def parse_salary(span: Sequence[Tok]) -> Dict:
    currency = None
    if span[0].kind == "sym" and span[0].norm in CUR_SYM:
        currency = CUR_SYM[span[0].norm]
    elif span[0].norm in CUR_WORD:
        currency = CUR_WORD[span[0].norm]
    vals: List[float] = []
    mults: List[Optional[str]] = []
    period = None
    for k, t in enumerate(span):
        if t.kind == "num":
            vals.append(to_number(t.norm))
            mults.append(span[k + 1].norm if k + 1 < len(span) and span[k + 1].norm in MULT else None)
        if t.norm in PERIOD:
            period = PERIOD[t.norm]
    if not vals:
        return {}
    if mults[-1] is not None:                      # "8-12 LPA": the unit written once covers the whole range
        mults = [m or mults[-1] for m in mults]
    for k, m in enumerate(mults):
        if m:
            vals[k] *= MULT[m]
            if m in ("lpa", "lakh", "lakhs", "lac", "lacs", "crore", "crores"):
                currency = currency or "INR"
                if m == "lpa":
                    period = period or "year"
    return {"min": min(vals), "max": max(vals), "currency": currency, "period": period}


# =========================================================================== CERTIFICATION
_CERT_WORDS = {"certification", "certifications", "certificate", "certificates", "certified", "license", "licence", "licensure"}
_CERT_PREFIX_STOP = {"valid", "current", "relevant", "required", "preferred", "any", "such", "of", "in", "and", "or", "with",
                     "a", "an", "the", "professional", "new", "other", "similar", "applicable", "appropriate", "state"} | set(STOPWORDS)


def build_certification_nfa() -> TokenNFA:
    n = TokenNFA("certification")
    q0 = n.start
    p1, p2, p3, pc, pc1, pc2, pc3, c1, c2, c3, c4 = (n.new_state() for _ in range(11))
    prefix = lambda t: is_alpha_word(t) and t.norm not in _CERT_PREFIX_STOP and t.norm not in _CERT_WORDS
    word = lambda t: is_alpha_word(t) and t.norm not in _FIELD_BOUNDARY
    cw = in_set(_CERT_WORDS)
    n.edge(q0, "PREFIX", prefix, p1)
    n.edge(p1, "PREFIX", prefix, p2)
    n.edge(p2, "PREFIX", prefix, p3)
    for s in (p1, p2, p3):
        n.edge(s, "CERT-WORD", cw, pc)
    n.edge(pc, "WORD", word, pc1)
    n.edge(pc1, "WORD", word, pc2)
    n.edge(pc2, "WORD", word, pc3)
    n.edge(q0, "'certified'", in_set({"certified"}), c1)
    n.edge(c1, "WORD", word, c2)
    n.edge(c2, "WORD", word, c3)
    n.edge(c3, "WORD", word, c4)
    for s in (pc, pc1, pc2, pc3, c2, c3, c4):
        n.accept(s, "certification", 1)
    return n


# =========================================================================== JOB ROLE
_LEVEL_W = {"senior", "sr", "junior", "jr", "lead", "principal", "staff", "associate", "head", "chief", "entry", "intern",
            "trainee", "assistant", "executive", "graduate"}
_DOMAIN_W = {"data", "machine", "learning", "deep", "software", "backend", "frontend", "front", "back", "end", "full", "stack",
             "devops", "cloud", "ml", "ai", "nlp", "business", "intelligence", "bi", "big", "web", "mobile", "android", "ios",
             "java", "python", "net", "sql", "database", "network", "security", "cyber", "qa", "test", "automation", "systems",
             "system", "it", "information", "technology", "product", "project", "program", "marketing", "sales", "financial",
             "finance", "accounting", "hr", "human", "resources", "customer", "service", "clinical", "registered", "medical",
             "computer", "vision", "analytics", "research", "applied", "embedded", "site", "reliability", "platform", "solutions",
             "technical", "support", "operations", "quality", "assurance", "ui", "ux", "digital", "healthcare", "tax", "audit",
             "credit", "risk", "investment", "banking", "staff", "cloud", "infrastructure", "etl", "mlops", "analytical"}
_ROLE_NOUN = {"engineer", "scientist", "analyst", "developer", "programmer", "architect", "administrator", "manager",
              "consultant", "specialist", "designer", "researcher", "tester", "director", "officer", "technician", "associate",
              "representative", "coordinator", "assistant", "nurse", "accountant", "auditor", "physician", "therapist",
              "pharmacist", "recruiter", "strategist", "lead", "executive", "advisor", "adviser", "controller", "underwriter",
              "banker", "planner", "supervisor", "technologist", "statistician", "economist", "intern"}


def build_role_nfa() -> TokenNFA:
    n = TokenNFA("job_role")
    q0 = n.start
    q1, d1, d2, d3, d4, qR, qLv = (n.new_state() for _ in range(7))
    lv, dom, noun = in_set(_LEVEL_W), in_set(_DOMAIN_W), in_set(_ROLE_NOUN)
    n.edge(q0, "LEVEL", lv, q1)
    n.edge(q0, "DOMAIN", dom, d1)
    n.edge(q1, "DOMAIN", dom, d1)
    n.edge(d1, "DOMAIN", dom, d2)
    n.edge(d2, "DOMAIN", dom, d3)
    n.edge(d3, "DOMAIN", dom, d4)
    for s in (q0, q1, d1, d2, d3, d4):
        n.edge(s, "ROLE-NOUN", noun, qR)
    n.edge(qR, "level-number", lambda t: t.norm in {"i", "ii", "iii", "iv", "1", "2", "3"}, qLv)
    n.accept(qR, "job_role", 1)
    n.accept(qLv, "job_role", 2)
    return n


# =========================================================================== PHRASE DICTIONARIES (Aho-Corasick)
def key_tokens(phrase: str) -> Tuple[str, ...]:
    return tuple(light(t.norm) for t in tokenize(phrase) if t.kind in ("word", "abbr", "num"))


def build_phrase_ac(mapping: Dict[str, object]) -> AhoCorasick:
    ac = AhoCorasick()
    for phrase, payload in mapping.items():
        ac.add(key_tokens(phrase), payload)
    return ac.build()


EMPLOYMENT_PHRASES = {
    "full time": "full-time", "fulltime": "full-time", "part time": "part-time", "parttime": "part-time",
    "contract": "contract", "contractor": "contract", "contractual": "contract", "permanent": "permanent",
    "temporary": "temporary", "internship": "internship", "intern": "internship", "freelance": "freelance",
    "casual": "casual", "seasonal": "seasonal", "per diem": "per-diem",
}
# seniority wording -> typical experience band (years).  Used to interpret QUERIES such as "entry level ML jobs".
LEVEL_PHRASES = {
    "entry level": ("entry", 0, 2), "fresher": ("entry", 0, 2), "freshers": ("entry", 0, 2), "junior": ("entry", 0, 2),
    "trainee": ("entry", 0, 1), "beginner": ("entry", 0, 2), "mid level": ("mid", 3, 5), "intermediate": ("mid", 3, 5),
    "senior": ("senior", 5, None), "sr": ("senior", 5, None), "experienced": ("senior", 3, None),
}
