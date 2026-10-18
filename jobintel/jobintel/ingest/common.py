"""Helpers shared by all dataset loaders: city/country normalisation, company names, experience & salary parsing,
title-keyword clusters, seniority -> experience mapping."""
from __future__ import annotations

import datetime as dt
import re
import unicodedata
from typing import Dict, List, Optional, Tuple

from jobintel.extraction.lexicon import normalize_tag
from jobintel.processing.patterns import build_salary_nfa, parse_salary
from jobintel.processing.text import tokenize

# --------------------------------------------------------------------------- geography
CITY_ALIASES = {
    "bangalore": "Bengaluru", "bengaluru": "Bengaluru", "bengalore": "Bengaluru",
    "hyderabad": "Hyderabad", "secunderabad": "Hyderabad",
    "gurgaon": "Gurugram", "gurugram": "Gurugram",
    "delhi": "Delhi NCR", "delhi ncr": "Delhi NCR", "ncr": "Delhi NCR", "new delhi": "Delhi NCR", "delhi / ncr": "Delhi NCR",
    "mumbai": "Mumbai", "mumbai suburban": "Mumbai", "bombay": "Mumbai",
    "kochi": "Kochi", "cochin": "Kochi", "trivandrum": "Thiruvananthapuram", "thiruvananthapuram": "Thiruvananthapuram",
    "mysore": "Mysuru", "mysuru": "Mysuru", "madras": "Chennai", "calcutta": "Kolkata", "pune": "Pune",
    "remote": "Remote", "work from home": "Remote", "wfh": "Remote",
}
COUNTRY_ALIASES = {
    "india": "India", "usa": "United States", "us": "United States", "united states": "United States", "u.s.": "United States",
    "uk": "United Kingdom", "gb": "United Kingdom", "united kingdom": "United Kingdom", "singapore": "Singapore", "sg": "Singapore",
    "canada": "Canada", "ca": "Canada", "gr": "Greece", "greece": "Greece", "de": "Germany", "germany": "Germany",
    "au": "Australia", "australia": "Australia", "nz": "New Zealand", "in": "India", "ie": "Ireland", "fr": "France",
    "nl": "Netherlands", "be": "Belgium", "es": "Spain", "pl": "Poland", "ph": "Philippines", "ro": "Romania", "dk": "Denmark",
}
_PAREN = re.compile(r"\([^)]*\)")


NON_CITIES = {"any location", "anywhere", "pan india", "all areas", "various locations", "multiple locations", "all india", "other"}


def canon_city(raw: str) -> Optional[str]:
    n = _PAREN.sub(" ", raw or "")
    n = re.sub(r"\s+", " ", n).strip(" .,-").lower()
    if not n or n in NON_CITIES:
        return None
    return CITY_ALIASES.get(n) or n.title()


def parse_naukri_location(raw: str) -> Tuple[List[str], Optional[str]]:
    """'Bangalore/Bengaluru, Hyderabad/Secunderabad' -> (['Bengaluru','Hyderabad'], None)."""
    cities: List[str] = []
    country = None
    for chunk in (raw or "").split(","):
        for piece in chunk.split("/"):
            key = _PAREN.sub(" ", piece).strip().lower()
            if key in COUNTRY_ALIASES and key not in CITY_ALIASES:
                country = COUNTRY_ALIASES[key]
                continue
            c = canon_city(piece)
            if c and c not in cities:
                cities.append(c)
    return cities, country


def parse_us_style_location(raw: str) -> Tuple[Optional[str], Optional[str]]:
    """'San Diego, CA' -> ('San Diego', 'CA')."""
    parts = [p.strip() for p in (raw or "").split(",") if p.strip()]
    if not parts:
        return None, None
    return parts[0].title() if parts[0].isupper() else parts[0], (parts[1] if len(parts) > 1 else None)


def parse_emscad_location(raw: str) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    """'US, CA, San Francisco' -> (country, region, city)."""
    parts = [p.strip() for p in (raw or "").split(",")]
    country = COUNTRY_ALIASES.get(parts[0].lower(), parts[0] or None) if parts and parts[0] else None
    region = parts[1] if len(parts) > 1 and parts[1] else None
    city = parts[2].title() if len(parts) > 2 and parts[2] else None
    return country, region, city


# --------------------------------------------------------------------------- companies
_LEGAL = {"pvt", "ltd", "private", "limited", "inc", "llc", "corp", "corporation", "co", "company", "pte", "llp", "plc", "gmbh", "lp", "incorporated"}


def normalize_company(name: str) -> str:
    name = "".join(ch for ch in unicodedata.normalize("NFKD", name or "") if not unicodedata.combining(ch))
    toks = [t for t in re.sub(r"[^a-z0-9& ]+", " ", name.lower()).split() if t not in _LEGAL]
    return " ".join(toks) or (name or "").strip().lower()


# Names that stand for "no employer given".  Treating them as one company would create a fake employer owning thousands of
# unrelated postings (and a fake PageRank authority).  Compared after normalize_company().
PLACEHOLDER_COMPANIES = {"undisclosed", "confidential", "not disclosed", "not specified", "unknown", "n a", "na", "private advertiser", "anonymous",
                         "unnamed", "hiring", "various", "multiple", "none", "null", "private", "name withheld", "withheld"}


def is_placeholder_company(name) -> bool:
    return not name or normalize_company(name) in PLACEHOLDER_COMPANIES


# --------------------------------------------------------------------------- experience / seniority
def parse_exp_range(text: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """'3-6' -> (3,6); '1 year  exp' -> (1,1); '5+' -> (5,None)."""
    if not text:
        return None, None
    nums = [float(x) for x in re.findall(r"\d+(?:\.\d+)?", str(text))]
    if not nums:
        return None, None
    if len(nums) >= 2:
        lo, hi = nums[0], nums[1]
        return (lo, hi) if hi >= lo else (hi, lo)
    return (nums[0], None) if "+" in str(text) else (nums[0], nums[0])


SENIORITY_EXP = {  # APPROXIMATE bands; stored with experience_source='seniority_map' so it is never mistaken for real data
    "entry level": (0, 2), "internship": (0, 1), "associate": (1, 4), "mid-senior level": (3, 10), "mid senior": (3, 10),
    "director": (8, None), "executive": (10, None), "junior executive": (0, 2), "senior executive": (5, None),
    "executive": (3, None), "manager": (5, None), "senior management": (8, None), "non-executive": (0, 3),
    "fresh/entry level": (0, 2), "professional": (3, None), "middle management": (6, None), "junior": (0, 2), "senior": (5, None),
}


def seniority_to_exp(s: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    if not s:
        return None, None
    return SENIORITY_EXP.get(s.strip().lower(), (None, None))


# --------------------------------------------------------------------------- salary (uses the salary FSA)
_SAL_NFA = build_salary_nfa()


def parse_salary_text(text: Optional[str], currency: Optional[str] = None) -> Dict:
    """Runs the salary finite automaton; `currency` overrides the placeholder ('$' is USD/SGD/AUD... depending on source)."""
    if not text or "undisclosed" in str(text).lower():
        return {}
    toks = tokenize(str(text))
    ms = _SAL_NFA.scan(toks)
    if not ms:
        return {}
    v = parse_salary(toks[ms[0].start:ms[0].end])
    if v and currency:
        v["currency"] = currency
    return v


def parse_numeric_range(text: Optional[str]) -> Tuple[Optional[float], Optional[float]]:
    """EMSCAD salary_range '35000-40000'; '0-0' and garbage ('Jun-90') -> (None, None)."""
    m = re.fullmatch(r"\s*(\d+)\s*-\s*(\d+)\s*", str(text or ""))
    if not m:
        return None, None
    lo, hi = float(m.group(1)), float(m.group(2))
    if hi == 0 or lo > hi:
        return None, None
    return lo, hi


# --------------------------------------------------------------------------- title-keyword clusters (HEURISTIC)
_CLUSTER_RX = [   # first match wins; the first four are the clusters named in the project brief, the rest split the former 'other'
    ("data_ai", r"data scien|data analy|data engineer|machine learning|\bml\b|\bai\b|artificial intelligence|analytics|business intelligence|big data|data architect|\bnlp\b|computer vision"),
    ("software_it", r"software|developer|programmer|devops|full[- ]?stack|back[- ]?end|front[- ]?end|cloud|network engineer|systems? admin|cyber|security analyst|\bit\b|qa engineer|web dev|\bjava\b|python|\.net|\bsre\b|database"),
    ("finance", r"accountant|accounting|financial|finance|auditor|banking|banker|\btax\b|bookkeep|controller|investment|actuar|credit|loan|underwrit"),
    ("healthcare", r"nurse|\brn\b|\blpn\b|physician|medical|clinical|therapist|pharmac|dental|health|patient|caregiver|\bcna\b"),
    ("education", r"teacher|lecturer|professor|tutor|trainer|instructor|educator|faculty|\bschool\b|academic|curriculum|early intervention|childcare|nursery|teaching"),
    ("engineering_manufacturing", r"mechanical|electrical|civil|chemical|manufactur|production|maintenance|technician|machinist|welder|fitter|construction|site engineer|process engineer|industrial|hvac|\bplant\b|electronics|\bqa\b|quality (?:control|assurance|engineer)"),
    ("sales_marketing", r"sales|marketing|business development|account (?:executive|manager)|\bbrand|\bseo\b|content (?:writer|creator)|social media|copywriter|advertis|public relations|merchandis|e-?commerce"),
    ("hr_admin", r"human resources|\bhr\b|recruit|talent|payroll|administrat|receptionist|office|secretar|executive assistant|clerk|data entry|customer (?:service|support|care)|call cent|back office|operations"),
    ("retail_hospitality", r"retail|store|cashier|waiter|waitress|chef|cook|kitchen|barista|restaurant|hotel|hospitality|housekeep|cleaner|bartender|server"),
    ("logistics_transport", r"driver|delivery|warehouse|logistics|supply chain|forklift|courier|shipping|freight|transport|dispatcher"),
    ("legal_public", r"lawyer|legal|paralegal|counsel|attorney|public service|civil service|policy|government"),
]
_CLUSTER_COMPILED = [(n, re.compile(p)) for n, p in _CLUSTER_RX]
CLUSTERS = [n for n, _ in _CLUSTER_RX] + ["other"]


def cluster_for_title(title: Optional[str]) -> str:
    t = (title or "").lower()
    for name, rx in _CLUSTER_COMPILED:
        if rx.search(t):
            return name
    return "other"


def to_date(s) -> Optional[dt.date]:
    if s is None:
        return None
    for fmt in ("%d %b %Y", "%Y-%m-%d", "%Y-%m-%d %H:%M:%S", "%Y-%m-%d %H:%M:%S%z"):
        try:
            return dt.datetime.strptime(str(s).strip(), fmt).date()
        except ValueError:
            continue
    try:
        return dt.datetime.fromisoformat(str(s).strip()).date()
    except ValueError:
        return None
