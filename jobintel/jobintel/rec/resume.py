"""Resume analyser: sections, contact-level entities (rule-based NER) and the Phase-1 extractor for skills / education / roles.

WHAT IS IMPLEMENTED (and what is not)
* Sections are found from heading lines (summary / experience / education / skills / projects / certifications ...).
* Entities (rule-based NER, finite patterns - NOT a trained model): EMAIL, PHONE, URL, PERSON (a position+shape heuristic for the name line), ORG (a capitalised
  phrase ending in a company / institution word), DATE_RANGE; skills, degrees, certifications, roles and locations come from `Extractor` (finite automata + dictionaries).
* Experience years: an explicit statement ("5+ years of experience") when present, otherwise the UNION of the date ranges found outside the education / certification
  sections ("Jan 2019 - Mar 2022", "06/2019 - 12/2020", "2018 - 2021", "... - Present").  Overlapping ranges are merged, so two parallel jobs are not double counted.
  A year-only end counts to the start of that year ("2018 - 2021" = 3 years).  Both numbers are returned, and a large disagreement is reported as a warning.
* Skill 'evidence' is descriptive (listed in a skills section / mentioned n times), not a calibrated probability.
* Not implemented here: scanned resumes / OCR, PDF or DOCX parsing (text input only), a trained NER model.  The precision / recall of the rules has NOT been measured on a
  labelled resume set; the tests check hand-written cases only."""
from __future__ import annotations

import datetime as dt
import re
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from ..extraction.extractor import Extractor

MONTHS = {"jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6, "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12}
_MONTH = r"(?:jan(?:uary)?|feb(?:ruary)?|mar(?:ch)?|apr(?:il)?|may|jun(?:e)?|jul(?:y)?|aug(?:ust)?|sep(?:t(?:ember)?)?|oct(?:ober)?|nov(?:ember)?|dec(?:ember)?)"
_YEAR = r"(?:19|20)\d{2}"


def _point(p: str) -> str:
    return rf"(?:(?P<{p}m>{_MONTH})\.?,?\s+(?P<{p}y>{_YEAR})|(?P<{p}nm>0?[1-9]|1[0-2])\s*[/.\-]\s*(?P<{p}ny>{_YEAR})|(?P<{p}yy>{_YEAR}))"


DATE_RANGE = re.compile(rf"(?<![\w/])(?:{_point('s')})\s*(?:-|\u2013|\u2014|to|until|till)\s*(?:{_point('e')}|(?P<now>present|current|now|till\s+date|to\s+date|ongoing))(?![\w])", re.I)
EMAIL = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
PHONE = re.compile(r"(?<![\w])(?:\+?\d{1,3}[\s\-]?)?(?:\(?\d{2,5}\)?[\s\-]?)?\d{3,5}[\s\-]?\d{4,5}(?![\w])")
URL = re.compile(r"(?:https?://|www\.)[^\s,;|)]+|(?:linkedin\.com|github\.com)/[^\s,;|)]+", re.I)
ORG_SUFFIX = (r"(?:Pvt\.?\s+Ltd\.?|Private\s+Limited|Limited|Ltd\.?|Inc\.?|LLC|LLP|Corp\.?|Corporation|Technologies|Technology|Systems|Solutions|Services|"
              r"Consulting|Labs|Group|Bank|University|Institute|College|Company|Analytics|Software|Infotech)")
ORG = re.compile(rf"\b((?:[A-Z][\w&.'\-]*[ \t]+){{0,4}}[A-Z][\w&.'\-]*[ \t]+{ORG_SUFFIX})(?![\w])")           # [ \t]: a name never spans a line break
EDU_ORG = re.compile(r"\b((?:[A-Z][\w&.'\-]*[ \t]+){0,3}(?:Institute|University|College|School|Academy)[ \t]+of[ \t]+(?:[A-Z][\w&.'\-]*[ \t]*){1,3})")
NOT_ORG = {"information technology", "computer technology", "data analytics", "computer science", "information systems", "management services"}
HEADINGS = {
    "summary": "summary", "professional summary": "summary", "objective": "summary", "career objective": "summary", "profile": "summary", "about me": "summary", "about": "summary",
    "experience": "experience", "work experience": "experience", "professional experience": "experience", "employment history": "experience", "work history": "experience",
    "education": "education", "academic background": "education", "academics": "education", "qualifications": "education", "academic qualifications": "education",
    "skills": "skills", "technical skills": "skills", "key skills": "skills", "core competencies": "skills", "skills summary": "skills", "it skills": "skills",
    "projects": "projects", "academic projects": "projects", "personal projects": "projects",
    "certifications": "certifications", "certificates": "certifications", "licenses": "certifications", "courses": "certifications",
    "achievements": "other", "awards": "other", "interests": "other", "hobbies": "other", "languages": "other", "references": "other",
}
NON_NAME_WORDS = {"engineer", "developer", "manager", "analyst", "resume", "curriculum", "vitae", "cv", "profile", "summary", "objective", "scientist", "intern", "student",
                  "consultant", "designer", "architect", "specialist", "contact", "address", "email", "phone", "mobile"}
EDU_LINE = re.compile(r"\b(bachelor|b\.?\s?tech|b\.?\s?e\b|b\.?\s?sc|master|m\.?\s?tech|m\.?\s?sc|mca|bca|mba|ph\.?d|diploma|university|college|school|institute)\b", re.I)


@dataclass
class Entity:
    label: str
    text: str
    start: int
    end: int
    source: str = "rule"


@dataclass
class ResumeProfile:
    name: Optional[str] = None
    emails: List[str] = field(default_factory=list)
    phones: List[str] = field(default_factory=list)
    urls: List[str] = field(default_factory=list)
    sections: Dict[str, str] = field(default_factory=dict)
    skills: List[Dict] = field(default_factory=list)
    experience_years: Optional[float] = None
    experience_source: Optional[str] = None            # 'stated' | 'date_ranges' | None
    stated_experience_years: Optional[float] = None
    computed_experience_years: Optional[float] = None
    date_ranges: List[Dict] = field(default_factory=list)
    education: List[Dict] = field(default_factory=list)
    best_education: Optional[str] = None
    certifications: List[str] = field(default_factory=list)
    roles: List[str] = field(default_factory=list)
    locations: List[str] = field(default_factory=list)
    organisations: List[str] = field(default_factory=list)
    entities: List[Entity] = field(default_factory=list)
    n_words: int = 0
    warnings: List[str] = field(default_factory=list)
    raw_text: str = ""                                 # the analysed text (used by the DL text space; left out of to_dict)

    def skill_weights(self) -> Dict[str, float]:
        return {s["name"]: float(s["weight"]) for s in self.skills}

    def to_dict(self) -> Dict:
        d = asdict(self)
        d.pop("raw_text", None)
        d["sections"] = {k: (v[:200] + "..." if len(v) > 200 else v) for k, v in self.sections.items()}
        return d


def _endpoint(m: "re.Match", p: str) -> Optional[Tuple[int, Optional[int]]]:
    g = m.groupdict()
    if g.get(p + "m"):
        return int(g[p + "y"]), MONTHS[g[p + "m"][:3].lower()]
    if g.get(p + "nm"):
        return int(g[p + "ny"]), int(g[p + "nm"])
    if g.get(p + "yy"):
        return int(g[p + "yy"]), None
    return None


def merge_months(intervals: List[Tuple[int, int]]) -> int:
    """total months covered by the UNION of [start, end) month intervals."""
    total, cur_s, cur_e = 0, None, None
    for s, e in sorted(intervals):
        if cur_e is None or s > cur_e:
            if cur_e is not None:
                total += cur_e - cur_s
            cur_s, cur_e = s, e
        else:
            cur_e = max(cur_e, e)
    if cur_e is not None:
        total += cur_e - cur_s
    return total


class ResumeAnalyzer:
    def __init__(self, extractor: Optional[Extractor] = None):
        self.extractor = extractor or Extractor()

    # ----------------------------------------------------------------------------------------------- sections
    @staticmethod
    def split_sections(text: str) -> Tuple[Dict[str, str], List[Tuple[str, int, int]]]:
        spans: List[Tuple[str, int, int]] = []
        marks: List[Tuple[str, int, int]] = []           # (section, heading_start, body_start)
        pos = 0
        for line in text.splitlines(keepends=True):
            raw = line.strip()
            norm = re.sub(r"[^a-z& ]", "", raw.lower()).strip()
            if raw and len(raw) <= 40 and norm in HEADINGS and len(raw.split()) <= 4:
                marks.append((HEADINGS[norm], pos, pos + len(line)))
            pos += len(line)
        sections: Dict[str, str] = {}
        for i, (sec, _, body) in enumerate(marks):
            end = marks[i + 1][1] if i + 1 < len(marks) else len(text)
            sections[sec] = (sections.get(sec, "") + "\n" + text[body:end]).strip()
            spans.append((sec, body, end))
        return sections, spans

    # ----------------------------------------------------------------------------------------------- experience
    def date_ranges(self, text: str, spans: List[Tuple[str, int, int]], today: dt.date) -> Tuple[List[Dict], List[Tuple[int, int]]]:
        found, intervals = [], []
        skip = [(a, b) for sec, a, b in spans if sec in ("education", "certifications")]
        for m in DATE_RANGE.finditer(text):
            line_start = text.rfind("\n", 0, m.start()) + 1
            line_end = text.find("\n", m.end())
            line = text[line_start: line_end if line_end >= 0 else len(text)]
            in_edu = any(a <= m.start() < b for a, b in skip) or (not spans and EDU_LINE.search(line) is not None)
            sp = _endpoint(m, "s")
            if sp is None:
                continue
            s = sp[0] * 12 + ((sp[1] - 1) if sp[1] else 0)
            if m.group("now"):
                e, label_end = today.year * 12 + today.month, "present"
            else:
                ep = _endpoint(m, "e")
                if ep is None:
                    continue
                e, label_end = ep[0] * 12 + (ep[1] if ep[1] else 0), f"{ep[0]}" + (f"-{ep[1]:02d}" if ep[1] else "")
            ok = e > s and (e - s) <= 50 * 12 and sp[0] >= 1970 and s <= today.year * 12 + today.month
            rec = {"text": m.group(0), "start": m.start(), "end": m.end(), "months": (e - s) if ok else None, "counted": bool(ok and not in_edu),
                   "reason": None if ok and not in_edu else ("education / certification context" if in_edu else "implausible range")}
            found.append(rec)
            if rec["counted"]:
                intervals.append((s, e))
        return found, intervals

    # ----------------------------------------------------------------------------------------------- main
    def analyze(self, text: str, *, today: Optional[dt.date] = None) -> ResumeProfile:
        today = today or dt.date.today()
        text = text or ""
        prof = ResumeProfile(n_words=len(text.split()), raw_text=text)
        if prof.n_words < 40:
            prof.warnings.append("very short text (< 40 words): extraction is unreliable")
        sections, spans = self.split_sections(text)
        prof.sections = sections
        # ---- contact-level entities
        for lab, rx, store in (("EMAIL", EMAIL, prof.emails), ("URL", URL, prof.urls)):
            for m in rx.finditer(text):
                store.append(m.group(0))
                prof.entities.append(Entity(lab, m.group(0), m.start(), m.end()))
        for m in PHONE.finditer(text):
            digits = re.sub(r"\D", "", m.group(0))
            if 10 <= len(digits) <= 13 and not DATE_RANGE.search(m.group(0)):
                prof.phones.append(m.group(0).strip())
                prof.entities.append(Entity("PHONE", m.group(0).strip(), m.start(), m.end()))
        # ---- name (heuristic: first plausible line near the top)
        pos = 0
        seen = 0
        for line in text.splitlines(keepends=True):
            s = line.strip()
            if s:
                seen += 1
                words = s.replace(",", " ").split()
                norm = re.sub(r"[^a-z& ]", "", s.lower()).strip()
                if (2 <= len(words) <= 4 and norm not in HEADINGS and not re.search(r"[\d@]", s) and all(re.fullmatch(r"[A-Za-z][A-Za-z.'\-]*", w) for w in words)
                        and all(w[0].isupper() for w in words) and not ({w.lower() for w in words} & NON_NAME_WORDS)):
                    prof.name = s
                    start = pos + line.index(s)
                    prof.entities.append(Entity("PERSON", s, start, start + len(s), source="heuristic"))
                    break
                if seen >= 6:
                    break
            pos += len(line)
        if prof.name is None:
            prof.warnings.append("name not found (heuristic looks only at the first lines)")
        # organisations: company-suffix names + 'X Institute/University of Y'; NOT inside the certifications section, NOT degree phrases ('in Information Technology')
        cert_spans = [(a_, b_) for sec, a_, b_ in spans if sec == "certifications"]
        edu = [(m.start(1), m.end(1), m.group(1).strip()) for m in EDU_ORG.finditer(text) if not any(a_ <= m.start(1) < b_ for a_, b_ in cert_spans)]
        cands = list(edu)
        for m in ORG.finditer(text):
            s_, e_, t_ = m.start(1), m.end(1), m.group(1)
            if (any(a_ <= s_ < b_ for a_, b_ in cert_spans) or any(es <= s_ and e_ <= ee for es, ee, _ in edu) or text[max(0, s_ - 4):s_].lower().endswith(" in ")
                    or t_.lower() in NOT_ORG):
                continue
            cands.append((s_, e_, t_))
        for s_, e_, t_ in sorted(cands):
            if t_ not in prof.organisations:
                prof.organisations.append(t_)
            prof.entities.append(Entity("ORG", t_, s_, s_ + len(t_)))
        # ---- extractor: skills, education, certifications, roles, locations, stated experience
        ex = self.extractor.extract(text)
        listed = {s["name"] for s in self.extractor.extract(sections["skills"], skills_only=True)["skills"]} if sections.get("skills") else set()
        for s in ex["skills"]:
            in_list = s["name"] in listed
            weight = 1.0 if in_list else (0.8 if s["mentions"] >= 2 else 0.6)
            prof.skills.append({"name": s["name"], "type": s["type"], "mentions": s["mentions"], "weight": weight,
                                "evidence": "listed in the skills section" if in_list else f"mentioned {s['mentions']}x in the text"})
        prof.education = ex["education"]
        prof.best_education = self.extractor.best_education(ex["education"]) if ex["education"] else None
        # the Phase-1 certification pattern can run across line breaks / headings on resumes: keep single-line matches and read the certifications section line by line
        certs = [c.strip() for c in ex["certifications"] if c.strip() and "\n" not in c.strip()]
        for line in sections.get("certifications", "").splitlines():
            t = re.sub(r"^[\s\-*\u2022\u00b7]+", "", line).strip()
            if 3 <= len(t) <= 120 and t.lower() not in {c.lower() for c in certs}:
                certs.append(t)
        uniq = lambda xs: [x for i, x in enumerate(xs) if x.lower() not in {y.lower() for y in xs[:i]}]
        prof.certifications, prof.roles, prof.locations = uniq(certs), uniq(ex["roles"]), ex["locations"]
        # ---- experience years
        found, intervals = self.date_ranges(text, spans, today)
        prof.date_ranges = found
        if intervals:
            prof.computed_experience_years = round(merge_months(intervals) / 12.0, 1)
            prof.entities += [Entity("DATE_RANGE", r["text"], r["start"], r["end"]) for r in found if r["counted"]]
        st = ex.get("experience")
        if st and st.get("explicit"):
            prof.stated_experience_years = float(st.get("min") if st.get("min") is not None else 0.0)
        if prof.stated_experience_years is not None:
            prof.experience_years, prof.experience_source = prof.stated_experience_years, "stated"
        elif prof.computed_experience_years is not None:
            prof.experience_years, prof.experience_source = prof.computed_experience_years, "date_ranges"
        else:
            prof.warnings.append("no experience statement or date ranges found: experience unknown (not assumed to be 0)")
        if prof.stated_experience_years is not None and prof.computed_experience_years is not None and abs(prof.stated_experience_years - prof.computed_experience_years) > 2:
            prof.warnings.append(f"stated experience ({prof.stated_experience_years:g} y) and date ranges ({prof.computed_experience_years:g} y) disagree by more than 2 years; the stated value is used")
        if not prof.skills:
            prof.warnings.append("no skills recognised (the skill lexicon only knows the skills of the job datasets)")
        return prof

    def analyze_file(self, path: Path, **kw) -> ResumeProfile:
        path = Path(path)
        if path.suffix.lower() not in (".txt", ".md", ""):
            raise ValueError(f"{path.suffix} is not supported yet: only plain-text resumes are parsed (PDF/DOCX/scanned input is a later step)")
        return self.analyze(path.read_text(encoding="utf-8", errors="replace"), **kw)
