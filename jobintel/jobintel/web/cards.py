"""Turning engine results into what a job card shows: a short, honest reason for the match, the skill chips, and an excerpt where the posting has real text.

Everything here is derived from the engine's own output (``SearchResult.reasons``, ``skills``, ``passage``) and from the index metadata.  Nothing is scored or invented here: the
number-heavy signals stay in the engine and the logs; the card shows only a sentence a job seeker can read.
"""
from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Sequence

from .ui import employer

PER_PAGE = 10
_TERM = re.compile(r"^✓ '(.+)' matched$")
MAX_CHIPS = 8
MAX_REASON_ITEMS = 4


@dataclass
class Card:
    job_id: int
    title: str
    company: Optional[str]
    location: Optional[str]
    experience: Optional[str]
    salary: Optional[str]
    salary_note: Optional[str]
    source: str
    excerpt_html: Optional[str]
    have: List[str] = field(default_factory=list)       # skills the search asked for that this job lists
    other: List[str] = field(default_factory=list)      # other skills the job lists
    missing: List[str] = field(default_factory=list)    # skills the search asked for that this job does not list
    reason: Optional[str] = None
    exp_min: Optional[float] = None


def location_text(city: Optional[str], cities: Sequence[str], country: Optional[str]) -> Optional[str]:
    city = (city or "").strip()
    extra = max(0, len(cities or []) - 1)
    if not city:
        return (country or "").title() or None
    out = city + (f" and {extra} more" if extra else "")
    return f"{out}, {country.title()}" if country and not extra else out


def match_reason(reasons: Sequence[str], *, browse: bool = False) -> Optional[str]:
    """One short sentence from the engine's reasons: the skills (or words) that matched, the place, the experience band.  None when there is nothing true to say."""
    skills = [r[len("✓ skill: "):] for r in reasons if r.startswith("✓ skill: ")]
    words = [m.group(1) for r in reasons if (m := _TERM.match(r))]
    parts: List[str] = []
    if skills:
        parts.append("Lists " + ", ".join(skills[:MAX_REASON_ITEMS]) + (f" and {len(skills) - MAX_REASON_ITEMS} more" if len(skills) > MAX_REASON_ITEMS else ""))
    elif words:
        parts.append("Mentions " + ", ".join(words[:MAX_REASON_ITEMS]))
    for r in reasons:
        if r.startswith("✓ location matches "):
            parts.append("in " + r[len("✓ location matches "):])
        elif r == "✓ experience band fits":
            parts.append("experience fits the range you asked for")
    return " · ".join(parts) or ("Matches your filters" if browse else None)


def build_card(x, ix, *, browse: bool = False) -> Card:
    """One result -> one card.  ``ix`` supplies the numeric fields the result only carries as text (experience for sorting, whether a salary has a currency)."""
    di = ix.doc_index(x.job_id)
    exp_min = salary_note = None
    if di is not None:
        v = ix.meta["exp_min"][di]
        exp_min = None if v is None or (isinstance(v, float) and math.isnan(v)) else float(v)
        if x.salary and not ix.meta["sal_cur"][di]:
            salary_note = "currency not stated"
    hit = [r[len("✓ skill: "):] for r in x.reasons if r.startswith("✓ skill: ")]
    miss = [r[len("✗ skill missing: "):] for r in x.reasons if r.startswith("✗ skill missing: ")]
    other = [s for s in (x.skills or []) if s not in hit and s not in miss]
    room = max(0, MAX_CHIPS - len(hit) - min(len(miss), 2))
    excerpt = x.passage["html"] if x.passage and x.text_quality != "skills_only" else None
    return Card(job_id=x.job_id, title=x.title, company=employer(x.company), location=location_text(x.city, x.cities, x.country), experience=x.experience, salary=x.salary,
                salary_note=salary_note, source=x.source, excerpt_html=excerpt, have=hit[:MAX_CHIPS], other=other[:room], missing=miss[:2],
                reason=match_reason(x.reasons, browse=browse), exp_min=exp_min)


SORTS = {"relevance": "Best match", "exp_low": "Experience: lowest first", "exp_high": "Experience: highest first"}


def sort_cards(cards: List[Card], key: str) -> List[Card]:
    """Re-order the retrieved pool.  Python's sort is stable, so jobs with the same experience keep their relevance order; jobs that state no experience go last either way."""
    if key == "exp_low":
        return sorted(cards, key=lambda c: (c.exp_min is None, c.exp_min if c.exp_min is not None else 0.0))
    if key == "exp_high":
        return sorted(cards, key=lambda c: (c.exp_min is None, -(c.exp_min if c.exp_min is not None else 0.0)))
    return cards


def paginate(items: Sequence[Any], page: int, per_page: int = PER_PAGE) -> Dict[str, Any]:
    total = len(items)
    pages = max(1, math.ceil(total / per_page))
    page = min(max(1, page), pages)
    lo = (page - 1) * per_page
    return {"items": list(items[lo:lo + per_page]), "page": page, "pages": pages, "total": total, "first": lo + 1 if total else 0, "last": min(total, lo + per_page)}


def with_links(pg: Dict[str, Any], make_url) -> Dict[str, Any]:
    """Add ready-made page URLs to a paginate() result: ``make_url(n)`` returns the address of page n."""
    pg["prev"] = make_url(pg["page"] - 1) if pg["page"] > 1 else None
    pg["next"] = make_url(pg["page"] + 1) if pg["page"] < pg["pages"] else None
    pg["numbers"] = [(n, make_url(n)) for n in range(1, pg["pages"] + 1)]
    return pg
