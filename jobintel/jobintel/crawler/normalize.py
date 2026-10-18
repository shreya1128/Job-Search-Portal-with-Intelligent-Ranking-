"""Turns the RAW strings scraped from a page (by BeautifulSoup OR Scrapy) into one JobRecord-compatible dict,
so both crawl engines produce identical output."""
from __future__ import annotations

import hashlib
import re
from typing import Dict, List, Optional

from jobintel.ingest import common as C
from jobintel.processing.text import clean_description


def to_record_dict(raw: Dict, source: str = "fixture_crawl") -> Dict:
    desc = clean_description(raw.get("description") or "")
    lo, hi = C.parse_exp_range(raw.get("experience_text"))
    sal = C.parse_salary_text(raw.get("salary_text"), "INR")
    cities, _ = C.parse_naukri_location(raw.get("location") or "")
    url = raw["url"]
    skills = [s.strip() for s in raw.get("skills") or [] if s and s.strip()]
    return dict(
        source=source, external_id=hashlib.sha1(url.encode()).hexdigest()[:16], title=(raw.get("title") or "").strip() or "Unknown role",
        company=(raw.get("company") or "").strip() or None, location_raw=raw.get("location"), city=cities[0] if cities else None, cities=cities,
        country=raw.get("country") or "India", description=desc or None, skills=skills, experience_min=lo, experience_max=hi,
        experience_text=raw.get("experience_text"), experience_source="dataset" if lo is not None else None, education=raw.get("education"),
        salary_min=sal.get("min"), salary_max=sal.get("max"), salary_currency=sal.get("currency"), salary_period=sal.get("period"),
        employment_type=raw.get("employment_type"), category=raw.get("category"), posting_date=(raw.get("date_posted") or None), url=url,
        text_quality="full" if desc else "skills_only")
