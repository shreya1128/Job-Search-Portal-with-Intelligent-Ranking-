"""BeautifulSoup extraction pipeline:   HTML -> DOM -> job fields -> clean job document  (+ outgoing typed links).

Strategy per page (most reliable first):
   1. schema.org JSON-LD  <script type="application/ld+json">  with @type JobPosting
   2. CSS-selector profile (SiteProfile) - add a profile per new source to make the crawler modular
Links are classified by CSS class so the crawler can build the Company -> Career page -> Job -> Related job graph."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from urllib.parse import urljoin

from bs4 import BeautifulSoup


@dataclass
class SiteProfile:
    """CSS selectors for one website layout.  Register new sources in SITE_PROFILES."""
    title: str = "h1.job-title"
    company: str = ".company-name"
    location: str = ".job-location"
    description: str = ".job-description"
    skills: str = "ul.skills li"
    experience: str = ".job-experience"
    education: str = ".job-education"
    salary: str = ".job-salary"
    employment_type: str = ".job-type"
    category: str = ".job-category"
    date_posted: str = "time.posted"


SITE_PROFILES: Dict[str, SiteProfile] = {"default": SiteProfile()}

LINK_CLASSES = {"job-link": "job", "related-link": "related_job", "company-link": "company", "careers-link": "careers", "next-page": "next"}


@dataclass
class Link:
    url: str
    kind: str   # job | related_job | company | careers | next | other


@dataclass
class ParsedPage:
    page_type: str
    job: Optional[Dict] = None
    company: Optional[Dict] = None
    links: List[Link] = field(default_factory=list)


def _text(node) -> Optional[str]:
    return node.get_text(" ", strip=True) if node else None


def _jsonld_job(soup: BeautifulSoup) -> Optional[Dict]:
    for tag in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(tag.string or "")
        except (ValueError, TypeError):
            continue
        for item in (data if isinstance(data, list) else [data]):
            if isinstance(item, dict) and item.get("@type") == "JobPosting":
                return item
    return None


def _from_jsonld(d: Dict) -> Dict:
    loc = d.get("jobLocation") or {}
    addr = (loc.get("address") if isinstance(loc, dict) else {}) or {}
    org = d.get("hiringOrganization") or {}
    sal = d.get("baseSalary") or {}
    val = sal.get("value") if isinstance(sal, dict) else {}
    sal_txt = None
    if isinstance(val, dict) and val.get("minValue") is not None:
        sal_txt = f"{sal.get('currency', 'INR')} {val.get('minValue')} - {val.get('maxValue')} per {str(val.get('unitText', 'YEAR')).lower()}"
    exp = d.get("experienceRequirements")
    skills = d.get("skills")
    return dict(title=d.get("title"), company=org.get("name") if isinstance(org, dict) else None,
                location=addr.get("addressLocality") if isinstance(addr, dict) else None, description=d.get("description"),
                skills=[s.strip() for s in skills.split(",")] if isinstance(skills, str) else (skills or []),
                experience_text=exp if isinstance(exp, str) else None, education=d.get("educationRequirements"),
                salary_text=sal_txt, employment_type=d.get("employmentType"), category=d.get("occupationalCategory"),
                date_posted=d.get("datePosted"))


def _from_css(soup: BeautifulSoup, p: SiteProfile) -> Dict:
    sel = lambda css: soup.select_one(css)
    desc_node = sel(p.description)
    return dict(title=_text(sel(p.title)), company=_text(sel(p.company)), location=_text(sel(p.location)),
                description=desc_node.decode_contents() if desc_node else None,
                skills=[_text(li) for li in soup.select(p.skills)], experience_text=_text(sel(p.experience)), education=_text(sel(p.education)),
                salary_text=_text(sel(p.salary)), employment_type=_text(sel(p.employment_type)), category=_text(sel(p.category)),
                date_posted=(sel(p.date_posted).get("datetime") if sel(p.date_posted) else None))


def parse_page(html: str, url: str, profile: Optional[SiteProfile] = None) -> ParsedPage:
    soup = BeautifulSoup(html, "lxml")
    profile = profile or SITE_PROFILES["default"]
    body = soup.body
    ptype = (body.get("data-page-type") if body else None) or "other"
    links = []
    for a in soup.find_all("a", href=True):
        kind = next((LINK_CLASSES[c] for c in a.get("class", []) if c in LINK_CLASSES), "other")
        links.append(Link(urljoin(url, a["href"]), kind))
    page = ParsedPage(ptype, links=links)
    if ptype == "job" or _jsonld_job(soup):
        ld = _jsonld_job(soup)
        raw = _from_jsonld(ld) if ld else _from_css(soup, profile)
        css = _from_css(soup, profile) if ld else {}
        for k, v in css.items():                      # fill gaps in JSON-LD with CSS values
            if not raw.get(k) and v:
                raw[k] = v
        if raw.get("title"):
            raw["url"] = url
            raw["_extracted_via"] = "json-ld" if ld else "css"
            page.job = raw
            page.page_type = "job"
    elif ptype == "company":
        h1 = soup.find("h1")
        page.company = dict(name=_text(h1), url=url, about=_text(soup.select_one(".company-about")))
    return page
