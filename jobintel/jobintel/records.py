"""Source-independent record produced by every loader / crawler and consumed by the ingestion pipeline."""
from __future__ import annotations

import datetime as dt
from dataclasses import dataclass, field
from typing import Any, Optional


@dataclass
class JobRecord:
    source: str
    external_id: str
    title: str
    company: Optional[str] = None
    company_profile: Optional[str] = None
    company_name_is_derived: bool = False
    location_raw: Optional[str] = None
    city: Optional[str] = None
    cities: list = field(default_factory=list)
    region: Optional[str] = None
    country: Optional[str] = None
    description: Optional[str] = None            # ORIGINAL text exactly as supplied (may contain HTML)
    skills: list = field(default_factory=list)   # raw tags supplied by the dataset
    experience_min: Optional[float] = None
    experience_max: Optional[float] = None
    experience_text: Optional[str] = None
    experience_source: Optional[str] = None      # dataset | extracted | seniority_map
    education: Optional[str] = None
    salary_min: Optional[float] = None
    salary_max: Optional[float] = None
    salary_currency: Optional[str] = None
    salary_period: Optional[str] = None
    employment_type: Optional[str] = None
    seniority: Optional[str] = None
    category: Optional[str] = None
    industry: Optional[str] = None
    posting_date: Optional[dt.date] = None
    url: Optional[str] = None
    is_fraudulent: bool = False
    text_quality: str = "full"                   # full | skills_only | preprocessed | none
    extras: dict = field(default_factory=dict)
