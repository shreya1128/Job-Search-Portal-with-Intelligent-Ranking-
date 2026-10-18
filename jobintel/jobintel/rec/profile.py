"""UserProfile: what the recommenders need to know about a person (skills with weights, experience, preferences, interaction history)."""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional

from .resume import ResumeProfile


@dataclass
class UserProfile:
    skills: Dict[str, float] = field(default_factory=dict)         # canonical skill -> weight in (0, 1]
    experience_years: Optional[float] = None
    preferred_role: Optional[str] = None
    locations: List[str] = field(default_factory=list)
    categories: List[str] = field(default_factory=list)
    employment: Optional[str] = None
    interactions: Dict[int, float] = field(default_factory=dict)   # job_id -> event weight (view 1 / click 2 / save 3 / apply 4)
    document: Optional[Any] = None

    @classmethod
    def from_resume(cls, r: ResumeProfile, *, preferred_role: Optional[str] = None, locations: Optional[List[str]] = None, **kw) -> "UserProfile":
        return cls(skills=r.skill_weights(), experience_years=r.experience_years, preferred_role=preferred_role or (r.roles[0] if r.roles else None),
                   locations=list(locations if locations is not None else r.locations), **kw)

    @classmethod
    def from_form(cls, skills: List[str], **kw) -> "UserProfile":
        return cls(skills={s.strip().lower(): 1.0 for s in skills if s.strip()}, **kw)

    def text(self) -> str:
        top = sorted(self.skills, key=lambda s: -self.skills[s])[:12]
        return " ".join(([self.preferred_role] if self.preferred_role else []) + top)
