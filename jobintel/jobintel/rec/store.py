"""Persistence of Phase 6 results in the Phase 1 tables (users, user_profiles, user_skills, job_interactions, recommendations, skill_gaps).
Synthetic users / interactions are written with is_synthetic = True and a username prefix 'synthetic_', and are never mixed into real-user logic."""
from __future__ import annotations

from typing import Dict, Iterable, List, Optional, Sequence

from sqlalchemy import delete, select
from sqlalchemy.orm import Session

from ..db import models as M
from .hybrid import RecommendationResponse
from .interactions import SyntheticData
from .profile import UserProfile as Profile
from .resume import ResumeProfile
from .skillgap import GapReport


def create_user(s: Session, username: str, email: str, *, synthetic: bool = False) -> int:
    u = M.User(username=username, email=email, password_hash="!no-login", is_synthetic=synthetic)     # placeholder: the web UI (Phase 7) sets real password hashes
    s.add(u)
    s.flush()
    return int(u.user_id)


def skill_ids(s: Session, names: Iterable[str]) -> Dict[str, int]:
    names = list(set(names))
    if not names:
        return {}
    rows = s.execute(select(M.Skill.canonical_name, M.Skill.skill_id).where(M.Skill.canonical_name.in_(names))).all()
    return {n: int(i) for n, i in rows}


def save_profile(s: Session, user_id: int, profile: Profile, resume: Optional[ResumeProfile] = None, *, resume_text: Optional[str] = None, filename: Optional[str] = None) -> Dict:
    up = s.execute(select(M.UserProfile).where(M.UserProfile.user_id == user_id)).scalar_one_or_none() or M.UserProfile(user_id=user_id)
    up.experience_years = profile.experience_years
    up.preferred_role = profile.preferred_role
    up.preferred_location = profile.locations[0] if profile.locations else None
    up.preferred_categories = profile.categories or None
    if resume is not None:
        up.full_name, up.education = resume.name, resume.best_education
        up.resume_filename, up.resume_text = filename, resume_text
    s.add(up)
    ids = skill_ids(s, profile.skills)
    s.execute(delete(M.UserSkill).where(M.UserSkill.user_id == user_id))
    src = "resume" if resume is not None else "manual"
    for name, sid in ids.items():
        s.add(M.UserSkill(user_id=user_id, skill_id=sid, source=src, confidence=float(profile.skills[name])))
    s.flush()
    return {"saved_skills": len(ids), "not_in_skills_table": sorted(set(profile.skills) - set(ids))}


def save_recommendations(s: Session, user_id: int, resp: RecommendationResponse) -> int:
    for r in resp.results:
        g = lambda k: r.signals.get(k)
        s.add(M.Recommendation(user_id=user_id, job_id=r.job_id, position=r.rank, ir_score=g("ir"), dl_score=g("dl"), content_score=g("content"), collaborative_score=g("collaborative"),
                               knowledge_score=g("knowledge"), skill_match_score=g("skill_match"), preference_score=g("preference"), pagerank_score=g("pagerank"),
                               final_score=r.final_score, weights=resp.weights_effective))
    s.flush()
    return len(resp.results)


def save_skill_gaps(s: Session, user_id: int, job_id: int, report: GapReport) -> Dict:
    ids = skill_ids(s, [r.skill for r in report.rows])
    n = 0
    for r in report.rows:
        if r.skill in ids:
            s.add(M.SkillGap(user_id=user_id, job_id=job_id, skill_id=ids[r.skill], status=r.status, importance=r.importance, market_frequency=r.market_frequency))
            n += 1
    s.flush()
    return {"saved": n, "not_in_skills_table": [r.skill for r in report.rows if r.skill not in ids]}


def write_synthetic(s: Session, data: SyntheticData, catalog, prefix: str = "synthetic") -> List[int]:
    """one SYNTHETIC user per persona + its generated interactions (is_synthetic = True on both)."""
    uids = []
    for p in data.personas:
        uids.append(create_user(s, f"{prefix}_{p['user']:05d}", f"{prefix}_{p['user']:05d}@example.invalid", synthetic=True))
    for u, d, v, c, sv, ap in zip(data.user, data.doc, data.viewed, data.clicked, data.saved, data.applied):
        s.add(M.JobInteraction(user_id=uids[int(u)], job_id=int(catalog.ix.job_ids[int(d)]), viewed=bool(v), clicked=bool(c), saved=bool(sv), applied=bool(ap), is_synthetic=True))
    s.flush()
    return uids
