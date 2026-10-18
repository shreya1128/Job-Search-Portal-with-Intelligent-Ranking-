"""Building a recommender ``UserProfile`` for the current visitor from the three places it can come from: a form, an analysed resume (kept server-side for a few hours) and the saved profile."""
from __future__ import annotations

import datetime as dt
from typing import Dict, List, Optional, Sequence, Tuple

from flask import session
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db import models as M
from ...rec import store as rec_store
from ...rec.profile import UserProfile
from ._common import ext, user_interactions

RESUME_KEY = "resume_token"


def saved_profile(s: Session, user_id: int) -> Optional[UserProfile]:
    up = s.execute(select(M.UserProfile).where(M.UserProfile.user_id == user_id)).scalar_one_or_none()
    rows = s.execute(select(M.Skill.canonical_name, M.UserSkill.confidence).join(M.UserSkill, M.UserSkill.skill_id == M.Skill.skill_id).where(M.UserSkill.user_id == user_id)).all()
    skills = {n: float(c) for n, c in rows}
    inter = user_interactions(s, user_id)
    if up is None and not skills:
        return None
    return UserProfile(skills=skills, experience_years=up.experience_years if up else None, preferred_role=up.preferred_role if up else None,
                       locations=[up.preferred_location] if up and up.preferred_location else [], categories=list(up.preferred_categories or []) if up else [],
                       interactions=inter, document=(up.resume_text if up and up.resume_text else None))


def session_resume() -> Optional[Dict]:
    """The resume analysed in this browser session: {'profile': ResumeProfile, 'document': DocumentText, 'filename': str} or None (expired / never uploaded)."""
    return ext().store.get(session.get(RESUME_KEY))


def normalize_skills(lex, names: Sequence[str]) -> Tuple[Dict[str, float], List[str]]:
    """Typed skill names -> canonical skills of the lexicon (weight 1.0); names the lexicon does not know are returned, never dropped silently."""
    known: Dict[str, float] = {}
    unknown: List[str] = []
    for n in names:
        c = lex.normalize(n)
        if c:
            known[c] = 1.0
        else:
            unknown.append(n)
    return known, unknown


# ---------------------------------------------------------------------------------------------------------------------------- the Profile page (Phase 7, stage 2)
# Only what the database can store today: name, education, summary, years of experience, target role, ONE preferred location, skills (and the resume fields, which the Resume page will
# fill).  There is no column for certifications, several locations or an employment-type preference, so the page does not offer them.
LIMITS = dict(full_name=120, education=120, summary=1000, role=120, location=120)

STEPS = (("skills", "Skills", "Add the skills you have."),
         ("years", "Years of experience", "Say how many years of work experience you have (0 is fine)."),
         ("role", "Target role", "Name the role you are looking for."),
         ("location", "Preferred location", "Add the city or country you want to work in."),
         ("education", "Education", "Add your highest education."),
         ("resume", "Resume analysed", "Upload and review a resume in the Resume Analyzer."))


def load_profile_form(s: Session, user_id: int) -> Dict:
    """Everything the Profile page shows, read from the user's own rows."""
    up = s.execute(select(M.UserProfile).where(M.UserProfile.user_id == user_id)).scalar_one_or_none()
    rows = s.execute(select(M.Skill.canonical_name, M.UserSkill.source).join(M.UserSkill, M.UserSkill.skill_id == M.Skill.skill_id)
                     .where(M.UserSkill.user_id == user_id).order_by(M.Skill.canonical_name)).all()
    return dict(full_name=(up.full_name or "") if up else "", education=(up.education or "") if up else "", summary=(up.summary or "") if up else "",
                years=up.experience_years if up else None, role=(up.preferred_role or "") if up else "", location=(up.preferred_location or "") if up else "",
                skills=[n for n, _ in rows], from_resume=[n for n, src in rows if src == "resume"],
                resume_filename=up.resume_filename if up else None, updated_at=up.updated_at if up else None, exists=up is not None)


def profile_steps(data: Dict) -> List[Dict]:
    """The completion checklist, computed from what is stored.  A step is done only when its value is really there; nothing here is a score or an estimate."""
    done = dict(skills=bool(data["skills"]), years=data["years"] is not None, role=bool(data["role"]), location=bool(data["location"]), education=bool(data["education"]),
                resume=bool(data["resume_filename"]))
    return [dict(key=k, label=label, hint=hint, done=done[k]) for k, label, hint in STEPS]


def sync_skills(s: Session, user_id: int, weights: Dict[str, float]) -> Dict:
    """Make the user's skill rows equal ``weights`` (canonical name -> weight).  Skills that stay keep their stored source and weight (a skill read from a resume stays 'resume');
    removed ones are deleted; new ones are 'manual'.  Names that have no row in the skills table are returned, never silently dropped."""
    ids = rec_store.skill_ids(s, weights)
    existing = {int(r.skill_id): r for r in s.execute(select(M.UserSkill).where(M.UserSkill.user_id == user_id)).scalars()}
    keep = set(ids.values())
    for sid, row in existing.items():
        if sid not in keep:
            s.delete(row)
    for name, sid in ids.items():
        if sid not in existing:
            s.add(M.UserSkill(user_id=user_id, skill_id=sid, source="manual", confidence=float(weights[name])))
    s.flush()
    return {"saved_skills": len(ids), "not_in_skills_table": sorted(set(weights) - set(ids))}


def save_profile_form(s: Session, user_id: int, *, full_name: str, education: str, summary: str, years: Optional[float], role: str, location: str,
                      weights: Dict[str, float]) -> Dict:
    """Store the Profile form.  The resume fields of the profile row are not touched."""
    up = s.execute(select(M.UserProfile).where(M.UserProfile.user_id == user_id)).scalar_one_or_none() or M.UserProfile(user_id=user_id)
    up.full_name, up.education, up.summary = full_name or None, education or None, summary or None
    up.experience_years, up.preferred_role, up.preferred_location = years, role or None, location or None
    s.add(up)
    s.flush()
    return sync_skills(s, user_id, weights)


def export_user_data(s: Session, user_id: int) -> Dict:
    """Everything stored about one user, for the 'download my data' link.  Job postings are identified by id and title only (they are public data, not the user's)."""
    u = s.get(M.User, user_id)
    form = load_profile_form(s, user_id)
    up = s.execute(select(M.UserProfile).where(M.UserProfile.user_id == user_id)).scalar_one_or_none()
    inter = s.execute(select(M.JobInteraction, M.Job.title).join(M.Job, M.Job.job_id == M.JobInteraction.job_id).where(M.JobInteraction.user_id == user_id)
                      .order_by(M.JobInteraction.last_event_at.desc())).all()
    searches = s.execute(select(M.Search).where(M.Search.user_id == user_id).order_by(M.Search.created_at.desc())).scalars().all()
    iso = lambda d: d.isoformat() if d is not None else None                         # noqa: E731
    return {
        "exported_at": dt.datetime.now().isoformat(timespec="seconds"),
        "account": {"username": u.username, "email": u.email, "created_at": iso(u.created_at)},
        "profile": {"full_name": form["full_name"] or None, "education": form["education"] or None, "summary": form["summary"] or None, "years_of_experience": form["years"],
                    "target_role": form["role"] or None, "preferred_location": form["location"] or None, "updated_at": iso(form["updated_at"]),
                    "resume_filename": form["resume_filename"], "resume_text": up.resume_text if up else None},
        "skills": [{"skill": n, "source": "resume" if n in form["from_resume"] else "manual"} for n in form["skills"]],
        "jobs": [{"job_id": int(i.job_id), "title": t, "viewed": bool(i.viewed), "opened_from_search": bool(i.clicked), "saved": bool(i.saved), "marked_applied": bool(i.applied),
                  "last_activity": iso(i.last_event_at)} for i, t in inter],
        "searches": [{"query": x.query_text, "filters": x.filters, "results": int(x.n_results), "at": iso(x.created_at)} for x in searches],
        "recommendations": [{"job_id": int(r.job_id), "position": int(r.position), "final_score": float(r.final_score), "weights": r.weights, "created_at": iso(r.created_at)}
                            for r in s.execute(select(M.Recommendation).where(M.Recommendation.user_id == user_id).order_by(M.Recommendation.created_at.desc())).scalars()],
        "skill_gaps": [{"job_id": int(g.job_id), "job_title": title, "skill": skill, "status": g.status, "importance": g.importance, "market_frequency": g.market_frequency, "created_at": iso(g.created_at)}
                       for g, title, skill in s.execute(select(M.SkillGap, M.Job.title, M.Skill.canonical_name).join(M.Job, M.Job.job_id == M.SkillGap.job_id).join(M.Skill, M.Skill.skill_id == M.SkillGap.skill_id).where(M.SkillGap.user_id == user_id).order_by(M.SkillGap.created_at.desc()))],
    }
