"""Profile-to-job / profile-to-role skill gap analysis."""
from __future__ import annotations

import logging
from flask import Blueprint, flash, redirect, render_template, request, url_for
from sqlalchemy import delete
from ..auth import load_current_user
from ..backend import Unavailable
from ..forms import Params
from ...db import models as M
from ...rec import store as rec_store
from ...rec.profile import UserProfile
from ...rec.skillgap import analyze_gap
from ._common import backend, db_session, public_reason
from ._profile import normalize_skills, saved_profile, session_resume

bp = Blueprint("skillgap", __name__)
log = logging.getLogger("jobintel.web")


def _jobs(ix, limit=120):
    rows = []
    for i, jid in enumerate(ix.job_ids):
        title = ix.meta["title"][i]
        if not title:
            continue
        rows.append({"id": int(jid), "title": str(title), "company": str(ix.meta["company"][i] or ""), "city": str(ix.meta["city"][i] or "")})
    rows.sort(key=lambda r: r["title"].lower())
    return rows[:limit]


def _candidate_profile(p, be, user):
    sr = session_resume()
    base = UserProfile.from_resume(sr["profile"]) if sr else None
    if base is None and user:
        try:
            with db_session() as s:
                base = saved_profile(s, user.user_id)
        except Exception:
            log.exception("could not load saved profile for skill gap")
    skills_text = p.text("skills", 1500)
    role = p.text("role", 120)
    if skills_text:
        known, unknown = normalize_skills(be.engine.lex, [x.strip() for x in skills_text.split(",") if x.strip()])
        if base:
            known = {**base.skills, **known}
    else:
        known, unknown = (dict(base.skills), []) if base else ({}, [])
    if not (known or role or (base and base.preferred_role)):
        return None, unknown
    prof = UserProfile(skills=known, experience_years=base.experience_years if base else None,
                       preferred_role=role or (base.preferred_role if base else None), locations=list(base.locations) if base else [])
    return prof, unknown


@bp.route("/skill-gap", methods=["GET", "POST"])
def skill_gap():
    be, user = backend(), load_current_user()
    p = Params(request.values)
    ctx = {"jobs": [], "job_id": p.text("job_id", 20), "role": p.text("role", 120), "skills": p.text("skills", 1500), "report": None,
           "unknown": [], "errors": [], "unavailable": None, "logged_in": bool(user)}
    try:
        ix = be.index
        cat = be.catalog
        ctx["jobs"] = _jobs(ix)
        profile, unknown = _candidate_profile(p, be, user)
        ctx["unknown"] = unknown
        if request.method == "POST" or request.args.get("job_id") or request.args.get("role"):
            if profile is None:
                ctx["errors"].append("Add at least one recognised skill or a target role. You can also analyse a resume first.")
            job_di = None
            job_id = None
            if ctx["job_id"]:
                try:
                    job_id = int(ctx["job_id"])
                    job_di = ix.doc_index(job_id)
                except (TypeError, ValueError):
                    job_di = None
                if job_di is None:
                    ctx["errors"].append("Choose a job from the list. That job is not in the searchable catalogue.")
            role_text = ctx["role"].strip() or (profile.preferred_role if profile else None)
            if not job_id and not role_text:
                ctx["errors"].append("Choose a target job or enter a target role.")
            if profile and not ctx["errors"]:
                kg, kg_error = be.try_get("kg")
                report = analyze_gap(cat, kg, profile.skills, job_di=job_di if job_id else None, role_text=None if job_id else role_text)
                ctx["report"] = report
                ctx["kg_unavailable"] = bool(kg_error)
                if user and job_id:
                    try:
                        with db_session() as s:
                            s.execute(delete(M.SkillGap).where(M.SkillGap.user_id == user.user_id, M.SkillGap.job_id == job_id))
                            rec_store.save_skill_gaps(s, user.user_id, job_id, report)
                            s.commit()
                    except Exception:
                        log.exception("could not persist skill gap")
                        flash("The analysis is shown, but it could not be saved to your workspace.", "warning")
    except Unavailable as e:
        ctx["unavailable"] = public_reason(e, "Skill-gap analysis")
    except Exception:
        log.exception("skill gap analysis failed")
        ctx["unavailable"] = "Skill-gap analysis could not be completed. Try again later; the site operator can inspect the server log."
    return render_template("skillgap.html", **ctx)
