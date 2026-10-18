"""Accounts: login, registration, the personal workspace, Saved Jobs, the Profile page and account settings (change password, download my data, delete my account)."""
from __future__ import annotations

import json
import math
from typing import Dict, List, Optional, Sequence

from flask import Blueprint, Response, abort, flash, redirect, render_template, request, session, url_for
from sqlalchemy import func, select
from sqlalchemy.orm import Session

from ...db import models as M
from ..auth import (authenticate, change_password, create_account, delete_account, load_current_user, login_required, password_stamp, safe_next, start_session,
                    validate_registration, verify_password)
from ..backend import Unavailable
from ..cards import location_text, with_links
from ..forms import Params
from ..ui import employer
from ._common import backend, db_session, ext, public_reason, record_interaction
from ._profile import LIMITS, export_user_data, load_profile_form, normalize_skills, profile_steps, save_profile_form

bp = Blueprint("account", __name__)

SAVED_PER_PAGE = 10
SAVED_STATUS = {"all": "All saved jobs", "open": "Not marked as applied", "applied": "Marked as applied"}


# ------------------------------------------------------------------------------------------------------------------------ login, registration, logout
@bp.route("/login", methods=["GET", "POST"])
def login():
    if load_current_user():
        return redirect(url_for("account.dashboard"))
    nxt = request.values.get("next", "")
    if request.method == "POST":
        p = Params(request.form)
        ident, password = p.text("identifier", 255), request.form.get("password", "")
        key = (request.remote_addr or "?", ident.lower())
        thr = ext().throttle
        if thr.blocked(key):
            flash("Too many failed attempts. Wait a few minutes and try again.", "error")
            return render_template("login.html", next=nxt, identifier=ident), 429
        with db_session() as s:
            user = authenticate(s, ident, password)
        if user is None:
            thr.hit(key)
            flash("User name or password is not correct.", "error")
            return render_template("login.html", next=nxt, identifier=ident), 401
        thr.clear(key)
        start_session(user.user_id, user.stamp)
        return redirect(safe_next(nxt))
    return render_template("login.html", next=nxt, identifier="")


@bp.route("/register", methods=["GET", "POST"])
def register():
    if not ext().settings.open_registration:
        abort(403, description="Registration is closed on this server.")
    if request.method == "POST":
        f = request.form
        p = Params(f)
        username, email = p.text("username", 40), p.text("email", 254)
        errs = validate_registration(username, email, f.get("password", ""), f.get("confirm", ""))
        if not errs:
            try:
                with db_session() as s:
                    uid = create_account(s, username, email, f["password"])
                    stamp = password_stamp(s.get(M.User, uid).password_hash)
                    s.commit()
            except ValueError as e:
                errs.append(str(e))
            else:
                start_session(uid, stamp)
                flash("Account created. Add your skills and target role on the Profile page.", "info")
                return redirect(url_for("account.dashboard"))
        return render_template("register.html", errors_list=errs, username=username, email=email), 400
    return render_template("register.html", errors_list=[], username="", email="")


@bp.route("/logout", methods=["POST"])
def logout():
    session.clear()
    flash("You are logged out.", "info")
    return redirect(url_for("main.landing"))


# ------------------------------------------------------------------------------------------------------------------------ job lists (saved, applied, viewed)
def _experience(text: Optional[str], lo: Optional[float], hi: Optional[float]) -> Optional[str]:
    if text:
        return text
    if lo is not None and hi is not None:
        return f"{lo:g} to {hi:g} years"
    return None


def job_rows(s: Session, user_id: int, flags: Sequence, *, limit: Optional[int] = None, offset: int = 0) -> List[Dict]:
    """The user's jobs whose interaction row satisfies ``flags``, newest activity first.  Only this user's rows are ever selected (``user_id`` is always part of the filter)."""
    q = (select(M.Job.job_id, M.Job.title, M.Job.city, M.Job.cities, M.Job.country, M.Job.experience_text, M.Job.experience_min, M.Job.experience_max, M.Job.employment_type,
                M.Job.is_fraudulent, M.Company.name, M.Company.name_is_derived, M.JobInteraction.saved, M.JobInteraction.applied, M.JobInteraction.last_event_at)
         .join(M.JobInteraction, M.JobInteraction.job_id == M.Job.job_id).outerjoin(M.Company, M.Company.company_id == M.Job.company_id)
         .where(M.JobInteraction.user_id == user_id, *flags).order_by(M.JobInteraction.last_event_at.desc(), M.Job.job_id).offset(offset))
    if limit is not None:
        q = q.limit(limit)
    return [dict(job_id=int(r[0]), title=r[1], location=location_text(r[2], r[3] or [], r[4]), experience=_experience(r[5], r[6], r[7]), employment=r[8], flagged=bool(r[9]),
                 company=employer(None if r[11] else r[10]), saved=bool(r[12]), applied=bool(r[13]), at=r[14]) for r in s.execute(q)]


def attach_skills(s: Session, rows: List[Dict], mine: set, per_job: int = 6) -> None:
    """Add ``have`` (skills the posting lists that you also list) and ``other`` (further listed skills) to each row, from the posting's own skill rows."""
    ids = [r["job_id"] for r in rows]
    by_job: Dict[int, List[str]] = {}
    if ids:
        for jid, name in s.execute(select(M.JobSkill.job_id, M.Skill.canonical_name).join(M.Skill, M.Skill.skill_id == M.JobSkill.skill_id).where(M.JobSkill.job_id.in_(ids))
                                   .order_by(M.JobSkill.job_id, M.JobSkill.mentions.desc(), M.Skill.canonical_name)):
            by_job.setdefault(int(jid), []).append(name)
    for r in rows:
        names = by_job.get(r["job_id"], [])
        r["have"] = [n for n in names if n in mine][:per_job]
        r["other"] = [n for n in names if n not in mine][: max(0, per_job - len(r["have"]))]


def _count(s: Session, user_id: int, *flags) -> int:
    return int(s.execute(select(func.count()).select_from(M.JobInteraction).where(M.JobInteraction.user_id == user_id, *flags)).scalar_one())


def _back(default: str) -> Response:
    target = Params(request.form).text("next", 300)
    return redirect(target if target.startswith("/") and not target.startswith("//") and "\\" not in target else default)


# ------------------------------------------------------------------------------------------------------------------------ Saved Jobs
@bp.route("/saved")
@login_required
def saved():
    user = load_current_user()
    p = Params(request.args)
    status = p.choice("status", list(SAVED_STATUS), "all")
    want_page = p.integer("page", 1, 100000, 1) or 1
    flags: List = [M.JobInteraction.saved.is_(True)]
    if status == "applied":
        flags.append(M.JobInteraction.applied.is_(True))
    elif status == "open":
        flags.append(M.JobInteraction.applied.is_(False))
    with db_session() as s:
        n_saved = _count(s, user.user_id, M.JobInteraction.saved.is_(True))
        total = _count(s, user.user_id, *flags)
        pages = max(1, math.ceil(total / SAVED_PER_PAGE))
        page = min(want_page, pages)
        rows = job_rows(s, user.user_id, flags, limit=SAVED_PER_PAGE, offset=(page - 1) * SAVED_PER_PAGE)
        mine = {n for (n,) in s.execute(select(M.Skill.canonical_name).join(M.UserSkill, M.UserSkill.skill_id == M.Skill.skill_id).where(M.UserSkill.user_id == user.user_id))}
        attach_skills(s, rows, mine)
    lo = (page - 1) * SAVED_PER_PAGE
    pg = {"items": rows, "page": page, "pages": pages, "total": total, "first": lo + 1 if total else 0, "last": min(total, lo + SAVED_PER_PAGE)}
    pg = with_links(pg, lambda n: url_for("account.saved", status=None if status == "all" else status, page=n if n > 1 else None))
    return render_template("saved.html", pg=pg, status=status, statuses=SAVED_STATUS, n_saved=n_saved, has_skills=bool(mine), here=request.full_path.rstrip("?"), errors_list=p.errors)


def _own_saved_row(s: Session, user_id: int, job_id: int) -> Optional[M.JobInteraction]:
    return s.execute(select(M.JobInteraction).where(M.JobInteraction.user_id == user_id, M.JobInteraction.job_id == job_id, M.JobInteraction.saved.is_(True))).scalar_one_or_none()


@bp.route("/saved/<int:job_id>/remove", methods=["POST"])
@login_required
def saved_remove(job_id: int):
    user = load_current_user()
    with db_session() as s:
        if _own_saved_row(s, user.user_id, job_id) is None:
            flash("That job is not in your saved list.", "info")
        else:
            title = s.get(M.Job, job_id).title
            record_interaction(s, user.user_id, job_id, "unsave")
            s.commit()
            flash(f"Removed “{title}” from your saved jobs.", "info")
    return _back(url_for("account.saved"))


@bp.route("/saved/<int:job_id>/applied", methods=["POST"])
@login_required
def saved_applied(job_id: int):
    """Mark (or unmark) a saved job as applied.  This is the user's own note: Jobintel does not see employers and tracks no further status."""
    user = load_current_user()
    on = request.form.get("value") == "1"
    with db_session() as s:
        if _own_saved_row(s, user.user_id, job_id) is None:
            flash("That job is not in your saved list.", "info")
        else:
            title = s.get(M.Job, job_id).title
            record_interaction(s, user.user_id, job_id, "apply" if on else "unapply")
            s.commit()
            flash(f"Marked “{title}” as applied." if on else f"Removed the applied mark from “{title}”.", "info")
    return _back(url_for("account.saved"))


# ------------------------------------------------------------------------------------------------------------------------ Profile
def _form_from_stored(d: Dict) -> Dict:
    return dict(full_name=d["full_name"], education=d["education"], summary=d["summary"], role=d["role"], location=d["location"], skills_text=", ".join(d["skills"]),
                years="" if d["years"] is None else f"{d['years']:g}")


@bp.route("/profile", methods=["GET", "POST"])
@login_required
def profile():
    user = load_current_user()
    be = backend()
    errors: List[str] = []
    with db_session() as s:
        stored = load_profile_form(s, user.user_id)
    form = _form_from_stored(stored)
    status = 200
    if request.method == "POST":
        p = Params(request.form)
        vals = dict(full_name=p.text("full_name", LIMITS["full_name"]), education=p.text("education", LIMITS["education"]), summary=p.text("summary", LIMITS["summary"]),
                    role=p.text("role", LIMITS["role"]), location=p.text("location", LIMITS["location"]))
        years = p.number("years", 0, 60, label="Years of experience")
        skills_in = p.csv("skills", 60, 60)
        form = dict(vals, skills_text=request.form.get("skills", "")[:1500], years=str(request.form.get("years", ""))[:10])
        weights, unknown, skills_checked = {}, [], True
        try:
            weights, unknown = normalize_skills(be.engine.lex, skills_in)
        except Unavailable as e:
            public_reason(e, "The skill list")
            skills_checked = False
        if p.errors:
            errors, status = list(p.errors), 400
        else:
            with db_session() as s:
                if skills_checked:
                    res = save_profile_form(s, user.user_id, years=years, weights=weights, **vals)
                else:                                                  # the skill list is not loaded: store the rest, leave the stored skills as they are
                    keep = {n: 1.0 for n in stored["skills"]}
                    res = save_profile_form(s, user.user_id, years=years, weights=keep, **vals)
                s.commit()
            msg = "Profile saved."
            if skills_checked:
                msg = f"Profile saved with {res['saved_skills']} skill{'s' if res['saved_skills'] != 1 else ''}."
                bad = unknown + res["not_in_skills_table"]
                if bad:
                    msg += f" Not recognised as skills, so not stored: {', '.join(bad)}."
            else:
                msg += " Your skills were not changed because the skill list could not be loaded right now; try again in a minute."
            flash(msg, "info")
            return redirect(url_for("account.profile"))
    return render_template("profile.html", form=form, stored=stored, steps=profile_steps(stored), errors_list=errors, limits=LIMITS), status


bp.add_url_rule("/dashboard/profile", endpoint="profile_legacy", view_func=profile, methods=["POST"])      # the address the profile form had before the Profile page existed


# ------------------------------------------------------------------------------------------------------------------------ account settings
def _account_page(errors: Optional[List[str]] = None, status: int = 200, **ctx):
    user = load_current_user()
    with db_session() as s:
        u = s.get(M.User, user.user_id)
        since = u.created_at
    return render_template("account.html", since=since, errors_list=errors or [], **ctx), status


@bp.route("/account")
@login_required
def account():
    return _account_page()


def _throttled(user_id: int, what: str):
    key = (request.remote_addr or "?", f"{what}:{user_id}")
    return key, ext().throttle


@bp.route("/account/password", methods=["POST"])
@login_required
def account_password():
    user = load_current_user()
    f = request.form
    key, thr = _throttled(user.user_id, "password")
    if thr.blocked(key):
        return _account_page(["Too many wrong attempts. Wait a few minutes and try again."], 429, open_section="password")
    with db_session() as s:
        errs = change_password(s, user.user_id, f.get("current", ""), f.get("new", ""), f.get("confirm", ""))
        if not errs:
            stamp = password_stamp(s.get(M.User, user.user_id).password_hash)
            s.commit()
        else:
            s.rollback()
    if errs:
        if errs == ["Your current password is not correct."]:
            thr.hit(key)
        return _account_page(errs, 400, open_section="password")
    thr.clear(key)
    session["pwv"] = stamp                                     # this browser stays logged in; every other one is logged out at its next request
    flash("Password changed. Other browsers where you were logged in have been logged out.", "info")
    return redirect(url_for("account.account"))


@bp.route("/account/export")
@login_required
def account_export():
    user = load_current_user()
    with db_session() as s:
        data = export_user_data(s, user.user_id)
    resp = Response(json.dumps(data, indent=2, ensure_ascii=False, default=str), mimetype="application/json")
    resp.headers["Content-Disposition"] = 'attachment; filename="jobintel-my-data.json"'
    resp.headers["Cache-Control"] = "no-store"
    return resp


@bp.route("/account/delete", methods=["POST"])
@login_required
def account_delete():
    user = load_current_user()
    f = request.form
    key, thr = _throttled(user.user_id, "delete")
    if thr.blocked(key):
        return _account_page(["Too many wrong attempts. Wait a few minutes and try again."], 429, open_section="delete")
    errs: List[str] = []
    if not Params(f).flag("understood"):
        errs.append("Tick the box to confirm that you understand this cannot be undone.")
    with db_session() as s:
        if not errs and not verify_password(s, user.user_id, f.get("password", "")):
            thr.hit(key)
            errs.append("Your password is not correct.")
        if errs:
            return _account_page(errs, 400, open_section="delete")
        delete_account(s, user.user_id)
        s.commit()
    session.clear()
    flash("Your account and the data stored with it were deleted.", "info")
    return redirect(url_for("main.landing"))


# ------------------------------------------------------------------------------------------------------------------------ the workspace
@bp.route("/dashboard")
@login_required
def dashboard():
    user = load_current_user()
    with db_session() as s:
        data = load_profile_form(s, user.user_id)
        steps = profile_steps(data)
        saved_flag, applied_flag, viewed_flag = M.JobInteraction.saved.is_(True), M.JobInteraction.applied.is_(True), M.JobInteraction.viewed.is_(True)
        counts = dict(saved=_count(s, user.user_id, saved_flag), applied=_count(s, user.user_id, applied_flag), viewed=_count(s, user.user_id, viewed_flag))
        saved_jobs = job_rows(s, user.user_id, [saved_flag], limit=5)
        applied_jobs = job_rows(s, user.user_id, [applied_flag], limit=5)
        viewed_jobs = job_rows(s, user.user_id, [viewed_flag], limit=6)
        searches = [dict(q=r[0], n=r[1], at=r[2]) for r in s.execute(select(M.Search.query_text, M.Search.n_results, M.Search.created_at)
                                                                    .where(M.Search.user_id == user.user_id).order_by(M.Search.created_at.desc()).limit(6))]
        last = s.execute(select(func.max(M.Recommendation.created_at)).where(M.Recommendation.user_id == user.user_id)).scalar_one()
        recs = []
        if last is not None:
            # "the last run" = the rows sharing the newest timestamp; compared inside the database so that the timestamp is never round-tripped through Python (SQLite and MySQL format it differently)
            newest = select(func.max(M.Recommendation.created_at)).where(M.Recommendation.user_id == user.user_id).scalar_subquery()
            q = (select(M.Recommendation.position, M.Job.job_id, M.Job.title, M.Company.name, M.Company.name_is_derived).join(M.Job, M.Job.job_id == M.Recommendation.job_id)
                 .outerjoin(M.Company, M.Company.company_id == M.Job.company_id).where(M.Recommendation.user_id == user.user_id, M.Recommendation.created_at == newest)
                 .order_by(M.Recommendation.position).limit(5))
            recs = [dict(rank=r[0], job_id=r[1], title=r[2], company=employer(None if r[4] else r[3])) for r in s.execute(q)]
        gap_jobs: Dict[int, Dict] = {}
        for jid, title, st, n in s.execute(select(M.SkillGap.job_id, M.Job.title, M.SkillGap.status, func.count()).join(M.Job, M.Job.job_id == M.SkillGap.job_id)
                                           .where(M.SkillGap.user_id == user.user_id).group_by(M.SkillGap.job_id, M.Job.title, M.SkillGap.status)):
            gap_jobs.setdefault(jid, {"job_id": jid, "title": title, "matching": 0, "missing": 0})[st] = int(n)
    return render_template("dashboard.html", data=data, steps=steps, n_done=sum(1 for x in steps if x["done"]), counts=counts, saved=saved_jobs, applied=applied_jobs, viewed=viewed_jobs,
                           searches=searches, recs=recs, rec_time=last, gap_jobs=list(gap_jobs.values())[:5])
