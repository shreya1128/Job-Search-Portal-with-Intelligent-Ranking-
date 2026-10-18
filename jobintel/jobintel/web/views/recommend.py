"""Recommendations blueprint — the 11-signal hybrid recommender (Phase 6 → Phase 8, live).

Routes: GET /recommendations (form + results), POST /recommendations/save.
The DL signal uses DLSkillSimilarity v2 (denoising autoencoder, 3,887/3,951 docs covered).
KG and CF signals fall back gracefully when unavailable (cold-start safe)."""
from __future__ import annotations

import logging
from typing import Dict, List, Optional, Tuple

from flask import Blueprint, abort, flash, redirect, render_template, request, session, url_for

from ...rec import store as rec_store
from ...rec.dlsim import DLSimilarity
from ...rec.hybrid import DEFAULT_REC_WEIGHTS, HybridRecommender
from ...rec.profile import UserProfile
from ..auth import load_current_user, login_required
from ..backend import Unavailable
from ..forms import Params
from ..signals import REC_SIGNALS, contributions
from ._common import backend, db_session, ext, interaction_state
from ._profile import normalize_skills, saved_profile, session_resume

bp = Blueprint("recommend", __name__)
log = logging.getLogger("jobintel.web")
K_CHOICES = [5, 10, 20]


def build_profile(p: Params, be, user) -> Tuple[Optional[UserProfile], Dict]:
    """Profile for this request + what it was built from.  Form values win over the base profile (the form is pre-filled from it, so editing a field overrides it)."""
    src = p.choice("src", ["auto", "form", "resume", "saved"], "auto")
    meta: Dict = {"source": None, "notes": [], "unknown_skills": [], "base_skills": []}
    base: Optional[UserProfile] = None
    if src in ("auto", "resume"):
        sr = session_resume()
        if sr is not None:
            base = UserProfile.from_resume(sr["profile"])
            meta["source"] = f"the resume you analysed ({sr['filename']})"
    if base is None and src in ("auto", "saved") and user is not None:
        with db_session() as s:
            base = saved_profile(s, user.user_id)
        if base is not None:
            meta["source"] = "your saved profile"
    if base is not None:
        meta["base_skills"] = sorted(base.skills, key=lambda k: -base.skills[k])
    skills_in = p.csv("skills", 60, 60)
    role, cities, years = p.text("role", 120), p.csv("city", 3, 80), p.number("years", 0, 60, label="Years of experience")
    employment = p.text("employment", 40)
    if skills_in:
        weights, unknown = normalize_skills(be.engine.lex, skills_in)
        meta["unknown_skills"] = unknown
        # skills that came from the base profile keep their resume weights; typed ones count fully
        weights = {k: (base.skills[k] if base is not None and k in base.skills else v) for k, v in weights.items()}
        meta["source"] = (meta["source"] + ", edited in the form") if meta["source"] else "the form"
    else:
        weights = dict(base.skills) if base is not None else {}
    if not (weights or role or (base is not None)):
        return None, meta
    prof = UserProfile(skills=weights, experience_years=years if years is not None else (base.experience_years if base else None),
                       preferred_role=role or (base.preferred_role if base else None), locations=cities or (base.locations if base else []),
                       categories=base.categories if base else [], employment=employment or None, interactions=dict(base.interactions) if base else {},
                       document=base.document if base else None)
    if user is not None and not prof.interactions:
        from ._common import user_interactions
        with db_session() as s:
            prof.interactions = user_interactions(s, user.user_id)
    return prof, meta


def recommender(be, weights: Dict[str, float]) -> Tuple[HybridRecommender, Dict[str, str]]:
    """The hybrid recommender wired to every part that is available; the reasons for the parts that are not."""
    reasons: Dict[str, str] = {}
    kg, r = be.try_get("kg")
    if r:
        reasons["knowledge"] = r
    cf, r = be.try_get("cf")
    if r:
        reasons["collaborative"] = r
    dl, r = be.try_get("dl")
    if r:
        reasons["dl"] = r
    return HybridRecommender(be.engine, be.catalog, kg, cf=cf, dl=dl if dl is not None else DLSimilarity(reason=reasons.get("dl")), weights=weights), reasons


@bp.route("/recommendations")
def recommendations():
    be = backend()
    user = load_current_user()
    p = Params(request.args)
    ctx = dict(k_choices=K_CHOICES, rec_signals=REC_SIGNALS, default_weights=DEFAULT_REC_WEIGHTS, weights_set={}, resp=None, prof=None, meta=None, problems=[], unavail=None,
               contribs={}, states={}, token=None, form={}, has_resume=session_resume() is not None, has_saved=False, cf_info=None, run=False)
    try:
        be.engine
    except Unavailable as e:
        ctx["unavail"] = str(e)
        ctx["form"] = {"role": "", "skills": "", "city": "", "years": "", "employment": "", "strict": False, "k": 10, "src": "auto"}
        return render_template("recommend.html", **ctx)
    prof, meta = build_profile(p, be, user)
    ctx["has_saved"] = user is not None
    ctx["has_saved_profile"] = False
    if user is not None:
        try:
            with db_session() as s:
                ctx["has_saved_profile"] = saved_profile(s, user.user_id) is not None
        except Exception:
            log.exception("could not inspect saved profile")
    ctx["form"] = {"role": p.text("role", 120) or (prof.preferred_role if prof else "") or "", "skills": p.text("skills", 1500) or ", ".join(meta["base_skills"]),
                   "city": p.text("city", 240) or ", ".join(prof.locations if prof else []), "years": p.text("years", 10) or ("" if not prof or prof.experience_years is None else f"{prof.experience_years:g}"),
                   "employment": p.text("employment", 40), "strict": p.flag("strict"), "k": p.choice("k", K_CHOICES, 10), "src": p.choice("src", ["auto", "form", "resume", "saved"], "auto")}
    ctx["meta"], ctx["prof"] = meta, prof
    ctx["weights_set"] = {s: v for s in DEFAULT_REC_WEIGHTS if (v := p.number(f"w_{s}", 0, 1, label=f"weight {s}")) is not None}
    run = p.flag("run") or p.choice("src", ["auto", "form", "resume", "saved"], None) in ("resume", "saved")
    ctx["run"] = run
    if run and prof is None:
        p.errors.append("Add some skills or a role, or analyse a resume first.")
    ctx["problems"] = list(p.errors)
    if run and prof is not None and not p.errors:
        weights = {**DEFAULT_REC_WEIGHTS, **ctx["weights_set"]}
        if not any(weights.values()):
            ctx["problems"].append("At least one weight must be above zero.")
        else:
            try:
                rec, reasons = recommender(be, weights)
                if rec.dl.available:
                    with be.tf_lock:
                        resp = rec.recommend(prof, k=ctx["form"]["k"], strict_location=ctx["form"]["strict"])
                else:
                    resp = rec.recommend(prof, k=ctx["form"]["k"], strict_location=ctx["form"]["strict"])
                for sig, why in reasons.items():                      # name the real reason a part is missing instead of the recommender's generic one
                    if sig in resp.inactive and resp.inactive[sig] in ("no collaborative model fitted", "no knowledge graph loaded", "DL similarity not available", "") or resp.inactive.get(sig, "").startswith("no encoder"):
                        resp.inactive[sig] = why
                ctx["resp"] = resp
                ctx["contribs"] = {x.job_id: contributions(x.signals, resp.weights_effective) for x in resp.results}
                ctx["cf_info"] = be.cf_info
                if user and resp.results:
                    with db_session() as s:
                        ctx["states"] = interaction_state(s, user.user_id, [x.job_id for x in resp.results])
                if resp.results and user:
                    ctx["token"] = ext().store.put({"resp": resp, "user_id": user.user_id})
            except Unavailable as e:
                ctx["unavail"] = str(e)
    return render_template("recommend.html", **ctx)


@bp.route("/recommendations/save", methods=["POST"])
@login_required
def save():
    user = load_current_user()
    entry = ext().store.get(request.form.get("token"))
    if not entry or entry["user_id"] != user.user_id:
        flash("These recommendations have expired. Run them again before saving.", "error")
        return redirect(url_for("recommend.recommendations"))
    with db_session() as s:
        n = rec_store.save_recommendations(s, user.user_id, entry["resp"])
        s.commit()
    ext().store.drop(request.form.get("token"))
    flash(f"Saved {n} recommendations to your dashboard.", "info")
    return redirect(url_for("account.dashboard"))
