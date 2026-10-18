"""Find Jobs (search, filters, sorting, pagination) and the job-details page.

The ranking is the hybrid retrieval model from Phase 3.  The page fetches the best ``POOL`` matches once and pages through them ten at a time; sorting by experience re-orders that
pool, which the page says.  Model choice, ranking weights and cluster routing are still accepted as URL parameters (and by /api/search) but the page does not offer them:
a job seeker has no use for them.  Per-signal scores stay in the engine's output and the search log; the card shows one plain sentence instead (see cards.py).
"""
from __future__ import annotations

import itertools
import logging
import math
import re
from typing import Dict, List, Optional, Tuple

from flask import Blueprint, abort, redirect, render_template, request, url_for
from sqlalchemy import select

from ...db import models as M
from ...ir import passages as PS
from ...ir.engine import MODELS
from ..auth import load_current_user
from ..backend import Unavailable
from ..cards import PER_PAGE, SORTS, build_card, paginate, sort_cards, with_links
from ..forms import Params
from ..signals import SEARCH_SIGNALS
from ..ui import AREAS, area_label, employer, safe_url, source_label
from ._common import backend, db_session, ext, interaction_state, public_reason, record_interaction
from ._profile import session_resume, saved_profile

bp = Blueprint("search", __name__)
log = logging.getLogger("jobintel.web")

POOL = 100                                                     # best matches fetched per search; the pages are cut from these
K_CHOICES = [10, 20, 50]                                       # accepted by the JSON API
EXPERIENCE_BANDS = {"0-1": ("Fresher, up to 1 year", 0.0, 1.0), "1-3": ("1 to 3 years", 1.0, 3.0), "3-6": ("3 to 6 years", 3.0, 6.0), "6-10": ("6 to 10 years", 6.0, 10.0),
                    "10-": ("10 years or more", 10.0, None)}
SUGGESTIONS = ["data analyst", "python developer", "machine learning engineer", "accountant", "customer service", "react frontend developer"]


def _location(raw: str, countries: List[str]) -> Tuple[Optional[str], Optional[str]]:
    """What a person types into 'Location' -> (city, country).  'Pune, India' is the city Pune; a bare country name is a country filter."""
    text = raw.split(",")[0].strip()
    if not text:
        return None, None
    if text.lower() in {c.lower() for c in countries}:
        return None, text
    return text, None


def parse_search(p: Params, sources_allowed: List[str], countries: Optional[List[str]] = None) -> Dict:
    q = p.text("q", 200)
    filters: Dict = {}
    city, country = _location(p.text("city", 80), countries or [])
    if city:
        filters["city"] = city
    country = p.text("country", 80) or country
    if country:
        filters["country"] = country
    area = p.choice("area", list(AREAS), None)
    if area:
        filters["cluster"] = area
    for key in ("role", "employment_type"):
        v = p.text(key, 80)
        if v:
            filters[key] = v
    skills = p.csv("skills", 10, 40)
    if skills:
        filters["skills"] = skills
    band = p.choice("exp", list(EXPERIENCE_BANDS), None)
    lo = p.number("min_exp", 0, 60, label="Minimum experience")
    hi = p.number("max_exp", 0, 60, label="Maximum experience")
    if lo is not None and hi is not None and lo > hi:
        p.errors.append("Minimum experience is larger than maximum experience")
        lo = hi = None
    if band and lo is None and hi is None:
        _, lo, hi = EXPERIENCE_BANDS[band]
    if lo is not None:
        filters["min_experience"] = lo
    if hi is not None:
        filters["max_experience"] = hi
    sal = p.number("min_salary", 0, 1e9, label="Minimum salary")
    if sal is not None:
        filters["min_salary"] = sal
    src = p.many("source", sources_allowed)
    if src:
        filters["sources"] = src
    weights = {s: v for s in SEARCH_SIGNALS if (v := p.number(f"w_{s}", 0, 1, label=f"weight {s}")) is not None}
    return {"q": q, "filters": filters, "model": p.choice("model", MODELS, "hybrid"), "k": p.choice("k", K_CHOICES, 10), "route": p.flag("route"), "keep_dupes": p.flag("dupes"),
            "compare": p.flag("compare"), "weights": weights, "sort": p.choice("sort", list(SORTS), "relevance"), "page": p.integer("page", 1, 1000, 1) or 1,
            "exp": band if band in EXPERIENCE_BANDS else None}


def log_search(be, user_id: Optional[int], q: str, filters: Dict, model: str, resp) -> Optional[int]:
    """Store the search and its displayed results (searches / search_results).  Best effort: a database problem never breaks the search itself."""
    try:
        with db_session() as s:
            row = M.Search(user_id=user_id, query_text=q[:1000], parsed_query=resp.parsed.to_dict() if resp.parsed else None, filters=filters or None, ranking_model=model,
                           n_results=len(resp.results), latency_ms=resp.timings_ms.get("total"))
            s.add(row)
            s.flush()
            for x in resp.results:
                sg = x.signals or {}
                s.add(M.SearchResult(search_id=row.search_id, job_id=x.job_id, position=x.rank, lm_score=sg.get("lm"), dfr_score=sg.get("dfr"), passage_score=sg.get("passage"),
                                     pagerank_score=sg.get("pagerank"), skill_score=sg.get("skill"), location_score=sg.get("location"), experience_score=sg.get("experience"),
                                     final_score=float(x.score) if math.isfinite(x.score) else 0.0, explanation={"reasons": x.reasons, "raw": {k: (v if v == v else None) for k, v in x.raw.items()}}))
            sid = int(row.search_id)
            s.commit()
            return sid
    except Exception:                                    # noqa: BLE001
        log.exception("could not log the search")
        return None


def overlap_table(rows: List[Dict], models: List[str], k: int) -> List[Dict]:
    """Share of the top-k jobs two models have in common (Jaccard), from the comparison rows.  Used by the evaluation tooling; not shown on the job-seeker pages."""
    sets = {m: {r["job_id"] for r in rows if r["ranks"].get(m, 10 ** 9) <= k} for m in models}
    out = []
    for a, b in itertools.combinations(models, 2):
        u = sets[a] | sets[b]
        out.append({"a": a, "b": b, "jaccard": (len(sets[a] & sets[b]) / len(u)) if u else None})
    return out


_UNKNOWN = re.compile(r"^terms not found in the index \(ignored\): \[(.*)\]$")


def friendly_warnings(warnings: List[str]) -> List[str]:
    """The engine's notes in words for a visitor (no list syntax, no talk of 'the index')."""
    out: List[str] = []
    for w in warnings:
        m = _UNKNOWN.match(w)
        if m:
            words = [x.strip().strip("'\"") for x in m.group(1).split(",") if x.strip()]
            out.append("These words do not appear in any posting, so they were ignored: " + ", ".join(words) + ".")
        elif w.startswith("none of the query terms occur"):
            out.append("None of those words appear in any posting. Try a more general word or a job title.")
        elif w.startswith("no text terms in the query"):
            continue                                                        # shown as the page's own 'no search words' note
        elif w.startswith("query has no searchable words"):
            out.append("There are no searchable words in that. Try a job title or a skill.")
        else:
            out.append(w)
    return out


def _args_for(prm: Dict, **override) -> Dict:
    """URL parameters that rebuild this search (used for the pager and the 'remove filter' links); empty values are left out."""
    f = prm["filters"]
    args: Dict = {"q": prm["q"] or None, "city": prm.get("city_raw") or None, "country": f.get("country"), "area": f.get("cluster"), "employment_type": f.get("employment_type"),
                  "role": f.get("role"), "skills": ", ".join(f["skills"]) if f.get("skills") else None, "exp": prm.get("exp"), "source": f.get("sources"),
                  "sort": prm["sort"] if prm["sort"] != "relevance" else None, "min_exp": None, "max_exp": None}
    if not prm.get("exp"):
        args["min_exp"], args["max_exp"] = f.get("min_experience"), f.get("max_experience")
    args.update(override)
    return {k: v for k, v in args.items() if v not in (None, "", [])}


def _active_filters(prm: Dict) -> List[Dict]:
    f, out = prm["filters"], []

    def add(label: str, *keys: str) -> None:
        out.append({"label": label, "url": url_for("search.search", **_args_for(prm, page=None, **{k: None for k in keys}))})
    if prm.get("city_raw") and "city" in f:
        add(f"Location: {f['city']}", "city")
    if f.get("country"):
        add(f"Country: {f['country'].title()}", "country")
    if f.get("cluster"):
        add(f"Area: {area_label(f['cluster'])}", "area")
    if f.get("employment_type"):
        add(f"Type: {f['employment_type']}", "employment_type")
    if f.get("role"):
        add(f"Title contains: {f['role']}", "role")
    if f.get("skills"):
        add("Lists: " + ", ".join(f["skills"]), "skills")
    if prm.get("exp"):
        add(f"Experience: {EXPERIENCE_BANDS[prm['exp']][0]}", "exp")
    elif f.get("min_experience") is not None or f.get("max_experience") is not None:
        add("Experience range", "min_exp", "max_exp")
    if f.get("sources"):
        add("Source: " + ", ".join(source_label(s) for s in f["sources"]), "source")
    return out


@bp.route("/jobs")
def search():
    be = backend()
    p = Params(request.args)
    facets, unavail = None, None
    try:
        facets = be.facets()
    except Unavailable as e:
        unavail = public_reason(e, "Job search")
    prm = parse_search(p, [s for s, _ in facets["sources"]] if facets else ["emscad", "linkedin", "naukri"], [c for c, _ in facets["countries"]] if facets else [])
    loc_text = p.text("city", 80)
    prm["city_raw"] = loc_text if "city" in prm["filters"] else ""      # only a real city rebuilds the 'city' parameter; a country typed there becomes the country filter
    q, filters = prm["q"], prm["filters"]
    searched = bool(q or filters)
    ctx = dict(prm=prm, loc_text=loc_text, facets=facets, areas=AREAS, bands=EXPERIENCE_BANDS, sorts=SORTS, searched=searched, unavail=unavail, problems=list(p.errors), page=None, states={}, args={}, active=[],
               notes=[], total_matches=0, ms=None, browse=False, suggestions=SUGGESTIONS, area_counts=dict(facets["areas"]) if facets else {}, no_results=False, pool=POOL)
    if not searched or unavail:
        return render_template("search.html", **ctx)
    user = load_current_user()
    try:
        eng, ix = be.engine, be.index
        resp = eng.search(q or "jobs", k=POOL, model=prm["model"], filters=filters, weights=prm["weights"] if prm["model"] == "hybrid" else None, explain=True,
                          collapse_duplicates=not prm["keep_dupes"], cluster_routing=prm["route"])
    except Unavailable as e:
        ctx["unavail"] = public_reason(e, "Job search")
        return render_template("search.html", **ctx)
    if resp.error:
        ctx["problems"].append(resp.error)
        return render_template("search.html", **ctx)
    browse = bool(resp.parsed is not None and not resp.parsed.terms)
    cards = sort_cards([build_card(x, ix, browse=browse) for x in resp.results], prm["sort"])
    pg = with_links(paginate(cards, prm["page"], PER_PAGE), lambda n: url_for("search.search", **_args_for(prm, page=n if n > 1 else None)))
    ctx.update(page=pg, total_matches=resp.n_candidates, ms=resp.timings_ms.get("total"), browse=browse, notes=friendly_warnings(resp.warnings), active=_active_filters(prm),
               args=_args_for(prm), no_results=not cards)
    if ext().settings.log_searches and pg["page"] == 1 and prm["sort"] == "relevance":
        log_search(be, user.user_id if user else None, q or "jobs", filters, prm["model"], resp)
    if user and pg["items"]:
        with db_session() as s:
            ctx["states"] = interaction_state(s, user.user_id, [c.job_id for c in pg["items"]])
    return render_template("search.html", **ctx)


@bp.route("/search")
def legacy_search():
    """The old address of the search page; links and bookmarks keep working."""
    return redirect(url_for("search.search", **request.args.to_dict(flat=False)), code=301)


# ---------------------------------------------------------------------------------------------------------------------------- job details
def _fmt_salary(j) -> Tuple[Optional[str], Optional[str]]:
    """(text, note).  The note says when the posting gives numbers without a currency, which most of the salary data does."""
    if j.salary_min is None and j.salary_max is None:
        return None, None
    lo, hi = j.salary_min, j.salary_max
    rng = f"{lo:,.0f}" if hi is None or lo == hi else f"{lo:,.0f} to {hi:,.0f}" if lo is not None else f"up to {hi:,.0f}"
    text = f"{j.salary_currency or ''} {rng}{' per ' + j.salary_period if j.salary_period else ''}".strip()
    return text, (None if j.salary_currency else "currency not stated")


def fit_for(job_skills: List[str], job_exp: Tuple[Optional[float], Optional[float]], profile) -> Optional[Dict]:
    """How a visitor's own profile lines up with this posting: skills they list / lack, and whether their experience sits inside the range.  Only real, stated facts are compared;
    returns None when there is no profile or the posting lists no skills and no experience."""
    if profile is None:
        return None
    mine = {s.lower() for s in profile.skills}
    have = [s for s in job_skills if s.lower() in mine]
    lack = [s for s in job_skills if s.lower() not in mine]
    exp_note = None
    lo, hi = job_exp
    yrs = profile.experience_years
    if yrs is not None and lo is not None:
        if yrs < lo:
            exp_note = f"The posting asks for {lo:g}{'+' if hi is None else ' to ' + format(hi, 'g')} years; your profile says {yrs:g}."
        elif hi is not None and yrs > hi:
            exp_note = f"The posting asks for {lo:g} to {hi:g} years; your profile says {yrs:g}, which is above that range."
        else:
            exp_note = f"Your {yrs:g} years of experience fit the {lo:g}{'+' if hi is None else ' to ' + format(hi, 'g')} years asked for."
    if not job_skills and exp_note is None:
        return None
    return {"have": have, "lack": lack, "n_skills": len(job_skills), "experience": exp_note}


@bp.route("/job/<int:job_id>")
def job_detail(job_id: int):
    be = backend()
    q = Params(request.args).text("q", 200)
    ref = request.args.get("ref")
    user = load_current_user()
    with db_session() as s:
        job = s.get(M.Job, job_id)
        if job is None:
            abort(404)
        company = s.get(M.Company, job.company_id) if job.company_id else None
        skills = s.execute(select(M.Skill.canonical_name, M.Skill.skill_type, M.JobSkill.origin, M.JobSkill.mentions).join(M.JobSkill, M.JobSkill.skill_id == M.Skill.skill_id)
                           .where(M.JobSkill.job_id == job_id).order_by(M.JobSkill.mentions.desc(), M.Skill.canonical_name)).all()
        dup_of = s.get(M.Job, job.duplicate_of) if job.duplicate_of else None
        state: Dict = {}
        profile = None
        if user:
            record_interaction(s, user.user_id, job_id, "click" if ref == "search" else "view")
            state = interaction_state(s, user.user_id, [job_id]).get(job_id, {})
            s.commit()
            profile = saved_profile(s, user.user_id)
        text = job.description_clean or ""
        salary, salary_note = _fmt_salary(job)
        info = dict(job_id=job.job_id, title=job.title, company=employer(company.name if company and not company.name_is_derived else None), city=job.city, cities=job.cities or [],
                    country=job.country, location_raw=job.location_raw, employment=job.employment_type, seniority=job.seniority, category=job.category, industry=job.industry,
                    education=job.education,
                    experience=job.experience_text or (f"{job.experience_min:g} to {job.experience_max:g} years" if job.experience_min is not None and job.experience_max is not None else None),
                    experience_source=job.experience_source, exp_range=(job.experience_min, job.experience_max), salary=salary, salary_note=salary_note, posting_date=job.posting_date,
                    apply_url=safe_url(job.url), text_quality=job.text_quality, fraudulent=bool(job.is_fraudulent), cluster=job.cluster, n_tokens=job.n_tokens)
        source = s.execute(select(M.CrawlerSource.name).where(M.CrawlerSource.source_id == job.source_id)).scalar_one_or_none()
        dup_info = {"job_id": dup_of.job_id, "title": dup_of.title} if dup_of else None
    info["source"] = source
    fit_source = "your saved profile" if profile is not None else None
    sr = session_resume()                                       # a resume analysed in this browser session (the Resume page) takes precedence over the saved profile
    if sr is not None:
        try:
            from ...rec.profile import UserProfile
            profile, fit_source = UserProfile.from_resume(sr["profile"]), "the resume you analysed"
        except Exception:                                       # noqa: BLE001 - a profile that cannot be read is simply not used
            log.exception("could not build a profile from the session resume")
    # query-term highlighting with the same stemming the index uses
    body_html = None
    if q and text:
        try:
            stems = set(be.engine.processor.parse(q).terms)
            if stems:
                from markupsafe import Markup
                body_html = Markup("".join("<p>" + "<br>".join(PS.highlight(line, stems) for line in blk.splitlines()) + "</p>" for blk in text.split("\n\n") if blk.strip()))
        except Unavailable:
            pass
    similar, similar_note = [], None
    try:
        ix, cat = be.index, be.catalog
        di = ix.doc_index(job_id)
        if di is None:
            similar_note = "This posting is not in the search index (it repeats another posting, is flagged, or has no text), so related jobs are not worked out for it."
        else:
            for d in cat.similar_jobs(di, 6):
                m = cat.meta(int(d))
                similar.append({"job_id": int(ix.job_ids[int(d)]), "title": m.get("title"), "company": employer(m.get("company")), "city": m.get("city")})
    except Unavailable as e:
        similar_note = public_reason(e, "Related jobs")
    fit = fit_for([n for n, *_ in skills], info["exp_range"], profile)
    return render_template("job.html", job=info, skills=[{"name": n, "type": t, "origin": o, "mentions": m} for n, t, o, m in skills], text=text, body_html=body_html, q=q, state=state,
                           similar=similar, similar_note=similar_note, duplicate_of=dup_info, fit=fit, fit_source=fit_source, has_profile=profile is not None,
                           back=request.referrer if request.referrer and request.host in (request.referrer or "") else None)
