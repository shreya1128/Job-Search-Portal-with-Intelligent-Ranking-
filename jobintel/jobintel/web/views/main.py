"""Home, About, Privacy, Terms and the system-status page."""
from __future__ import annotations

import importlib.metadata
import platform
from typing import Dict, List

import flask
from flask import Blueprint, render_template

from ..ui import AREAS
from ._common import backend, ext

bp = Blueprint("main", __name__)

# The home page is kept short on purpose (search first, six job areas, three example postings).  The skills-in-demand bars (_demand, from data/analytics/market.json) and the popular
# roles (Backend.popular_roles) are computed and kept for the Career Insights page (stage 4); they are not shown on Home.
DEMAND_SHOWN = 8
HOME_AREAS = 6
HOME_EXAMPLES = 3


def _demand(market: Dict, n: int = DEMAND_SHOWN) -> List[Dict]:
    """The skills most often listed by postings, from the stored MapReduce result (technical skills and tools)."""
    merged: Dict[str, int] = {}
    for typ in ("technical", "tool"):
        for name, c in market.get("top_skills_by_type", {}).get(typ, []):
            merged[name] = merged.get(name, 0) + int(c)
    rows = sorted(merged.items(), key=lambda x: (-x[1], x[0]))[:n]
    total = max(1, int(market.get("total_jobs") or 1))
    top = rows[0][1] if rows else 1
    return [{"skill": k, "jobs": v, "share": v / total, "width": 100 * v / top} for k, v in rows]


@bp.route("/")
def landing():
    be = backend()
    areas, examples = [], []
    try:
        areas = [(k, n) for k, n in be.area_counts() if k in AREAS and k != "other"][:HOME_AREAS]        # the biggest few; Find Jobs lists them all
        examples = be.example_jobs()[:HOME_EXAMPLES]
    except Exception:                                                         # noqa: BLE001 - the home page still renders without the database facts
        flask.current_app.logger.warning("home page facts unavailable", exc_info=True)
    return render_template("landing.html", areas=areas, examples=examples)


@bp.route("/about")
def about():
    sources = []
    try:
        from sqlalchemy import func, select
        from sqlalchemy.orm import Session
        from ...db import models as M
        with Session(backend().db) as s:
            sources = [(str(n), int(c)) for n, c in s.execute(select(M.CrawlerSource.name, func.count(M.Job.job_id)).join(M.Job, M.Job.source_id == M.CrawlerSource.source_id)
                                                           .where(M.Job.duplicate_of.is_(None), M.Job.is_fraudulent.is_(False)).group_by(M.CrawlerSource.name)
                                                           .order_by(func.count(M.Job.job_id).desc())).all()]
    except Exception:                                                         # noqa: BLE001
        flask.current_app.logger.warning("about page: source counts unavailable", exc_info=True)
    return render_template("about.html", sources=sources)


@bp.route("/privacy")
def privacy():
    return render_template("privacy.html")


@bp.route("/terms")
def terms():
    return render_template("terms.html")


@bp.route("/health")
def health():
    be = backend()
    st = be.status()
    files = []
    d = ext().settings.data_dir
    for label, rel in (("Search index", "index/job_index.pkl"), ("Job text embeddings", "dl/embeddings_text.npz"), ("Job skill embeddings", "dl/embeddings_skills.npz"),
                       ("Classifier (best LSTM)", "dl/models/lstm_s43.keras"), ("Skill autoencoder", "dl/models/ae_denoising.keras"), ("Market analytics", "analytics/market.json")):
        files.append({"label": label, "path": f"data/{rel}", "present": (d / rel).exists()})
    return render_template("health.html", status=st, files=files, versions={"Python": platform.python_version(), "Flask": importlib.metadata.version("flask")}, show_detail=bool(ext().settings.debug))
