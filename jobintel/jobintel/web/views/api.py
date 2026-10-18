"""JSON endpoints used by the pages (and handy for tests / other clients): interactions, health, search, skill suggestions, job-function prediction."""
from __future__ import annotations

import html
import time
from typing import Dict

from flask import Blueprint, jsonify, request
from sqlalchemy import select

from ...db import models as M
from ..auth import load_current_user
from ..backend import Unavailable
from ..forms import Params
from ._common import ACTIONS, backend, db_session, record_interaction
from .search import parse_search

bp = Blueprint("api", __name__, url_prefix="/api")


@bp.route("/health")
def health():
    be = backend()
    st = be.status()
    ok_db = False
    try:
        be.db
        ok_db = True
    except Unavailable:
        pass
    return jsonify(ok=ok_db, resources=st)


@bp.route("/interact", methods=["POST"])
def interact():
    user = load_current_user()
    if user is None:
        return jsonify(error="log in first"), 401
    body = request.get_json(silent=True) or {}
    action, job_id = body.get("action"), body.get("job_id")
    if action not in ACTIONS or not isinstance(job_id, int) or isinstance(job_id, bool):
        return jsonify(error="expected {job_id: int, action: one of %s}" % ", ".join(ACTIONS)), 400
    try:
        with db_session() as s:
            record_interaction(s, user.user_id, job_id, action)
            s.commit()
    except LookupError:
        return jsonify(error="no such job"), 404
    return jsonify(ok=True, job_id=job_id, action=action)


@bp.route("/search")
def search_json():
    """The same search as the page, as JSON (explanations and signals included)."""
    be = backend()
    p = Params(request.args)
    prm = parse_search(p, ["emscad", "linkedin", "naukri"])
    if p.errors:
        return jsonify(error="; ".join(p.errors)), 400
    if not prm["q"] and not prm["filters"]:
        return jsonify(error="give q and / or a filter"), 400
    try:
        resp = be.engine.search(prm["q"] or "jobs", k=prm["k"], model=prm["model"], filters=prm["filters"], weights=prm["weights"] if prm["model"] == "hybrid" else None,
                                collapse_duplicates=not prm["keep_dupes"], cluster_routing=prm["route"])
    except Unavailable as e:
        return jsonify(error=str(e)), 503
    return jsonify(resp.to_dict())


@bp.route("/skills")
def skills():
    """Skill names starting with ``prefix`` (from the loaded index, most common first) for form hints."""
    prefix = Params(request.args).text("prefix", 40).lower()
    try:
        names = be_facets_skills()
    except Unavailable as e:
        return jsonify(error=str(e)), 503
    return jsonify([n for n in names if n.startswith(prefix)][:15] if prefix else names[:15])


def be_facets_skills():
    return backend().facets()["skills"]


@bp.route("/job/<int:job_id>/function")
def job_function(job_id: int):
    """Neural job-function prediction for one posting (the Phase-5 LSTM).  Returns an HTML snippet for the job page."""
    be = backend()
    with db_session() as s:
        text = s.execute(select(M.Job.description_clean).where(M.Job.job_id == job_id)).scalar_one_or_none()
    if text is None:
        return jsonify(error="no such job"), 404
    try:
        enc = be.encoder
        with be.tf_lock:
            out = enc.predict_category([text], top=3)[0]
    except Unavailable as e:
        return jsonify(html=f'<p class="text-soft small mb-0">The network is not available: {html.escape(str(e))}</p>')
    if not out.get("top"):
        return jsonify(html='<p class="text-soft small mb-0">The text has no words the network knows.</p>')
    rows = "".join(f'<tr><td>{html.escape(t["category"])}</td><td class="num">{100 * t["probability"]:.0f}%</td></tr>' for t in out["top"])
    return jsonify(html=f'<table class="table table-tight mb-1"><tbody>{rows}</tbody></table>'
                        f'<p class="form-text mb-0">Model {html.escape(enc.name)}; trained on seven EMSCAD job functions, test accuracy about 55%. A job outside these seven functions is still forced into one of them.</p>')
