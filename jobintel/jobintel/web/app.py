"""Flask application factory.

    from jobintel.web import create_app
    app = create_app()                      # settings from the environment, real index / database
    app = create_app(settings, backend)     # tests inject a Backend built on a tiny fixture

Pages (blueprints): main (home, about, privacy, terms, status), search (find jobs, job details), recommend, market (career insights), skillgap, resume, account (log in, saved jobs, profile, workspace), api.
"""
from __future__ import annotations

import logging
import threading
from datetime import timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Optional

from flask import Flask, jsonify, render_template, request
from werkzeug.exceptions import HTTPException

from .auth import load_current_user
from .backend import Backend
from .security import Throttle, init_security
from .session_store import EphemeralStore
from .settings import WebSettings
from .signals import SIGNAL_ORDER, SIGNALS
from .ui import init_ui

log = logging.getLogger("jobintel.web")

# The seven items of the main navigation, in this order (product direction).  (label, endpoint, endpoints on which the item shows as the current page)
NAV_ITEMS = [
    ("Home", "main.landing", ("main.landing",)),
    ("Find Jobs", "search.search", ("search.search", "search.job_detail")),
    ("Recommendations", "recommend.recommendations", ("recommend.recommendations",)),
    ("Resume", "resume.analyzer", ("resume.analyzer",)),
    ("Career Insights", "market.market", ("market.market", "skillgap.skill_gap")),
    ("Saved Jobs", "account.saved", ("account.saved",)),
    ("Profile", "account.profile", ("account.profile",)),
]

ERROR_TEXT = {
    400: ("Request not accepted", "The request could not be processed.", "Check the form and try again."),
    401: ("Log in first", "This page needs an account.", ""),
    403: ("Not allowed", "You do not have access to this page.", ""),
    404: ("Page not found", "There is nothing at this address.", "The job may have been removed, or the link is mistyped."),
    405: ("Method not allowed", "This address does not accept that kind of request.", ""),
    413: ("File too large", "The upload is bigger than the limit.", "Choose a smaller file."),
    429: ("Too many attempts", "Wait a few minutes and try again.", ""),
    500: ("Something went wrong", "Something unexpected happened on our side. It has been logged.", "Reload the page. If it keeps happening, tell the site operator."),
}


def create_app(settings: Optional[WebSettings] = None, backend: Optional[Backend] = None, **config) -> Flask:
    settings = settings or WebSettings.from_env()
    here = Path(__file__).parent
    app = Flask(__name__, static_folder=str(here / "static"), template_folder=str(here / "templates"))
    app.secret_key = settings.secret_key
    app.config.update(
        MAX_CONTENT_LENGTH=settings.max_upload_mb * 1024 * 1024,
        SESSION_COOKIE_HTTPONLY=True, SESSION_COOKIE_SAMESITE="Lax",
        PERMANENT_SESSION_LIFETIME=timedelta(hours=settings.session_hours),
        TESTING=settings.testing, JSON_SORT_KEYS=False, TEMPLATES_AUTO_RELOAD=settings.debug,
    )
    app.config.update(config)
    be = backend or Backend(settings)
    app.extensions["jobintel"] = SimpleNamespace(backend=be, settings=settings, store=EphemeralStore(), throttle=Throttle())

    init_security(app)
    init_ui(app)
    app.jinja_env.globals.update(signals=SIGNALS, signal_order=SIGNAL_ORDER)

    @app.context_processor
    def _globals():
        return {"current_user": load_current_user(), "nav_items": NAV_ITEMS, "index_stats": be.index_summary(), "settings": settings}

    from .views import account, api, main, market, recommend, resume, search, skillgap      # views/experiments.py is parked, not registered (see its docstring)
    for mod in (main, search, recommend, market, skillgap, resume, account, api):
        app.register_blueprint(mod.bp)

    @app.errorhandler(HTTPException)
    def _http(e: HTTPException):
        code = e.code or 500
        if request.path.startswith("/api/"):
            return jsonify(error=e.description or e.name), code
        title, msg, hint = ERROR_TEXT.get(code, (e.name, e.description or "", ""))
        if code == 400 and e.description and e.description != "Bad Request: The browser (or proxy) sent a request that this server could not understand.":
            msg = e.description
        return render_template("error.html", code=code, title=title, message=msg, hint=hint), code

    @app.errorhandler(Exception)
    def _boom(e: Exception):
        log.exception("unhandled error on %s", request.path)
        if request.path.startswith("/api/"):
            return jsonify(error="internal error"), 500
        title, msg, hint = ERROR_TEXT[500]
        return render_template("error.html", code=500, title=title, message=msg, hint=hint), 500

    if settings.warmup and not settings.testing and backend is None:
        threading.Thread(target=be.warm_up, name="jobintel-warmup", daemon=True).start()
    return app
