"""Request-level protection: CSRF tokens for every state-changing request, security headers, and a tiny in-memory throttle for login attempts.

CSRF:  a random token is kept in the signed session cookie; every POST must send it back (form field ``csrf_token`` or header ``X-CSRF-Token``).
Headers: a Content-Security-Policy that only allows this site's own scripts / styles / fonts (all front-end libraries are bundled in static/vendor), no framing, no sniffing.
"""
from __future__ import annotations

import hmac
import secrets
import threading
import time
from collections import defaultdict, deque
from typing import Deque, Dict, Tuple

from flask import Flask, abort, request, session

SAFE_METHODS = {"GET", "HEAD", "OPTIONS"}
CSP = ("default-src 'self'; img-src 'self' data: blob:; style-src 'self' 'unsafe-inline'; script-src 'self'; font-src 'self'; connect-src 'self'; "
       "frame-ancestors 'none'; base-uri 'self'; form-action 'self'; object-src 'none'")


def csrf_token() -> str:
    tok = session.get("_csrf")
    if not tok:
        tok = secrets.token_urlsafe(32)
        session["_csrf"] = tok
    return tok


def init_security(app: Flask) -> None:
    app.jinja_env.globals["csrf_token"] = csrf_token

    @app.before_request
    def _check_csrf():
        if request.method in SAFE_METHODS or app.config.get("CSRF_DISABLED"):
            return None
        sent = request.form.get("csrf_token") or request.headers.get("X-CSRF-Token") or ""
        expected = session.get("_csrf", "")
        if not expected or not hmac.compare_digest(sent, expected):
            abort(400, description="The form has expired or is not valid. Reload the page and try again.")
        return None

    @app.after_request
    def _headers(resp):
        resp.headers.setdefault("Content-Security-Policy", CSP)
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("X-Frame-Options", "DENY")
        resp.headers.setdefault("Referrer-Policy", "same-origin")
        if request.endpoint and not request.endpoint.startswith("static"):
            resp.headers.setdefault("Cache-Control", "no-store" if session.get("user_id") else "private, max-age=0")
        return resp


class Throttle:
    """At most ``limit`` events per ``window`` seconds for a key (login attempts per address + user name).  In-memory: resets on restart, which is enough for a single-process server."""

    def __init__(self, limit: int = 8, window: float = 300.0):
        self.limit, self.window = limit, window
        self._hits: Dict[Tuple[str, str], Deque[float]] = defaultdict(deque)
        self._lock = threading.Lock()

    def _trim(self, q: Deque[float], now: float) -> None:
        while q and now - q[0] > self.window:
            q.popleft()

    def blocked(self, key: Tuple[str, str]) -> bool:
        now = time.time()
        with self._lock:
            q = self._hits[key]
            self._trim(q, now)
            return len(q) >= self.limit

    def hit(self, key: Tuple[str, str]) -> None:
        with self._lock:
            self._hits[key].append(time.time())

    def clear(self, key: Tuple[str, str]) -> None:
        with self._lock:
            self._hits.pop(key, None)
