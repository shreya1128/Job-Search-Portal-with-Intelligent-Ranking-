"""Accounts: registration, login, the logged-in user for a request, and the ``login_required`` decorator.

Passwords are stored as salted hashes (werkzeug, scrypt / pbkdf2 depending on the platform) in ``users.password_hash``; the plain password is never stored or logged.
Generated demo users (``is_synthetic``) and rows with the placeholder hash written by the Phase-6 scripts cannot log in."""
from __future__ import annotations

import hashlib
import hmac
import re
from dataclasses import dataclass
from functools import wraps
from typing import Dict, List, Optional

from flask import current_app, flash, g, redirect, request, session, url_for
from sqlalchemy import delete, or_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session
from werkzeug.security import check_password_hash, generate_password_hash

from ..db import models as M

USERNAME_RE = re.compile(r"^[A-Za-z0-9_.-]{3,40}$")
EMAIL_RE = re.compile(r"^[^@\s]{1,64}@[^@\s]{1,190}\.[^@\s]{2,24}$")
PLACEHOLDER_HASH = "!no-login"
_DUMMY_HASH = generate_password_hash("not-a-real-password")      # compared against when the user does not exist (constant-time-ish login)


def password_stamp(password_hash: Optional[str]) -> str:
    """A short fingerprint of the stored password hash, kept in the session cookie next to the user id.  When the password changes the fingerprint changes, so every other browser
    that is still logged in with the old password is logged out on its next request.  It reveals nothing usable: it is a truncated SHA-256 of a salted hash."""
    return hashlib.sha256((password_hash or "").encode("utf-8")).hexdigest()[:20]


@dataclass
class CurrentUser:
    user_id: int
    username: str
    email: str
    role: str
    stamp: str = ""


def start_session(user_id: int, stamp: str) -> None:
    """A fresh session for a user who has just proved who they are (the old one, including its CSRF token and any stored resume, is discarded)."""
    session.clear()
    session["user_id"] = int(user_id)
    session["pwv"] = stamp
    session.permanent = True


def validate_registration(username: str, email: str, password: str, confirm: str) -> List[str]:
    errs: List[str] = []
    if not USERNAME_RE.match(username or ""):
        errs.append("User name: 3-40 characters, letters, digits, dot, dash or underscore.")
    if username.lower().startswith("synthetic_"):
        errs.append("User names starting with 'synthetic_' are reserved for generated demo users.")
    if not EMAIL_RE.match(email or ""):
        errs.append("Enter a valid e-mail address.")
    if len(password or "") < 8:
        errs.append("Password: at least 8 characters.")
    if len(password or "") > 128:
        errs.append("Password: at most 128 characters.")
    if password and username and password.lower() == username.lower():
        errs.append("Password must not be your user name.")
    if password != confirm:
        errs.append("The two passwords differ.")
    return errs


def create_account(s: Session, username: str, email: str, password: str) -> int:
    u = M.User(username=username, email=email.lower(), password_hash=generate_password_hash(password), role="user", is_synthetic=False)
    s.add(u)
    try:
        s.flush()
    except IntegrityError as e:                                       # unique user name / e-mail
        s.rollback()
        raise ValueError("That user name or e-mail address is already registered.") from e
    return int(u.user_id)


def authenticate(s: Session, identifier: str, password: str) -> Optional[CurrentUser]:
    ident = (identifier or "").strip()
    row = s.execute(select(M.User).where(or_(M.User.username == ident, M.User.email == ident.lower()))).scalar_one_or_none()
    stored = row.password_hash if row is not None else _DUMMY_HASH
    placeholder = stored in (PLACEHOLDER_HASH, "")
    ok = check_password_hash(_DUMMY_HASH if placeholder else stored, password or "")      # always one hash check, whether or not the user exists
    if row is None or placeholder or not ok or row.is_synthetic:
        return None
    return CurrentUser(int(row.user_id), row.username, row.email, row.role, password_stamp(row.password_hash))


def _check_new_password(new: str, confirm: str, username: str, current: str = "") -> List[str]:
    errs: List[str] = []
    if len(new or "") < 8:
        errs.append("New password: at least 8 characters.")
    if len(new or "") > 128:
        errs.append("New password: at most 128 characters.")
    if new and username and new.lower() == username.lower():
        errs.append("The new password must not be your user name.")
    if new and current and new == current:
        errs.append("The new password must be different from the current one.")
    if new != confirm:
        errs.append("The two new passwords differ.")
    return errs


def change_password(s: Session, user_id: int, current: str, new: str, confirm: str) -> List[str]:
    """Change the password of ``user_id``.  Returns the list of problems (empty = changed).  The caller commits and refreshes the session stamp."""
    row = s.get(M.User, int(user_id))
    if row is None or row.is_synthetic or row.password_hash in (PLACEHOLDER_HASH, "") or not check_password_hash(row.password_hash, current or ""):
        return ["Your current password is not correct."]
    errs = _check_new_password(new, confirm, row.username, current)
    if errs:
        return errs
    row.password_hash = generate_password_hash(new)
    s.flush()
    return []


def verify_password(s: Session, user_id: int, password: str) -> bool:
    row = s.get(M.User, int(user_id))
    return bool(row is not None and not row.is_synthetic and row.password_hash not in (PLACEHOLDER_HASH, "") and check_password_hash(row.password_hash, password or ""))


def delete_account(s: Session, user_id: int) -> Dict[str, int]:
    """Remove a user and everything stored about them: profile, skills, interactions (saved / applied / viewed), search history with its result rows, stored recommendations and skill
    gaps.  Every table is deleted from explicitly, so the result does not depend on the database enforcing ON DELETE CASCADE.  Job postings are not touched.  The caller commits."""
    uid = int(user_id)
    counts: Dict[str, int] = {}
    mine = select(M.Search.search_id).where(M.Search.user_id == uid)
    counts["search_results"] = s.execute(delete(M.SearchResult).where(M.SearchResult.search_id.in_(mine))).rowcount or 0
    counts["searches"] = s.execute(delete(M.Search).where(M.Search.user_id == uid)).rowcount or 0
    for name, model in (("recommendations", M.Recommendation), ("skill_gaps", M.SkillGap), ("interactions", M.JobInteraction), ("skills", M.UserSkill), ("profile", M.UserProfile)):
        counts[name] = s.execute(delete(model).where(model.user_id == uid)).rowcount or 0
    counts["user"] = s.execute(delete(M.User).where(M.User.user_id == uid, M.User.is_synthetic.is_(False))).rowcount or 0
    s.flush()
    return counts


def load_current_user() -> Optional[CurrentUser]:
    """The logged-in user of this request (cached on ``g``), or None."""
    if "current_user" in g.__dict__:
        return g.current_user
    g.current_user = None
    uid = session.get("user_id")
    if uid:
        try:
            with Session(current_app.extensions["jobintel"].backend.db) as s:
                row = s.get(M.User, int(uid))
                stamp = password_stamp(row.password_hash) if row is not None else ""
                if row is not None and not row.is_synthetic and hmac.compare_digest(str(session.get("pwv", "")), stamp):
                    g.current_user = CurrentUser(int(row.user_id), row.username, row.email, row.role, stamp)
                else:                                                # deleted, demo user, or the password changed since this browser logged in
                    session.pop("user_id", None)
                    session.pop("pwv", None)
        except Exception:                                             # noqa: BLE001 - database down: behave as logged out, the page shows its own message
            current_app.logger.exception("could not load the logged-in user")
    return g.current_user


def login_required(view):
    @wraps(view)
    def wrapped(*a, **kw):
        if load_current_user() is None:
            flash("Log in to see this page.", "info")
            return redirect(url_for("account.login", next=request.full_path.rstrip("?")))
        return view(*a, **kw)
    return wrapped


def safe_next(target: Optional[str]) -> str:
    """Only redirect to a path on this site (never to another host)."""
    if target and target.startswith("/") and not target.startswith("//") and "\\" not in target:
        return target
    return url_for("account.dashboard")
