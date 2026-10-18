"""Helpers shared by the views: access to the backend, database sessions, the interaction log and the profile of the current visitor."""
from __future__ import annotations

import datetime as dt
from typing import Dict, Iterable, List, Optional

import logging

from flask import current_app
from sqlalchemy import select
from sqlalchemy.orm import Session

from ...db import models as M

ACTIONS = ("view", "click", "save", "unsave", "apply", "unapply")
log = logging.getLogger("jobintel.web")


def public_reason(err: Exception, what: str = "This part of Jobintel") -> str:
    """What a visitor is told when a resource is missing.  The real reason (it may name files or commands) goes to the server log and, in debug mode, the status page, never into a page."""
    log.warning("%s unavailable: %s", what, err)
    return f"{what} is not available right now. Please try again in a minute. If it keeps happening, the site operator can see the cause in the server log."


def ext():
    return current_app.extensions["jobintel"]


def backend():
    return ext().backend


def db_session() -> Session:
    return Session(backend().db)


def record_interaction(s: Session, user_id: int, job_id: int, action: str) -> None:
    """Upsert the (user, job) row of ``job_interactions``: view / click set their flag, save / apply toggle theirs.  Real users only: is_synthetic stays False."""
    if action not in ACTIONS:
        raise ValueError(f"unknown action {action!r}")
    if s.get(M.Job, job_id) is None:
        raise LookupError("no such job")
    row = s.execute(select(M.JobInteraction).where(M.JobInteraction.user_id == user_id, M.JobInteraction.job_id == job_id)).scalar_one_or_none()
    if row is None:
        row = M.JobInteraction(user_id=user_id, job_id=job_id, viewed=False, clicked=False, saved=False, applied=False, is_synthetic=False)
        s.add(row)
    if action == "view":
        row.viewed = True
    elif action == "click":
        row.viewed = row.clicked = True
    elif action == "save":
        row.viewed = row.saved = True
    elif action == "unsave":
        row.saved = False
    elif action == "apply":
        row.viewed = row.applied = True
    elif action == "unapply":
        row.applied = False
    row.last_event_at = dt.datetime.now()
    s.flush()


def interaction_state(s: Session, user_id: Optional[int], job_ids: Iterable[int]) -> Dict[int, Dict[str, bool]]:
    ids = [int(j) for j in job_ids]
    if not user_id or not ids:
        return {}
    rows = s.execute(select(M.JobInteraction).where(M.JobInteraction.user_id == user_id, M.JobInteraction.job_id.in_(ids))).scalars().all()
    return {int(r.job_id): {"saved": bool(r.saved), "applied": bool(r.applied), "viewed": bool(r.viewed)} for r in rows}


def user_interactions(s: Session, user_id: int) -> Dict[int, float]:
    """{job_id: event weight} of a real user (view 1 / click 2 / save 3 / apply 4), the form the collaborative signal expects."""
    from ...rec.interactions import EVENT_WEIGHTS
    out: Dict[int, float] = {}
    for r in s.execute(select(M.JobInteraction).where(M.JobInteraction.user_id == user_id)).scalars():
        w = EVENT_WEIGHTS["apply"] if r.applied else EVENT_WEIGHTS["save"] if r.saved else EVENT_WEIGHTS["click"] if r.clicked else EVENT_WEIGHTS["view"] if r.viewed else 0.0
        if w > 0:
            out[int(r.job_id)] = float(w)
    return out
