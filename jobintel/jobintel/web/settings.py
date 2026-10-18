"""Web-application settings.  Everything that can differ between machines comes from environment variables (or a .env file loaded by the run script); nothing secret is in the code.

    SECRET_KEY            signs the session cookie.  If it is missing a random key is generated for this process only (sessions end when the server restarts) and a warning is logged.
    JOBINTEL_INDEX        path of the inverted index           (default: <DATA_DIR>/index/job_index.pkl)
    JOBINTEL_DEBUG        1 = Flask debug pages (never on a shared machine)
    JOBINTEL_WARMUP       1 (default) = load the neural models in a background thread at start-up so that the first recommendation is not slow; 0 = load on first use
    JOBINTEL_PARTS        >1 = serve searches through the partitioned engine with that many index partitions (threads); default 1
    JOBINTEL_LOG_SEARCHES 1 (default) = store every search (and its top results) in the `searches` / `search_results` tables
    JOBINTEL_OPEN_REGISTRATION  1 (default) = anyone can create an account; 0 = registration closed
    MAX_UPLOAD_MB         largest accepted resume upload (default 8, hard ceiling 15)
The database connection uses the same DB_* / DATABASE_URL variables as every script (see .env.example).
"""
from __future__ import annotations

import logging
import os
import secrets
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from ..config import DATA_DIR

log = logging.getLogger("jobintel.web")


def _flag(name: str, default: bool) -> bool:
    v = os.environ.get(name)
    if v is None or v == "":
        return default
    return v.strip().lower() in ("1", "true", "yes", "on")


@dataclass
class WebSettings:
    secret_key: str = ""
    debug: bool = False
    testing: bool = False
    data_dir: Path = field(default_factory=lambda: Path(DATA_DIR))
    index_path: Optional[Path] = None
    warmup: bool = True
    parts: int = 1
    log_searches: bool = True
    open_registration: bool = True
    max_upload_mb: int = 8
    session_hours: int = 12
    secret_is_ephemeral: bool = False

    def __post_init__(self) -> None:
        self.data_dir = Path(self.data_dir)
        if self.index_path is None:
            self.index_path = self.data_dir / "index" / "job_index.pkl"
        self.index_path = Path(self.index_path)
        self.max_upload_mb = max(1, min(int(self.max_upload_mb), 15))
        if not self.secret_key:
            self.secret_key = secrets.token_hex(32)
            self.secret_is_ephemeral = True

    @classmethod
    def from_env(cls, **overrides) -> "WebSettings":
        env = os.environ
        kw = dict(
            secret_key=env.get("SECRET_KEY", ""),
            debug=_flag("JOBINTEL_DEBUG", False),
            index_path=Path(env["JOBINTEL_INDEX"]) if env.get("JOBINTEL_INDEX") else None,
            warmup=_flag("JOBINTEL_WARMUP", True),
            parts=max(1, int(env.get("JOBINTEL_PARTS", "1") or 1)),
            log_searches=_flag("JOBINTEL_LOG_SEARCHES", True),
            open_registration=_flag("JOBINTEL_OPEN_REGISTRATION", True),
            max_upload_mb=int(env.get("MAX_UPLOAD_MB", "8") or 8),
        )
        kw.update(overrides)
        s = cls(**kw)
        if s.secret_is_ephemeral and not s.testing:
            log.warning("SECRET_KEY is not set: a random key is used, so logins end whenever the server restarts. Set SECRET_KEY in your .env file.")
        return s
