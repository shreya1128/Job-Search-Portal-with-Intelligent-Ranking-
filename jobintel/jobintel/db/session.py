"""Engine / session helpers. SQLite connections get foreign keys switched on so tests behave like MySQL."""
from __future__ import annotations

from functools import lru_cache

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session, sessionmaker

from jobintel.config import database_url


@lru_cache(maxsize=4)
def get_engine(url: str | None = None) -> Engine:
    url = url or database_url()
    kwargs = {"pool_pre_ping": True}
    if url.startswith("mysql"):
        kwargs.update(pool_recycle=1800)
    engine = create_engine(url, **kwargs)
    if url.startswith("sqlite"):
        @event.listens_for(engine, "connect")
        def _fk_on(dbapi_conn, _):
            dbapi_conn.execute("PRAGMA foreign_keys=ON")
    return engine


def get_session(url: str | None = None) -> Session:
    return sessionmaker(bind=get_engine(url), expire_on_commit=False)()
