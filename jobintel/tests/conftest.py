import json
import os

import pytest
from sqlalchemy import create_engine, event

from jobintel.db.models import Base
from jobintel.sample_data import generate


@pytest.fixture()
def engine(tmp_path):
    url = os.environ.get("JOBINTEL_TEST_DB_URL")
    if url:
        # Opt-in: run the test on a real MySQL / MariaDB server instead of SQLite, e.g.
        #   JOBINTEL_TEST_DB_URL=mysql+pymysql://user:pw@127.0.0.1:3306/jobintel_webtest   python -m pytest tests/test_web_account.py
        # The database is EMPTIED (every table dropped) before and after each test, so it must be a scratch database; any name without "test" in it is refused.
        eng = create_engine(url)
        if "test" not in (eng.url.database or "").lower():
            pytest.fail(f"JOBINTEL_TEST_DB_URL points at '{eng.url.database}': the test database name must contain 'test' because its tables are dropped")
        Base.metadata.drop_all(eng)
        Base.metadata.create_all(eng)
        yield eng
        Base.metadata.drop_all(eng)
        eng.dispose()
        return
    eng = create_engine(f"sqlite:///{tmp_path/'t.sqlite'}")

    @event.listens_for(eng, "connect")
    def _fk(dbapi, _):
        dbapi.execute("PRAGMA foreign_keys=ON")
    Base.metadata.create_all(eng)
    yield eng
    eng.dispose()


@pytest.fixture(scope="session")
def sample_jobs():
    return generate(60, seed=7)


@pytest.fixture(scope="session")
def extractor():
    from jobintel.extraction.extractor import Extractor
    from jobintel.extraction.lexicon import SkillLexicon
    return Extractor(SkillLexicon.from_seed())   # seed lexicon: tests never depend on the big datasets
