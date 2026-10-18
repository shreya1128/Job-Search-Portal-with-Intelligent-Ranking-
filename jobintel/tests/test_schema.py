import pytest
from sqlalchemy import insert
from sqlalchemy.exc import IntegrityError

from jobintel.db.models import ALL_TABLES, Base, CrawlerSource, Job
from jobintel.db.export_schema import mermaid_er, mysql_ddl

REQUIRED = {"users", "user_profiles", "skills", "user_skills", "companies", "jobs", "job_skills", "job_passages", "job_terms", "job_links",
            "job_pagerank", "job_interactions", "searches", "search_results", "recommendations", "skill_gaps", "crawler_sources",
            "model_experiments", "ir_experiments", "knowledge_relationships"}


def test_all_required_tables_exist():
    assert REQUIRED <= set(ALL_TABLES) and len(ALL_TABLES) == 21


def test_foreign_keys_and_checks_enforced(engine):
    with engine.begin() as c:
        c.execute(insert(CrawlerSource).values(name="s", source_type="dataset"))
    with pytest.raises(IntegrityError):                       # FK: unknown source
        with engine.begin() as c:
            c.execute(insert(Job).values(source_id=999, external_id="x", title="t"))
    with pytest.raises(IntegrityError):                       # CHECK: experience_max < experience_min
        with engine.begin() as c:
            c.execute(insert(Job).values(source_id=1, external_id="x", title="t", experience_min=5, experience_max=2))
    with engine.begin() as c:
        c.execute(insert(Job).values(source_id=1, external_id="ok", title="t"))
    with pytest.raises(IntegrityError):                       # UNIQUE(source_id, external_id)
        with engine.begin() as c:
            c.execute(insert(Job).values(source_id=1, external_id="ok", title="t2"))


def test_ddl_and_er_export():
    ddl = mysql_ddl()
    assert ddl.count("CREATE TABLE") == 21 and "ENGINE=InnoDB" in ddl and "LONGTEXT" in ddl and "FOREIGN KEY" in ddl
    assert mermaid_er().startswith("erDiagram") and "jobs ||--o{ job_skills" in mermaid_er()
