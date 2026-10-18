from sqlalchemy import func, select, text

from jobintel.db.models import Company, Job, JobSkill, Skill
from jobintel.ingest import knowledge as K
from jobintel.ingest.load_db import Ingestor
from jobintel.ingest.loaders import load_jsonl
from jobintel.records import JobRecord
from jobintel.sample_data import write_jsonl
from jobintel.extraction.lexicon import SkillLexicon


def _rec(**kw):
    base = dict(source="emscad", external_id="1", title="Data Scientist", company="Acme", city="Pune", location_raw="Pune",
                description="<p>We need a Data Scientist with 3-5 years of experience in Python and SQL. Bachelor's degree in statistics. Rs. 8-12 LPA. Full-time.</p>",
                text_quality="full")
    base.update(kw)
    return JobRecord(**base)


def test_original_and_processed_both_stored_and_fields_extracted(engine, extractor):
    ing = Ingestor(engine, extractor, extractor.lex, verbose=False)
    ing.ingest("emscad", [_rec()])
    with engine.connect() as c:
        j = c.execute(select(Job)).mappings().one()
        skills = {r[0]: r[1] for r in c.execute(select(Skill.canonical_name, JobSkill.origin).join(JobSkill, JobSkill.skill_id == Skill.skill_id))}
    assert "<p>" in j["description_original"] and "<p>" not in j["description_clean"] and "python" in j["processed_text"]
    assert (j["experience_min"], j["experience_max"], j["experience_source"]) == (3, 5, "extracted")
    assert j["education"].startswith("bachelor") and j["salary_min"] == 800000 and j["salary_period"] == "year" and j["employment_type"] == "full-time"
    assert j["cluster"] == "data_ai" and skills["python"] == "extracted" and skills["sql"] == "extracted"


def test_dataset_values_beat_extracted_and_tags_merge(engine, extractor):
    ing = Ingestor(engine, extractor, extractor.lex, verbose=False)
    ing.ingest("naukri", [_rec(source="naukri", experience_min=1, experience_max=2, experience_source="dataset", skills=["Python", "Tableau"], text_quality="full")])
    with engine.connect() as c:
        j = c.execute(select(Job)).mappings().one()
        origins = dict(c.execute(select(Skill.canonical_name, JobSkill.origin).join(JobSkill, JobSkill.skill_id == Skill.skill_id)).all())
    assert (j["experience_min"], j["experience_max"], j["experience_source"]) == (1, 2, "dataset")
    assert origins["python"] == "both" and origins["tableau"] == "dataset_tag" and origins["sql"] == "extracted"


def test_duplicates_flagged_only_for_same_title_employer_place(engine, extractor):
    long = "We need a Data Scientist with 3 years of experience in Python and SQL. " * 10
    recs = [_rec(external_id="1", description=long), _rec(external_id="2", description=long),                      # exact repost -> duplicate
            _rec(external_id="3", description=long, city="Mumbai", location_raw="Mumbai"),                         # same text, other city -> distinct
            _rec(external_id="4", description=long, company="Other Corp"),                                         # same text, other employer -> distinct
            _rec(external_id="5", description=long, title="ML Engineer")]                                          # same text, other title -> distinct
    Ingestor(engine, extractor, extractor.lex, verbose=False).ingest("emscad", recs)
    with engine.connect() as c:
        dup = dict(c.execute(select(Job.external_id, Job.duplicate_of)).all())
        ids = dict(c.execute(select(Job.external_id, Job.job_id)).all())
    assert dup["2"] == ids["1"] and dup["1"] is None and dup["3"] is None and dup["4"] is None and dup["5"] is None


def test_replace_is_idempotent_and_unique_external_ids(engine, extractor):
    ing = Ingestor(engine, extractor, extractor.lex, verbose=False)
    recs = [_rec(external_id="same", description=None, skills=["python"], text_quality="skills_only")] * 3
    ing.ingest("naukri", recs)
    ing.ingest("naukri", recs)
    with engine.connect() as c:
        assert c.execute(select(func.count()).select_from(Job)).scalar_one() == 3
        assert c.execute(select(func.count(func.distinct(Job.external_id)))).scalar_one() == 3


def test_sample_data_end_to_end(engine, extractor, sample_jobs, tmp_path):
    p = tmp_path / "s.jsonl"
    write_jsonl(p, sample_jobs)
    st = Ingestor(engine, extractor, extractor.lex, verbose=False).ingest("sample_synthetic", load_jsonl(p, "sample_synthetic"))
    assert st["records"] == 60 and st["quality_full"] == 60 and st["skills_extracted"] > 100
    with engine.connect() as c:
        assert c.execute(text("SELECT COUNT(*) FROM jobs WHERE processed_text IS NULL OR description_original IS NULL")).scalar_one() == 0
        assert c.execute(text("SELECT s.source_type FROM crawler_sources s")).scalar_one() == "synthetic"     # clearly labelled


def test_curated_knowledge_graph_matches_the_brief(engine):
    K.load_curated(engine, SkillLexicon.from_seed())
    with engine.connect() as c:
        ess = {r[0] for r in c.execute(text("""SELECT s.canonical_name FROM knowledge_relationships r JOIN occupations o ON o.occupation_id=r.subject_id
            JOIN skills s ON s.skill_id=r.object_id WHERE o.name='Machine Learning Engineer' AND r.relation='requires_essential'"""))}
        py = {r[0] for r in c.execute(text("""SELECT b.canonical_name FROM knowledge_relationships r JOIN skills a ON a.skill_id=r.subject_id
            JOIN skills b ON b.skill_id=r.object_id WHERE a.canonical_name='python' AND r.relation='broader_than'"""))}
    assert {"python", "machine learning", "sql", "tensorflow", "docker"} <= ess
    assert {"pandas", "numpy", "scikit-learn"} <= py
