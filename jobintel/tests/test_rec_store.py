import os
import sys

import pytest
from sqlalchemy import create_engine, func, select
from sqlalchemy.orm import Session

sys.path.insert(0, os.path.dirname(__file__))
from jobintel.db import models as M
from jobintel.extraction.extractor import Extractor
from jobintel.extraction.lexicon import SkillLexicon
from jobintel.rec import store
from jobintel.rec.catalog import ItemCatalog
from jobintel.rec.hybrid import HybridRecommender
from jobintel.rec.interactions import SyntheticConfig, generate
from jobintel.rec.knowledge import KnowledgeGraph
from jobintel.rec.profile import UserProfile
from jobintel.rec.resume import ResumeAnalyzer
from jobintel.rec.skillgap import analyze_gap
from rec_fixtures import make_text_engine
from test_rec_resume import RESUME


@pytest.fixture()
def db():
    eng = create_engine("sqlite://")
    M.Base.metadata.create_all(eng)
    with Session(eng) as s:
        for n in ("sql", "excel", "power bi", "python", "tableau", "data analysis"):
            s.add(M.Skill(name=n, canonical_name=n))
        s.commit()
        yield s


def test_profile_and_user_skills_roundtrip(db):
    an = ResumeAnalyzer(Extractor(SkillLexicon.from_seed(), cities=["pune"]))
    r = an.analyze(RESUME)
    prof = UserProfile.from_resume(r, preferred_role="data analyst")
    uid = store.create_user(db, "priya", "priya@example.com")
    out = store.save_profile(db, uid, prof, r, resume_text=RESUME, filename="priya.txt")
    db.commit()
    up = db.execute(select(M.UserProfile).where(M.UserProfile.user_id == uid)).scalar_one()
    assert up.full_name == "Priya Sharma" and up.preferred_role == "data analyst" and up.resume_filename == "priya.txt"
    n = db.execute(select(func.count()).select_from(M.UserSkill).where(M.UserSkill.user_id == uid)).scalar()
    assert n == out["saved_skills"] > 0
    store.save_profile(db, uid, prof, r)                                         # saving again replaces, never duplicates
    db.commit()
    assert db.execute(select(func.count()).select_from(M.UserSkill).where(M.UserSkill.user_id == uid)).scalar() == n


def test_recommendations_and_gaps_are_stored_with_their_signals(db):
    eng = make_text_engine()
    cat = ItemCatalog(eng.index)
    kg = KnowledgeGraph.from_dicts({"Data analyst": {"essential": ["sql", "excel", "data analysis"], "optional": ["power bi", "tableau"]}})
    prof = UserProfile.from_form(["sql", "excel"], preferred_role="data analyst")
    resp = HybridRecommender(eng, cat, kg).recommend(prof, k=4)
    uid = store.create_user(db, "u1", "u1@example.com")
    assert store.save_recommendations(db, uid, resp) == len(resp.results)
    db.commit()
    rows = db.execute(select(M.Recommendation).where(M.Recommendation.user_id == uid).order_by(M.Recommendation.position)).scalars().all()
    assert [r.job_id for r in rows] == [x.job_id for x in resp.results]
    assert rows[0].final_score == pytest.approx(resp.results[0].final_score) and rows[0].dl_score is None and rows[0].content_score is not None
    assert rows[0].weights == resp.weights_effective
    rep = analyze_gap(cat, kg, prof.skills, job_di=0)
    res = store.save_skill_gaps(db, uid, resp.results[0].job_id, rep)
    db.commit()
    assert res["saved"] == db.execute(select(func.count()).select_from(M.SkillGap).where(M.SkillGap.user_id == uid)).scalar() > 0
    assert {g.status for g in db.execute(select(M.SkillGap)).scalars()} <= {"matching", "missing"}


def test_synthetic_users_and_interactions_are_flagged(db):
    from rec_fixtures import make_family_index
    cat = ItemCatalog(make_family_index())
    data = generate(cat, SyntheticConfig(n_users=12, seed=3))
    uids = store.write_synthetic(db, data, cat)
    db.commit()
    assert len(uids) == 12
    assert all(u.is_synthetic and u.username.startswith("synthetic_") for u in db.execute(select(M.User)).scalars())
    n = db.execute(select(func.count()).select_from(M.JobInteraction)).scalar()
    assert n == data.n_pairs and db.execute(select(func.count()).select_from(M.JobInteraction).where(M.JobInteraction.is_synthetic == False)).scalar() == 0   # noqa: E712
