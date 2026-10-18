import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))

from jobintel.rec.catalog import ItemCatalog
from jobintel.rec.knowledge import KnowledgeGraph
from jobintel.rec.skillgap import analyze_gap
from rec_fixtures import JOBS, make_index

KG = KnowledgeGraph.from_dicts(
    {"Data analyst": {"essential": ["sql", "excel", "data analysis"], "optional": ["power bi", "tableau", "python"]},
     "Machine learning engineer": {"essential": ["python", "machine learning"], "optional": ["tensorflow", "docker"]},
     "Accountant": {"essential": ["accounting", "excel", "gst"], "optional": ["tally"]}},
    related={"power bi": {"tableau": 0.7}, "sql": {"mysql": 0.8}})


@pytest.fixture(scope="module")
def cat():
    return ItemCatalog(make_index())


def pos(cat, title):
    return [i for i, t in enumerate(cat.ix.meta["title"]) if t == title][0]


def test_content_scores_rank_the_right_family_first(cat):
    s = cat.content_scores({"sql": 1.0, "excel": 1.0, "data analysis": 1.0})
    order = np.argsort(-s)
    assert cat.ix.meta["title"][order[0]] == "Data Analyst"
    assert s[pos(cat, "Frontend Developer")] == 0.0
    assert s[pos(cat, "Warehouse Assistant")] == 0.0                         # no skills -> no content evidence
    assert np.all((s >= 0) & (s <= 1.0 + 1e-6))


def test_rare_skills_count_more_than_common_ones(cat):
    assert cat.idf[cat.sidx["tensorflow"]] > cat.idf[cat.sidx["sql"]]


def test_coverage_is_share_of_job_requirements_and_unknown_for_skill_less_jobs(cat):
    cov = cat.coverage({"sql": 1.0, "excel": 1.0, "data analysis": 1.0, "power bi": 1.0})
    assert cov[pos(cat, "Data Analyst")] == pytest.approx(1.0)
    assert 0 < cov[pos(cat, "Senior Data Analyst")] < 1.0
    from jobintel.ir.ranker import UNKNOWN_SCORE
    assert cov[pos(cat, "Warehouse Assistant")] == UNKNOWN_SCORE
    assert cat.content_scores({"nonexistent skill": 1.0}) is None and cat.coverage({"nonexistent skill": 1.0}) is None


def test_similar_jobs_and_title_lookup_and_market_frequency(cat):
    i = pos(cat, "Data Analyst")
    sim = cat.similar_jobs(i, 3)
    assert cat.ix.meta["title"][sim[0]] == "Senior Data Analyst" and i not in sim
    assert len(cat.jobs_by_title("data analyst")) == 2
    f = cat.market_frequency(cat.jobs_by_title("data analyst"))
    assert f["sql"] == 1.0 and f["tableau"] == 0.5


def test_kg_expand_rank_and_title_match():
    w, added = KG.expand({"power bi": 1.0, "sql": 1.0})
    assert w["tableau"] == pytest.approx(0.35) and added["tableau"] == "power bi" and w["power bi"] == 1.0
    r = KG.rank_occupations({"sql": 1.0, "excel": 1.0, "data analysis": 1.0})
    assert r[0]["occupation"] == "Data analyst" and r[0]["missing_essential"] == []
    assert KG.match_title("data analyst")[0]["occupation"] == "Data analyst"
    assert KG.match_title("zzz qqq") == []


def test_knowledge_scores_cold_start_from_role_only(cat):
    s, info = KG.knowledge_scores(cat, {}, preferred_role="Accountant")
    assert info["occupations"][0]["from"] == "preferred role"
    assert cat.ix.meta["title"][int(np.argmax(s))] in ("Accountant", "Tax Accountant")
    none, info = KG.knowledge_scores(cat, {}, preferred_role=None)
    assert none is None and "reason" in info                                  # nothing to go on -> says so instead of inventing a score


def test_knowledge_scores_from_skills(cat):
    s, info = KG.knowledge_scores(cat, {"python": 1.0, "machine learning": 1.0})
    assert cat.ix.meta["title"][int(np.argmax(s))] == "Machine Learning Engineer"


def test_gap_for_a_job_marks_missing_and_bridges_related_skills(cat):
    i = pos(cat, "Senior Data Analyst")
    rep = analyze_gap(cat, KG, {"sql": 1.0, "excel": 1.0, "power bi": 1.0}, job_di=i)
    miss = {r.skill: r for r in rep.missing}
    assert {"tableau", "python", "data analysis"} <= set(miss)
    assert miss["tableau"].related_held == ["power bi"]                       # holds a related skill -> reported as a bridge
    assert rep.occupation == "Data analyst" and 0 < rep.skill_coverage < 1
    assert all(rep.rows[k].importance >= rep.rows[k + 1].importance for k in range(len(rep.rows) - 1))
    full = analyze_gap(cat, KG, {s: 1.0 for s in cat.skills}, job_di=i)
    assert full.skill_coverage == pytest.approx(1.0) and not full.missing


def test_gap_for_a_role_and_argument_errors(cat):
    rep = analyze_gap(cat, KG, {"sql": 1.0}, role_text="data analyst", min_market_freq=0.5)
    assert any(r.skill == "excel" and r.status == "missing" for r in rep.rows)
    empty = analyze_gap(cat, KG, {"sql": 1.0}, role_text="astronaut")
    assert empty.skill_coverage is None and any("no job title" in w for w in empty.warnings)
    with pytest.raises(ValueError):
        analyze_gap(cat, KG, {}, job_di=0, role_text="x")
    with pytest.raises(ValueError):
        analyze_gap(cat, KG, {})
