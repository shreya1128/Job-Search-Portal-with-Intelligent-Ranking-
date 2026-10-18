import math
import numpy as np
import pytest

from jobintel.extraction.extractor import Extractor
from jobintel.extraction.lexicon import SkillLexicon
from jobintel.ir import passages as PS
from jobintel.ir.engine import MODELS, MemoryDocStore, SearchEngine
from jobintel.ir.index import IndexDoc, InvertedIndex
from jobintel.ir.pagerank import LinkGraph, pagerank, related_job_edges
from jobintel.ir.query import QueryDFA, QueryProcessor
from jobintel.ir.ranker import combine, experience_score, location_score, minmax
from jobintel.processing.text import index_tokens

LEX = SkillLexicon.from_seed()
EXT = Extractor(LEX, cities=["Pune", "Mumbai", "Bengaluru", "Delhi NCR"])

JOBS = [
    dict(title="Machine Learning Engineer", city="Pune", exp=(0, 2), skills=["python", "machine learning", "sql"],
         text="About the role:\nWe build ranking models.\n\nRequirements:\n• Python, SQL and machine learning experience\n• Good communication\n\nBenefits:\n• Health insurance"),
    dict(title="Data Analyst", city="Mumbai", exp=(2, 4), skills=["sql", "excel", "tableau"],
         text="Responsibilities:\n• Build dashboards in Tableau\n• Write SQL queries and Excel reports\n\nRequirements:\n• 2-4 years of experience"),
    dict(title="Registered Nurse", city="Pune", exp=(1, 3), skills=["nursing", "patient care"],
         text="Provide patient care in the ward.\n\nRequirements:\n• Nursing degree and BLS certification"),
    dict(title="Backend Developer", city="Bengaluru", exp=(3, 6), skills=["python", "sql", "docker"],
         text="Develop services in Python and SQL. Deploy with Docker. Machine learning exposure is a plus."),
    dict(title="Senior Data Scientist", city="Pune", exp=(5, 9), skills=["python", "machine learning", "tensorflow"],
         text="Lead machine learning projects with Python and TensorFlow.\n\nRequirements:\n• 5+ years of experience\n• Deep learning"),
]


def make_engine(pagerank_values=None, clusters=None):
    docs = [IndexDoc(100 + i, index_tokens(j["title"] + "\n" + j["text"]), dict(
        source="toy", title=j["title"], company=j.get("company", f"Co{i}"), city=j["city"], cities=[j["city"]], country="India", exp_min=j["exp"][0], exp_max=j["exp"][1],
        sal_min=None, sal_max=None, sal_cur=None, sal_period=None, employment=None, cluster=(clusters[i] if clusters else "x"), category=None, text_quality="full", skills=j["skills"],
        posting_date=None, seniority=None, external_id=f"toy-{i}")) for i, j in enumerate(JOBS)]
    ix = InvertedIndex.build(docs)
    if pagerank_values is not None:
        ix.pagerank = np.array(pagerank_values, dtype=float)
    store = MemoryDocStore({100 + i: (j["title"], j["text"]) for i, j in enumerate(JOBS)})
    return SearchEngine(ix, store, EXT, lexicon=LEX)


# --------------------------------------------------------------------------- query processing
def test_query_dfa():
    ok = lambda q: QueryDFA.validate(q)[0]
    assert ok("Python machine learning") and ok("C++ / C# developer") and ok("data engineer (5+ years) $80k")
    assert not ok("") and not ok("   ") and not ok("....") and not ok("python\x00") and not ok("<script>") and not ok("x" * 201) and not ok(None)
    assert QueryDFA.validate("a;b")[1].startswith("unsupported character")


def test_query_parsing_example():
    p = QueryProcessor(EXT).parse("Entry level machine learning jobs in Pune")
    assert p.valid and p.skills == ["machine learning"] and p.locations == ["Pune"]
    assert p.level["name"] == "entry" and p.experience == {"min": 0, "max": 2, "source": "level"}
    assert p.terms == ["machin", "learn"]                           # 'entry level', 'Pune', 'jobs' and 'in' are not matched as text
    q2 = QueryProcessor(EXT).parse("python developer 3 years experience")
    assert q2.experience == {"min": 3, "max": 3, "source": "stated"} and "year" not in q2.terms
    bad = QueryProcessor(EXT).parse("<b>")
    assert not bad.valid and bad.error


# --------------------------------------------------------------------------- passages
def test_segmentation_sections_offsets_and_best_passage():
    text = JOBS[0]["text"]
    ps = PS.segment(text)
    assert [p.section for p in ps] == ["About the role", "Requirements", "Benefits"] or len(ps) >= 3
    for p in ps:
        assert text[p.start:p.end] == p.text and p.n_words > 0
    qt = [("python", 1, 0.05), ("sql", 1, 0.05), ("machin", 1, 0.05), ("learn", 1, 0.05)]
    sc = PS.score_passages(ps, qt)
    best, _ = PS.best_passage(ps, sc)
    assert best.section == "Requirements" and "Python" in best.text


def test_tiny_fragments_are_folded_into_the_neighbouring_passage():
    text = "Requirements:\n" + " ".join(["word"] * 70) + "\nOk\n" + " ".join(["more"] * 60)
    ps = PS.segment(text)
    assert all(p.n_words >= 8 for p in ps) and any("Ok" in p.text for p in ps)
    for p in ps:
        assert text[p.start:p.end] == p.text


def test_long_text_is_split_and_highlighting_is_safe():
    long = " ".join(f"Sentence number {i} talks about databases and SQL tuning." for i in range(40))
    ps = PS.segment(long)
    assert len(ps) > 3 and all(p.n_words <= 90 for p in ps) and "".join(p.text for p in ps).replace(" ", "") != ""
    h = PS.highlight("Skills: Python, machine learning & <b>SQL</b>", {"python", "machin", "learn", "sql"})
    assert "<mark>Python</mark>" in h and "<mark>machine learning</mark>" in h and "&lt;b&gt;" in h and "<b>" not in h
    assert PS.segment("") == [] and PS.highlight("", {"x"}) == ""


# --------------------------------------------------------------------------- PageRank
def _dense_pr(n, edges, d=0.85):
    P = np.zeros((n, n))
    for a, b, w in edges:
        P[a, b] += w
    for i in range(n):
        s = P[i].sum()
        P[i] = P[i] / s if s else np.full(n, 1.0 / n)
    return np.linalg.solve(np.eye(n) - d * P.T, np.full(n, (1 - d) / n))


def test_pagerank_matches_the_linear_system_solution():
    edges = [(0, 1, 1), (1, 2, 1), (2, 0, 1), (2, 3, 2), (3, 0, 1), (4, 0, 1)]          # node 5 is dangling and isolated
    res = pagerank(6, [e[0] for e in edges], [e[1] for e in edges], [e[2] for e in edges])
    assert res.converged and math.isclose(res.scores.sum(), 1.0, rel_tol=1e-9)
    assert np.allclose(res.scores, _dense_pr(6, edges), atol=1e-8)


def test_pagerank_cycle_is_uniform_and_hub_wins():
    cyc = pagerank(4, [0, 1, 2, 3], [1, 2, 3, 0])
    assert np.allclose(cyc.scores, 0.25)
    star = pagerank(5, [1, 2, 3, 4], [0, 0, 0, 0])
    assert star.scores[0] == star.scores.max() and star.scores[0] > 2 * star.scores[1]


def test_link_graph_company_career_job():
    g = LinkGraph()
    for j in (1, 2):
        g.edge(("company", 7), ("career", 7))
        g.edge(("career", 7), ("job", j))
        g.edge(("job", j), ("company", 7))
    g.edge(("job", 1), ("job", 2), 2.0)
    pr = g.run()
    assert math.isclose(sum(pr.values()), 1.0, rel_tol=1e-9) and pr[("job", 2)] > pr[("job", 1)]      # job 2 is linked from job 1 as well


def test_related_job_edges_use_rare_shared_skills_within_cluster():
    rows = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4]); cols = np.array([0, 1, 0, 1, 2, 3, 2, 3, 0])
    cluster = np.array([0, 0, 1, 1, 1])
    s, d, w = related_job_edges(rows, cols, 5, 4, cluster, k=2, min_df=2, max_df=10, min_shared_idf=0.1)
    pairs = set(zip(s.tolist(), d.tolist()))
    assert (0, 1) in pairs and (1, 0) in pairs and (2, 3) in pairs
    assert (1, 3) not in pairs and (4, 0) not in pairs and all(cluster[a] == cluster[b] for a, b in pairs)   # job 4 shares skill 0 but lives in another cluster


# --------------------------------------------------------------------------- ranker
def test_normalisation_and_weight_renormalisation():
    assert np.allclose(minmax(np.array([1.0, 2.0, 3.0])), [0, 0.5, 1]) and np.allclose(minmax(np.array([5.0, 5.0])), 0.5)
    final, eff = combine({"a": np.array([1.0, 0.0]), "b": None, "c": np.array([0.0, 1.0])}, {"a": 3, "b": 5, "c": 1})
    assert math.isclose(sum(eff.values()), 1.0) and set(eff) == {"a", "c"} and np.allclose(final, [0.75, 0.25])
    assert experience_score(0, 2, 0, 2) == 1.0 and experience_score(5, float("nan"), 0, 2) == pytest.approx(1 / 4) and location_score(["Pune"], "Pune", ["pune"]) == 1.0


# --------------------------------------------------------------------------- engine end to end
def test_all_models_return_explained_results():
    eng = make_engine()
    for m in MODELS:
        r = eng.search("python machine learning", k=3, model=m)
        assert r.error is None and 1 <= len(r.results) <= 3 and all(x.reasons for x in r.results), m
    top = eng.search("python machine learning sql", k=3, model="hybrid").results[0]
    assert top.title == "Machine Learning Engineer" and top.passage and "<mark>" in top.passage["html"] and top.passage["section"] == "Requirements"
    assert any(s.startswith("✓ 'python'") for s in top.reasons)


def test_nonsense_and_invalid_queries():
    eng = make_engine()
    assert eng.search("<script>alert(1)</script>").error and eng.search("").error
    r = eng.search("zzzxxqq")
    assert r.results == [] and r.error is None
    assert "not found" in " ".join(eng.search("python zzzxxqq").warnings)
    with pytest.raises(ValueError):
        eng.search("python", model="bm25")


def test_hybrid_weights_drop_inactive_signals_and_location_matters():
    eng = make_engine(pagerank_values=[0.1, 0.1, 0.1, 0.1, 0.6])
    r1 = eng.search("machine learning python", model="hybrid")
    assert "location" not in r1.weights and "skill" in r1.weights and math.isclose(sum(r1.weights.values()), 1.0)
    r2 = eng.search("machine learning python jobs in Pune", model="hybrid")
    assert "location" in r2.weights and all("Pune" in x.cities for x in r2.results[:2])
    r3 = eng.search("machine learning python", model="hybrid", weights={"pagerank": 100.0, "lm": 0.0, "dfr": 0.0, "passage": 0.0, "skill": 0.0})
    assert r3.results[0].title == "Senior Data Scientist"                    # all weight on PageRank -> the high-authority document


def test_filters_are_hard_constraints():
    eng = make_engine()
    assert {x.city for x in eng.search("python sql", filters={"city": "Pune"}).results} <= {"Pune"}
    assert [x.title for x in eng.search("python sql", filters={"skills": ["docker"]}).results] == ["Backend Developer"]
    ex = eng.search("python", filters={"min_experience": 0, "max_experience": 2}).results
    assert ex and all(x.experience in ("0-2 yrs",) for x in ex)
    assert eng.search("python", filters={"role": "nurse"}).results == []
    assert eng.search("python", filters={"city": "Atlantis"}).results == []


def test_browse_mode_without_text_terms():
    eng = make_engine(pagerank_values=[0.1, 0.1, 0.4, 0.1, 0.3])
    r = eng.search("jobs in Pune", model="hybrid")
    assert r.results and all(x.city == "Pune" for x in r.results) and len(r.results) == 3 and "no text terms" in " ".join(r.warnings)   # only the Pune postings
    assert r.results[0].title == "Registered Nurse"                                                                                    # highest PageRank among them
    assert eng.search("jobs in Pune", model="lm").results == []


def test_same_query_is_deterministic_and_fast_enough():
    eng = make_engine()
    a = [x.job_id for x in eng.search("python sql", k=5).results]
    assert a == [x.job_id for x in eng.search("python sql", k=5).results]
    assert eng.search("python sql").timings_ms["total"] < 500


def test_collapse_duplicates_keeps_the_best_of_identical_postings():
    dup = dict(JOBS[0], company="Co0")                      # same title, same employer, same city, same text = a reposted ad
    JOBS.append(dup)
    try:
        eng = make_engine()
        plain = [x.title for x in eng.search("machine learning python sql", k=6, model="lm").results]
        collapsed = [x.title for x in eng.search("machine learning python sql", k=6, model="lm", collapse_duplicates=True).results]
    finally:
        JOBS.pop()
    assert plain.count("Machine Learning Engineer") == 2 and collapsed.count("Machine Learning Engineer") == 1
