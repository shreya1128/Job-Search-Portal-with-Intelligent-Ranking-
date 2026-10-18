import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.dirname(__file__))
from jobintel.rec.dlsim import DLSimilarity
from jobintel.rec.hybrid import DEFAULT_REC_WEIGHTS, SIGNALS, HybridRecommender
from jobintel.rec.interactions import ItemKNN
from jobintel.rec.knowledge import KnowledgeGraph
from jobintel.rec.profile import UserProfile
from rec_fixtures import JOBS, make_text_engine

KG = KnowledgeGraph.from_dicts({"Data analyst": {"essential": ["sql", "excel", "data analysis"], "optional": ["power bi", "tableau", "python"]},
                                "Accountant": {"essential": ["accounting", "excel", "gst"], "optional": ["tally"]}})


@pytest.fixture(scope="module")
def engine():
    return make_text_engine()


def titles(resp):
    return [r.title for r in resp.results]


def test_the_default_weights_cover_the_eleven_signals_and_sum_to_one():
    assert len(SIGNALS) == 11 and set(DEFAULT_REC_WEIGHTS) == set(SIGNALS)
    assert sum(DEFAULT_REC_WEIGHTS.values()) == pytest.approx(1.0)
    with pytest.raises(ValueError):
        HybridRecommender(make_text_engine(), weights={"bogus": 1.0})


def test_cold_start_profile_gets_recommendations_and_honest_inactive_list(engine):
    rec = HybridRecommender(engine, kg=KG)
    p = UserProfile.from_form(["sql", "excel", "power bi"], preferred_role="data analyst", locations=["Pune"], experience_years=2)
    r = rec.recommend(p, k=5)
    assert titles(r)[0] in ("Data Analyst", "Senior Data Analyst") and r.n_candidates >= 5
    assert {"dl", "collaborative", "pagerank"} <= set(r.inactive)                                  # no embeddings, no history, no PageRank in this toy index
    assert "no collaborative model" in r.inactive["collaborative"] and "embeddings" in r.inactive["dl"]
    assert sum(r.weights_effective.values()) == pytest.approx(1.0, abs=1e-4)                       # dropped weights were re-normalised
    assert not ({"dl", "collaborative", "pagerank"} & set(r.weights_effective))


def test_final_score_is_exactly_the_weighted_sum_of_its_signals(engine):
    rec = HybridRecommender(engine, kg=KG)
    r = rec.recommend(UserProfile.from_form(["sql", "excel"], preferred_role="data analyst"), k=6)
    for x in r.results:
        assert x.final_score == pytest.approx(sum(r.weights_effective[s] * x.signals[s] for s in r.weights_effective), abs=1e-9)
        assert sum(x.contributions.values()) == pytest.approx(x.final_score, abs=1e-9)
        assert all(0.0 <= v <= 1.0 for v in x.signals.values() if v is not None)
    assert [x.final_score for x in r.results] == sorted((x.final_score for x in r.results), reverse=True)
    assert [x.rank for x in r.results] == list(range(1, len(r.results) + 1))


def test_weights_can_be_changed_and_zero_weight_removes_a_signal(engine):
    rec = HybridRecommender(engine, kg=KG)
    p = UserProfile.from_form(["sql", "excel"], preferred_role="data analyst", locations=["Pune"])
    r = rec.recommend(p, k=5, weights={"preference": 0.0})
    assert "preference" not in r.weights_effective and r.inactive["preference"] == "weight is 0"


def test_seen_jobs_are_excluded(engine):
    rec = HybridRecommender(engine, kg=KG)
    p = UserProfile.from_form(["sql", "excel"], preferred_role="data analyst")
    first = rec.recommend(p, k=3).results[0]
    again = rec.recommend(p, k=3, exclude_job_ids=[first.job_id])
    assert first.job_id not in {x.job_id for x in again.results}


def test_skill_explanations(engine):
    rec = HybridRecommender(engine, kg=KG)
    r = rec.recommend(UserProfile.from_form(["sql", "excel", "power bi"], preferred_role="data analyst"), k=3)
    da = [x for x in r.results if x.title == "Data Analyst"][0]
    assert set(da.matched_skills) == {"sql", "excel", "power bi"} and da.missing_skills == ["data analysis"]
    assert any(s.startswith("missing") for s in da.reasons)


def test_preference_signal_rewards_the_preferred_city(engine):
    rec = HybridRecommender(engine)
    r = rec.recommend(UserProfile.from_form(["sql", "excel"], locations=["Mumbai"]), k=12)
    assert all(x.signals["preference"] is not None for x in r.results)
    mumbai = [x.signals["preference"] for x in r.results if x.city == "Mumbai"]
    other = [x.signals["preference"] for x in r.results if x.city != "Mumbai"]
    assert min(mumbai) > max(other) - 1e-9 or np.mean(mumbai) > np.mean(other)


def test_collaborative_signal_switches_on_with_history(engine):
    cf = ItemKNN(engine.index.n_docs, shrink=1.0, min_item_count=2).fit([0, 0, 1, 1, 2, 2, 3, 3], [0, 1] * 4, [4.0] * 8)      # 4 users liked docs 0 and 1 (Data Analyst + Senior Data Analyst)
    rec = HybridRecommender(engine, kg=KG, cf=cf)
    cold = rec.recommend(UserProfile.from_form(["sql"], preferred_role="analyst"), k=5)
    assert "cold start" in cold.inactive["collaborative"]                                           # a model exists, but this user has no history
    warm = rec.recommend(UserProfile.from_form(["sql"], preferred_role="analyst", interactions={100: 4.0}), k=5, exclude_job_ids=[100])
    assert "collaborative" in warm.weights_effective
    assert warm.results[0].job_id == 101 and warm.results[0].signals["collaborative"] == pytest.approx(1.0)


class FakeEncoder:
    """TEST DOUBLE (hash embedding) - checks the generic text adapter wiring only; it is not a model and proves nothing about DL quality."""
    def embed(self, texts):
        out = np.zeros((len(texts), 32))
        for i, t in enumerate(texts):
            for w in t.lower().split():
                out[i, hash(w) % 32] += 1.0
        return out


def test_generic_text_adapter_is_inactive_without_artifacts_and_active_with_them(engine, tmp_path):
    assert not DLSimilarity.load_default(tmp_path).available and "not found" in DLSimilarity.load_default(tmp_path).reason
    enc = FakeEncoder()
    E = enc.embed([" ".join(JOBS[i][2]) + " " + JOBS[i][0] for i in range(len(JOBS))])
    rec = HybridRecommender(engine, kg=KG, dl=DLSimilarity.from_arrays(enc, engine.index.job_ids, E))
    r = rec.recommend(UserProfile.from_form(["docker", "kubernetes", "aws"], preferred_role="devops engineer"), k=3)
    assert "dl" in r.weights_effective and r.results[0].title == "DevOps Engineer" and abs(sum(r.weights_effective.values()) - 1.0) < 1e-4


def _tiny_skill_encoder(vocab):
    """a small UNTRAINED Keras encoder with a layer named 'latent' - TEST DOUBLE for the wiring (random weights: no quality claim)."""
    keras = pytest.importorskip("keras")
    keras.utils.set_random_seed(0)
    inp = keras.Input(shape=(len(vocab),))
    z = keras.layers.Dense(8, name="latent")(inp)
    return keras.Model(inp, z)


def test_skill_space_adapter_maps_by_name_and_source_key_and_uses_neutral_for_missing(engine):
    from jobintel.rec.catalog import ItemCatalog
    from jobintel.rec.dlsim import DLSkillSimilarity
    from jobintel.ir.ranker import UNKNOWN_SCORE
    cat = ItemCatalog(engine.index)
    vocab = sorted({s for j in JOBS for s in j[2]})
    enc = _tiny_skill_encoder(vocab)
    X = np.zeros((len(JOBS), len(vocab)), dtype=np.float32)
    for i, j in enumerate(JOBS):
        for s in j[2]:
            X[i, vocab.index(s)] = 1.0
    Z = np.asarray(enc.predict(X, verbose=0))
    has = [i for i, j in enumerate(JOBS) if j[2]]                              # the skill-less warehouse job gets NO embedding row
    dl = DLSkillSimilarity(enc, vocab, cat, source=["test"] * len(has), external_id=[str(i) for i in has], vectors=Z[has])
    assert dl.available and dl.n_docs_covered == len(has)
    rec = HybridRecommender(engine, cat, KG, dl=dl)
    p = UserProfile.from_form(["docker", "kubernetes", "aws", "jenkins"], preferred_role="devops engineer")
    z = dl.profile_vector(p.skills)
    assert z is not None and abs(np.linalg.norm(z) - 1.0) < 1e-5
    # the adapter's raw score for a job equals the cosine of the two latent codes it was built from (mapping by name and by (source, external_id) is right)
    di = [i for i, t in enumerate(engine.index.meta["title"]) if t == "DevOps Engineer"][0]
    raw = dl.score_docs(p, cat, [di])[0]
    assert raw == pytest.approx(float(Z[di] / np.linalg.norm(Z[di]) @ z), abs=1e-5)
    wh = [i for i, t in enumerate(engine.index.meta["title"]) if t == "Warehouse Assistant"][0]
    assert np.isnan(dl.score_docs(p, cat, [wh])[0])
    r = rec.recommend(p, k=12, restrict_docs=list(range(len(JOBS))))
    assert "dl" in r.weights_effective and 0 < r.dl_coverage < 1
    wh_row = [x for x in r.results if x.title == "Warehouse Assistant"]
    assert not wh_row or wh_row[0].signals["dl"] == pytest.approx(UNKNOWN_SCORE)       # no embedding -> neutral


def test_skill_space_adapter_is_inactive_without_files_or_known_skills(engine, tmp_path):
    from jobintel.rec.catalog import ItemCatalog
    from jobintel.rec.dlsim import DLSkillSimilarity
    cat = ItemCatalog(engine.index)
    off = DLSkillSimilarity.from_files(tmp_path, cat)
    assert not off.available and "not found" in off.reason
    rec = HybridRecommender(engine, cat, KG, dl=off)
    r = rec.recommend(UserProfile.from_form(["sql"], preferred_role="data analyst"), k=3)
    assert "dl" in r.inactive and r.dl_coverage is None
    vocab = ["sql"]
    dl = DLSkillSimilarity(_tiny_skill_encoder(vocab), vocab, cat, ["test"], ["0"], np.ones((1, 8)))
    assert dl.profile_vector({"underwater basket weaving": 1.0}) is None


def test_restrict_docs_ranks_exactly_that_candidate_set(engine):
    rec = HybridRecommender(engine, kg=KG)
    keep = [0, 2, 4, 7]
    r = rec.recommend(UserProfile.from_form(["sql", "excel"], preferred_role="data analyst"), k=20, restrict_docs=keep, collapse_duplicates=False)
    assert {x.job_id for x in r.results} == {int(engine.index.job_ids[d]) for d in keep}


def test_empty_and_unknown_profiles_do_not_crash(engine):
    rec = HybridRecommender(engine, kg=KG)
    r = rec.recommend(UserProfile(), k=5)
    assert r.results == [] and r.warnings
    r2 = rec.recommend(UserProfile.from_form(["underwater basket weaving"]), k=5)
    assert r2.results == [] or all(x.final_score >= 0 for x in r2.results)
    assert any("not present" in w for w in r2.warnings)


def test_strict_location_returns_only_the_requested_city_and_the_soft_mode_warns(engine):
    rec = HybridRecommender(engine, kg=KG)
    p = UserProfile.from_form(["docker", "kubernetes", "aws"], preferred_role="devops engineer", locations=["Pune"])
    soft = rec.recommend(p, k=1)
    assert soft.results[0].city == "Bengaluru" and any("strict_location" in w for w in soft.warnings)       # the DevOps job is in Bengaluru: location did not force a Pune job
    strict = rec.recommend(p, k=5, strict_location=True)
    assert strict.results and all(r.city == "Pune" for r in strict.results) and not strict.warnings
    nowhere = rec.recommend(UserProfile.from_form(["sql"], locations=["Atlantis"]), k=3, strict_location=True)
    assert nowhere.results == [] and any("no job in" in w for w in nowhere.warnings)
