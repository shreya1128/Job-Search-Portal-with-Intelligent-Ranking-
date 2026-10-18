import pytest

from jobintel.ir.index import IndexDoc, InvertedIndex
from jobintel.mapreduce import jobs as J
from jobintel.mapreduce.framework import run_mapreduce, stable_hash, sum_reducer

WORDS = [("a b a", 1), ("b c", 2), ("a a a d", 3)] * 40


def wc_mapper(rec):
    for w in rec[0].split():
        yield w, 1


def wc_combiner(k, vs):
    return sum(vs)


EXPECT = {"a": 5 * 40, "b": 2 * 40, "c": 40, "d": 40}


def test_wordcount_matches_a_plain_counter_with_and_without_combiner():
    plain = run_mapreduce(WORDS, wc_mapper, sum_reducer, n_workers=1, chunk_size=7)
    comb = run_mapreduce(WORDS, wc_mapper, sum_reducer, n_workers=1, chunk_size=7, combiner=wc_combiner)
    assert plain.output == comb.output == EXPECT
    assert comb.stats["pairs_after_combine"] < plain.stats["pairs_after_combine"] == plain.stats["pairs_emitted"]       # the combiner shrinks the shuffle


import multiprocessing as mp

@pytest.mark.parametrize("method", [m for m in ("fork", "spawn") if m in mp.get_all_start_methods()])
def test_parallel_workers_give_the_same_result(method):
    r = run_mapreduce(WORDS, wc_mapper, sum_reducer, n_workers=2, chunk_size=10, combiner=wc_combiner, start_method=method)
    assert r.output == EXPECT and r.stats["workers"] == 2 and r.stats["reducers"] == 2


def test_stable_hash_is_process_independent():
    assert stable_hash(("skill", "technical", "python")) == stable_hash(("skill", "technical", "python"))
    assert stable_hash("a") != stable_hash("b")


def rec(job_id, cluster, cities, exp, month, skills, source="naukri", company=None):
    return dict(job_id=job_id, source=source, cluster=cluster, cities=cities, exp_min=exp, month=month, company=company, category=None, salary_min=None,
                salary_cur=None, salary_period=None, skills=skills)


def test_market_job_counts_pairs_and_distributions():
    recs = [rec(1, "data_ai", ["Pune"], 2, "2019-05", [("python", "technical"), ("sql", "technical"), ("teamwork", "soft")]),
            rec(2, "data_ai", ["Pune", "Mumbai"], 4, "2019-06", [("python", "technical"), ("sql", "technical")]),
            rec(3, "healthcare", ["Pune"], 0.5, "2019-06", [("nursing", "knowledge"), ("python", "technical")])] * 30
    r = run_mapreduce(recs, J.market_mapper, sum_reducer, n_workers=1, combiner=J.sum_combiner)
    o = r.output
    assert o[("jobs",)] == 90 and o[("skill", "technical", "python")] == 90 and o[("skill", "soft", "teamwork")] == 30
    assert o[("pair", "python", "sql")] == 60 and ("pair", "nursing", "python") not in o            # pairs only among technical/tool skills
    assert o[("cluster_skill", "data_ai", "python")] == 60 and o[("city_skill", "Mumbai", "python")] == 30
    assert o[("exp_skill", "1-3 yrs", "python")] == 30 and o[("exp_skill", "3-5 yrs", "sql")] == 30 and o[("exp_skill", "0-1 yrs", "nursing")] == 30
    assert ("skill", "soft", "teamwork") in o and not any(k[0] == "cluster_skill" and k[2] == "teamwork" for k in o)        # soft skills stay out of role tables
    a = J.assemble_market(o)
    assert a["total_jobs"] == 90 and a["top_skills_by_type"]["technical"][0] == ("python", 90)
    assert a["distributions"]["city"]["Pune"] == 90 and a["distributions"]["month"] == {"2019-06": 60, "2019-05": 30}
    pr = next(p for p in a["top_pairs_by_count"] if (p["a"], p["b"]) == ("python", "sql"))
    assert pr["jobs"] == 60 and pr["lift"] == round(60 * 90 / (90 * 60), 2)


def test_inverted_index_built_by_mapreduce_equals_the_real_index():
    docs = {0: "python sql python", 1: "sql java", 2: "python java java docker", 3: "nurse care"}
    ix = InvertedIndex.build(IndexDoc(i, t.split(), {"skills": [], "cities": None, "city": None}) for i, t in docs.items())
    r = run_mapreduce([(i, t.split()) for i, t in docs.items()], J.index_mapper, J.index_reducer, n_workers=2, chunk_size=1)
    assert len(r.output) == ix.n_terms
    for term, postings in r.output.items():
        tid = ix.term_id(term)
        d, f = ix.postings(tid)
        assert postings == list(zip(d.tolist(), f.tolist())) and sum(x[1] for x in postings) == ix.cf[tid] and len(postings) == ix.df[tid]
