import numpy as np
import pytest

from jobintel.ir import models as M
from jobintel.ir.cluster import ClusterRouter
from jobintel.ir.distributed import DistributedEngine, ProcessExecutor, ResultMerger, SequentialExecutor
from jobintel.ir.engine import MODELS, MemoryDocStore, SearchEngine
from jobintel.ir.index import IndexDoc, InvertedIndex
from jobintel.ir.partition import build_partitions, replicate, save_partitions
from jobintel.processing.text import index_tokens
from tests.test_ir_engine import EXT, JOBS, LEX, make_engine

CLUSTERS = ["data_ai", "data_ai", "healthcare", "software_it", "data_ai"]
QUERIES = ["python machine learning", "python sql", "machine learning", "nurse patient care", "docker python sql", "learning"]


def mono():
    return make_engine(pagerank_values=[0.1, 0.2, 0.3, 0.15, 0.25], clusters=CLUSTERS)


def same(a, b):
    assert [x.job_id for x in a.results] == [x.job_id for x in b.results]
    assert np.allclose([x.score for x in a.results], [x.score for x in b.results], atol=1e-9)
    assert a.n_candidates == b.n_candidates


# --------------------------------------------------------------------------- partitions
@pytest.mark.parametrize("strategy", ["range", "roundrobin"])
@pytest.mark.parametrize("p", [1, 2, 3, 5])
def test_partitions_cover_the_collection_exactly(p, strategy):
    ix = mono().index
    parts = build_partitions(ix, p, strategy)
    assert sum(x.n_local for x in parts) == ix.n_docs
    assert sorted(np.concatenate([x.global_idx for x in parts]).tolist()) == list(range(ix.n_docs))
    for t in range(ix.n_terms):
        docs = np.sort(np.concatenate([x.global_idx[x.postings(t)[0]] for x in parts]))
        assert docs.tolist() == ix.postings(t)[0].tolist()
        tf = {int(d): int(f) for x in parts for d, f in zip(x.global_idx[x.postings(t)[0]], x.postings(t)[1])}
        assert tf == dict(zip(ix.postings(t)[0].tolist(), ix.postings(t)[1].tolist()))
    assert all(x.cf is parts[0].cf and x.n_docs == ix.n_docs and x.total_tokens == ix.total_tokens for x in parts)       # global statistics everywhere


def test_partition_scores_equal_monolithic_scores():
    ix = mono().index
    qt = [(ix.term_id("python"), 1), (ix.term_id("sql"), 1)]
    allc = M.candidates(ix, qt)
    for fn in (lambda i, c: M.unigram_lm(i, qt, c), lambda i, c: M.dfr(i, qt, c), lambda i, c: M.keyword_baseline(i, qt, c)):
        full = dict(zip(allc.tolist(), fn(ix, allc)))
        for part in build_partitions(ix, 3):
            c = M.candidates(part, qt)
            for g, s in zip(part.global_idx[c], fn(part, c)):
                assert abs(full[int(g)] - s) < 1e-12


# --------------------------------------------------------------------------- distributed == single-process (every model, filters, routing)
@pytest.mark.parametrize("executor", ["sequential", "threads"])
@pytest.mark.parametrize("strategy", ["range", "roundrobin"])
@pytest.mark.parametrize("p", [1, 2, 3, 5])
def test_distributed_ranking_is_identical_to_single_process(p, strategy, executor):
    ref = mono()
    with DistributedEngine.build(ref.index, ref.docstore, p, executor, strategy, extractor=EXT, lexicon=LEX) as dist:
        for q in QUERIES:
            for m in MODELS:
                same(ref.search(q, k=5, model=m), dist.search(q, k=5, model=m))
        for kw in ({"filters": {"city": "Pune"}}, {"filters": {"min_experience": 0, "max_experience": 3}}, {"cluster_routing": True}):
            for m in ("lm", "dfr", "lm_bigram", "hybrid"):
                same(ref.search("python machine learning sql", k=5, model=m, **kw), dist.search("python machine learning sql", k=5, model=m, **kw))


def test_process_executor_with_spawn_matches_and_reports_timings(tmp_path):
    ref = mono()
    parts = build_partitions(ref.index, 3)
    ex = ProcessExecutor(save_partitions(parts, tmp_path), start_method="spawn")          # 'spawn' = what Windows uses
    from jobintel.ir.distributed import QueryCoordinator
    dist = DistributedEngine(ref.index, ref.docstore, QueryCoordinator(ex), extractor=EXT, lexicon=LEX)
    try:
        for m in MODELS:
            r = dist.search("python machine learning", k=4, model=m)
            same(ref.search("python machine learning", k=4, model=m), r)
        t = dist.search("python sql", model="lm").timings_ms
        assert {"exec_ms", "nodes_max_ms", "nodes_sum_ms", "merge_ms"} <= set(t) and t["nodes_sum_ms"] >= t["nodes_max_ms"] > 0
    finally:
        dist.close()
    assert all(not p.is_alive() for p in ex.procs)


def test_merger_is_deterministic_on_ties():
    a = {"n_candidates": 2, "docs": np.array([4, 0]), "score": np.array([1.0, 1.0])}
    b = {"n_candidates": 1, "docs": np.array([2]), "score": np.array([1.0])}
    st = ResultMerger.merge("lm", [a, b], 3)
    assert st.ranked.tolist() == [0, 2, 4] and st.n_candidates == 3          # equal scores -> lower document index first, whatever the node order
    assert ResultMerger.merge("lm", [{"n_candidates": 0, "docs": np.zeros(0, dtype=np.int64)}], 3).ranked.tolist() == []


def test_replicated_collection_has_consistent_statistics():
    ix = mono().index
    big = replicate(ix, 3)
    assert len(big.doc_len) == 3 * ix.n_docs and int(big.cf.sum()) == 3 * ix.total_tokens
    t = ix.term_id("python")
    a, b = big.indptr[t], big.indptr[t + 1]
    docs, tf = big.post_doc[a:b], big.post_tf[a:b]
    assert len(docs) == 3 * ix.df[t] and np.all(np.diff(docs) > 0) and int(tf.sum()) == 3 * ix.cf[t]
    parts = build_partitions(big, 4)
    assert sum(p.n_local for p in parts) == len(big.doc_len)


# --------------------------------------------------------------------------- cluster routing
def test_cluster_router_routes_to_the_matching_cluster():
    eng = mono()
    r = ClusterRouter.from_index(eng.index)
    qt = lambda text: [(eng.index.term_id(w), 1) for w in index_tokens(text) if eng.index.term_id(w) >= 0]
    nurse = r.route(qt("nurse patient care"), mu=1.0)            # mu=1: this toy collection has 3 healthcare tokens (the default mu=500 is sized for 7 M)
    assert nurse.clusters[0] == "healthcare" and nurse.posteriors["healthcare"] > 0.9 and nurse.doc_fraction < 0.5
    tech = r.route(qt("python machine learning"), mu=1.0)
    assert tech.clusters[0] in ("data_ai", "software_it") and "healthcare" not in tech.clusters
    assert r.docs_in(["healthcare"]).tolist() == [2] and r.docs_in([]).tolist() == []


def test_cluster_routing_restricts_the_search_and_reports_it():
    eng = mono()
    plain = eng.search("python machine learning", model="lm", k=5)
    routed = eng.search("python machine learning", model="lm", k=5, cluster_routing=True)
    assert routed.routing and routed.routing["clusters"] and routed.n_candidates <= plain.n_candidates
    assert eng.search("nurse patient", model="lm", cluster_routing=True).results[0].title == "Registered Nurse"
