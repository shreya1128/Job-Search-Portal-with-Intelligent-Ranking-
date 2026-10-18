#!/usr/bin/env python3
"""Phase-3 test gate on the REAL index: (1) partitioned/parallel search returns EXACTLY the single-process ranking, (2) cluster routing is
measured (search-space reduction AND what it costs in lost results), (3) MapReduce output equals SQL, (4) a small sequential-vs-parallel
benchmark.  Writes docs/phase3_report.md."""
import statistics as st
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import DATA_DIR, ROOT
from jobintel.db.session import get_engine
from jobintel.extraction.extractor import Extractor
from jobintel.ir import models as M
from jobintel.ir.benchmark import BENCH_QUERIES, machine_info, run_benchmark, to_markdown
from jobintel.ir.distributed import DistributedEngine, ProcessExecutor, QueryCoordinator
from jobintel.ir.engine import MODELS, DbDocStore, SearchEngine
from jobintel.ir.index import InvertedIndex
from jobintel.ir.partition import build_partitions, save_partitions
from jobintel.ir.query import QueryProcessor
from jobintel.mapreduce import jobs as J
from jobintel.mapreduce.framework import run_mapreduce, sum_reducer
from jobintel.mapreduce.market import load_market_records, sql_skill_counts

QUERIES = ["python machine learning sql", "Entry level machine learning jobs in Pune", "data analyst sql excel", "registered nurse", "accountant tax",
           "senior data scientist tensorflow deep learning", "java spring boot microservices", "devops engineer docker kubernetes aws", "customer service representative",
           "financial analyst forecasting excel", "python developer 3 years experience", "business intelligence power bi", "nursing patient care icu",
           "react javascript frontend developer", "cybersecurity analyst", "project manager agile scrum", "primary school teacher", "warehouse forklift operator"]
def main():
    out, fails = ["# Phase 3 verification report\n"], []


    def check(name, ok, detail=""):
        out.append(f"- {'PASS' if ok else 'FAIL'} - {name}" + (f" ({detail})" if detail else ""))
        if not ok:
            fails.append(name)


    e = get_engine()
    ix = InvertedIndex.load(DATA_DIR / "index" / "job_index.pkl")
    ref = SearchEngine(ix, DbDocStore(e))
    info = machine_info()
    out.append(f"Machine: {info['cpu_cores']} CPU core(s) - {info['platform']} - Python {info['python']} - NumPy {info['numpy']}\n\nIndex: `{ix.stats()}`\n")

    # ============================================================================================ 1. partitions + parallel == single process
    out.append("## 1. Partitioned / parallel search equals single-process search\n")
    parts = build_partitions(ix, 4)
    check("4 range partitions cover every document exactly once", sum(p.n_local for p in parts) == ix.n_docs and len(set(np.concatenate([p.global_idx for p in parts]).tolist())) == ix.n_docs,
          "sizes " + str([p.n_local for p in parts]))
    check("postings are partitioned without loss", sum(len(p.post_doc) for p in parts) == len(ix.post_doc), f"{len(ix.post_doc):,} postings")
    configs = [("sequential", 4, "range"), ("sequential", 3, "roundrobin"), ("threads", 4, "range"), ("processes", 2, "range")]
    n_cmp = bad = 0
    t0 = time.time()
    for executor, p, strat in configs:
        with DistributedEngine.build(ix, DbDocStore(e), p, executor, strat, start_method="spawn" if executor == "processes" else None) as dist:
            cases = [(q, m, {}) for q in QUERIES[:10] for m in MODELS] + [(q, m, {"filters": {"city": "Pune"}}) for q in QUERIES[:4] for m in ("lm", "dfr", "lm_bigram", "hybrid")] \
                    + [(q, m, {"cluster_routing": True}) for q in QUERIES[:6] for m in ("lm", "dfr", "hybrid")]
            for q, m, kw in cases:
                a, b = ref.search(q, k=10, model=m, explain=False, **kw), dist.search(q, k=10, model=m, explain=False, **kw)
                n_cmp += 1
                ok = [x.job_id for x in a.results] == [x.job_id for x in b.results] and np.allclose([x.score for x in a.results], [x.score for x in b.results], atol=1e-9) and a.n_candidates == b.n_candidates
                bad += not ok
                if not ok:
                    print("MISMATCH", executor, p, strat, q, m, kw)
    check("distributed ranking is IDENTICAL to the single-process ranking (job ids, scores, candidate counts)", bad == 0,
          f"{n_cmp} comparisons over {len(configs)} configurations (sequential P=4, round-robin P=3, threads P=4, real processes P=2 started with 'spawn'), all 6 models, with filters and cluster routing; {time.time() - t0:.0f}s")

    # ============================================================================================ 2. cluster routing
    out.append("\n## 2. Cluster-based retrieval (route the query to the most likely clusters, then search only those)\n")
    router = ref._router()
    out.append("Cluster sizes: " + ", ".join(f"{c} {len(router.docs_in([c])):,}" for c in sorted(router.labels, key=lambda c: -len(router.docs_in([c])))) +
               "\n\nClusters are title-keyword heuristics (the four named in the brief plus eight more), not ground truth.\n")
    qp = QueryProcessor(Extractor())
    rows = []
    outside_total = 0
    for q in QUERIES:
        full_t, rout_t = [], []
        for _ in range(5):
            full = ref.search(q, k=10, model="lm", explain=False)
            rout = ref.search(q, k=10, model="lm", explain=False, cluster_routing=True)
            full_t.append(full.timings_ms["total"]); rout_t.append(rout.timings_ms["total"])
        ids_f, ids_r = [x.job_id for x in full.results], [x.job_id for x in rout.results]
        clusters = rout.routing["clusters"] if rout.routing else []
        in_c = set(router.docs_in(clusters).tolist()) if clusters else set()
        kept = sum(ix.doc_index(j) in in_c for j in ids_f)
        outside_total += sum(ix.meta["cluster"][ix.doc_index(x.job_id)] not in clusters for x in rout.results)
        rows.append(dict(q=q, clusters=clusters, post=(rout.routing or {}).get("posteriors", {}), frac=(rout.routing or {}).get("doc_fraction", 1.0), cand_f=full.n_candidates, cand_r=rout.n_candidates,
                         ms_f=st.median(full_t), ms_r=st.median(rout_t), overlap=len(set(ids_f) & set(ids_r)) / max(1, len(ids_f)), kept=kept / max(1, len(ids_f))))
    out.append("| query | clusters searched (posterior of the best) | docs searched | candidates full -> routed | LM ms full / routed | top-10 overlap | full top-10 still reachable |\n|---|---|---|---|---|---|---|")
    for r in rows:
        best = max(r["post"].values()) if r["post"] else 0
        out.append(f"| {r['q'][:44]} | {', '.join(r['clusters'])} ({best:.2f}) | {100 * r['frac']:.0f}% | {r['cand_f']:,} -> {r['cand_r']:,} | {r['ms_f']:.1f} / {r['ms_r']:.1f} | {r['overlap']:.2f} | {r['kept']:.2f} |")
    mean = lambda k: st.mean(r[k] for r in rows)
    out.append(f"\n**Mean over {len(rows)} queries:** {100 * mean('frac'):.0f}% of the documents searched, candidates reduced by {100 * (1 - mean('cand_r') / mean('cand_f')):.0f}%, "
               f"top-10 overlap with the full search {mean('overlap'):.2f}, share of the full top-10 that lies inside the routed clusters {mean('kept'):.2f}. "
               f"Overlap below 1 means routing LOST some results the full search returned (they live in clusters that were not searched) - this is the price of the smaller search space; "
               f"whether the lost results were relevant is a Phase-4 question.")
    check("routing returns clusters and shrinks the search space for every query", all(r["clusters"] and r["frac"] < 1 for r in rows))
    check("routing never returns a document from a cluster that was not chosen", outside_total == 0, f"checked {sum(1 for _ in rows)} queries x 10 results")

    # ============================================================================================ 3. MapReduce == SQL
    out.append("\n## 3. MapReduce (skill analysis and inverted index) equals SQL / the real index\n")
    recs = load_market_records(e)
    r1 = run_mapreduce(recs, J.market_mapper, sum_reducer, n_workers=1, combiner=J.sum_combiner)
    mr = {(k[2], k[1]): v for k, v in r1.output.items() if k[0] == "skill"}
    sq = sql_skill_counts(e)
    check("MapReduce skill counts are identical to SQL GROUP BY", mr == sq, f"{len(sq):,} (skill,type) counts over {len(recs):,} jobs")
    r2 = run_mapreduce(recs[:20000], J.market_mapper, sum_reducer, n_workers=2, combiner=J.sum_combiner, start_method="spawn")
    r3 = run_mapreduce(recs[:20000], J.market_mapper, sum_reducer, n_workers=1, combiner=J.sum_combiner)
    check("2 worker processes (spawn) give the same output as 1 worker", r2.output == r3.output, f"{r3.stats['keys']:,} keys")
    out.append(f"- map/shuffle/reduce stats (1 worker, all jobs): `{r1.stats}`; the combiner removed {100 * (1 - r1.stats['pairs_after_combine'] / r1.stats['pairs_emitted']):.0f}% of the emitted pairs before the shuffle")
    m = 5000
    docs = [(i, [ix.terms[t] for t in ix.doc_tokens(i)]) for i in range(m)]
    ri = run_mapreduce(docs, J.index_mapper, J.index_reducer, n_workers=1, chunk_size=1000)
    bad = 0
    for term, postings in ri.output.items():
        tid = ix.term_id(term)
        d, f = ix.postings(tid)
        keep = d < m
        bad += postings != list(zip(d[keep].tolist(), f[keep].tolist()))
    check("inverted index built by MapReduce equals the real index (first 5,000 documents)", bad == 0 and len(ri.output) == sum(1 for t in range(ix.n_terms) if ix.postings(t)[0].size and ix.postings(t)[0][0] < m), f"{len(ri.output):,} terms")

    # ============================================================================================ 4. benchmark
    out.append("\n## 4. Sequential vs parallel retrieval (first stage: candidates + scoring + merge)\n")
    pq = []
    for q in BENCH_QUERIES:
        p = qp.parse(q)
        pq.append((p.qterms(ix)[0], p.sequence(ix)))
    rows_b = run_benchmark(ix, pq, sizes=(1,), parts=(1, 2, 4), executors=("sequential", "threads", "processes"), models=("lm", "hybrid"), reps=2, progress=lambda *_: None)
    out.append(to_markdown(rows_b, info))
    out.append("\n*Speedup/efficiency are MEASURED wall-clock. 'Ideal speedup' is an ESTIMATE from the measured per-node times (what P free cores would give if communication were free).* "
               f"This machine has {info['cpu_cores']} core(s); with fewer cores than partitions the measured speedup cannot exceed ~1 - run `python scripts/benchmark_parallel.py` on your own machine for the real figure. "
               "`docs/parallel_benchmark.md` also covers collections 4x larger.")
    check("every executor returned the same candidate counts", len({(r['parts'], r['model'], round(r['candidates'])) for r in rows_b}) == len({(r['parts'], r['model']) for r in rows_b}))
    out.append(f"\n**Result: {'ALL CHECKS PASSED' if not fails else 'FAILED: ' + ', '.join(fails)}**")
    (ROOT / "docs" / "phase3_report.md").write_text("\n".join(out), encoding="utf-8")
    print("\n".join(out))
    sys.exit(1 if fails else 0)


if __name__ == "__main__":      # required: 'spawn' (Windows) re-imports this file in every worker process
    main()
