#!/usr/bin/env python3
"""Job-market skill analysis with the MapReduce engine, validated against SQL.

   python scripts/run_mapreduce.py                       # runs with 1 worker and with --workers, validates, writes data/analytics/market.json
   python scripts/run_mapreduce.py --workers 4 --index-sample 20000"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import DATA_DIR
from jobintel.db.session import get_engine
from jobintel.ir.index import InvertedIndex
from jobintel.mapreduce import jobs as J
from jobintel.mapreduce.framework import run_mapreduce, sum_reducer
from jobintel.mapreduce.market import load_market_records, sql_scalar, sql_skill_counts


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=min(4, mp.cpu_count()))
    ap.add_argument("--out", type=Path, default=DATA_DIR / "analytics" / "market.json")
    ap.add_argument("--index-sample", type=int, default=10000, help="documents for the MapReduce-built inverted index check (0 = skip)")
    ap.add_argument("--start-method", default=None, help="fork | spawn (default: the platform default)")
    a = ap.parse_args()
    e = get_engine()
    t0 = time.time()
    recs = load_market_records(e)
    print(f"records: {len(recs):,} jobs read from MySQL in {time.time() - t0:.1f}s   (cpu cores: {mp.cpu_count()})")

    runs = {}
    for w in sorted({1, a.workers}):
        r = run_mapreduce(recs, J.market_mapper, sum_reducer, n_workers=w, chunk_size=2000, combiner=J.sum_combiner, start_method=a.start_method)
        runs[w] = r
        print(f"  workers={w}: {r.stats}")
    base = runs[1]
    for w, r in runs.items():
        assert r.output == base.output, f"result with {w} workers differs from the sequential run"
    print(f"  all runs produce IDENTICAL output ({base.stats['keys']:,} keys). Combiner removed "
          f"{100 * (1 - base.stats['pairs_after_combine'] / base.stats['pairs_emitted']):.0f}% of the {base.stats['pairs_emitted']:,} emitted pairs before the shuffle.")
    if a.workers > 1:
        print(f"  measured speedup map+reduce with {a.workers} workers on {mp.cpu_count()} core(s): {runs[1].stats['total_s'] / runs[a.workers].stats['total_s']:.2f}x")

    # ---- validation against SQL ---------------------------------------------------------------------------------------
    o = base.output
    mr = {(k[2], k[1]): v for k, v in o.items() if k[0] == "skill"}
    sq = sql_skill_counts(e)
    assert mr == sq, "MapReduce skill counts differ from SQL GROUP BY"
    print(f"VALIDATION 1: {len(sq):,} (skill, type) counts are IDENTICAL to the SQL GROUP BY result")
    pairs = [("python", "sql"), ("machine learning", "python"), ("java", "sql")]
    for x, y in pairs:
        n_sql = sql_scalar(e, f"""SELECT COUNT(*) FROM (SELECT js.job_id FROM job_skills js JOIN skills k USING(skill_id) JOIN jobs j USING(job_id) JOIN crawler_sources s USING(source_id)
                  WHERE {__import__('jobintel.mapreduce.market', fromlist=['WHERE']).WHERE} AND k.skill_type IN ('technical','tool') AND k.canonical_name IN (:x,:y)
                  GROUP BY js.job_id HAVING COUNT(DISTINCT k.canonical_name) = 2) t""", x=x, y=y)
        n_mr = o.get(("pair",) + tuple(sorted((x, y))), 0)
        print(f"VALIDATION 2: jobs listing both '{x}' and '{y}': MapReduce {n_mr:,} vs SQL {n_sql:,} -> {'OK' if n_mr == n_sql else 'MISMATCH'}")
        assert n_mr == n_sql
    cl_sql = sql_scalar(e, f"""SELECT COUNT(DISTINCT j.job_id) FROM job_skills js JOIN skills k USING(skill_id) JOIN jobs j USING(job_id) JOIN crawler_sources s USING(source_id)
              WHERE {__import__('jobintel.mapreduce.market', fromlist=['WHERE']).WHERE} AND k.canonical_name='python' AND j.cluster='data_ai' AND k.skill_type IN ('technical','tool','knowledge','certification')""")
    print(f"VALIDATION 3: 'python' in the data_ai cluster: MapReduce {o[('cluster_skill', 'data_ai', 'python')]:,} vs SQL {cl_sql:,}")
    assert o[("cluster_skill", "data_ai", "python")] == cl_sql

    # ---- inverted index via MapReduce -------------------------------------------------------------------------------------
    ixp = DATA_DIR / "index" / "job_index.pkl"
    if a.index_sample and ixp.exists():
        ix = InvertedIndex.load(ixp)
        m = min(a.index_sample, ix.n_docs)
        docs = [(i, [ix.terms[t] for t in ix.doc_tokens(i)]) for i in range(m)]
        r = run_mapreduce(docs, J.index_mapper, J.index_reducer, n_workers=a.workers, chunk_size=1000, start_method=a.start_method)
        bad = 0
        for term, postings in r.output.items():
            tid = ix.term_id(term)
            d, f = ix.postings(tid)
            keep = d < m
            bad += postings != list(zip(d[keep].tolist(), f[keep].tolist()))
        n_expected = sum(1 for t in range(ix.n_terms) if ix.postings(t)[0].size and ix.postings(t)[0][0] < m)
        print(f"VALIDATION 4: inverted index of the first {m:,} documents built by MapReduce: {len(r.output):,} terms (expected {n_expected:,}), "
              f"{bad} posting lists differ from the real index  [{r.stats['total_s']}s, {r.stats['pairs_emitted']:,} pairs]")
        assert bad == 0 and len(r.output) == n_expected

    a.out.parent.mkdir(parents=True, exist_ok=True)
    market = J.assemble_market(base.output)
    market["_meta"] = {"records": len(recs), "mapreduce_stats": base.stats, "note": "synthetic and fraudulent postings excluded; skills by role/city/experience/month use technical, tool, knowledge and certification skills only"}
    a.out.write_text(json.dumps(market, indent=1, default=list), encoding="utf-8")
    print("wrote", a.out)
    print("top technical skills:", market["top_skills_by_type"].get("technical", [])[:10])
    print("top pairs by lift:", [(p["a"], p["b"], p["jobs"], p["lift"]) for p in market["top_pairs_by_lift"][:5]])


if __name__ == "__main__":
    main()
