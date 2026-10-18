#!/usr/bin/env python3
"""Sequential vs parallel retrieval benchmark.  RUN THIS ON YOUR OWN MACHINE: the speedup depends on your CPU cores.

   python scripts/benchmark_parallel.py                       # sizes x1 and x4, P = 1,2,4
   python scripts/benchmark_parallel.py --sizes 1 4 8 --parts 1 2 4 8
x4 / x8 repeat the real collection 4 / 8 times (new document ids) ONLY to study how the cost grows with collection size.
Results: data/benchmarks/parallel.json and docs/parallel_benchmark.md"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import DATA_DIR, ROOT
from jobintel.extraction.extractor import Extractor
from jobintel.ir.benchmark import BENCH_QUERIES, machine_info, run_benchmark, to_markdown
from jobintel.ir.index import InvertedIndex
from jobintel.ir.query import QueryProcessor

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--index", type=Path, default=DATA_DIR / "index" / "job_index.pkl")
    ap.add_argument("--sizes", type=int, nargs="*", default=[1, 4])
    ap.add_argument("--parts", type=int, nargs="*", default=[1, 2, 4])
    ap.add_argument("--executors", nargs="*", default=["sequential", "threads", "processes"])
    ap.add_argument("--models", nargs="*", default=["lm", "hybrid"])
    ap.add_argument("--reps", type=int, default=3)
    ap.add_argument("--start-method", default=None)
    a = ap.parse_args()
    ix = InvertedIndex.load(a.index)
    qp = QueryProcessor(Extractor())
    pq = []
    for q in BENCH_QUERIES:
        p = qp.parse(q)
        pq.append((p.qterms(ix)[0], p.sequence(ix)))
    info = machine_info()
    print("machine:", info)
    rows = run_benchmark(ix, pq, a.sizes, a.parts, a.executors, a.models, a.reps, a.start_method)
    out = DATA_DIR / "benchmarks"
    out.mkdir(parents=True, exist_ok=True)
    (out / "parallel.json").write_text(json.dumps({"machine": info, "rows": rows}, indent=1), encoding="utf-8")
    md = "# Sequential vs parallel retrieval benchmark\n\n" + to_markdown(rows, info)
    (ROOT / "docs" / "parallel_benchmark.md").write_text(md, encoding="utf-8")
    print("\n" + md)
