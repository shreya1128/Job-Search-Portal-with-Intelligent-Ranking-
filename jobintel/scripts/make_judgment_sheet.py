#!/usr/bin/env python3
"""Build the relevance-judgment sheet you label by hand (pooled top results of several systems, shuffled, system names hidden).

  python scripts/make_judgment_sheet.py                       # -> data/eval/judgments_to_label.xlsx (+ .csv, pool.json)
  python scripts/make_judgment_sheet.py --depth 8             # smaller pool = fewer rows to label
  python scripts/make_judgment_sheet.py --extend labeled.xlsx # later: only the jobs that the labeled sheet does not cover yet
                                                             # (needed when you evaluate additional systems / new settings)"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import DATA_DIR
from jobintel.db.session import get_engine
from jobintel.eval import judgments as J
from jobintel.eval.queries import load_queries
from jobintel.eval.systems import POOL_SYSTEMS, build_systems
from jobintel.ir.engine import DbDocStore
from jobintel.ir.index import InvertedIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--queries", type=Path, default=DATA_DIR / "eval" / "queries.json")
    ap.add_argument("--index", type=Path, default=DATA_DIR / "index" / "job_index.pkl")
    ap.add_argument("--depth", type=int, default=10, help="top-N of every pooled system goes into the pool")
    ap.add_argument("--systems", nargs="*", default=None, help="systems to pool (default: baseline, LM, DFR, hybrid, hybrid without PageRank; add e.g. lm_bigram later with --extend)")
    ap.add_argument("--out", type=Path, default=DATA_DIR / "eval" / "judgments_to_label.xlsx")
    ap.add_argument("--pool", type=Path, default=DATA_DIR / "eval" / "pool.json")
    ap.add_argument("--extend", type=Path, nargs="*", help="already-labeled sheet(s): output only the jobs they do not cover")
    a = ap.parse_args()
    queries = load_queries(a.queries)
    ix = InvertedIndex.load(a.index)
    store = DbDocStore(get_engine())
    systems = build_systems(ix, store, k=a.depth)
    names = a.systems or POOL_SYSTEMS
    pool = J.build_pool(systems, queries, names, a.depth)
    only = None
    if a.extend:
        have = {}
        for f in a.extend:
            for q, d in J.read_labels(f)[0].items():
                have.setdefault(q, set()).update(d)
        only = {q: {j for j in p["job_ids"] if J.keys_of(ix, [j]).pop() not in have.get(q, set())} for q, p in pool.items()}
        if a.pool.exists():                                         # keep the old pool membership too
            old = J.load_pool(a.pool)
            for q, p in pool.items():
                p["job_ids"] = list(dict.fromkeys(old.get(q, {}).get("job_ids", []) + p["job_ids"]))
    rows = J.sheet_rows(pool, ix, store, only)
    J.save_pool(pool, a.pool)
    J.write_sheet(rows, queries, a.out)
    J.write_csv(rows, a.out.with_suffix(".csv"))
    per = [sum(1 for r in rows if r["query_id"] == q["id"]) for q in queries]
    print(f"systems pooled: {names} (depth {a.depth})")
    print(f"{len(rows)} jobs to label for {len(queries)} queries (per query: min {min(per)}, mean {sum(per) / len(per):.1f}, max {max(per)})")
    print("wrote", a.out, "and", a.out.with_suffix(".csv"), "| pool:", a.pool)


if __name__ == "__main__":
    main()
