#!/usr/bin/env python3
"""Compute the IR evaluation from your labeled sheet.

  python scripts/evaluate_ir.py --judgments labeled.xlsx --run-label manual-1
  python scripts/evaluate_ir.py --judgments labeled.xlsx extension.xlsx --run-label manual-2     # several sheets are merged
Writes docs/phase4_report.md and stores every (system, query) result in the ir_experiments table."""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import DATA_DIR, ROOT
from jobintel.db.session import get_engine
from jobintel.eval import judgments as J
from jobintel.eval import runner as R
from jobintel.eval.queries import load_queries
from jobintel.eval.report import render
from jobintel.eval.systems import MAIN_SYSTEMS, build_systems
from jobintel.ir.engine import DbDocStore
from jobintel.ir.index import InvertedIndex


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--judgments", type=Path, nargs="+", required=True)
    ap.add_argument("--run-label", default="manual-1")
    ap.add_argument("--relevance-source", default="manual", choices=["manual", "assistant_judged", "category_proxy"])
    ap.add_argument("--thr", type=int, default=1, choices=[1, 2], help="grade counted as relevant: 1 = partially or clearly, 2 = clearly only")
    ap.add_argument("--queries", type=Path, default=DATA_DIR / "eval" / "queries.json")
    ap.add_argument("--index", type=Path, default=DATA_DIR / "index" / "job_index.pkl")
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--systems", nargs="*", default=None)
    ap.add_argument("--allow-partial", action="store_true", help="evaluate even if some pooled jobs are unlabeled (unlabeled count as NOT relevant -> biased)")
    ap.add_argument("--skip-unlabeled-queries", action="store_true", help="evaluate only the queries you labeled completely (a query that is labeled only in part is still an error)")
    ap.add_argument("--no-db", action="store_true")
    ap.add_argument("--out", type=Path, default=ROOT / "docs" / "phase4_report.md")
    a = ap.parse_args()
    labels, stats, unl = {}, {"judged": 0, "unlabeled": 0, "grade_counts": {0: 0, 1: 0, 2: 0}}, {}
    for f in a.judgments:
        lab, st = J.read_labels(f)
        for q, d in lab.items():
            labels.setdefault(q, {}).update(d)
        for q, n in st["unlabeled_by_query"].items():
            unl[q] = unl.get(q, 0) + n
    if a.skip_unlabeled_queries:
        partial = sorted(q for q, n in unl.items() if n and labels.get(q))
        if partial:
            sys.exit(f"queries labeled only in part: {partial} - finish them or leave them completely blank.")
        skipped = sorted(q for q, n in unl.items() if n and not labels.get(q))
        print(f"skipping completely unlabeled queries: {skipped}")
        unl = {}
    stats["unlabeled"] = sum(unl.values())
    stats["judged"] = sum(len(d) for d in labels.values())
    stats["grade_counts"] = {g: sum(1 for d in labels.values() for v in d.values() if v == g) for g in (0, 1, 2)}
    if stats["unlabeled"] and not a.allow_partial:
        sys.exit(f"{stats['unlabeled']} jobs in the sheet have no relevance label - finish labeling (or use --allow-partial, which biases the results).")
    queries = load_queries(a.queries)
    ix = InvertedIndex.load(a.index)
    labels, missing = J.resolve_labels(labels, ix)
    if missing:
        print(f"note: {missing} labeled jobs do not exist in this index (different data?) and are ignored")
    stats["judged"] = sum(len(d) for d in labels.values())
    stats["grade_counts"] = {g: sum(1 for d in labels.values() for v in d.values() if v == g) for g in (0, 1, 2)}
    systems = build_systems(ix, DbDocStore(get_engine()), k=a.depth)
    names = a.systems or list(systems)
    df = R.evaluate(systems, names, queries, labels, a.thr, progress=lambda n: print("  evaluated", n))
    md = render(df, queries, stats, a.relevance_source, a.thr, a.depth, a.run_label, ix.n_docs)
    a.out.write_text(md, encoding="utf-8")
    if not a.no_db:
        n = R.store_results(get_engine(), df, a.run_label, a.relevance_source, ix.n_docs, {"thr": a.thr, "depth": a.depth})
        print(f"stored {n} rows in ir_experiments (run_label {a.run_label})")
    print(md.split("## 2.")[0])
    print("full report:", a.out)


if __name__ == "__main__":
    main()
