#!/usr/bin/env python3
"""Command-line job search over the built index.

  python scripts/search_cli.py "python machine learning sql"
  python scripts/search_cli.py "Entry level machine learning jobs in Pune" --model hybrid --k 5
  python scripts/search_cli.py "data analyst" --city Pune --min-exp 0 --max-exp 2 --model dfr
  python scripts/search_cli.py "python developer" --compare            # all models side by side"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import DATA_DIR
from jobintel.db.session import get_engine
from jobintel.ir.distributed import DistributedEngine
from jobintel.ir.engine import MODELS, DbDocStore, SearchEngine
from jobintel.ir.index import InvertedIndex

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("query")
    ap.add_argument("--k", type=int, default=10)
    ap.add_argument("--model", default="hybrid", choices=MODELS)
    ap.add_argument("--index", type=Path, default=DATA_DIR / "index" / "job_index.pkl")
    ap.add_argument("--city"); ap.add_argument("--country"); ap.add_argument("--role"); ap.add_argument("--skills", nargs="*")
    ap.add_argument("--min-exp", type=float); ap.add_argument("--max-exp", type=float); ap.add_argument("--min-salary", type=float)
    ap.add_argument("--employment-type"); ap.add_argument("--sources", nargs="*"); ap.add_argument("--text-quality", nargs="*")
    ap.add_argument("--keep-duplicates", action="store_true", help="do not collapse identical title+employer+city postings");
    ap.add_argument("--route", action="store_true", help="cluster-based retrieval: search only the clusters the query is routed to (EXPERIMENTAL: can miss results)")
    ap.add_argument("--parts", type=int, default=1, help="partition the index and search the partitions in parallel")
    ap.add_argument("--executor", default="processes", choices=["sequential", "threads", "processes"])
    ap.add_argument("--compare", action="store_true"); ap.add_argument("--json", action="store_true"); ap.add_argument("--no-explain", action="store_true")
    a = ap.parse_args()
    filters = {k: v for k, v in dict(city=a.city, country=a.country, role=a.role, skills=a.skills, min_experience=a.min_exp, max_experience=a.max_exp,
                                     min_salary=a.min_salary, employment_type=a.employment_type, sources=a.sources, text_quality=a.text_quality).items() if v not in (None, [])}
    ix = InvertedIndex.load(a.index)
    eng = DistributedEngine.build(ix, DbDocStore(get_engine()), a.parts, a.executor) if a.parts > 1 else SearchEngine(ix, DbDocStore(get_engine()))
    if a.compare:
        out = eng.compare_models(a.query, k=a.k, filters=filters)
        print(json.dumps(out, indent=1, default=str) if a.json else "")
        if not a.json:
            ms = out["models"]
            print(f"{'job':44s} {'LM':>5s} {'DFR':>5s} | " + " ".join(f"{m:>9s}" for m in ms))
            for r in out["rows"]:
                print(f"{(r['title'] or '')[:30] + ' / ' + (r['company'] or '')[:11]:44s} {r.get('lm_score', ''):>5} {r.get('dfr_score', ''):>5} | " + " ".join(f"{r['ranks'].get(m, '-'):>9}" for m in ms))
            print("latency ms:", out["timings_ms"])
        sys.exit(0)
    r = eng.search(a.query, k=a.k, model=a.model, filters=filters, explain=not a.no_explain, collapse_duplicates=not a.keep_duplicates, cluster_routing=a.route)
    if a.json:
        print(json.dumps(r.to_dict(), indent=1, default=str)); sys.exit(0)
    print(f"query: {a.query!r} | model={a.model} | candidates={r.n_candidates:,} of {r.n_docs:,} | {r.timings_ms.get('total')} ms")
    if r.parsed and r.parsed.valid:
        p = r.parsed
        print(f"parsed: terms={p.terms} skills={p.skills} locations={p.locations} experience={p.experience}")
    if r.weights:
        print("weights used:", {k: round(v, 3) for k, v in r.weights.items()})
    if r.routing:
        print("routing:", r.routing)
    for w in r.warnings + ([r.error] if r.error else []):
        print("note:", w)
    for x in r.results:
        print(f"\n#{x.rank}  {x.title}  |  {x.company or '-'}  |  {x.city or '-'}  [{x.source}, {x.text_quality}]  score={x.score:.3f}" + (f"  exp={x.experience}" if x.experience else "") + (f"  salary={x.salary}" if x.salary else ""))
        if x.signals:
            print("   signals:", {k: round(v, 2) for k, v in x.signals.items() if v is not None})
        for line in x.reasons[:7]:
            print("   ", line)
        if x.passage:
            print("    passage:", re.sub(r"</?mark>", "**", re.sub(r"\s+", " ", x.passage["html"]))[:260])
    if a.parts > 1:
        eng.close()


if __name__ == "__main__":      # required: 'spawn' (Windows) re-imports this file in every worker process
    main()
