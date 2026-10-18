#!/usr/bin/env python3
"""Load MyCareersFuture (MCF) into an EXISTING Jobintel database and refresh everything that depends on it.  Idempotent: re-running changes nothing.

    python scripts/add_mcf.py                 # fetch -> ingest -> verify -> PageRank -> search index (+ passages / terms tables)
    python scripts/add_mcf.py --dry-run       # show the plan only
    python scripts/add_mcf.py --steps verify  # run selected steps (fetch, ingest, verify, pagerank, index, embeddings)
    python scripts/add_mcf.py --with-dl       # also rebuild the skill-space DL embeddings (needs data/dl/models/ae_denoising_v2.keras + ae_vocab_v2.json)

Steps
  fetch      download MCF unless the JSON is already present (env MCF_JSON or data/raw/mycareersfuture/); uses the fallback repository if the original is gone
  ingest     scripts/ingest_all.py --only mcf   - SKIPPED when the database already holds MCF rows (use --reingest to force)
  verify     compare the database with the numbers the project was originally built on (Phase 1 report): 20,298 records, 18,733 unique, 86% with salary, dates 2018-07-09..2019-06-18.
             A mismatch stops the run (exit 1): a partial or different dataset must not slip through.
  pagerank   scripts/compute_pagerank.py            index   scripts/build_index.py --write-db
After it: the market analytics (data/analytics/market.json) are NOT refreshed here - run  python scripts/run_mapreduce.py  (they were computed over the old collection)."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

STEPS = ["fetch", "ingest", "verify", "pagerank", "index", "embeddings"]
DEFAULT_STEPS = ["fetch", "ingest", "verify", "pagerank", "index"]
EXPECTED = {"records": 20298, "unique": 18733, "salary_share": (0.85, 0.87), "first_date": "2018-07-09", "last_date": "2019-06-18"}


def check_collection(stats: dict) -> list:
    """pure comparison of measured numbers with EXPECTED -> [(label, actual, expected, ok)]."""
    lo, hi = EXPECTED["salary_share"]
    share = stats["with_salary"] / stats["unique"] if stats["unique"] else 0.0
    return [("records in the database", stats["records"], EXPECTED["records"], stats["records"] == EXPECTED["records"]),
            ("unique (non-duplicate) records", stats["unique"], EXPECTED["unique"], stats["unique"] == EXPECTED["unique"]),
            ("share of unique records with salary", round(share, 3), f"{lo}-{hi}", lo <= share <= hi),
            ("first posting date", stats["first_date"], EXPECTED["first_date"], str(stats["first_date"]) == EXPECTED["first_date"]),
            ("last posting date", stats["last_date"], EXPECTED["last_date"], str(stats["last_date"]) == EXPECTED["last_date"])]


def mcf_stats(engine) -> dict:
    from sqlalchemy import text
    q = text("""SELECT COUNT(*) records, SUM(j.duplicate_of IS NULL AND j.is_fraudulent = 0) uniq,
                SUM(j.duplicate_of IS NULL AND j.is_fraudulent = 0 AND j.salary_min IS NOT NULL) sal, MIN(j.posting_date) d0, MAX(j.posting_date) d1
                FROM jobs j JOIN crawler_sources s USING(source_id) WHERE s.name = 'mcf'""")
    with engine.connect() as c:
        r = c.execute(q).one()
    return {"records": int(r[0] or 0), "unique": int(r[1] or 0), "with_salary": int(r[2] or 0), "first_date": r[3], "last_date": r[4]}


def index_tables_complete(passage_jobs: int, expected_jobs: int) -> bool:
    """build_index --write-db writes job_passages for every indexable FULL-TEXT job; an interrupted write (e.g. a time limit) leaves fewer - detected here."""
    return expected_jobs > 0 and passage_jobs == expected_jobs


def passage_counts(engine) -> tuple:
    from sqlalchemy import text
    with engine.connect() as c:
        have = c.execute(text("SELECT COUNT(DISTINCT job_id) FROM job_passages")).scalar() or 0
        want = c.execute(text("SELECT COUNT(*) FROM jobs WHERE duplicate_of IS NULL AND is_fraudulent = 0 AND text_quality = 'full'")).scalar() or 0
    return int(have), int(want)


def mcf_json_path() -> Path:
    from jobintel.config import RAW_DIR
    return Path(os.environ.get("MCF_JSON") or RAW_DIR / "mycareersfuture" / "mycareersfuture.json")


def build_plan(steps, reingest: bool, have_json: bool, rows_in_db: int) -> list:
    """[(step, will_run, reason)] - pure, so it can be tested without a database."""
    plan = []
    for s in steps:
        if s == "fetch":
            plan.append((s, not have_json, "MCF json already present" if have_json else "download MCF (fallback repository if needed)"))
        elif s == "ingest":
            skip = rows_in_db > 0 and not reingest
            plan.append((s, not skip, f"database already holds {rows_in_db:,} MCF rows (use --reingest to force)" if skip else "ingest MCF"))
        else:
            plan.append((s, True, {"verify": "compare with the original Phase 1 numbers", "pagerank": "recompute PageRank", "index": "rebuild the search index (+ passage / term tables)",
                                    "embeddings": "rebuild the skill-space DL embeddings for all sources"}[s]))
    return plan


def run(script: str, *args: str) -> None:
    cmd = [sys.executable, str(ROOT / "scripts" / script), *args]
    print("  $", " ".join(Path(c).name if c.startswith(str(ROOT)) else c for c in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--steps", nargs="+", choices=STEPS)
    ap.add_argument("--with-dl", action="store_true")
    ap.add_argument("--reingest", action="store_true")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()
    steps = a.steps or (DEFAULT_STEPS + (["embeddings"] if a.with_dl else []))
    from jobintel.db.session import get_engine
    eng = get_engine()
    st = mcf_stats(eng)
    plan = build_plan(steps, a.reingest, mcf_json_path().exists(), st["records"])
    print("PLAN")
    for s, go, why in plan:
        print(f"  {'RUN ' if go else 'skip'}  {s:10s} {why}")
    if a.dry_run:
        sys.exit(0)
    for s, go, _ in plan:
        print(f"\n== {s}")
        if not go:
            continue
        if s == "fetch":
            run("fetch_datasets.py", "--only", "mcf")
        elif s == "ingest":
            run("ingest_all.py", "--only", "mcf")
        elif s == "verify":
            checks = check_collection(mcf_stats(eng))
            for label, actual, exp, ok in checks:
                print(f"  {'OK  ' if ok else 'FAIL'}  {label:40s} {actual}  (expected {exp})")
            if not all(c[3] for c in checks):
                sys.exit("\nVERIFY FAILED: the MCF data in the database is not the dataset the project was built on - stopping before the index / embeddings are rebuilt.")
        elif s == "pagerank":
            run("compute_pagerank.py")
        elif s == "index":
            run("build_index.py", "--write-db")
            have, want = passage_counts(eng)
            print(f"  {'OK  ' if index_tables_complete(have, want) else 'FAIL'}  job_passages cover {have:,} of {want:,} full-text jobs")
            if not index_tables_complete(have, want):
                sys.exit("\nINDEX TABLES INCOMPLETE: the write was interrupted or failed - re-run  python scripts/add_mcf.py --steps index")
        elif s == "embeddings":
            run("build_dl_embeddings_v2.py")
    print("\nDONE.  Remember: data/analytics/market.json (web 'Career Insights') still reflects the previous collection -> python scripts/run_mapreduce.py")
