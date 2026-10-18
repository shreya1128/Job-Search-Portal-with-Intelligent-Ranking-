#!/usr/bin/env python3
"""Run a crawler.   Default: serve the local fixture site and crawl it (no internet needed).

  python scripts/run_crawler.py --engine bs4              # BeautifulSoup pipeline
  python scripts/run_crawler.py --engine scrapy           # Scrapy spider
  python scripts/run_crawler.py --engine bs4 --ingest     # ...and load jobs + link graph into MySQL
  python scripts/run_crawler.py --engine bs4 --seed https://example-job-site.test/   # your own PERMITTED source
Only crawl sites whose robots.txt / terms allow it - the fetcher enforces robots.txt, but terms of use are on you."""
import argparse, json, subprocess, sys
from dataclasses import asdict
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import DATA_DIR, FIXTURE_DIR, SAMPLE_DIR, ROOT
from jobintel.crawler.fixture_site import build_site, serve
from jobintel.crawler.pipeline import crawl
from jobintel.sample_data import generate, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--engine", choices=["bs4", "scrapy"], default="bs4")
ap.add_argument("--seed", help="start URL (default: the local fixture site)")
ap.add_argument("--out", type=Path, default=None)
ap.add_argument("--max-pages", type=int, default=2000)
ap.add_argument("--delay", type=float, default=0.0, help="seconds between requests to one host (use >=1 for real sites)")
ap.add_argument("--ingest", action="store_true", help="load the crawl output into MySQL")
a = ap.parse_args()
out = a.out or DATA_DIR / f"crawl_{a.engine}"

srv = None
seed = a.seed
if not seed:
    if not (FIXTURE_DIR / "index.html").exists():
        jobs = generate(300)
        write_jsonl(SAMPLE_DIR / "sample_jobs.jsonl", jobs)
        build_site(FIXTURE_DIR, jobs)
    srv, base = serve(FIXTURE_DIR)
    seed = base + "/index.html"
    print("fixture site at", base)
try:
    if a.engine == "bs4":
        st = crawl([seed], out, max_pages=a.max_pages, delay=a.delay)
        d = asdict(st)
        print(json.dumps({k: v for k, v in d.items() if k not in ("duplicate_urls",)}, indent=2))
    else:
        r = subprocess.run([sys.executable, "-m", "jobintel.crawler.run_scrapy", "--start", seed, "--out", str(out), "--max-pages", str(a.max_pages)],
                           cwd=ROOT, capture_output=True, text=True)
        print(r.stdout or r.stderr[-1500:])
finally:
    if srv:
        srv.shutdown()
if a.ingest:
    from jobintel.crawler.store import store_edges
    from jobintel.db.models import Base
    from jobintel.db.session import get_engine
    from jobintel.ingest.load_db import Ingestor
    from jobintel.ingest.loaders import load_jsonl
    e = get_engine()
    Base.metadata.create_all(e)
    print("ingest:", Ingestor(e, verbose=False).ingest("fixture_crawl", load_jsonl(out / "jobs.jsonl", "fixture_crawl")))
    print("edges :", store_edges(e, out))
