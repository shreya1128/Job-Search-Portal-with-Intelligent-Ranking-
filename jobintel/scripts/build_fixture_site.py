#!/usr/bin/env python3
"""Generate the synthetic sample jobs (if missing) and the static fixture website the crawlers run against."""
import argparse, json, sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from jobintel.config import FIXTURE_DIR, SAMPLE_DIR
from jobintel.crawler.fixture_site import build_site
from jobintel.sample_data import generate, write_jsonl

ap = argparse.ArgumentParser()
ap.add_argument("--n", type=int, default=300)
ap.add_argument("--out", type=Path, default=FIXTURE_DIR)
a = ap.parse_args()
jobs = generate(a.n)
write_jsonl(SAMPLE_DIR / "sample_jobs.jsonl", jobs)
print(json.dumps(build_site(a.out, jobs), indent=2))
