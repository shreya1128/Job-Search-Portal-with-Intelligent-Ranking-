import json
import subprocess
import sys
from pathlib import Path

import pytest
from sqlalchemy import func, select, text

from jobintel.crawler.fixture_site import build_site, serve
from jobintel.crawler.frontier import URLFrontier, canonicalize
from jobintel.crawler.parser import parse_page
from jobintel.crawler.pipeline import crawl
from jobintel.crawler.store import store_edges
from jobintel.db.models import JobLink
from jobintel.ingest.load_db import Ingestor
from jobintel.ingest.loaders import load_jsonl
from jobintel.processing.dedupe import content_hash


@pytest.fixture(scope="module")
def site(tmp_path_factory):
    from jobintel.sample_data import generate
    d = tmp_path_factory.mktemp("site")
    jobs = generate(80, seed=3)
    info = build_site(d, jobs, n_reposts=4)
    srv, base = serve(d)
    yield dict(dir=d, base=base, srv=srv, info=info)
    srv.shutdown()


def test_frontier_canonicalisation_and_limits():
    f = URLFrontier({"h:1"}, max_depth=1, max_pages=2)
    assert f.add("http://H:1/a?utm_source=x&b=1") and not f.add("http://h:1/a?b=1#frag")          # same page once
    assert not f.add("http://other/x") and not f.add("ftp://h:1/") and not f.add("http://h:1/deep", depth=5)
    assert f.rejected == {"offsite": 1, "too_deep": 1, "seen": 1, "bad_scheme": 1}
    assert canonicalize("http://LOCALHOST:80//jobs//1/?ref=2&z=1&a=2") == "http://localhost/jobs/1?a=2&z=1"


def test_parser_jsonld_beats_css_and_links_typed():
    html = ('<html><head><script type="application/ld+json">{"@type":"JobPosting","title":"ML Engineer","hiringOrganization":{"name":"Acme"},'
            '"jobLocation":{"address":{"addressLocality":"Pune"}},"skills":"Python, SQL","description":"<p>Build</p>"}</script></head>'
            '<body data-page-type="job"><h1 class="job-title">WRONG</h1><span class="job-experience">2-4 years</span>'
            '<a class="company-link" href="/c/">Acme</a><ul class="related-jobs"><li><a class="related-link" href="/jobs/2.html">x</a></li></ul></body></html>')
    p = parse_page(html, "http://h/jobs/1.html")
    assert p.job["title"] == "ML Engineer" and p.job["skills"] == ["Python", "SQL"] and p.job["_extracted_via"] == "json-ld"
    assert p.job["experience_text"] == "2-4 years"                                           # gap filled from CSS
    assert {(l.kind, l.url) for l in p.links} == {("company", "http://h/c/"), ("related_job", "http://h/jobs/2.html")}


def test_bs4_crawl_respects_robots_and_dedupes(site, tmp_path):
    st = crawl([site["base"] + "/index.html"], tmp_path / "out")
    assert st.blocked_by_robots == 1 and any("/private/" in u for u in st.blocked_urls)
    assert not any(p.startswith("/private/") for p in site["srv"].request_log)                   # never even requested
    assert st.jobs_extracted == site["info"]["unique_jobs"] and st.duplicates_skipped == site["info"]["reposts"]
    assert st.by_page_type["job"] == site["info"]["jobs"] and st.by_page_type["company"] == site["info"]["companies"]
    rows = [json.loads(l) for l in (tmp_path / "out" / "jobs.jsonl").read_text().splitlines()]
    assert all(r["title"] and r["description"] and r["skills"] and r["source"] == "fixture_crawl" for r in rows)
    types = {json.loads(l)["type"] for l in (tmp_path / "out" / "edges.jsonl").read_text().splitlines()}
    assert types == {"careers_to_job", "company_to_careers", "job_to_company", "related_job"}


def test_crawl_output_ingested_with_link_graph(site, tmp_path, engine, extractor):
    crawl([site["base"] + "/index.html"], tmp_path / "out")
    Ingestor(engine, extractor, extractor.lex, verbose=False).ingest("fixture_crawl", load_jsonl(tmp_path / "out" / "jobs.jsonl", "fixture_crawl"))
    r = store_edges(engine, tmp_path / "out")
    assert r["by_type"]["related_job"] > 0 and r["by_type"]["company_to_job"] > 0 and r["by_type"]["job_to_company"] > 0
    with engine.connect() as c:
        assert c.execute(select(func.count()).select_from(JobLink).where(JobLink.is_constructed == True)).scalar_one() == 0   # noqa: E712  (observed, not constructed)
        assert c.execute(text("SELECT COUNT(*) FROM companies WHERE career_page_url IS NOT NULL")).scalar_one() == site["info"]["companies"]


@pytest.mark.slow
def test_scrapy_matches_bs4(site, tmp_path):
    crawl([site["base"] + "/index.html"], tmp_path / "bs4")
    r = subprocess.run([sys.executable, "-m", "jobintel.crawler.run_scrapy", "--start", site["base"] + "/index.html", "--out", str(tmp_path / "sc")],
                       capture_output=True, text=True, timeout=120)
    assert r.returncode == 0, r.stderr[-800:]
    sig = lambda p: {content_hash(x["title"], x["company"], x["city"], x["description"]) for x in map(json.loads, (p / "jobs.jsonl").read_text().splitlines())}
    assert sig(tmp_path / "bs4") == sig(tmp_path / "sc")
    assert json.loads((tmp_path / "sc" / "crawl_stats.json").read_text())["robots_blocked"] == 1
