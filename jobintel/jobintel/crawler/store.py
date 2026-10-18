"""Store crawler output: jobs -> `jobs` (through the normal ingestion pipeline) and observed hyperlinks -> `job_links`."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

from sqlalchemy import delete, insert, select, update
from sqlalchemy.engine import Engine

from jobintel.db.models import Company, CrawlerSource, Job, JobLink
from jobintel.ingest.common import normalize_company


def store_edges(engine: Engine, crawl_dir: Path, source: str = "fixture_crawl") -> Dict:
    d = Path(crawl_dir)
    comp_lines = [json.loads(l) for l in (d / "companies.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()] if (d / "companies.jsonl").exists() else []
    edges = [json.loads(l) for l in (d / "edges.jsonl").read_text(encoding="utf-8").splitlines() if l.strip()]
    with engine.begin() as c:
        sid = c.execute(select(CrawlerSource.source_id).where(CrawlerSource.name == source)).scalar_one()
        job_by_url = {u: i for i, u in c.execute(select(Job.job_id, Job.url).where(Job.source_id == sid))}
        comp_ids = {n: i for i, n in c.execute(select(Company.company_id, Company.normalized_name))}
        page2company: Dict[str, int] = {}
        for co in comp_lines:
            cid = comp_ids.get(normalize_company(co["name"] or ""))
            if cid:
                page2company[co["url"]] = cid
                if co.get("careers_url"):
                    page2company[co["careers_url"]] = cid
                    c.execute(update(Company).where(Company.company_id == cid).values(career_page_url=co["careers_url"][:1000]))
        c.execute(delete(JobLink).where(JobLink.is_constructed == False))   # noqa: E712  (crawler-owned rows are rebuilt)
        rows, skipped = [], 0
        for e in edges:
            t = e["type"]
            if t == "careers_to_job" and e["src"] in page2company and e["dst"] in job_by_url:
                rows.append(dict(link_type="company_to_job", src_company_id=page2company[e["src"]], dst_job_id=job_by_url[e["dst"]], is_constructed=False, weight=1.0))
            elif t == "job_to_company" and e["src"] in job_by_url and e["dst"] in page2company:
                rows.append(dict(link_type="job_to_company", src_job_id=job_by_url[e["src"]], dst_company_id=page2company[e["dst"]], is_constructed=False, weight=1.0))
            elif t == "related_job" and e["src"] in job_by_url and e["dst"] in job_by_url:
                rows.append(dict(link_type="related_job", src_job_id=job_by_url[e["src"]], dst_job_id=job_by_url[e["dst"]], is_constructed=False, weight=1.0))
            elif t != "company_to_careers":     # the career page is a virtual node of the graph, nothing to store
                skipped += 1
        full = {"src_company_id": None, "src_job_id": None, "dst_company_id": None, "dst_job_id": None}
        rows = [{**full, **r} for r in rows]          # bulk insert needs identical keys in every row
        for i in range(0, len(rows), 2000):
            c.execute(insert(JobLink), rows[i:i + 2000])
    kinds: Dict[str, int] = {}
    for r in rows:
        kinds[r["link_type"]] = kinds.get(r["link_type"], 0) + 1
    return dict(edges_in_file=len(edges), stored=len(rows), by_type=kinds, skipped_unresolved=skipped)
