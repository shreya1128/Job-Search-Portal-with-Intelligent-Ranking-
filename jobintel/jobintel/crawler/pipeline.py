"""Crawler pipeline (the standard architecture, one box per step):

  URL seed -> URL FRONTIER -> FETCHER (robots.txt, politeness) -> PARSER (BeautifulSoup) -> DOCUMENT EXTRACTOR
           -> DUPLICATE DETECTION (exact hash + SimHash) -> STORAGE (jsonl) -> (ingest) -> INDEX
Besides jobs it records the observed hyperlinks (edges) that become the PageRank graph."""
from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Dict, List, Optional
from urllib.parse import urlsplit

from jobintel.crawler.fetcher import Fetcher
from jobintel.crawler.frontier import URLFrontier, canonicalize
from jobintel.crawler.normalize import to_record_dict
from jobintel.crawler.parser import SiteProfile, parse_page
from jobintel.processing.dedupe import NearDuplicateIndex, content_hash, normalize_for_hash, simhash
from jobintel.processing.text import index_tokens


@dataclass
class CrawlStats:
    engine: str = "bs4"
    pages_fetched: int = 0
    blocked_by_robots: int = 0
    errors: int = 0
    jobs_extracted: int = 0
    duplicates_skipped: int = 0
    edges: int = 0
    by_page_type: Dict[str, int] = field(default_factory=dict)
    blocked_urls: List[str] = field(default_factory=list)
    duplicate_urls: List[str] = field(default_factory=list)
    frontier_rejected: Dict[str, int] = field(default_factory=dict)
    seconds: float = 0.0


def edge_type(page_type: str, link_kind: str) -> Optional[str]:
    return {("careers", "job"): "careers_to_job", ("company", "careers"): "company_to_careers",
            ("job", "company"): "job_to_company", ("job", "related_job"): "related_job"}.get((page_type, link_kind))


class DuplicateFilter:
    """Same title + company + PLACE + (identical or near-identical text) = duplicate.  Same boilerplate for another city is a different listing."""

    def __init__(self):
        self.exact: Dict[str, str] = {}
        self.near = NearDuplicateIndex()
        self.meta: Dict[int, tuple] = {}
        self._n = 0

    def check(self, rec: Dict) -> Optional[str]:
        text = rec.get("description") or " ".join(rec.get("skills") or [])
        place = rec.get("city") or rec.get("location_raw")
        key = (normalize_for_hash(rec["title"]), normalize_for_hash(rec.get("company")), normalize_for_hash(place))
        h = content_hash(rec["title"], rec.get("company"), place, text)
        if h in self.exact:
            return self.exact[h]
        toks = index_tokens(text)
        if len(toks) >= 30:
            sim = simhash(toks)
            for c in self.near.candidates(sim):
                if self.meta[c][0] == key:
                    return self.meta[c][1]
            self._n += 1
            self.near.add(self._n, sim)
            self.meta[self._n] = (key, rec["url"])
        self.exact[h] = rec["url"]
        return None


def crawl(seeds: List[str], out_dir: Path, *, max_pages: int = 2000, max_depth: int = 10, delay: float = 0.0, respect_robots: bool = True,
          profile: Optional[SiteProfile] = None, source: str = "fixture_crawl", fetcher: Optional[Fetcher] = None) -> CrawlStats:
    t0 = time.time()
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    fetcher = fetcher or Fetcher(delay=delay, respect_robots=respect_robots)
    frontier = URLFrontier({urlsplit(s).netloc for s in seeds}, max_depth=max_depth, max_pages=max_pages)
    for s in seeds:
        frontier.add(s, 0)
    stats, dup = CrawlStats(engine="bs4"), DuplicateFilter()
    companies: Dict[str, Dict] = {}
    with open(out / "jobs.jsonl", "w", encoding="utf-8") as fj, open(out / "edges.jsonl", "w", encoding="utf-8") as fe:
        while True:
            nxt = frontier.pop()
            if nxt is None:
                break
            url, depth, parent = nxt
            res = fetcher.fetch(url)
            if res.blocked_by_robots:
                stats.blocked_by_robots += 1
                stats.blocked_urls.append(url)
                continue
            if res.html is None or (res.status and res.status >= 400):
                stats.errors += 1
                continue
            stats.pages_fetched += 1
            page = parse_page(res.html, url, profile)
            stats.by_page_type[page.page_type] = stats.by_page_type.get(page.page_type, 0) + 1
            for link in page.links:
                et = edge_type(page.page_type, link.kind)
                if et:
                    fe.write(json.dumps({"src": canonicalize(url), "dst": canonicalize(link.url), "type": et}) + "\n")
                    stats.edges += 1
                frontier.add(link.url, depth + 1, url)
            if page.company:
                careers = next((canonicalize(l.url) for l in page.links if l.kind == "careers"), None)
                companies[canonicalize(url)] = dict(name=page.company["name"], url=canonicalize(url), careers_url=careers)
            if page.job:
                rec = to_record_dict(page.job, source)
                rec["url"] = canonicalize(url)
                original = dup.check(rec)
                if original:
                    stats.duplicates_skipped += 1
                    stats.duplicate_urls.append(rec["url"])
                else:
                    stats.jobs_extracted += 1
                    fj.write(json.dumps(rec, ensure_ascii=False) + "\n")
    (out / "companies.jsonl").write_text("\n".join(json.dumps(c) for c in companies.values()), encoding="utf-8")
    stats.frontier_rejected = dict(frontier.rejected)
    stats.seconds = round(time.time() - t0, 2)
    (out / "crawl_stats.json").write_text(json.dumps(asdict(stats), indent=2), encoding="utf-8")
    return stats
