"""Scrapy implementation of the same crawler (the syllabus asks for BOTH Scrapy and BeautifulSoup).

Differences from the BeautifulSoup pipeline are deliberate: Scrapy's own engine does scheduling, the duplicate-URL
filter, robots.txt compliance (ROBOTSTXT_OBEY), depth limits and throttling; selection uses Scrapy CSS/XPath selectors;
item pipelines do de-duplication and storage (JSON Lines).
Run with   python -m jobintel.crawler.run_scrapy --start http://127.0.0.1:8765/ --out data/crawl_scrapy"""
from __future__ import annotations

import json
from pathlib import Path
from urllib.parse import urlsplit

import scrapy
from scrapy.exceptions import DropItem

from jobintel.crawler.frontier import canonicalize
from jobintel.crawler.normalize import to_record_dict
from jobintel.crawler.parser import _from_jsonld
from jobintel.crawler.pipeline import DuplicateFilter, edge_type

_KIND = {"job-link": "job", "related-link": "related_job", "company-link": "company", "careers-link": "careers", "next-page": "next"}


class FixtureJobsSpider(scrapy.Spider):
    name = "fixture_jobs"

    def __init__(self, start_url: str = "http://127.0.0.1:8765/", **kw):
        super().__init__(**kw)
        self.start_urls = [start_url]
        self.allowed_domains = [urlsplit(start_url).hostname]

    def parse(self, response):
        ptype = response.css("body::attr(data-page-type)").get() or "other"
        for a in response.css("a[href]"):
            href = a.attrib["href"]
            kind = next((v for c, v in _KIND.items() if c in (a.attrib.get("class") or "").split()), "other")
            et = edge_type(ptype, kind)
            if et:
                yield {"_kind": "edge", "src": canonicalize(response.url), "dst": canonicalize(response.urljoin(href)), "type": et}
            yield response.follow(href, callback=self.parse)         # dupe filter + robots + offsite handled by Scrapy
        if ptype == "company":
            careers = response.css("a.careers-link::attr(href)").get()
            yield {"_kind": "company", "name": response.css("h1::text").get(), "url": canonicalize(response.url),
                   "careers_url": canonicalize(response.urljoin(careers)) if careers else None}
        if ptype == "job":
            ld = response.css('script[type="application/ld+json"]::text').get()
            raw = None
            if ld:
                try:
                    data = json.loads(ld)
                    raw = _from_jsonld(data) if data.get("@type") == "JobPosting" else None
                except ValueError:
                    raw = None
            if raw is None:
                raw = dict(title=response.css("h1.job-title::text").get(), company=response.css(".company-name::text").get(),
                           location=response.css(".job-location::text").get(), description=response.css("div.job-description").xpath("string(.)").get() and "".join(response.css("div.job-description > *").getall()),
                           skills=response.css("ul.skills li::text").getall(), experience_text=response.css(".job-experience::text").get(),
                           education=response.css(".job-education::text").get(), employment_type=response.css(".job-type::text").get(),
                           category=response.css(".job-category::text").get(), date_posted=response.css("time.posted::attr(datetime)").get())
            if raw.get("title"):
                raw["url"] = canonicalize(response.url)
                yield {"_kind": "job", **to_record_dict(raw)}


class DedupePipeline:
    def __init__(self):
        self.dup = DuplicateFilter()
        self.duplicates = 0

    def process_item(self, item, spider):
        if item["_kind"] == "job" and self.dup.check(item):
            self.duplicates += 1
            raise DropItem("duplicate job")
        return item


class JsonLinesPipeline:
    FILES = {"job": "jobs.jsonl", "edge": "edges.jsonl", "company": "companies.jsonl"}

    @classmethod
    def from_crawler(cls, crawler):
        p = cls()
        p.out = Path(crawler.settings.get("JOBINTEL_OUT_DIR", "data/crawl_scrapy"))
        return p

    def open_spider(self, spider):
        self.out.mkdir(parents=True, exist_ok=True)
        self.fh = {k: open(self.out / v, "w", encoding="utf-8") for k, v in self.FILES.items()}

    def close_spider(self, spider):
        for f in self.fh.values():
            f.close()

    def process_item(self, item, spider):
        kind = item.pop("_kind")
        self.fh[kind].write(json.dumps(item, ensure_ascii=False) + "\n")
        return item
