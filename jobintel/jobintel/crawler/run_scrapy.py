"""Programmatic Scrapy runner:  python -m jobintel.crawler.run_scrapy --start URL --out DIR"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from scrapy.crawler import CrawlerProcess

from jobintel.crawler.fetcher import USER_AGENT
from jobintel.crawler.scrapy_spider import FixtureJobsSpider


def run(start: str, out: Path, max_pages: int = 2000) -> dict:
    settings = {
        "BOT_NAME": "jobintel", "USER_AGENT": USER_AGENT, "ROBOTSTXT_OBEY": True, "DOWNLOAD_DELAY": 0, "CONCURRENT_REQUESTS": 8,
        "DEPTH_LIMIT": 12, "CLOSESPIDER_PAGECOUNT": max_pages, "LOG_LEVEL": "WARNING", "TELNETCONSOLE_ENABLED": False,
        "RETRY_TIMES": 2, "REQUEST_FINGERPRINTER_IMPLEMENTATION": "2.7", "JOBINTEL_OUT_DIR": str(out),
        "ITEM_PIPELINES": {"jobintel.crawler.scrapy_spider.DedupePipeline": 100, "jobintel.crawler.scrapy_spider.JsonLinesPipeline": 300},
    }
    process = CrawlerProcess(settings)
    crawler = process.create_crawler(FixtureJobsSpider)
    process.crawl(crawler, start_url=start)
    process.start()
    st = crawler.stats.get_stats()
    summary = dict(engine="scrapy", pages_fetched=st.get("downloader/response_status_count/200", 0), robots_blocked=st.get("robotstxt/forbidden", 0),
                   offsite_filtered=st.get("offsite/filtered", 0), duplicate_requests_filtered=st.get("dupefilter/filtered", 0),
                   items_dropped_duplicate=st.get("item_dropped_count", 0), items_scraped=st.get("item_scraped_count", 0))
    (out / "crawl_stats.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", required=True)
    ap.add_argument("--out", default="data/crawl_scrapy")
    ap.add_argument("--max-pages", type=int, default=2000)
    a = ap.parse_args()
    print(json.dumps(run(a.start, Path(a.out), a.max_pages), indent=2))
