"""Polite HTTP fetcher: honours robots.txt (and Crawl-delay), sends an identifying User-Agent, throttles per host,
enforces timeouts / size limits and retries transient 5xx errors."""
from __future__ import annotations

import time
import urllib.robotparser as rp
from dataclasses import dataclass
from typing import Dict, Optional
from urllib.parse import urlsplit

import requests

USER_AGENT = "JobIntelBot/1.0 (academic project; polite crawler)"


@dataclass
class FetchResult:
    url: str
    status: Optional[int] = None
    html: Optional[str] = None
    content_type: Optional[str] = None
    error: Optional[str] = None
    blocked_by_robots: bool = False


class Fetcher:
    def __init__(self, user_agent: str = USER_AGENT, delay: float = 0.0, timeout: float = 10.0, respect_robots: bool = True,
                 max_bytes: int = 2_000_000, retries: int = 2, session: Optional[requests.Session] = None):
        self.ua, self.delay, self.timeout, self.respect_robots = user_agent, delay, timeout, respect_robots
        self.max_bytes, self.retries = max_bytes, retries
        self.session = session or requests.Session()
        self.session.headers.update({"User-Agent": user_agent, "Accept": "text/html,application/xhtml+xml"})
        self._robots: Dict[str, Optional[rp.RobotFileParser]] = {}
        self._last: Dict[str, float] = {}
        self.requests_made = 0

    # ---- robots.txt ---------------------------------------------------------------------------------------------
    def _robots_for(self, url: str) -> Optional[rp.RobotFileParser]:
        p = urlsplit(url)
        base = f"{p.scheme}://{p.netloc}"
        if base in self._robots:
            return self._robots[base]
        parser: Optional[rp.RobotFileParser] = rp.RobotFileParser()
        try:
            r = self.session.get(base + "/robots.txt", timeout=self.timeout)
            if r.status_code == 200:
                parser.parse(r.text.splitlines())
            elif r.status_code in (401, 403):
                parser.disallow_all = True
            else:                      # 404 etc.: no robots.txt -> everything allowed
                parser.allow_all = True
        except requests.RequestException:
            parser = None              # cannot read robots.txt -> be conservative
        self._robots[base] = parser
        return parser

    def allowed(self, url: str) -> bool:
        if not self.respect_robots:
            return True
        parser = self._robots_for(url)
        return bool(parser and parser.can_fetch(self.ua, url))

    def _throttle(self, url: str) -> None:
        host = urlsplit(url).netloc
        wait = self.delay
        parser = self._robots.get(f"{urlsplit(url).scheme}://{host}")
        if parser and parser.crawl_delay(self.ua):
            wait = max(wait, float(parser.crawl_delay(self.ua)))
        gap = time.time() - self._last.get(host, 0)
        if wait and gap < wait:
            time.sleep(wait - gap)
        self._last[host] = time.time()

    # ---- fetch ---------------------------------------------------------------------------------------------------
    def fetch(self, url: str) -> FetchResult:
        if not self.allowed(url):
            return FetchResult(url, blocked_by_robots=True, error="blocked by robots.txt")
        last_err = None
        for attempt in range(self.retries + 1):
            self._throttle(url)
            try:
                self.requests_made += 1
                r = self.session.get(url, timeout=self.timeout, stream=True)
                ctype = r.headers.get("Content-Type", "")
                if r.status_code >= 500 and attempt < self.retries:
                    time.sleep(0.2 * (attempt + 1))
                    continue
                if "html" not in ctype.lower() and "xml" not in ctype.lower():
                    return FetchResult(url, r.status_code, None, ctype, "non-HTML content")
                body = r.raw.read(self.max_bytes + 1, decode_content=True)
                if len(body) > self.max_bytes:
                    return FetchResult(url, r.status_code, None, ctype, "response too large")
                enc = r.encoding or "utf-8"
                return FetchResult(url, r.status_code, body.decode(enc, errors="replace"), ctype,
                                   None if r.status_code < 400 else f"HTTP {r.status_code}")
            except requests.RequestException as exc:
                last_err = str(exc)
                time.sleep(0.2 * (attempt + 1))
        return FetchResult(url, None, None, None, last_err or "request failed")
