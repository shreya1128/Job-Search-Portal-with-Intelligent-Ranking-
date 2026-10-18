"""URL frontier (crawler architecture:  URL seed -> FRONTIER -> fetcher -> parser -> ...).
Breadth-first queue with canonicalisation, host allow-list, depth limit and a global page budget."""
from __future__ import annotations

from collections import deque
from typing import Deque, Iterable, Optional, Set, Tuple
from urllib.parse import parse_qsl, urlencode, urlsplit, urlunsplit

_TRACKING = ("utm_", "ref", "fbclid", "gclid")


def canonicalize(url: str) -> str:
    p = urlsplit(url.strip())
    host = (p.hostname or "").lower()
    port = f":{p.port}" if p.port and not ((p.scheme == "http" and p.port == 80) or (p.scheme == "https" and p.port == 443)) else ""
    path = p.path or "/"
    while "//" in path:
        path = path.replace("//", "/")
    if len(path) > 1 and path.endswith("/"):
        path = path[:-1]
    q = [(k, v) for k, v in parse_qsl(p.query) if not k.lower().startswith(_TRACKING)]
    return urlunsplit((p.scheme.lower(), host + port, path, urlencode(sorted(q)), ""))  # fragment dropped


class URLFrontier:
    def __init__(self, allowed_hosts: Optional[Iterable[str]] = None, max_depth: int = 8, max_pages: int = 1000):
        self.allowed_hosts = {h.lower() for h in allowed_hosts} if allowed_hosts else None
        self.max_depth, self.max_pages = max_depth, max_pages
        self._queue: Deque[Tuple[str, int, Optional[str]]] = deque()
        self.seen: Set[str] = set()
        self.dispatched = 0
        self.rejected = {"offsite": 0, "too_deep": 0, "seen": 0, "bad_scheme": 0}

    def add(self, url: str, depth: int = 0, parent: Optional[str] = None) -> bool:
        p = urlsplit(url)
        if p.scheme not in ("http", "https"):
            self.rejected["bad_scheme"] += 1
            return False
        c = canonicalize(url)
        if self.allowed_hosts is not None and urlsplit(c).netloc not in self.allowed_hosts:
            self.rejected["offsite"] += 1
            return False
        if depth > self.max_depth:
            self.rejected["too_deep"] += 1
            return False
        if c in self.seen:
            self.rejected["seen"] += 1
            return False
        self.seen.add(c)
        self._queue.append((c, depth, parent))
        return True

    def pop(self) -> Optional[Tuple[str, int, Optional[str]]]:
        if not self._queue or self.dispatched >= self.max_pages:
            return None
        self.dispatched += 1
        return self._queue.popleft()

    def __len__(self) -> int:
        return len(self._queue)
