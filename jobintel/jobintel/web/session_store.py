"""Short-lived server-side store for things that are too big for the session cookie (an analysed resume, a recommendation request).

The browser only holds a random token.  Entries expire after ``ttl`` seconds and the store keeps at most ``max_items`` entries (oldest are dropped first).  Nothing is written to disk:
an uploaded resume of a visitor who is not logged in disappears when the entry expires or the server restarts."""
from __future__ import annotations

import secrets
import threading
import time
from collections import OrderedDict
from typing import Any, Optional


class EphemeralStore:
    def __init__(self, ttl: float = 7200.0, max_items: int = 200):
        self.ttl, self.max_items = ttl, max_items
        self._d: "OrderedDict[str, tuple[float, Any]]" = OrderedDict()
        self._lock = threading.Lock()

    def put(self, value: Any) -> str:
        token = secrets.token_urlsafe(18)
        now = time.time()
        with self._lock:
            self._purge(now)
            self._d[token] = (now, value)
            while len(self._d) > self.max_items:
                self._d.popitem(last=False)
        return token

    def get(self, token: Optional[str]) -> Optional[Any]:
        if not token:
            return None
        now = time.time()
        with self._lock:
            item = self._d.get(token)
            if item is None:
                return None
            if now - item[0] > self.ttl:
                del self._d[token]
                return None
            return item[1]

    def drop(self, token: Optional[str]) -> None:
        if token:
            with self._lock:
                self._d.pop(token, None)

    def _purge(self, now: float) -> None:
        for k in [k for k, (t, _) in self._d.items() if now - t > self.ttl]:
            del self._d[k]

    def __len__(self) -> int:
        return len(self._d)
