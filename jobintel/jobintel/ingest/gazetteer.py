"""Location gazetteer for the location automaton, learned from the ingested jobs (cities with enough postings)."""
from __future__ import annotations

import json
from pathlib import Path
from typing import List

from sqlalchemy import text
from sqlalchemy.engine import Engine

# city names that are also everyday English words: never auto-added to the gazetteer
AMBIGUOUS_CITIES = {"Reading", "Mobile", "Nice", "Orange", "Normal", "Mesa", "Marina", "Jackson", "Of", "Remote Work", "Other", "Hybrid", "Washington"}


def build_gazetteer(engine: Engine, path: Path, min_postings: int = 25) -> List[str]:
    with engine.connect() as c:
        rows = c.execute(text("SELECT city, COUNT(*) n FROM jobs WHERE city IS NOT NULL AND duplicate_of IS NULL GROUP BY city HAVING COUNT(*) >= :m ORDER BY n DESC"), {"m": min_postings}).all()
    cities = [r[0] for r in rows if r[0] and len(r[0]) > 2 and r[0] not in AMBIGUOUS_CITIES and not r[0].isupper()]
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(cities, ensure_ascii=False), encoding="utf-8")
    return cities
