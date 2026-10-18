#!/usr/bin/env python3
"""Re-apply the CURRENT title-keyword cluster rules (jobintel/ingest/common.py) to every job already in the database,
without re-ingesting. Rebuild the index afterwards (scripts/build_index.py)."""
import sys
from collections import Counter
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from sqlalchemy import text
from jobintel.db.session import get_engine
from jobintel.ingest.common import cluster_for_title

e = get_engine()
with e.connect() as c:
    rows = c.execute(text("SELECT job_id, title FROM jobs")).all()
groups = {}
for jid, title in rows:
    groups.setdefault(cluster_for_title(title), []).append(jid)
with e.begin() as c:
    for cl, ids in groups.items():
        for i in range(0, len(ids), 2000):
            c.execute(text("UPDATE jobs SET cluster = :c WHERE job_id IN :ids").bindparams(__import__("sqlalchemy").bindparam("ids", expanding=True)), {"c": cl, "ids": ids[i:i + 2000]})
print("clusters:", dict(Counter({k: len(v) for k, v in groups.items()}).most_common()))
